"""embedding / reranker（encoder 家族：bge / jina / nomic 等）格式间转换测试。

复刻 tools/ 下已验证的转换逻辑并做端到端回归：
- HF→ONNX（fp16）→ MLX（MLXEmbedders 命名 + 兄弟键量化 + 朝向正确）
- MLX→MLX（HF roberta/bert 命名 → MLXEmbedders 命名 + 量化，反量化对比 HF 基准）
- ONNX→ONNX（weight-only int8，embedding 对激活量化敏感，onnxruntime 对比余弦）
- 路由：onnx->onnx / onnx->mlx / mlx->mlx / hf->mlx

端到端用例按依赖可用性跳过（pytest.importorskip）。
"""
from __future__ import annotations

import shutil

import pytest

from modconv.backend import (
    is_encoder_family, probe_backend, safetensors_keys_hf_named,
)


class _Hooks:
    def should_cancel(self):
        return False

    def on_log(self, m):
        pass

    def on_file_started(self, *a):
        pass

    def on_file_progress(self, *a):
        pass

    def on_file_finished(self, *a):
        pass


# --------------------------------------------------------------------------
# 家族 / 命名风格判定
# --------------------------------------------------------------------------
def test_is_encoder_family():
    assert is_encoder_family({"model_type": "xlm-roberta"})
    assert is_encoder_family({"model_type": "bert"})
    assert is_encoder_family({"model_type": "nomic_bert"})
    assert is_encoder_family({"architectures": ["BertModel"]})
    assert is_encoder_family({"auto_map": {"AutoModel": "x"}})  # jina：仅 auto_map
    assert not is_encoder_family({"model_type": "qwen2"})
    assert not is_encoder_family({"model_type": "llama"})
    assert not is_encoder_family({})


def test_safetensors_keys_hf_named(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    hf_dir = tmp_path / "hf_named"
    hf_dir.mkdir()
    save_file({"roberta.encoder.layer.0.attention.self.query.weight":
               np.zeros((1, 1), dtype=np.float16)},
              str(hf_dir / "model.safetensors"))
    assert safetensors_keys_hf_named(hf_dir) is True

    mlx_dir = tmp_path / "mlx_named"
    mlx_dir.mkdir()
    save_file({"encoder.layers.0.attention.query_proj.weight":
               np.zeros((1, 1), dtype=np.float16)},
              str(mlx_dir / "model.safetensors"))
    assert safetensors_keys_hf_named(mlx_dir) is False

    empty = tmp_path / "empty"
    empty.mkdir()
    assert safetensors_keys_hf_named(empty) is None


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
def _hf_encoder_dir(tmp_path, model_type="bert", num_labels=0):
    transformers = pytest.importorskip("transformers")
    d = tmp_path / f"enc_{model_type}_{num_labels}"
    d.mkdir()
    if model_type == "bert":
        cfg = transformers.BertConfig(hidden_size=32, num_hidden_layers=2,
                                      num_attention_heads=4, intermediate_size=64,
                                      vocab_size=100, max_position_embeddings=64)
        if num_labels:
            cfg.num_labels = num_labels
            transformers.AutoModelForSequenceClassification.from_config(cfg) \
                .save_pretrained(str(d))
        else:
            transformers.AutoModel.from_config(cfg).save_pretrained(str(d))
    return d


def _hf_to_onnx_fp16(hf_dir, out_dir, kind):
    pytest.importorskip("torch")
    from modconv.onnx_backend import HfToOnnxBackend

    out_dir.mkdir(parents=True, exist_ok=True)
    r = HfToOnnxBackend(hf_dir, out_dir, quant_mode="fp16", model_kind=kind).run(_Hooks())
    assert r.success, r.message
    onnx_file = next(out_dir.glob("*.onnx"))
    shutil.copy(hf_dir / "config.json", out_dir / "config.json")
    return onnx_file


def test_routing_encoder(tmp_path):
    hf = _hf_encoder_dir(tmp_path, num_labels=0)
    p = probe_backend(hf, target="mlx")
    assert p.name == "mlx_libs" and p.task == "hf->mlx"

    # 已量化（config 有 quantization）→ mlx->mlx
    qdir = tmp_path / "quant_mlx"
    qdir.mkdir()
    (qdir / "config.json").write_text(
        '{"model_type": "xlm-roberta", "quantization": {"bits": 4, "group_size": 64}}')
    p2 = probe_backend(qdir, target="mlx")
    assert p2.name == "mlx_rekey" and p2.task == "mlx->mlx"

    # ONNX 文件 → onnx->mlx / onnx->onnx
    onx = tmp_path / "model.onnx"
    onx.write_bytes(b"")
    p3 = probe_backend(onx, target="mlx")
    assert p3.task == "onnx->mlx"
    p4 = probe_backend(onx, target="onnx")
    assert p4.task == "onnx->onnx"


# --------------------------------------------------------------------------
# 端到端
# --------------------------------------------------------------------------
@pytest.mark.parametrize("qmode", ["fp16", "int4"])
def test_hf_onnx_to_mlx_encoder(tmp_path, qmode):
    """HF→ONNX→MLX：MLXEmbedders 命名 + [out,in] 朝向 + 兄弟键量化。"""
    pytest.importorskip("mlx")
    import mlx.core as mx

    hf = _hf_encoder_dir(tmp_path)
    onx_dir = tmp_path / "onx"
    onnx_file = _hf_to_onnx_fp16(hf, onx_dir, "embedding")

    from modconv.mlx_backend import OnnxToMlxBackend
    out = tmp_path / f"mlx_{qmode}"
    r = OnnxToMlxBackend(onnx_file, out, quant_mode=qmode).run(_Hooks())
    assert r.success, r.message

    raw = mx.load(str(out / "model.safetensors"))
    # MLXEmbedders 命名
    assert "encoder.layers.0.attention.query_proj.weight" in raw
    assert "encoder.layers.0.linear1.weight" in raw
    # 朝向：反量化后 linear1 = [intermediate=64, hidden=32]（HF [out,in]）
    # （int4 下 .weight 是打包 uint32，形状 [64, 32*4/32]=[64,4] 属正常）
    if qmode == "fp16":
        assert raw["encoder.layers.0.linear1.weight"].shape == (64, 32)
    # embeddings.norm 存在（占位或真实值），无 pooler
    assert "embeddings.norm.weight" in raw
    assert not any("pooler" in k for k in raw)
    if qmode == "int4":
        assert raw["encoder.layers.0.linear1.weight"].dtype == mx.uint32
        base = "encoder.layers.0.linear1"
        assert f"{base}.scales" in raw and f"{base}.biases" in raw  # 兄弟键
        assert '"quantization"' in (out / "config.json").read_text()


def test_mlx_to_mlx_rekey(tmp_path):
    """HF 命名 MLX → MLXEmbedders 命名 + int4，反量化对比 HF 基准（朝向/命名验证）。"""
    pytest.importorskip("mlx")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import save_file
    from modconv.mlx_rekey_backend import MlxToMlxBackend

    hf = _hf_encoder_dir(tmp_path)
    model = pytest.importorskip("transformers").AutoModel.from_pretrained(str(hf))
    hf_sd = {k: v.detach().numpy().astype(np.float16) for k, v in model.state_dict().items()}
    src = tmp_path / "hf_named_mlx"
    src.mkdir()
    save_file(hf_sd, str(src / "model.safetensors"))
    shutil.copy(hf / "config.json", src / "config.json")

    out = tmp_path / "mlx_rekey"
    r = MlxToMlxBackend(src, out, quant_mode="int4").run(_Hooks())
    assert r.success, r.message
    raw = mx.load(str(out / "model.safetensors"))
    assert "encoder.layers.0.attention.query_proj.scales" in raw
    assert raw["encoder.layers.0.linear1.weight"].dtype == mx.uint32
    assert not any("pooler" in k for k in raw)

    # 反量化 vs HF 基准（捕捉朝向/命名错误——错则余弦暴跌）
    # hidden=32 不整除 64，量化时 group_size 回退为 32
    for hfk, mk in (("encoder.layer.0.intermediate.dense.weight",
                     "encoder.layers.0.linear1.weight"),
                    ("encoder.layer.0.attention.self.query.weight",
                     "encoder.layers.0.attention.query_proj.weight")):
        base = mk.rsplit(".weight", 1)[0]
        dq = np.array(mx.dequantize(raw[mk], raw[base + ".scales"],
                                    raw[base + ".biases"], 32, 4,
                                    "affine").astype(mx.float32))
        ref = hf_sd[hfk].astype(np.float32)
        cos = float((dq * ref).sum() / (np.linalg.norm(dq) * np.linalg.norm(ref)))
        assert cos > 0.98, f"{mk} 反量化余弦 {cos}"


def test_onnx_to_onnx_int8(tmp_path):
    """ONNX(fp16) → ONNX(int8 weight-only)：onnxruntime 对比 embedding 余弦。"""
    pytest.importorskip("onnxruntime")
    import numpy as np
    import onnxruntime as ort
    from modconv.onnx_backend import OnnxToOnnxBackend

    hf = _hf_encoder_dir(tmp_path)
    onx_dir = tmp_path / "onx"
    onnx_file = _hf_to_onnx_fp16(hf, onx_dir, "embedding")

    out = tmp_path / "int8"
    r = OnnxToOnnxBackend(onnx_file, out, quant_mode="int8").run(_Hooks())
    assert r.success, r.message
    int8_file = out / "model.onnx"
    assert int8_file.is_file()

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 100, size=(2, 16)).astype(np.int64)
    s1 = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(int8_file), providers=["CPUExecutionProvider"])
    e1 = s1.run(None, {"input_ids": ids})[0].astype(np.float32)
    e2 = s2.run(None, {"input_ids": ids})[0].astype(np.float32)

    def emb(x):  # mean pooling + L2（embedding 口径）
        p = x.mean(axis=1)
        return p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-9)

    cos = float((emb(e1) * emb(e2)).sum(axis=1).mean())
    assert cos > 0.99, f"int8 与 fp16 源余弦 {cos}"


def test_reranker_classifier_head(tmp_path):
    """XLM-R reranker：ONNX→MLX 保留 classifier.dense/out_proj，num_labels=1。"""
    pytest.importorskip("mlx")
    import mlx.core as mx

    hf = _hf_encoder_dir(tmp_path, model_type="bert", num_labels=1)
    onx_dir = tmp_path / "onx"
    onnx_file = _hf_to_onnx_fp16(hf, onx_dir, "reranker")

    from modconv.mlx_backend import OnnxToMlxBackend
    out = tmp_path / "mlx_reranker"
    r = OnnxToMlxBackend(onnx_file, out, quant_mode="fp16").run(_Hooks())
    assert r.success, r.message
    raw = mx.load(str(out / "model.safetensors"))
    assert "classifier.dense.weight" in raw
    assert "classifier.out_proj.weight" in raw
    assert '"num_labels": 1' in (out / "config.json").read_text()
