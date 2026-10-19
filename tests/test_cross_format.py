"""跨格式互转测试：MLX→ONNX / MLX→GGUF / GGUF→MLX / GGUF→ONNX，
以及 embedding（encoder）在 ONNX / MLX 间的互转与 HF→三种格式。

实现方式：把 MLX / GGUF 源归一化为临时 HF 目录（source_normalize），再复用
已有的 HF→{ONNX,MLX,GGUF} 后端。全部用例用 tiny 模型实测。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from modconv.converter import ConversionHooks


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


def _tiny_qwen_with_tokenizer(tmp_path: Path) -> Path:
    pytest.importorskip("transformers")
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    d = tmp_path / "qwen"
    d.mkdir()
    cfg = transformers.Qwen2Config(hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=300,
                                   max_position_embeddings=64,
                                   tie_word_embeddings=True)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(str(d))
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(
        ["hello world this is a tiny test of the model tokenization and generation example"],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"]))
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>").save_pretrained(str(d))
    return d


def _tiny_bert(tmp_path: Path) -> Path:
    pytest.importorskip("transformers")
    import transformers

    d = tmp_path / "bert"
    d.mkdir()
    cfg = transformers.BertConfig(hidden_size=32, num_hidden_layers=2,
                                  num_attention_heads=4, intermediate_size=64,
                                  vocab_size=100, max_position_embeddings=64)
    transformers.AutoModel.from_config(cfg).save_pretrained(str(d))
    return d


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a * b).sum() / (na * nb) if na and nb else 0.0)


def _emb_cos(a, b) -> float:
    def emb(x):
        p = x.mean(axis=1)
        return p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-9)
    return float((emb(a) * emb(b)).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# 源格式识别 / 路由
# ---------------------------------------------------------------------------
def test_detect_source_format(tmp_path):
    from modconv.backend import (SOURCE_FORMAT_GGUF, SOURCE_FORMAT_MLX,
                                 SOURCE_FORMAT_ONNX, detect_source_format)

    hf = _tiny_bert(tmp_path)
    assert detect_source_format(hf) != SOURCE_FORMAT_MLX  # HF 命名 → hf
    onx = tmp_path / "m.onnx"
    onx.write_bytes(b"x")
    assert detect_source_format(onx) == SOURCE_FORMAT_ONNX
    gg = tmp_path / "m.gguf"
    gg.write_bytes(b"x")
    assert detect_source_format(gg) == SOURCE_FORMAT_GGUF

    # 量化 config → MLX；MLXEmbedders 命名目录 → MLX
    q = tmp_path / "quant"
    q.mkdir()
    (q / "config.json").write_text('{"model_type":"qwen2","quantization":{"bits":4}}')
    assert detect_source_format(q) == SOURCE_FORMAT_MLX
    from safetensors.numpy import save_file
    e = tmp_path / "embedders"
    e.mkdir()
    (e / "config.json").write_text('{"model_type":"xlm-roberta"}')
    save_file({"encoder.layers.0.attention.query_proj.weight":
               np.zeros((4, 4), dtype=np.float16)}, str(e / "model.safetensors"))
    assert detect_source_format(e) == SOURCE_FORMAT_MLX


def test_probe_routing_new_targets(tmp_path):
    from modconv.backend import probe_backend

    hf = _tiny_qwen_with_tokenizer(tmp_path)
    onx = tmp_path / "m.onnx"
    onx.write_bytes(b"")
    gg = tmp_path / "m.gguf"
    gg.write_bytes(b"")
    mlx_dir = tmp_path / "mlxdir"
    mlx_dir.mkdir()
    (mlx_dir / "config.json").write_text(
        '{"model_type":"qwen2","quantization":{"bits":4,"group_size":64}}')

    # 源 × 目标 → 任务
    assert probe_backend(hf, target="onnx").task == "hf->onnx"
    assert probe_backend(onx, target="onnx").task == "onnx->onnx"
    assert probe_backend(mlx_dir, target="onnx").task == "mlx->onnx"
    assert probe_backend(gg, target="onnx").task == "gguf->onnx"
    assert probe_backend(onx, target="mlx").task == "onnx->mlx"
    assert probe_backend(gg, target="mlx").task == "gguf->mlx"
    assert probe_backend(mlx_dir, target="gguf").task == "mlx->gguf"
    assert probe_backend(hf, target="gguf").task == "hf->gguf"
    assert probe_backend(onx, target="gguf").task == "onnx->gguf"
    # GGUF→GGUF 暂不支持（明确报错而非静默）
    p = probe_backend(gg, target="gguf")
    assert p.task == "gguf->gguf" and p.available is False


# ---------------------------------------------------------------------------
# causal：HF→MLX→ONNX / →GGUF→MLX / →GGUF→ONNX
# ---------------------------------------------------------------------------
def test_causal_hf_to_onnx(tmp_path):
    """causal HF→ONNX（transformers 5.x 的 aten::diff 已由导出补丁绕过）。"""
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from modconv.onnx_backend import HfToOnnxBackend

    hf = _tiny_qwen_with_tokenizer(tmp_path)
    out = tmp_path / "onnx"
    out.mkdir()
    r = HfToOnnxBackend(hf, out, quant_mode="fp16", model_kind="causal").run(_Hooks())
    assert r.success, r.message
    assert (out / "config.json").is_file()  # 产物自包含，可作下游源
    import onnxruntime as ort
    ids = np.array([[1, 2, 3, 4]])
    s = ort.InferenceSession(str(next(out.glob("*.onnx"))),
                             providers=["CPUExecutionProvider"])
    o = s.run(None, {"input_ids": ids})[0]
    assert o.shape[0] == 1 and o.shape[1] == 4 and np.all(np.isfinite(o))


def test_mlx_to_onnx_causal(tmp_path):
    """MLX→ONNX：与 HF→ONNX 的 logits 余弦≈1（无损）。"""
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("mlx_lm")
    import onnxruntime as ort
    from modconv.mlx_libs_backend import HfToMlxBackend
    from modconv.onnx_backend import HfToOnnxBackend, MlxToOnnxBackend

    hf = _tiny_qwen_with_tokenizer(tmp_path)
    mlx = tmp_path / "mlx"
    assert HfToMlxBackend(hf, mlx, quant_mode="fp16").run(_Hooks()).success

    onx_hf = tmp_path / "onnx_hf"
    onx_hf.mkdir()
    assert HfToOnnxBackend(hf, onx_hf, quant_mode="fp16",
                           model_kind="causal").run(_Hooks()).success
    onx_mlx = tmp_path / "onnx_mlx"
    onx_mlx.mkdir()
    r = MlxToOnnxBackend(mlx, onx_mlx, quant_mode="fp16").run(_Hooks())
    assert r.success, r.message

    ids = np.array([[1, 2, 3, 4]])
    a = ort.InferenceSession(str(next(onx_hf.glob("*.onnx"))),
                             providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(next(onx_mlx.glob("*.onnx"))),
                             providers=["CPUExecutionProvider"])
    cos = _cos(a.run(None, {"input_ids": ids})[0],
               b.run(None, {"input_ids": ids})[0])
    assert cos > 0.99, cos


def test_gguf_to_mlx_and_onnx(tmp_path):
    """GGUF→MLX（权重+分词器重建）与 GGUF→ONNX（前向有限）。"""
    pytest.importorskip("gguf")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    from safetensors import safe_open
    from modconv.mlx_libs_backend import HfToMlxBackend
    from modconv.gguf_backend import GgufBackend
    from modconv.gguf_to_mlx_backend import GgufToMlxBackend
    from modconv.onnx_backend import GgufToOnnxBackend

    hf = _tiny_qwen_with_tokenizer(tmp_path)
    mlx = tmp_path / "mlx"
    assert HfToMlxBackend(hf, mlx, quant_mode="fp16").run(_Hooks()).success
    gg_dir = tmp_path / "gguf"
    r = GgufBackend(mlx, gg_dir, quant_mode="int4").run(_Hooks())
    assert r.success, r.message
    gguf = next(gg_dir.glob("*.gguf"))

    # GGUF → MLX
    mlx2 = tmp_path / "mlx_from_gguf"
    r2 = GgufToMlxBackend(gguf, mlx2, quant_mode="fp16").run(_Hooks())
    assert r2.success, r2.message
    st = next(mlx2.glob("*.safetensors"))
    with safe_open(str(st), framework="numpy") as f:
        keys = list(f.keys())
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.0.self_attn.q_proj.bias" in keys   # 偏置补齐自洽
    assert (mlx2 / "tokenizer.json").is_file()              # 由 GGUF 词表重建
    assert (mlx2 / "config.json").is_file()

    # GGUF → ONNX
    onx = tmp_path / "onnx_from_gguf"
    onx.mkdir()
    r3 = GgufToOnnxBackend(gguf, onx, quant_mode="fp16").run(_Hooks())
    assert r3.success, r3.message
    ids = np.array([[1, 2, 3, 4]])
    s = ort.InferenceSession(str(next(onx.glob("*.onnx"))),
                             providers=["CPUExecutionProvider"])
    o = s.run(None, {"input_ids": ids})[0]
    assert np.all(np.isfinite(o))


# ---------------------------------------------------------------------------
# embedding（encoder）：ONNX ↔ MLX 互转
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qmode", ["fp16", "int4"])
def test_encoder_onnx_mlx_roundtrip(tmp_path, qmode):
    """HF→ONNX→MLX(q)→ONNX：与原 ONNX 的 embedding 余弦高于阈值（无额外损失）。"""
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("mlx")
    import onnxruntime as ort
    from modconv.onnx_backend import HfToOnnxBackend, MlxToOnnxBackend
    from modconv.mlx_backend import OnnxToMlxBackend

    hf = _tiny_bert(tmp_path)
    onx = tmp_path / "onnx"
    onx.mkdir()
    assert HfToOnnxBackend(hf, onx, quant_mode="fp16",
                           model_kind="embedding").run(_Hooks()).success
    of = next(onx.glob("*.onnx"))
    s = ort.InferenceSession(str(of), providers=["CPUExecutionProvider"])
    ids = np.array([[1, 2, 3, 4, 5]])
    ref = s.run(None, {"input_ids": ids})[0].astype(np.float32)

    mlx = tmp_path / f"mlx_{qmode}"
    assert OnnxToMlxBackend(of, mlx, quant_mode=qmode).run(_Hooks()).success

    onx2 = tmp_path / f"onnx_from_mlx_{qmode}"
    onx2.mkdir()
    r = MlxToOnnxBackend(mlx, onx2, quant_mode="fp16",
                         model_kind="embedding").run(_Hooks())
    assert r.success, r.message
    s2 = ort.InferenceSession(str(next(onx2.glob("*.onnx"))),
                              providers=["CPUExecutionProvider"])
    got = s2.run(None, {"input_ids": ids})[0].astype(np.float32)
    cos = _emb_cos(ref, got)
    # fp16 应几乎无损；int4 源有正常量化损失但仍应很高
    assert cos > (0.999 if qmode == "fp16" else 0.95), (qmode, cos)


def test_mlx_reverse_naming_roundtrip(tmp_path):
    """MLXEmbedders 命名 → HF 命名 的反向映射可逆（rekey 的逆）。"""
    from modconv.mlx_rekey_backend import rekey
    from modconv.source_normalize import mlx_to_hf_name

    hf_keys = [
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.0.attention.self.query.bias",
        "bert.encoder.layer.0.attention.output.dense.weight",
        "bert.encoder.layer.0.attention.output.LayerNorm.weight",
        "bert.encoder.layer.0.intermediate.dense.weight",
        "bert.encoder.layer.0.output.dense.weight",
        "bert.encoder.layer.0.output.LayerNorm.weight",
        "bert.embeddings.LayerNorm.weight",
    ]
    for hk in hf_keys:
        mk = rekey(hk)
        assert mk is not None, hk
        back = mlx_to_hf_name(mk)
        # 去掉 bert. 前缀后应回到原 HF 键（rekey 会剥离前缀）
        assert back == hk[len("bert."):], (hk, mk, back)
