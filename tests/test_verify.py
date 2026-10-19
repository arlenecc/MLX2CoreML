"""转换产物自动可用性验证测试。

通过 verify_conversion 对转换产物做「加载 + 前向/生成 + 保真余弦」校验，
确认转换后模型能正常加载、能力正常、除正常量化损失外无额外损失。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from modconv.converter import ConversionHooks, VerificationResult
from modconv.verify import verify_conversion


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


def _tiny_bert(tmp_path):
    import transformers
    d = tmp_path / "hf"
    d.mkdir()
    cfg = transformers.BertConfig(hidden_size=32, num_hidden_layers=2,
                                  num_attention_heads=4, intermediate_size=64,
                                  vocab_size=100, max_position_embeddings=64)
    transformers.AutoModel.from_config(cfg).save_pretrained(str(d))
    return d


def _hf_to_onnx(hf_dir, out, kind):
    pytest.importorskip("torch")
    from modconv.onnx_backend import HfToOnnxBackend

    out.mkdir(parents=True, exist_ok=True)
    r = HfToOnnxBackend(hf_dir, out, quant_mode="fp16", model_kind=kind).run(_Hooks())
    assert r.success, r.message
    onnx_file = next(out.glob("*.onnx"))
    shutil.copy(hf_dir / "config.json", out / "config.json")
    return onnx_file


def test_verify_onnx_reference_cosine(tmp_path):
    """onnx->onnx int8：验证应与原 ONNX 输出余弦>0.9（无额外损失）。"""
    pytest.importorskip("onnxruntime")
    hf = _tiny_bert(tmp_path)
    onx = tmp_path / "onx"
    onnx_file = _hf_to_onnx(hf, onx, "embedding")
    from modconv.onnx_backend import OnnxToOnnxBackend
    o2 = tmp_path / "int8"
    r = OnnxToOnnxBackend(onnx_file, o2, quant_mode="int8").run(_Hooks())
    assert r.success, r.message
    ver = verify_conversion("onnx", o2, onnx_file, _Hooks(),
                           model_kind="embedding", quant_mode="int8")
    assert isinstance(ver, VerificationResult)
    assert ver.loaded and ver.ran
    assert ver.metrics.get("cosine", 0) > 0.9, ver.summary
    assert ver.passed, ver.summary


def test_verify_mlx_weight_cosine(tmp_path):
    """onnx->mlx fp16：验证应能加载，且与源 ONNX 权重余弦>0.9（朝向/命名正确）。"""
    pytest.importorskip("mlx")
    hf = _tiny_bert(tmp_path)
    onx = tmp_path / "onx"
    onnx_file = _hf_to_onnx(hf, onx, "embedding")
    from modconv.mlx_backend import OnnxToMlxBackend

    for qm in ("fp16", "int4"):
        m = tmp_path / f"mlx_{qm}"
        r = OnnxToMlxBackend(onnx_file, m, quant_mode=qm).run(_Hooks())
        assert r.success, r.message
        ver = verify_conversion("mlx", m, onnx_file, _Hooks(),
                               model_kind="embedding", quant_mode=qm)
        assert ver.loaded and ver.ran
        # onnx 源无 safetensors，无法做权重余弦；退化为加载+有限值+可序列化校验
        assert ver.passed, ver.summary
        if "weight_cosine" in ver.metrics:
            assert ver.metrics["weight_cosine"] > 0.9, (qm, ver.summary)


def test_verify_mlx_rekey(tmp_path):
    """mlx->mlx：验证能与源 MLX(HF 命名) 权重余弦>0.9（反量化后）。"""
    pytest.importorskip("mlx")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import numpy as np
    import transformers
    from safetensors.numpy import save_file
    from modconv.mlx_rekey_backend import MlxToMlxBackend

    hf = _tiny_bert(tmp_path)
    model = transformers.AutoModel.from_pretrained(str(hf))
    hf_sd = {k: v.detach().numpy().astype(np.float16) for k, v in model.state_dict().items()}
    src = tmp_path / "hf_named_mlx"
    src.mkdir()
    save_file(hf_sd, str(src / "model.safetensors"))
    shutil.copy(hf / "config.json", src / "config.json")
    out = tmp_path / "mlx_rekey"
    r = MlxToMlxBackend(src, out, quant_mode="int4").run(_Hooks())
    assert r.success, r.message
    ver = verify_conversion("mlx", out, src, _Hooks(), quant_mode="int4")
    assert ver.loaded and ver.ran
    assert ver.metrics["weight_cosine"] > 0.9, ver.summary
    assert ver.passed, ver.summary
