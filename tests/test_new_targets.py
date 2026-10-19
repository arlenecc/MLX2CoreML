"""新增转换目标（HF→ONNX / ONNX→MLX / HF→MLX / HF→GGUF）的测试。

- 路由/可用性/体积预估为轻量单测（不触发重型导入）。
- 端到端集成用例按依赖可用性跳过（pytest.importorskip）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from modconv.backend import (
    probe_backend, detect_source_format, hf_to_onnx_available,
    onnx_to_mlx_available, hf_to_mlx_available, hf_to_gguf_available,
)
from modconv.estimator import estimate_output
from modconv.scanner import scan_directory


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
# 路由 / 可用性 / 预估
# --------------------------------------------------------------------------
def _hf_dir(tmp_path: Path) -> Path:
    d = tmp_path / "hf"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
    return d


def _onnx_file(tmp_path: Path) -> Path:
    f = tmp_path / "model.onnx"
    f.write_bytes(b"")
    return f


def test_detect_source_format(tmp_path):
    assert detect_source_format(_hf_dir(tmp_path)) == "hf"
    assert detect_source_format(_onnx_file(tmp_path)) == "onnx"


def test_probe_routing_onnx(tmp_path):
    hf = _hf_dir(tmp_path)
    p = probe_backend(hf, target="onnx")
    assert p.name == "onnx" and p.task == "hf->onnx" and p.target == "onnx"
    assert p.available == hf_to_onnx_available()
    # ONNX 源 → onnx->onnx（weight-only int8，nomic 等 embedding 已验证）
    p2 = probe_backend(_onnx_file(tmp_path), target="onnx")
    assert p2.task == "onnx->onnx"
    assert p2.available == hf_to_onnx_available()


def test_probe_routing_mlx(tmp_path):
    # ONNX 源 → onnx->mlx（自定义）
    p = probe_backend(_onnx_file(tmp_path), target="mlx")
    assert p.name == "mlx" and p.task == "onnx->mlx"
    assert p.available == onnx_to_mlx_available()
    # HF 源 → hf->mlx（经 mlx 库）
    p2 = probe_backend(_hf_dir(tmp_path), target="mlx")
    assert p2.name == "mlx_libs" and p2.task == "hf->mlx"
    assert p2.available == hf_to_mlx_available()


def test_probe_routing_gguf(tmp_path):
    p = probe_backend(_hf_dir(tmp_path), target="gguf")
    assert p.name == "gguf" and p.task == "hf->gguf"
    assert p.available == hf_to_gguf_available()
    # ONNX 文件源 → onnx->gguf（权重取自 .onnx，结构/分词器取自同目录）
    p2 = probe_backend(_onnx_file(tmp_path), target="gguf")
    assert p2.name == "gguf" and p2.task == "onnx->gguf"
    assert p2.available == hf_to_gguf_available()


def test_scanner_single_file(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"x" * 123)
    scan = scan_directory(f)
    assert scan.count == 1
    assert scan.files[0].relpath == "model.onnx"
    assert scan.files[0].is_weight and scan.files[0].size == 123


def test_estimate_gguf(tmp_path):
    hf = _hf_dir(tmp_path)
    (hf / "model.safetensors").write_bytes(b"x" * 1000)
    scan = scan_directory(hf)
    e = estimate_output(scan, backend="gguf", compressed=True, nbits="int4")
    assert e.file_count == 1
    assert 0 < e.total_bytes < 1000  # int4 系数 < 1


# --------------------------------------------------------------------------
# 端到端集成（按依赖跳过）
# --------------------------------------------------------------------------
def _tiny_bert(tmp_path: Path) -> Path:
    transformers = pytest.importorskip("transformers")
    d = tmp_path / "bert"
    d.mkdir()
    cfg = transformers.BertConfig(hidden_size=32, num_hidden_layers=2,
                                  num_attention_heads=4, intermediate_size=64,
                                  vocab_size=100, max_position_embeddings=64)
    transformers.AutoModel.from_config(cfg).save_pretrained(str(d))
    return d


def _tiny_qwen(tmp_path: Path) -> Path:
    transformers = pytest.importorskip("transformers")
    d = tmp_path / "qwen"
    d.mkdir()
    cfg = transformers.Qwen2Config(hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=300,
                                   max_position_embeddings=64)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(str(d))
    return d


@pytest.mark.parametrize("qmode", ["fp16", "int4", "int8"])
def test_hf_onnx_then_mlx(tmp_path, qmode):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    import shutil
    from modconv.onnx_backend import HfToOnnxBackend
    from modconv.mlx_backend import OnnxToMlxBackend

    hf = _tiny_bert(tmp_path)
    onnx_out = tmp_path / "onnx"
    onnx_out.mkdir()
    r = HfToOnnxBackend(hf, onnx_out, quant_mode="fp16", model_kind="embedding").run(_Hooks())
    assert r.success, r.message
    onnx_file = next(onnx_out.glob("*.onnx"))
    shutil.copy(hf / "config.json", onnx_file.parent / "config.json")

    mlx_out = tmp_path / f"mlx_{qmode}"
    mlx_out.mkdir()
    r2 = OnnxToMlxBackend(onnx_file, mlx_out, quant_mode=qmode).run(_Hooks())
    assert r2.success, r2.message
    assert (mlx_out / "model.safetensors").is_file()


def test_hf_to_mlx_via_libs(tmp_path):
    pytest.importorskip("mlx_embeddings")
    from modconv.mlx_libs_backend import HfToMlxBackend

    hf = _tiny_bert(tmp_path)
    out = tmp_path / "mlxlib"
    r = HfToMlxBackend(hf, out, quant_mode="int8").run(_Hooks())
    assert r.success, r.message
    assert any(p.suffix in (".safetensors", ".npz") for p in out.glob("*"))


@pytest.mark.parametrize("qmode", ["fp16", "int8", "int4"])
def test_hf_to_gguf_loadable(tmp_path, qmode):
    pytest.importorskip("gguf")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from modconv.gguf_backend import GgufBackend

    hf = _tiny_qwen(tmp_path)
    # 构造小型 ByteLevel BPE 分词器（写 tokenizer.json）
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(
        ["hello world this is a tiny test of the model tokenization and generation example"],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"]))
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                            bos_token=None).save_pretrained(str(hf))

    out = tmp_path / f"gguf_{qmode}"
    r = GgufBackend(hf, out, quant_mode=qmode).run(_Hooks())
    assert r.success, r.message
    gguf = out / "qwen.gguf"
    assert gguf.is_file()

    # 用 llama_cpp 验证可加载（若可用）
    llama_cpp = pytest.importorskip("llama_cpp")
    llm = llama_cpp.Llama(model_path=str(gguf), n_ctx=32, verbose=False)
    assert llm.n_vocab() > 0


def _tiny_qwen_with_tokenizer(tmp_path: Path) -> Path:
    """_tiny_qwen + ByteLevel BPE tokenizer（GGUF 需 tokenizer.json）。"""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    hf = _tiny_qwen(tmp_path)
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(
        ["hello world this is a tiny test of the model tokenization and generation example"],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"]))
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                            bos_token=None).save_pretrained(str(hf))
    return hf


@pytest.mark.parametrize("qmode", ["fp16", "int4"])
def test_mlx_to_gguf(tmp_path, qmode):
    """MLX（fp16 / int4 量化）目录 → GGUF，验证 llama_cpp 可加载。"""
    pytest.importorskip("gguf")
    pytest.importorskip("mlx_lm")
    from modconv.mlx_libs_backend import HfToMlxBackend
    from modconv.gguf_backend import GgufBackend

    hf = _tiny_qwen_with_tokenizer(tmp_path)
    mlx = tmp_path / f"mlx_{qmode}"
    r = HfToMlxBackend(hf, mlx, quant_mode=qmode).run(_Hooks())
    assert r.success, r.message

    out = tmp_path / f"gguf_from_mlx_{qmode}"
    r2 = GgufBackend(mlx, out, quant_mode="int4").run(_Hooks())
    assert r2.success, r2.message
    gguf = out / f"mlx_{qmode}.gguf"
    assert gguf.is_file()
    llama_cpp = pytest.importorskip("llama_cpp")
    llm = llama_cpp.Llama(model_path=str(gguf), n_ctx=32, verbose=False)
    assert llm.n_vocab() > 0


def test_onnx_to_gguf(tmp_path):
    """ONNX 文件源 → GGUF：权重名带包装前缀（m.）仍可恢复 HF 命名。"""
    pytest.importorskip("gguf")
    pytest.importorskip("onnx")
    pytest.importorskip("torch")
    import shutil

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    hf = _tiny_qwen_with_tokenizer(tmp_path)
    # 用 HF 参数名合成带 m. 前缀的 ONNX（模拟包装导出），同目录放 config/tokenizer
    model = pytest.importorskip("transformers").AutoModelForCausalLM.from_config(
        __import__("transformers").Qwen2Config(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
            max_position_embeddings=64))
    sd = {f"m.{k}": v.detach().numpy() for k, v in model.named_parameters()}
    onxdir = tmp_path / "onx"
    onxdir.mkdir()
    inits = [numpy_helper.from_array(v.astype(np.float32), name=k) for k, v in sd.items()]
    graph = helper.make_graph(
        [], "m",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, None)],
        initializer=inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)]),
              str(onxdir / "model.onnx"))
    shutil.copy(hf / "config.json", onxdir / "config.json")
    shutil.copy(hf / "tokenizer.json", onxdir / "tokenizer.json")

    from modconv.gguf_backend import GgufBackend
    out = tmp_path / "gguf_from_onnx"
    r = GgufBackend(onxdir / "model.onnx", out, quant_mode="fp16").run(_Hooks())
    assert r.success, r.message
    gguf = out / "model.gguf"
    assert gguf.is_file()
    llama_cpp = pytest.importorskip("llama_cpp")
    llm = llama_cpp.Llama(model_path=str(gguf), n_ctx=32, verbose=False)
    assert llm.n_vocab() > 0
