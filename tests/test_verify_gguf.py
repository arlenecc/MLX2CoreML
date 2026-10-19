"""GGUF 产物自动可用性验证测试（独立进程：与 mlx/verify 的其他测试隔离，
避免 llama_cpp 与 MLX 原生库同进程触发 C 级 abort）。"""
from __future__ import annotations

import shutil

import pytest

from modconv.converter import ConversionHooks, VerificationResult


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


def test_verify_gguf_load(tmp_path):
    """gguf：验证 llama_cpp 能加载并生成（causal 冒烟）。"""
    pytest.importorskip("gguf")
    pytest.importorskip("llama_cpp")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    hf = tmp_path / "hf"
    hf.mkdir()
    cfg = transformers.Qwen2Config(hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=300,
                                   max_position_embeddings=64)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(str(hf))
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(["hello world this is a tiny test of the model"],
                           trainers.BpeTrainer(vocab_size=300,
                                               special_tokens=["<|endoftext|>"]))
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>").save_pretrained(str(hf))

    from modconv.mlx_libs_backend import HfToMlxBackend
    from modconv.gguf_backend import GgufBackend
    from modconv.verify import verify_conversion

    mlx = tmp_path / "mlx"
    assert HfToMlxBackend(hf, mlx, quant_mode="fp16").run(_Hooks()).success
    gg = tmp_path / "gguf"
    assert GgufBackend(mlx, gg, quant_mode="fp16").run(_Hooks()).success
    ver = verify_conversion("gguf", gg, mlx, _Hooks(), quant_mode="fp16")
    assert isinstance(ver, VerificationResult)
    assert ver.loaded and ver.ran
    assert ver.passed, ver.summary
