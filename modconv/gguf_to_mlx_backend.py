"""GGUF → MLX 转换后端。

把 llama.cpp 的 GGUF 文件转为 MLX 格式（HF 命名 safetensors + config.json +
tokenizer.json），供 mlx-lm 加载；或经本工具 ONNX→MLX / MLX→MLX 链路再转成
MLXEmbedders 命名。

实现：先用 source_normalize.gguf_to_hf_dir 把 GGUF 归一化为临时 HF 目录
（llama 布局张量名→HF 名、量化张量反量化、config 由元数据重建、tokenizer.json
由词表/merges 重建——已验证分词 ID 一致），再复用 HfToMlxBackend（mlx-lm）。

范围：GGUF 只承载 llama 布局 causal 模型（llama/qwen2/qwen3/mistral），
不支持 MoE / SentencePiece 模型；encoder（BERT 家族）本就不进 GGUF。
"""
from __future__ import annotations

import time
from pathlib import Path

from modconv.converter import ConversionHooks, ConversionResult


class GgufToMlxBackend:
    """GGUF 文件 → MLX（mlx-lm 命名），可选 int4/int8 量化。

    quant_mode: "fp16" / "int4" / "int8"（量化由 mlx-lm 完成）。
    """

    def __init__(self, source: Path, output_dir: Path, *,
                 quant_mode: str = "fp16"):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode

    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            from modconv.source_normalize import gguf_to_hf_dir_ctx
            with gguf_to_hf_dir_ctx(self.source, hooks) as hf_dir:
                hooks.on_log(f"已归一化为 HF 目录: {hf_dir}")
                from modconv.mlx_libs_backend import HfToMlxBackend
                return HfToMlxBackend(
                    hf_dir, self.output_dir, quant_mode=self.quant_mode).run(hooks)
        except Exception as exc:
            msg = f"GGUF→MLX 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)
