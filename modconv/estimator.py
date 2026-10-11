"""转换后输出的体积/数量预估。"""
from __future__ import annotations

from dataclasses import dataclass

from modconv.scanner import ScanResult

# 输出包中额外生成的清单文件（Manifest.json）数量与估算字节数
MANIFEST_FILE_COUNT = 1
MANIFEST_EXTRA_BYTES = 4096

# 真实转换（coremltools）产出的 .mlpackage 内部固定文件数（coremldata.bin /
# model.mlmodel / Metadata.json 等，与模型大小无关）
MLPACKAGE_FILE_COUNT = 3

# 真实转换的体积估算系数：权重先转 fp16（×0.5），再 4-bit palettization（×0.25）。
# 仅供粗略参考——palettization 受 coremltools weight_threshold（2048 元素）约束，
# 小权重会被跳过，LUT 与索引本身也占空间；非权重文件（config/tokenizer…）
# 不进入 Core ML 模型，不计入。
FP16_BYTES_FACTOR = 0.5
PALETTIZE_BYTES_FACTOR = 0.25


@dataclass(frozen=True)
class OutputEstimate:
    file_count: int
    total_bytes: int


def estimate_output(scan: ScanResult, *, backend: str = "package") -> OutputEstimate:
    """估算转换后 Core ML 输出的文件数量与总占用空间。

    backend="package"（默认）：打包口径——权重 1:1 无损存放，加上 Manifest.json
    的开销。
    backend="coremltools"：真实转换口径——只把权重写进 .mlpackage，按 fp16 +
    4-bit 压缩折算体积，辅助文件（config/tokenizer 等）不进模型。
    """
    if backend == "coremltools":
        weights = sum(f.size for f in scan.files if f.is_weight)
        return OutputEstimate(
            file_count=MLPACKAGE_FILE_COUNT if scan.count else 0,
            total_bytes=int(weights * FP16_BYTES_FACTOR * PALETTIZE_BYTES_FACTOR),
        )
    return OutputEstimate(
        file_count=scan.count + MANIFEST_FILE_COUNT,
        total_bytes=scan.total_bytes + MANIFEST_EXTRA_BYTES,
    )
