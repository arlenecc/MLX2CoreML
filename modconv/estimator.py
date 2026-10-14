"""转换后输出的体积/数量预估。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modconv.scanner import ScanResult

# 输出包中额外生成的清单文件（Manifest.json）数量与估算字节数
MANIFEST_FILE_COUNT = 1
MANIFEST_EXTRA_BYTES = 4096

# 真实转换（coremltools）产出的 .mlpackage 内部固定文件数（coremldata.bin /
# model.mlmodel / Metadata.json 等，与模型大小无关）
MLPACKAGE_FILE_COUNT = 3

# 真实转换的体积估算系数：权重先转 fp16（×0.5），再 palettization 压缩。
# 仅供粗略参考——palettization 受 coremltools weight_threshold（2048 元素）约束，
# 小权重会被跳过，LUT 与索引本身也占空间；实测 0.8B 模型 fp16→4-bit 后为源
# bf16 的约 0.23 倍（1.6GB→372MB），故 0.25 的理论系数偏低，4-bit 按 0.35 取值；
# 非权重文件（config/tokenizer…）不进入 Core ML 模型，不计入。
FP16_BYTES_FACTOR = 0.5
PALETTIZE_BYTES_FACTOR = 0.35

# 不同压缩位数的估算系数（相对 fp16 的体积倍数），覆盖 GUI 的 4/6/8-bit 选项；
# 未列出的位数回退到 PALETTIZE_BYTES_FACTOR。6/8-bit 的压缩比弱于 4-bit，
# 之前统一用 0.35 会系统性偏低估真实体积。
_PALETTIZE_PER_NBIT = {4: 0.35, 6: 0.42, 8: 0.5}


@dataclass(frozen=True)
class OutputEstimate:
    file_count: int
    total_bytes: int


def estimate_output(scan: ScanResult, *, backend: str = "package",
                    compressed: bool = True, nbits: Optional[int] = None) -> OutputEstimate:
    """估算转换后 Core ML 输出的文件数量与总占用空间。

    backend="package"（默认）：打包口径——权重 1:1 无损存放，加上 Manifest.json
    的开销。
    backend="coremltools"：真实转换口径——只把权重写进 .mlpackage，按 fp16（compressed
    =False）或 fp16 + palettization 折算体积，辅助文件（config/tokenizer 等）不进模型。
    nbits 指定压缩位数（4/6/8）时使用对应的压缩系数，使估算更贴近实际产物。
    """
    if backend == "coremltools":
        weights = sum(f.size for f in scan.files if f.is_weight)
        factor = FP16_BYTES_FACTOR
        if compressed:
            factor *= _PALETTIZE_PER_NBIT.get(nbits, PALETTIZE_BYTES_FACTOR) if nbits else PALETTIZE_BYTES_FACTOR
        return OutputEstimate(
            file_count=MLPACKAGE_FILE_COUNT if scan.count else 0,
            total_bytes=int(weights * factor),
        )
    return OutputEstimate(
        file_count=scan.count + MANIFEST_FILE_COUNT,
        total_bytes=scan.total_bytes + MANIFEST_EXTRA_BYTES,
    )
