"""通用工具函数。"""

from pathlib import Path
from typing import List

_UNITS = ("KB", "MB", "GB", "TB", "PB")


def is_appledouble(name: str) -> bool:
    """是否为 macOS AppleDouble 伴生文件（``._xxx``）。

    往 exFAT/FAT 等外接卷写入文件时，系统会为每个文件生成 ``._`` 前缀的伴随文件
    （存放扩展属性/资源分支）。直接 glob 会误选它们，导致 protobuf 解析失败、
    体积统计虚高等问题，所有文件发现逻辑都必须跳过。
    """
    return name.startswith("._")


def list_model_files(directory: Path, pattern: str = "*") -> List[Path]:
    """glob 目录内文件并跳过 AppleDouble 伴生文件（按名排序）。"""
    return sorted(p for p in Path(directory).glob(pattern)
                  if not is_appledouble(p.name))

# 混合注意力架构里属于"注意力"（而非 linear attention / GDN）的层类型。
# real_backend 用它推断最小上下文长度，kv_wrapper 用它判断是否需要静态 pad 前向，
# 两处必须一致：否则前者允许 max_seq_len=1 而后者会在构造时抛 ValueError。
# 放在无重量级依赖的本模块，避免 real_backend 为了它去 import torch。
ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention", "chunked_attention")


def format_size(size_bytes: int) -> str:
    """把字节数格式化为人类可读字符串，例如 1.50 KB。"""
    size_bytes = int(size_bytes)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in _UNITS:
        value /= 1024.0
        if value < 1024.0 or unit == _UNITS[-1]:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} {_UNITS[-1]}"  # pragma: no cover
