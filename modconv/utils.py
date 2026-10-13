"""通用工具函数。"""

_UNITS = ("KB", "MB", "GB", "TB", "PB")

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
