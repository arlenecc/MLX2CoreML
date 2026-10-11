"""通用工具函数。"""

_UNITS = ("KB", "MB", "GB", "TB", "PB")


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
