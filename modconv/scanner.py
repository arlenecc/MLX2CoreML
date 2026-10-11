"""MLX 模型目录扫描。

以流式方式遍历目录，返回文件清单、数量与总大小；
不读取文件内容，内存占用只与文件条目数相关。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# 视作模型权重文件的扩展名（.bin 为 HF 旧的 pytorch_model.bin 权重命名）
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".npz", ".gguf"}

# 扫描进度回调的上报间隔（每处理多少个文件上报一次）
SCAN_PROGRESS_EVERY = 256


@dataclass(frozen=True)
class FileInfo:
    relpath: str
    size: int
    is_weight: bool


@dataclass
class ScanResult:
    source: Optional[Path]
    files: List[FileInfo] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


def is_weight_file(name: str) -> bool:
    return Path(name).suffix.lower() in WEIGHT_SUFFIXES


def scan_directory(
    root: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> ScanResult:
    """扫描目录并返回 ScanResult；路径不存在或不是目录时抛出异常。

    progress_cb(processed_count, running_bytes) 每处理 SCAN_PROGRESS_EVERY 个
    文件上报一次，供 GUI 实时显示大目录的扫描进度。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")

    files: List[FileInfo] = []
    processed = 0
    running_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0  # 扫描期间文件被移除
            relpath = path.relative_to(root).as_posix()
            files.append(FileInfo(relpath, size, is_weight_file(name)))
            processed += 1
            running_bytes += size
            if progress_cb is not None and processed % SCAN_PROGRESS_EVERY == 0:
                progress_cb(processed, running_bytes)
    return ScanResult(source=root, files=files)
