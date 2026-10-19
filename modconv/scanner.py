"""MLX 模型目录扫描。

以流式方式遍历目录，返回文件清单、数量与总大小；
不读取文件内容，内存占用只与文件条目数相关。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from modconv.utils import is_appledouble

# 视作模型权重文件的扩展名（.bin 为 HF 旧的 pytorch_model.bin 权重命名；
# .onnx 为 ONNX 模型文件，HF→ONNX / ONNX→MLX 任务会产出/消费它）
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".npz", ".gguf", ".onnx"}

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

    用 os.scandir 直接遍历：DirEntry 自带 d_type，判定符号链接/目录通常无需
    系统调用，每个文件只剩一次 stat（原 os.walk + Path 写法是两次 stat 再加
    一次 relative_to 字符串运算）。显式栈代替递归，避免深目录爆栈。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    if root.is_file():
        # 单文件源（如 .onnx）：只统计该文件本身
        size = root.stat().st_size
        return ScanResult(source=root, files=[FileInfo(root.name, size, is_weight_file(root.name))])
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")

    files: List[FileInfo] = []
    processed = 0
    running_bytes = 0
    # (目录绝对路径, 该目录相对 root 的 posix 前缀)
    stack = [(str(root), "")]
    while stack:
        dirpath, prefix = stack.pop()
        subdirs = []
        plain_files = []  # [(文件名, 字节数)]，趁着 scandir 迭代器还活着就取完 stat
        try:
            with os.scandir(dirpath) as entries:
                for entry in entries:
                    try:
                        # 符号链接一律跳过（与 followlinks=False 语义一致：
                        # 不跟随、不重复统计）
                        if entry.is_symlink():
                            continue
                        if entry.is_dir():
                            subdirs.append(entry.name)
                            continue
                        try:
                            size = entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            size = 0  # 扫描期间文件被移除：仍列出，转换阶段会记录并跳过
                    except OSError:
                        continue  # 条目状态已失效
                    plain_files.append((entry.name, size))
        except OSError:
            continue  # 目录不可读（权限变化/被移除）：跳过，不让整次扫描失败
        # 跳过 macOS AppleDouble 伴生文件（._xxx）：写入 exFAT/FAT 外接卷时系统会
        # 为每个文件生成它们，计入会虚高体积统计并被当作权重文件拷贝。
        plain_files = [(n, s) for n, s in plain_files if not is_appledouble(n)]
        # 按文件名排序：保证扫描结果与文件系统返回顺序无关（进度、日志可复现）
        plain_files.sort(key=lambda item: item[0])
        for name, size in plain_files:
            files.append(FileInfo(prefix + name, size, is_weight_file(name)))
            processed += 1
            running_bytes += size
            if progress_cb is not None and processed % SCAN_PROGRESS_EVERY == 0:
                progress_cb(processed, running_bytes)
        # 栈是 LIFO：逆序压入，出栈顺序才是字典序（与旧实现一致）
        for name in sorted(subdirs, reverse=True):
            stack.append((os.path.join(dirpath, name), f"{prefix}{name}/"))
    return ScanResult(source=root, files=files)
