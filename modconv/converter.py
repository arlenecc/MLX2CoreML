"""MLX → Core ML 转换核心逻辑（不依赖 Qt，可独立测试）。

性能设计：
- 流式分块读写（默认 4 MiB 块），内存占用恒定，可处理任意大小的模型文件；
- 进度回调按字节间隔节流（默认 16 MiB），避免大文件时事件洪泛拖垮 GUI；
- 每个拷贝块之间检查取消标记，可即时中断长耗时转换。
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence

from modconv.scanner import FileInfo
from modconv.utils import format_size

CHUNK_SIZE = 4 * 1024 * 1024                     # 流式拷贝块大小
PROGRESS_EMIT_INTERVAL_BYTES = 16 * 1024 * 1024  # 进度回调最小字节间隔
MANIFEST_NAME = "Manifest.json"


@dataclass(frozen=True)
class VerificationResult:
    """转换后产物的可用性验证结果（转换完成自动执行）。

    - loaded/ran：模型能否用目标原生运行时加载、能否完成一次前向/生成；
    - passed：是否通过验证（加载+前向正常且能力无额外损失；skipped 时为 False）；
    - skipped：因依赖缺失/无法构建参考而跳过（非失败，仅未验证）；
    - metrics：量化损失等数值指标（如 cosine / max_diff / norm_ratio / has_nan）；
    - summary：面向用户的中文结论。
    """
    target: str
    loaded: bool
    ran: bool
    passed: bool
    skipped: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    summary: str = ""


@dataclass(frozen=True)
class ConversionResult:
    success: bool
    cancelled: bool
    message: str
    output_path: Optional[Path]
    files_done: int
    bytes_done: int
    elapsed_seconds: float
    verification: Optional[VerificationResult] = None


class ConversionHooks:
    """转换过程回调接口（鸭子类型实现即可）。"""

    def should_cancel(self) -> bool:
        return False

    def on_log(self, message: str) -> None:
        raise NotImplementedError

    def on_file_started(self, relpath: str, size: int, index: int, total_files: int) -> None:
        raise NotImplementedError

    def on_file_progress(self, relpath: str, file_done: int, file_total: int,
                         overall_done: int, overall_total: int) -> None:
        raise NotImplementedError

    def on_file_finished(self, relpath: str, overall_done: int) -> None:
        raise NotImplementedError


class ModelConverter:
    """把 MLX 模型目录打包转换为 Core ML .mlpackage 目录结构。

    输出布局：
        <输出目录>/<模型名>.mlpackage/
        ├── Manifest.json
        └── Data/
            ├── com.apple.CoreML/weights/<权重文件相对路径>
            └── mlx/<其余文件相对路径>
    """

    def __init__(self, source: Path, output_dir: Path, files: Sequence[FileInfo], *,
                 chunk_size: int = CHUNK_SIZE,
                 emit_interval: int = PROGRESS_EMIT_INTERVAL_BYTES):
        self.source = Path(source).expanduser()
        if not self.source.is_dir():
            raise ValueError(f"源目录不存在: {self.source}")
        self.source = self.source.resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.files: List[FileInfo] = list(files)
        self.chunk_size = max(64 * 1024, int(chunk_size))
        self.emit_interval = max(1, int(emit_interval))

    @property
    def package_dir(self) -> Path:
        return self.output_dir / f"{self.source.name}.mlpackage"

    def _build_manifest(self) -> str:
        payload = {
            "fileIdentifier": f"{self.source.name}.mlpackage",
            "author": "modconv",
            "specificationVersion": 1,
            "sourceFormat": "MLX",
            "note": "Weights are copied losslessly; model program generation "
                    "requires a coremltools backend.",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success: bool, cancelled: bool, message: str,
                   output: Optional[Path], files_done: int, bytes_done: int) -> ConversionResult:
            return ConversionResult(success, cancelled, message, output,
                                    files_done, bytes_done, time.monotonic() - start)

        if hooks.should_cancel():
            return finish(False, True, "转换已取消", None, 0, 0)

        if not self.files:
            message = "源目录中没有可转换的文件"
            hooks.on_log(message)
            return finish(False, False, message, None, 0, 0)

        manifest_bytes = self._build_manifest().encode("utf-8")
        total_bytes = sum(f.size for f in self.files) + len(manifest_bytes)
        package = self.package_dir

        hooks.on_log(f"目标输出包: {package}")
        if package.exists():
            hooks.on_log("输出包已存在，将先删除旧内容（覆盖模式）")
            shutil.rmtree(package)

        weights_root = package / "Data" / "com.apple.CoreML" / "weights"
        mlx_root = package / "Data" / "mlx"
        weights_root.mkdir(parents=True, exist_ok=True)
        mlx_root.mkdir(parents=True, exist_ok=True)

        overall_done = 0
        files_done = 0
        total_files = len(self.files)
        cancelled = False

        skipped = 0
        for index, info in enumerate(self.files, start=1):
            if hooks.should_cancel():
                cancelled = True
                break
            # relpath 直接参与路径拼接：拒绝绝对路径与越出输出包的 ../，
            # 避免写出到包外（库 API 可被外部直接传入自定义 FileInfo）
            relpath = PurePosixPath(info.relpath)
            if relpath.is_absolute() or ".." in relpath.parts:
                skipped += 1
                hooks.on_log(f"跳过非法相对路径: {info.relpath}")
                continue
            source_file = self.source / info.relpath
            if info.is_weight:
                dest = weights_root / info.relpath
            else:
                dest = mlx_root / info.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            hooks.on_file_started(info.relpath, info.size, index, total_files)
            try:
                copied = self._copy_stream(source_file, dest, info, hooks,
                                           overall_done, total_bytes)
            except OSError as exc:
                # 扫描后被删除/权限变化：记录并继续，避免单个文件让整次转换失败
                skipped += 1
                hooks.on_log(f"跳过无法读取的文件 {info.relpath}: {exc}")
                continue
            overall_done += copied
            if hooks.should_cancel():
                cancelled = True
                break
            files_done += 1
            hooks.on_file_finished(info.relpath, overall_done)

        if cancelled:
            message = f"转换已取消: 已完成 {files_done}/{total_files} 个文件"
            hooks.on_log(message)
            # 半成品包对调用方没有价值（缺 Manifest、权重不全），却可能占几 GB：
            # 取消即删除，避免反复取消后在磁盘上堆下一串废包
            shutil.rmtree(package, ignore_errors=True)
            return finish(False, True, message, None, files_done, overall_done)

        (package / MANIFEST_NAME).write_bytes(manifest_bytes)
        overall_done += len(manifest_bytes)
        message = (f"转换完成: {files_done}/{total_files} 个文件"
                   + (f"（跳过 {skipped} 个）" if skipped else "")
                   + f", 输出 {format_size(overall_done)}, "
                   f"耗时 {time.monotonic() - start:.1f} 秒")
        hooks.on_log(message)
        return finish(True, False, message, package, files_done, overall_done)

    def _copy_stream(self, src: Path, dest: Path, info: FileInfo,
                     hooks: ConversionHooks, overall_done: int, total_bytes: int) -> int:
        """流式拷贝单个文件，返回已拷贝字节数；期间响应取消并按字节间隔节流进度。"""
        copied = 0
        since_emit = 0
        with open(src, "rb") as fsrc, open(dest, "wb") as fdst:
            while True:
                if hooks.should_cancel():
                    break
                chunk = fsrc.read(self.chunk_size)
                if not chunk:
                    break
                fdst.write(chunk)
                copied += len(chunk)
                since_emit += len(chunk)
                if since_emit >= self.emit_interval or copied == info.size:
                    hooks.on_file_progress(info.relpath, copied, info.size,
                                           overall_done + copied, total_bytes)
                    since_emit = 0
        # 收尾上报：文件大小与扫描时不一致（如 stat 失败按 0 记）时，
        # copied == info.size 永不成立，靠这里保证进度条能走到 100%
        if since_emit and copied != info.size:
            hooks.on_file_progress(info.relpath, copied, max(copied, info.size),
                                   overall_done + copied, total_bytes)
        return copied
