"""QThread 工作线程：把扫描/转换放到后台线程，通过信号跨线程更新 UI，
保证 GUI 主线程不被耗时操作阻塞。"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import QThread, pyqtSignal

from modconv.backend import probe_backend
from modconv.converter import ConversionResult, ModelConverter
from modconv.real_backend import CoremltoolsBackend
from modconv.scanner import FileInfo, scan_directory


class _ScanCancelled(Exception):
    """扫描被取消：用于从扫描回调里提前退出（不算错误）。"""


class ScanWorker(QThread):
    """后台目录扫描。"""

    scan_progress = pyqtSignal(int, int)   # 已扫描文件数, 累计字节
    scan_finished = pyqtSignal(object)     # ScanResult
    scan_error = pyqtSignal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = Path(path)
        self._cancelled = False

    def cancel(self):
        """请求提前结束扫描（供关闭窗口等场景使用，避免线程无法退出）。"""
        self._cancelled = True

    def run(self):
        def _progress(count, size):
            if self._cancelled:
                raise _ScanCancelled
            self.scan_progress.emit(count, size)

        try:
            if self._cancelled:
                raise _ScanCancelled
            result = scan_directory(self._path, progress_cb=_progress)
        except _ScanCancelled:
            return  # 取消：不发 scan_finished，UI 状态由发起方负责
        except OSError as exc:
            self.scan_error.emit(str(exc))
            return
        self.scan_finished.emit(result)


class ConversionWorker(QThread):
    """后台模型转换。"""

    log_message = pyqtSignal(str)
    file_started = pyqtSignal(str, int, int, int)        # relpath, size, index, total
    file_progress = pyqtSignal(str, int, int, int, int)  # relpath, file_done, file_total, overall_done, overall_total
    file_finished = pyqtSignal(str, int)                 # relpath, overall_done
    overall_progress = pyqtSignal(int, int)              # overall_done, overall_total
    conversion_finished = pyqtSignal(object)             # ConversionResult

    def __init__(self, source: Path, output_dir: Path, files: Sequence[FileInfo],
                 parent=None, backend: str = "auto", max_seq_len: int | None = None,
                 **kwargs):
        super().__init__(parent)
        self._source = Path(source)
        self._output_dir = Path(output_dir)
        self._files = list(files)
        self._backend = backend
        self._max_seq_len = max_seq_len
        self._kwargs = kwargs
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        hooks = _WorkerHooks(self)
        result = None
        try:
            result = self._run_backend(hooks)
        except Exception as exc:  # 兜底：任何未捕获异常也必须通知 UI，否则
            # 主线程永远不会恢复（开始/浏览按钮永久禁用）
            result = ConversionResult(
                False, False, f"转换失败: {type(exc).__name__}: {exc}", None, 0, 0, 0.0)
        finally:
            if result is None:  # 理论上不可达：保证 UI 收到的永远是有效结果
                result = ConversionResult(False, False, "转换失败: 未知错误",
                                          None, 0, 0, 0.0)
            if result.success:
                self.overall_progress.emit(result.bytes_done, result.bytes_done)
            # conversion_finished 必须无条件发出：UI 靠它解除运行中状态
            self.conversion_finished.emit(result)

    def _run_backend(self, hooks) -> ConversionResult:
        if self._backend != "package":
            plan = probe_backend(self._source, requested=self._backend)
            hooks.on_log(plan.describe())
            if plan.name == "coremltools":
                try:
                    result = CoremltoolsBackend(self._source, self._output_dir,
                                                max_seq_len=self._max_seq_len).run(hooks)
                except Exception as exc:
                    # 真实转换失败直接报错，禁止降级回打包后端（打包产物不是
                    # 可用的 Core ML 模型，静默回退会掩盖失败）。
                    # 带上异常类型便于定位（Core ML 的错误常只带一个 dict 描述）
                    message = f"coremltools 转换失败: {type(exc).__name__}: {exc}"
                    hooks.on_log(message)
                    result = ConversionResult(False, False, message, None, 0, 0, 0.0)
                return result
        try:
            return ModelConverter(self._source, self._output_dir,
                                  self._files, **self._kwargs).run(hooks)
        except Exception as exc:  # 转换器内部异常统一兜底，避免线程崩溃
            return ConversionResult(False, False,
                                    f"转换失败: {type(exc).__name__}: {exc}",
                                    None, 0, 0, 0.0)


class _WorkerHooks:
    """把 ModelConverter 的回调桥接到 Qt 信号。"""

    def __init__(self, worker: ConversionWorker):
        self._worker = worker

    def should_cancel(self) -> bool:
        return self._worker._cancelled

    def on_log(self, message: str) -> None:
        self._worker.log_message.emit(message)

    def on_file_started(self, relpath, size, index, total_files):
        self._worker.file_started.emit(relpath, size, index, total_files)

    def on_file_progress(self, relpath, file_done, file_total, overall_done, overall_total):
        self._worker.file_progress.emit(relpath, file_done, file_total,
                                        overall_done, overall_total)
        self._worker.overall_progress.emit(overall_done, overall_total)

    def on_file_finished(self, relpath, overall_done):
        self._worker.file_finished.emit(relpath, overall_done)
