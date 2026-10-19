"""QThread 工作线程：把扫描/转换放到后台线程，通过信号跨线程更新 UI，
保证 GUI 主线程不被耗时操作阻塞。"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import Qt, QThread, pyqtSignal

from modconv.backend import BackendPlan, probe_backend
from modconv.converter import ConversionResult, ModelConverter
from modconv.real_backend import CoremltoolsBackend
from modconv.scanner import FileInfo, scan_directory


# 运行中的 QThread 必须至少保有一个 Python 强引用：QThread 在运行期间被销毁会
# 触发 Qt 的 qFatal → 进程直接 abort（Python 层既捕获不到也无法降级）。仅要求
# 调用方"自己记得保留引用"太脆弱——局部变量一出作用域就可能被 GC，而线程通常
# 还要再跑几毫秒。因此这里由基类统一登记/注销：start() 时登记，线程真正结束后
# （finished 已在主线程派发）注销，使任何用法都不会中途销毁线程。
_LIVE_WORKERS: set = set()


class _TrackedThread(QThread):
    """带生命周期保护的 QThread 基类（防止运行中被 GC 导致进程 abort）。"""

    def start(self, *args, **kwargs) -> None:
        _LIVE_WORKERS.add(self)
        # finished 在线程停止后由 Qt 发出；强制排队连接，让注销发生在主线程
        # 事件循环中——此时线程已停止，销毁才是安全的
        self.finished.connect(self._release, Qt.ConnectionType.QueuedConnection)
        super().start(*args, **kwargs)

    def _release(self) -> None:
        _LIVE_WORKERS.discard(self)


class _ScanCancelled(Exception):
    """扫描被取消：用于从扫描回调里提前退出（不算错误）。"""


class ScanWorker(_TrackedThread):
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
        except Exception as exc:
            # 未捕获异常逃出 run() 会让 PyQt 直接结束进程，任何意外都必须转成错误信号
            self.scan_error.emit(f"{type(exc).__name__}: {exc}")
            return
        self.scan_finished.emit(result)


class ConversionWorker(_TrackedThread):
    """后台模型转换。"""

    log_message = pyqtSignal(str)
    file_started = pyqtSignal(str, int, int, int)        # relpath, size, index, total
    file_progress = pyqtSignal(str, int, int, int, int)  # relpath, file_done, file_total, overall_done, overall_total
    file_finished = pyqtSignal(str, int)                 # relpath, overall_done
    overall_progress = pyqtSignal(int, int)              # overall_done, overall_total
    conversion_finished = pyqtSignal(object)             # ConversionResult

    def __init__(self, source: Path, output_dir: Path, files: Sequence[FileInfo],
                 parent=None, backend: str = "auto", target: str = "coreml",
                 max_seq_len: int | None = None,
                 quantize_nbits: int | None = 4, quant_mode: str = "fp16",
                 model_kind: str = "auto", **kwargs):
        super().__init__(parent)
        self._source = Path(source)
        self._output_dir = Path(output_dir)
        self._files = list(files)
        self._backend = backend
        self._target = target
        self._max_seq_len = max_seq_len
        # 权重压缩位数：只有真实转换（Core ML）后端用得到；打包后端忽略
        self._quantize_nbits = quantize_nbits
        # 量化模式（跨后端统一）：fp16 / int4 / int8（Core ML 另支持 4bit/6bit/8bit 调色板）
        self._quant_mode = quant_mode
        # HF→ONNX 的模型类型提示：auto / causal / embedding / reranker
        self._model_kind = model_kind
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
            # 转换成功后自动验证产物可用性（打包后端无可推理产物，跳过）
            if result.success and self._backend != "package" and result.output_path:
                try:
                    from modconv.verify import verify_conversion
                    ver = verify_conversion(
                        self._target, result.output_path, self._source, hooks,
                        model_kind=self._model_kind, quant_mode=self._quant_mode)
                    result = ConversionResult(
                        result.success, result.cancelled, result.message,
                        result.output_path, result.files_done, result.bytes_done,
                        result.elapsed_seconds, verification=ver)
                except Exception as exc:  # 验证异常不覆盖转换结果
                    try:
                        hooks.on_log(f"验证步骤异常（已忽略）: {type(exc).__name__}: {exc}")
                    except Exception:
                        pass
            if result.success:
                self.overall_progress.emit(result.bytes_done, result.bytes_done)
            # conversion_finished 必须无条件发出：UI 靠它解除运行中状态
            self.conversion_finished.emit(result)

    def _run_backend(self, hooks) -> ConversionResult:
        # 强制打包：跳过探测，直接 1:1 拷贝（不可直接推理）
        if self._backend == "package":
            plan = BackendPlan("package", "用户指定打包后端", available=True)
        else:
            plan = probe_backend(self._source, requested=self._backend, target=self._target)
        hooks.on_log(plan.describe())
        if not plan.available:
            # 依赖缺失 / 源-目标不匹配：明确报错，不静默降级为打包产物
            return ConversionResult(False, False,
                                   f"无法执行该转换: {plan.reason}", None, 0, 0, 0.0)
        try:
            if plan.name == "coremltools":
                # Core ML 的量化模式映射到 n-bit 调色板（palettization）
                _QMODE_TO_NBITS = {"fp16": None, "4bit": 4, "6bit": 6, "8bit": 8}
                nbits = _QMODE_TO_NBITS.get(self._quant_mode, self._quantize_nbits)
                return CoremltoolsBackend(self._source, self._output_dir,
                                          max_seq_len=self._max_seq_len,
                                          quantize_nbits=nbits).run(hooks)
            if plan.name == "onnx":
                if plan.task == "onnx->onnx":
                    from modconv.onnx_backend import OnnxToOnnxBackend
                    return OnnxToOnnxBackend(self._source, self._output_dir,
                                             quant_mode=self._quant_mode).run(hooks)
                if plan.task == "mlx->onnx":
                    from modconv.onnx_backend import MlxToOnnxBackend
                    return MlxToOnnxBackend(self._source, self._output_dir,
                                            quant_mode=self._quant_mode,
                                            model_kind=self._model_kind).run(hooks)
                if plan.task == "gguf->onnx":
                    from modconv.onnx_backend import GgufToOnnxBackend
                    return GgufToOnnxBackend(self._source, self._output_dir,
                                            quant_mode=self._quant_mode).run(hooks)
                from modconv.onnx_backend import HfToOnnxBackend
                return HfToOnnxBackend(self._source, self._output_dir,
                                       quant_mode=self._quant_mode,
                                       model_kind=self._model_kind).run(hooks)
            if plan.name == "mlx":
                from modconv.mlx_backend import OnnxToMlxBackend
                return OnnxToMlxBackend(self._source, self._output_dir,
                                        quant_mode=self._quant_mode).run(hooks)
            if plan.name == "mlx_rekey":
                from modconv.mlx_rekey_backend import MlxToMlxBackend
                return MlxToMlxBackend(self._source, self._output_dir,
                                       quant_mode=self._quant_mode).run(hooks)
            if plan.name == "mlx_libs":
                if plan.task == "gguf->mlx":
                    from modconv.gguf_to_mlx_backend import GgufToMlxBackend
                    return GgufToMlxBackend(self._source, self._output_dir,
                                            quant_mode=self._quant_mode).run(hooks)
                from modconv.mlx_libs_backend import HfToMlxBackend
                return HfToMlxBackend(self._source, self._output_dir,
                                      quant_mode=self._quant_mode).run(hooks)
            if plan.name == "gguf":
                from modconv.gguf_backend import GgufBackend
                return GgufBackend(self._source, self._output_dir,
                                   quant_mode=self._quant_mode).run(hooks)
            # 兜底：1:1 打包（不可直接推理）
            return ModelConverter(self._source, self._output_dir,
                                  self._files, **self._kwargs).run(hooks)
        except Exception as exc:
            # 真实转换失败直接报错（含 Core ML 那种只带 dict 描述的错误），禁止静默回退
            message = f"转换失败: {type(exc).__name__}: {exc}"
            hooks.on_log(message)
            return ConversionResult(False, False, message, None, 0, 0, 0.0)


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
