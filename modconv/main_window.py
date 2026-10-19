"""PyQt6 主窗口。

布局：
- 上方 1/3：配置区（MLX 模型目录、Core ML 输出目录）；
- 下方 2/3：处理过程显示区（统计行 + 开始/取消按钮 + 日志）；
- 状态栏：总体处理进度。

所有耗时操作（目录扫描、模型转换）都在后台线程执行，
通过信号跨线程更新 UI，保证大模型转换时界面不卡顿。
"""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QEventLoop, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from modconv.backend import SUPPORTED_MODEL_TYPES, detect_model_type, real_backend_available
from modconv.estimator import estimate_output
from modconv.scanner import ScanResult
from modconv.utils import format_size
from modconv.workers import ConversionWorker, ScanWorker


def _ask_existing_directory(parent, caption: str, default: str) -> str:
    """目录选择对话框（独立函数，便于测试替换）。"""
    return QFileDialog.getExistingDirectory(parent, caption, default)


# 关闭窗口时等待转换线程退出的上限：超过则拒绝关闭（避免销毁运行中的 QThread 导致崩溃）
_CLOSE_WAIT_MS = 30_000
# 扫描线程较短，但大目录也可能数秒；取消后仍需等它真正退出
_SCAN_WAIT_MS = 3_000
# 手动输入源目录后的扫描防抖间隔
_SCAN_DEBOUNCE_MS = 400


class MainWindow(QMainWindow):
    scan_completed = pyqtSignal(object)       # ScanResult
    conversion_finished = pyqtSignal(object)  # ConversionResult

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLX → Core ML 模型转换工具")
        self.resize(1100, 780)
        self.setMinimumSize(860, 560)

        self._scan_result: ScanResult | None = None
        self._scan_source: Path | None = None
        self._scan_gen = 0
        self._scan_workers: list[ScanWorker] = []
        # _conv_worker：当前（或最近一次）转换，供"取消"使用；
        # _conv_workers：所有尚未退出的转换线程。二者分开是因为 conversion_finished
        # 是在线程 run() 内部发出的，此时线程还要再跑几毫秒才真正退出——若在那
        # 一刻就把唯一引用置空，关闭窗口会漏掉它（销毁运行中的 QThread 会崩溃）。
        self._conv_worker: ConversionWorker | None = None
        self._conv_workers: list[ConversionWorker] = []
        self._pending_start = False
        self._splitter_configured = False

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        outer.addWidget(self.splitter)

        self.splitter.addWidget(self._build_config_panel())
        self.splitter.addWidget(self._build_process_panel())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)

        self._build_status_bar()

    # ------------------------------------------------------------------ #
    # 界面构建
    # ------------------------------------------------------------------ #
    def _build_config_panel(self) -> QWidget:
        panel = QGroupBox("转换配置")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 12, 12, 12)

        grid.addWidget(QLabel("模型目录:"), 0, 0)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择模型目录（HF 模型或 ONNX 文件）")
        grid.addWidget(self.source_edit, 0, 1)
        self.source_browse_btn = QPushButton("浏览...")
        self.source_browse_btn.clicked.connect(self._browse_source)
        grid.addWidget(self.source_browse_btn, 0, 2)
        self.onnx_browse_btn = QPushButton("选 ONNX 文件...")
        self.onnx_browse_btn.clicked.connect(self._browse_onnx_source)
        grid.addWidget(self.onnx_browse_btn, 0, 3)
        self.gguf_browse_btn = QPushButton("选 GGUF 文件...")
        self.gguf_browse_btn.clicked.connect(self._browse_gguf_source)
        grid.addWidget(self.gguf_browse_btn, 0, 4)

        grid.addWidget(QLabel("输出目录:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择转换产物的存放目录")
        grid.addWidget(self.output_edit, 1, 1)
        self.output_browse_btn = QPushButton("浏览...")
        self.output_browse_btn.clicked.connect(self._browse_output)
        grid.addWidget(self.output_browse_btn, 1, 2)

        # 目标格式：决定走哪条转换链路（Core ML / ONNX / MLX / GGUF）
        grid.addWidget(QLabel("目标格式:"), 2, 0)
        self.target_combo = QComboBox()
        self._TARGET_ITEMS = [("Core ML (.mlpackage)", "coreml"),
                              ("ONNX (.onnx)", "onnx"),
                              ("MLX (Apple)", "mlx"),
                              ("GGUF (llama.cpp)", "gguf")]
        for label, data in self._TARGET_ITEMS:
            self.target_combo.addItem(label, data)
        self.target_combo.setCurrentIndex(0)
        self.target_combo.currentIndexChanged.connect(lambda _: self._on_target_changed())
        grid.addWidget(self.target_combo, 2, 1)

        # 最大上下文长度：仅 Core ML（带 KV cache 的 stateful 模型）需要
        self.max_seq_label = QLabel("最大上下文长度:")
        grid.addWidget(self.max_seq_label, 3, 0)
        self.max_seq_combo = QComboBox()
        self.max_seq_combo.setEditable(True)
        self._MAX_SEQ_PRESETS = [
            ("自动（≤512）", None), ("2k", 2048), ("4k", 4096), ("8k", 8192),
            ("16k", 16384), ("32k", 32768), ("64k", 65536),
            ("128k", 131072), ("256k", 262144),
        ]
        self.max_seq_combo.addItems([label for label, _ in self._MAX_SEQ_PRESETS])
        self.max_seq_combo.setCurrentIndex(0)  # 默认「自动（≤512）」
        self.max_seq_combo.setToolTip(
            "转换输出的最大上下文长度（token 数），仅 Core ML 目标需要。\n"
            "选「自动」时取 min(模型位置编码上限, 512)，最稳妥。")
        grid.addWidget(self.max_seq_combo, 3, 1)

        # 权重压缩：选项随目标格式自适应（Core ML 用 n-bit；ONNX/MLX 用 int4/int8）
        grid.addWidget(QLabel("权重压缩:"), 4, 0)
        self.quant_combo = QComboBox()
        self._QUANT_COREML = [("fp16（不压缩）", "fp16"),
                              ("4-bit", "4bit"), ("6-bit", "6bit"), ("8-bit", "8bit")]
        self._QUANT_ONNX_MLX = [("fp16（不压缩）", "fp16"),
                                ("int8", "int8"), ("int4", "int4")]
        self.quant_combo.currentIndexChanged.connect(lambda _: self._refresh_estimate())
        grid.addWidget(self.quant_combo, 4, 1)

        # 模型类型：仅 HF→ONNX 时需要（自动判定或手动指定）
        self.kind_label = QLabel("模型类型 (HF→ONNX):")
        grid.addWidget(self.kind_label, 5, 0)
        self.kind_combo = QComboBox()
        self._KIND_ITEMS = [("自动", "auto"), ("文本生成", "causal"),
                            ("嵌入", "embedding"), ("重排序", "reranker")]
        for label, data in self._KIND_ITEMS:
            self.kind_combo.addItem(label, data)
        self.kind_combo.setCurrentIndex(0)
        grid.addWidget(self.kind_combo, 5, 1)

        # 初始化目标相关的可选项与可用性
        self._refresh_quant_options()
        self._on_target_changed()

        # 手动输入路径后延迟触发扫描：连续敲键盘/多次回车会各起一次全目录扫描
        # （旧扫描无法取消，还会互相抢占），防抖后只保留最后一次
        self._scan_debounce = QTimer(self)
        self._scan_debounce.setSingleShot(True)
        self._scan_debounce.setInterval(_SCAN_DEBOUNCE_MS)
        self._scan_debounce.timeout.connect(self._refresh_scan)
        self.source_edit.editingFinished.connect(lambda: self._scan_debounce.start())
        return panel

    def _build_process_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)

        stats_row = QHBoxLayout()
        self.source_stats_label = QLabel("MLX 模型目录: 未扫描")
        self.estimate_stats_label = QLabel("预计 Core ML 输出: —")
        stats_row.addWidget(self.source_stats_label)
        stats_row.addWidget(self.estimate_stats_label)
        stats_row.addStretch(1)
        self.start_button = QPushButton("开始转换")
        self.start_button.setMinimumWidth(120)
        self.start_button.clicked.connect(self._on_start_clicked)
        stats_row.addWidget(self.start_button)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        stats_row.addWidget(self.cancel_button)
        layout.addLayout(stats_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("处理过程将显示在这里…")
        layout.addWidget(self.log_view, 1)
        return panel

    def _build_status_bar(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedWidth(260)
        self.status_label = QLabel("就绪")
        status_bar = self.statusBar()
        status_bar.addWidget(QLabel(" 总进度:"))
        status_bar.addWidget(self.progress_bar)
        status_bar.addPermanentWidget(self.status_label)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitter_configured:
            self._splitter_configured = True
            QTimer.singleShot(0, self._apply_splitter_ratio)

    def _apply_splitter_ratio(self):
        """按 1/3 : 2/3 分配配置区与处理过程显示区。"""
        total = self.splitter.height()
        if total > 0:
            top = max(1, total // 3)
            self.splitter.setSizes([top, total - top])

    # ------------------------------------------------------------------ #
    # 扫描
    # ------------------------------------------------------------------ #
    def _browse_source(self):
        default = self.source_edit.text() or str(Path.home())
        path = _ask_existing_directory(self, "选择 MLX 模型目录", default)
        if path:
            self.source_edit.setText(path)
            self._refresh_scan()

    def _browse_onnx_source(self):
        """选择单个 .onnx 文件作为源（GGUF/MLX 目标支持 ONNX 源）。"""
        default = str(Path(self.source_edit.text() or Path.home()).parent)
        path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 模型文件", default, "ONNX (*.onnx)")
        if path:
            self.source_edit.setText(path)
            self._refresh_scan()

    def _browse_gguf_source(self):
        """选择单个 .gguf 文件作为源（MLX/ONNX 目标支持 GGUF 源）。"""
        default = str(Path(self.source_edit.text() or Path.home()).parent)
        path, _ = QFileDialog.getOpenFileName(self, "选择 GGUF 模型文件", default, "GGUF (*.gguf)")
        if path:
            self.source_edit.setText(path)
            self._refresh_scan()

    def _browse_output(self):
        default = self.output_edit.text() or str(Path.home())
        path = _ask_existing_directory(self, "选择 Core ML 输出目录", default)
        if path:
            self.output_edit.setText(path)

    def _refresh_scan(self):
        # 直接调用（开始转换前补扫描 / 浏览按钮选目录）时取消待触发的防抖，
        # 避免紧接着又起一次重复的扫描
        self._scan_debounce.stop()
        text = self.source_edit.text().strip()
        # 任何"放弃本次扫描"的分支都要递增 _scan_gen：否则仍在飞行的旧扫描
        # 结果回来后会覆盖这里刚写入的状态（UI 显示已失效目录的统计）
        self._scan_gen += 1
        # 上一次扫描的结果已因 gen 递增而作废，继续跑只会白占 CPU/IO、和本次
        # 扫描抢磁盘带宽（大目录上很明显），无论本次是否真的再起扫描都先取消
        for stale in self._scan_workers:
            if self._thread_alive(stale):
                stale.cancel()
        if not text:
            self._scan_result = None
            self._scan_source = None
            self._pending_start = False
            self.source_stats_label.setText("源模型目录: 未扫描")
            self.estimate_stats_label.setText("预计 Core ML 输出: —")
            return
        path = Path(text).expanduser()
        if not path.is_dir():
            self._scan_result = None
            self._scan_source = None
            self._pending_start = False
            self.source_stats_label.setText("源模型目录: 无效路径")
            self.estimate_stats_label.setText("预计 Core ML 输出: —")
            self._set_status(f"无效的模型目录: {path}")
            return
        gen = self._scan_gen
        self.source_stats_label.setText("正在扫描源模型目录…")
        worker = ScanWorker(path)
        worker.scan_progress.connect(lambda c, b, g=gen: self._on_scan_progress(g, c, b))
        worker.scan_finished.connect(lambda r, g=gen: self._on_scan_finished(g, r))
        worker.scan_error.connect(lambda m, g=gen: self._on_scan_error(g, m))
        self._scan_workers = [w for w in self._scan_workers if self._thread_alive(w)]
        self._scan_workers.append(worker)
        # 线程结束后立刻从持有列表移除，避免长期保留 QThread 对象（含其扫描结果）
        worker.finished.connect(lambda w=worker: self._forget_scan_worker(w))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _forget_scan_worker(self, worker) -> None:
        try:
            self._scan_workers.remove(worker)
        except ValueError:
            pass

    def _forget_conv_worker(self, worker) -> None:
        try:
            self._conv_workers.remove(worker)
        except ValueError:
            pass

    def _on_scan_progress(self, gen: int, count: int, size: int):
        if gen != self._scan_gen:
            return
        self.source_stats_label.setText(f"扫描中… 已发现 {count} 个文件, {format_size(size)}")

    def _on_scan_error(self, gen: int, message: str):
        if gen != self._scan_gen:
            return
        self._scan_result = None
        # 扫描失败时丢弃"扫描完成后自动开始转换"的挂起请求，
        # 否则它会在下一次（可能由用户主动触发的）扫描后被意外消费
        self._pending_start = False
        self.source_stats_label.setText(f"扫描失败: {message}")

    def _selected_quant_mode(self) -> str:
        """当前压缩下拉框对应的量化档位字符串（fp16/4bit/6bit/8bit/int4/int8）。"""
        return self.quant_combo.currentData()

    def _refresh_quant_options(self):
        """按目标格式刷新压缩下拉框可选项（Core ML 用 n-bit；ONNX/MLX 用 int4/int8）。"""
        target = self.target_combo.currentData()
        items = self._QUANT_COREML if target == "coreml" else self._QUANT_ONNX_MLX
        self.quant_combo.blockSignals(True)
        self.quant_combo.clear()
        for label, mode in items:
            self.quant_combo.addItem(label, mode)
        # 默认：Core ML 选 4-bit（沿用历史默认）；ONNX/MLX 选 fp16（不压缩）
        self.quant_combo.setCurrentIndex(1 if target == "coreml" else 0)
        self.quant_combo.blockSignals(False)

    def _on_target_changed(self):
        target = self.target_combo.currentData()
        self._refresh_quant_options()
        coreml = target == "coreml"
        self.max_seq_combo.setEnabled(coreml)
        self.max_seq_label.setEnabled(coreml)
        onnx = target == "onnx"
        self.kind_combo.setEnabled(onnx)
        self.kind_label.setEnabled(onnx)
        self._refresh_estimate()

    def _refresh_estimate(self):
        """按当前扫描结果 + 目标格式 + 压缩档位刷新体积预估。"""
        scan = self._scan_result
        if scan is None:
            return
        target = self.target_combo.currentData()
        mode = self._selected_quant_mode()
        if target == "coreml":
            # 体积预估按实际后端口径：架构受支持「且」真实转换依赖齐备时走真实
            # 转换，否则按打包 1:1。只做轻量读取（config.json / find_spec）。
            model_type = detect_model_type(scan.source)
            real_ok = model_type in SUPPORTED_MODEL_TYPES and real_backend_available()
            backend = "coremltools" if real_ok else "package"
            nbits = {"fp16": None, "4bit": 4, "6bit": 6, "8bit": 8}.get(mode)
            estimate = estimate_output(scan, backend=backend,
                                       compressed=(nbits is not None), nbits=nbits)
            note = ("（打包口径）" if backend == "package"
                    else f"（{nbits}-bit 压缩估算）" if nbits
                    else "（fp16 未压缩估算）")
            label = "预计 Core ML 输出"
        else:
            backend = target  # "onnx" / "mlx" / "gguf"
            estimate = estimate_output(scan, backend=backend,
                                       compressed=(mode != "fp16"), nbits=mode)
            note = "" if mode == "fp16" else f"（{mode} 量化估算）"
            label = {"onnx": "预计 ONNX 输出", "mlx": "预计 MLX 输出",
                     "gguf": "预计 GGUF 输出"}[target]
        self.estimate_stats_label.setText(
            f"{label}: ≈{estimate.file_count} 个文件, "
            f"{format_size(estimate.total_bytes)}{note}")

    def _on_scan_finished(self, gen: int, result: ScanResult):
        if gen != self._scan_gen:
            return
        self._scan_result = result
        self._scan_source = result.source
        self.source_stats_label.setText(
            f"源模型目录: {result.count} 个文件, {format_size(result.total_bytes)}")
        self._refresh_estimate()
        # 不支持/无法识别的架构、或真实转换依赖缺失，都会在转换时才暴露
        # （转换失败，或静默退回打包产物），扫描阶段就提示，避免用户等半天才
        # 发现走了兜底路径
        model_type = detect_model_type(result.source)
        if model_type is None and result.source.is_file():
            pass  # 单文件源（.onnx）：没有 config.json，跳过架构提示
        elif model_type not in SUPPORTED_MODEL_TYPES:
            arch = model_type or "未识别（缺少或非法的 config.json）"
            hint = (f"架构 '{arch}' 暂不支持真实转换，将使用打包后端"
                    if model_type else f"{arch}，将使用打包后端")
            self._log(hint)
            self._set_status(hint)
        elif not real_backend_available():
            hint = ("真实转换依赖（torch / transformers / coremltools）缺失，"
                    "将使用打包后端（产物不可直接推理）")
            self._log(hint)
            self._set_status(hint)
        self.scan_completed.emit(result)
        if self._pending_start:
            self._pending_start = False
            self._begin_conversion()

    # ------------------------------------------------------------------ #
    # 转换
    # ------------------------------------------------------------------ #
    def _on_start_clicked(self):
        # 上一轮转换的线程可能还在收尾（结果已到但未退出），此时也不能再起一次：
        # 两个转换线程同时写同一个输出目录会互相破坏
        if self._conv_worker is not None or self._conv_workers:
            return
        source = self._source_path_from_input()
        if source is None:
            return
        output = self._output_path_from_input()
        if output is None:
            return
        if not self._validate_output(source, output):
            return
        if self._scan_result is None or self._scan_source != source:
            self._pending_start = True
            self._refresh_scan()
            self._set_status("正在扫描模型目录，扫描完成后自动开始转换…")
            return
        self._begin_conversion()

    def _source_path_from_input(self) -> Path | None:
        text = self.source_edit.text().strip()
        if not text:
            self._reject_start("请先选择模型目录、.onnx 或 .gguf 文件")
            return None
        source = Path(text).expanduser()
        # 目录（HF/MLX/ONNX）或单个 .onnx / .gguf 文件均可作为源
        if not source.is_dir() and not (
                source.is_file() and source.suffix.lower() in (".onnx", ".gguf")):
            self._reject_start(f"模型目录或文件不存在: {source}")
            return None
        return source

    def _output_path_from_input(self) -> Path | None:
        text = self.output_edit.text().strip()
        if not text:
            self._reject_start("请先选择 Core ML 输出目录")
            return None
        return Path(text).expanduser()

    def _validate_output(self, source: Path, output: Path) -> bool:
        if self._is_output_inside_source(source, output):
            self._reject_start("输出目录不能与源目录相同，也不能位于源目录内部")
            return False
        return self._ensure_output_dir(output)

    def _reject_start(self, message: str):
        self._set_status(message)
        self._log(message)

    def _max_seq_len_from_input(self) -> int | None:
        """解析最大上下文长度：预设档位 / 自定义整数（支持 k、m 后缀）。

        返回 None 表示自动（空或「自动」项）；正整数表示自定义；
        非法输入弹窗提示并返回哨兵 -1，调用方据此拦截启动。
        """
        text = self.max_seq_combo.currentText().strip()
        if not text or "自动" in text:
            return None
        mult, low = 1, text.lower()
        if low.endswith("k"):
            mult, text = 1024, text[:-1]
        elif low.endswith("m"):
            mult, text = 1024 * 1024, text[:-1]
        try:
            value = int(text) * mult
        except ValueError:
            value = 0
        if value <= 0:
            self._reject_start(
                f"最大上下文长度无效: {text}（应为正整数/预设档位，留空或选「自动」为自动）")
            return -1
        return value

    @staticmethod
    def _is_output_inside_source(source: Path, output: Path) -> bool:
        try:
            src = source.resolve()
            out = output.resolve()
        except OSError:
            src, out = source, output
        return out == src or src in out.parents

    def _ensure_output_dir(self, output: Path) -> bool:
        if output.is_dir():
            return True
        try:
            output.mkdir(parents=True, exist_ok=True)
            self._log(f"已创建输出目录: {output}")
            return True
        except OSError as exc:
            self._reject_start(f"无法创建输出目录: {exc}")
            return False

    def _begin_conversion(self):
        if self._scan_result is None or self._conv_worker is not None or self._conv_workers:
            return
        # 最大上下文长度：None = 自动；-1 = 非法（已在解析时弹窗），直接拦截
        max_seq_len = self._max_seq_len_from_input()
        if max_seq_len == -1:
            return
        source = self._scan_result.source
        # 扫描期间输出目录输入框是可编辑的，由 _pending_start 触发时必须重新校验：
        # 否则用户改成一个位于源目录内的路径就能绕过 _on_start_clicked 的检查
        output = self._output_path_from_input()
        if output is None:
            return
        if not self._validate_output(source, output):
            return
        self._set_running(True)
        self.progress_bar.setValue(0)
        self._log(f"开始转换: {source.name} → {output}")
        target = self.target_combo.currentData()
        quant_mode = self._selected_quant_mode()
        model_kind = self.kind_combo.currentData() if target == "onnx" else "auto"
        worker = ConversionWorker(source, output, self._scan_result.files,
                                  target=target, max_seq_len=max_seq_len,
                                  quant_mode=quant_mode, model_kind=model_kind)
        self._conv_worker = worker
        self._conv_workers.append(worker)
        worker.log_message.connect(self._log)
        worker.file_started.connect(self._on_file_started)
        worker.file_progress.connect(self._on_file_progress)
        worker.overall_progress.connect(self._on_overall_progress)
        worker.conversion_finished.connect(self._on_conversion_finished)
        # 线程真正退出后才解除持有：此后 deleteLater 已经排好队，交回 Qt 即可。
        # 不再用"退役列表"长期保留——那些 worker 还攥着整份 FileInfo 清单，
        # 大目录上是实打实的内存滞留。
        worker.finished.connect(lambda w=worker: self._forget_conv_worker(w))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_file_started(self, relpath: str, size: int, index: int, total: int):
        self._set_status(f"({index}/{total}) 正在转换: {relpath} ({format_size(size)})")
        self._log(f"[{index}/{total}] 转换 {relpath} ({format_size(size)})")

    def _on_file_progress(self, relpath: str, file_done: int, file_total: int,
                          overall_done: int, overall_total: int):
        overall_pct = overall_done * 100 // max(1, overall_total)
        if relpath.startswith("阶段"):
            # coremltools 后端以转换阶段为进度单位，没有"当前文件"概念
            self._set_status(f"{relpath} ({overall_pct}%)")
            return
        file_pct = (file_done * 100 // file_total) if file_total else 100
        self._set_status(f"当前文件: {relpath} {file_pct}% (总计 {overall_pct}%)")

    def _on_overall_progress(self, done: int, total: int):
        pct = int(done * 100 / total) if total else 100
        self.progress_bar.setValue(min(100, pct))

    def _on_conversion_finished(self, result):
        # 线程此刻可能仍在退出途中，引用仍留在 _conv_workers 里，不能在这里丢弃
        self._conv_worker = None
        self._pending_start = False
        self._set_running(False)
        prefix = "✓ " if result.success else ("⏹ " if result.cancelled else "✗ ")
        self._log(prefix + result.message)
        # 自动验证结果（转换完成后对产物做可用性/保真校验）
        if result.success and result.verification is not None:
            v = result.verification
            if v.skipped:
                self._log(f"  验证: 跳过（{v.summary}）")
            elif v.passed:
                self._log(f"  验证通过 ✓ {v.summary}")
            else:
                self._log(f"  验证未通过 ✗ {v.summary}")
        self._set_status(result.message)
        if result.success:
            self.progress_bar.setValue(100)
        self.conversion_finished.emit(result)

    def _on_cancel_clicked(self):
        # _conv_worker 已置空但线程还没退出时（结果已到、收尾中），仍要能取消
        targets = self._conv_workers or ([self._conv_worker] if self._conv_worker else [])
        if not targets:
            return
        for worker in targets:
            worker.cancel()
        self.cancel_button.setEnabled(False)
        self._log("已请求取消转换，等待当前块结束…")

    # ------------------------------------------------------------------ #
    # 公共小工具
    # ------------------------------------------------------------------ #
    def _log(self, message: str):
        # 只在用户没有手动往上翻看历史时自动跟随到底部，避免打断阅读
        bar = self.log_view.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")
        if follow:
            bar.setValue(bar.maximum())

    def _set_status(self, message: str):
        self.status_label.setText(message)

    def _set_running(self, running: bool):
        for widget in (self.source_edit, self.output_edit, self.max_seq_combo,
                       self.quant_combo, self.source_browse_btn,
                       self.output_browse_btn, self.start_button):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    # ------------------------------------------------------------------ #
    # 关闭保护
    # ------------------------------------------------------------------ #
    @staticmethod
    def _thread_alive(worker) -> bool:
        """线程是否仍在运行；C++ 对象已被 deleteLater 销毁时视为已结束。"""
        try:
            return worker.isRunning()
        except RuntimeError:
            return False

    @staticmethod
    def _wait_for_thread(worker, budget_ms: int) -> bool:
        """等待线程退出，期间持续处理事件循环（避免界面长时间无响应）。

        单次 wait(预算) 会把主线程整段阻塞到超时，期间窗口无法重绘、macOS 显示
        转圈；改为小步等待 + processEvents，既不改变语义也能保持界面可重绘。
        """
        step = 100
        remaining = budget_ms
        while remaining > 0:
            if worker.wait(min(step, remaining)):
                return True
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, step)
            remaining -= step
        return worker.wait(0)

    def closeEvent(self, event):
        """关闭窗口前确保后台线程已退出：QThread 运行中被销毁会导致进程崩溃。"""
        # 防抖定时器还有一次待触发的扫描：先停掉，避免关闭后又起一个线程
        self._scan_debounce.stop()
        # 已发出 conversion_finished、但线程尚未真正退出的转换线程也必须等：
        # 只看 _conv_worker 会漏掉这段窗口期
        conv_workers = list(self._conv_workers)
        if self._conv_worker is not None and self._conv_worker not in conv_workers:
            conv_workers.append(self._conv_worker)
        if conv_workers:
            self._log("窗口关闭：已请求取消转换，等待后台线程退出…")
        for worker in conv_workers:
            worker.cancel()
            if not self._wait_for_thread(worker, _CLOSE_WAIT_MS):
                # 深度转换阶段（反量化/图转换）不可中断，超时则拒绝关闭
                event.ignore()
                self._set_status("后台转换尚未退出（当前阶段不可中断），请稍后再关闭窗口")
                return
        # 扫描线程也要先请求取消再等待：大目录扫描可能远超 2 秒，
        # 单纯 wait 超时后接受关闭会让线程在对象销毁后继续运行而崩溃
        for scan_worker in list(self._scan_workers):
            if not self._thread_alive(scan_worker):
                continue
            scan_worker.cancel()
            if not self._wait_for_thread(scan_worker, _SCAN_WAIT_MS):
                event.ignore()
                self._set_status("后台扫描尚未退出，请稍后再关闭窗口")
                return
        event.accept()
