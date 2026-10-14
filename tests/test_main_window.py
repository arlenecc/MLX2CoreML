import pytest

from modconv.main_window import MainWindow


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _prepare_scan(qtbot, window, model_dir):
    window.source_edit.setText(str(model_dir))
    window._refresh_scan()
    qtbot.waitUntil(lambda: window._scan_result is not None, timeout=5000)


def test_widgets_exist(window):
    assert window.source_edit is not None
    assert window.output_edit is not None
    assert window.start_button.text() == "开始转换"
    assert window.cancel_button.text() == "取消"
    assert window.start_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert window.log_view.isReadOnly()


def test_layout_split_is_one_third_two_thirds(qtbot, window):
    """配置区约占 1/3，处理过程显示区约占 2/3。"""
    window.resize(990, 750)
    window.show()
    qtbot.wait(200)
    sizes = window.splitter.sizes()
    total = sum(sizes)
    assert total > 0
    ratio = sizes[0] / total
    assert 0.26 < ratio < 0.40


def test_start_requires_paths(qtbot, window):
    window._on_start_clicked()
    assert window._conv_worker is None
    assert "请先选择" in window.status_label.text()
    assert "请先选择" in window.log_view.toPlainText()


def test_browse_source_triggers_scan(qtbot, window, model_dir, monkeypatch):
    monkeypatch.setattr(
        "modconv.main_window._ask_existing_directory",
        lambda *args, **kwargs: str(model_dir),
    )
    window.source_browse_btn.click()
    qtbot.waitUntil(lambda: window._scan_result is not None, timeout=5000)
    assert window.source_edit.text() == str(model_dir)
    assert "5 个文件" in window.source_stats_label.text()
    assert "预计" in window.estimate_stats_label.text()


def test_full_conversion(qtbot, window, model_dir, tmp_path):
    out = tmp_path / "out"
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(out))
    with qtbot.waitSignal(window.conversion_finished, timeout=20000) as blocker:
        window.start_button.click()
    result = blocker.args[0]
    assert result.success is True
    pkg = out / "tiny-llm.mlpackage"
    assert (pkg / "Manifest.json").is_file()
    assert (pkg / "Data" / "com.apple.CoreML" / "weights" / "sub" / "model.safetensors").is_file()
    assert window.progress_bar.value() == 100
    assert "转换完成" in window.log_view.toPlainText()
    assert window.start_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert window._conv_worker is None


def test_conversion_creates_missing_output_dir(qtbot, window, model_dir, tmp_path):
    out = tmp_path / "deep" / "out"
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(out))
    with qtbot.waitSignal(window.conversion_finished, timeout=20000):
        window.start_button.click()
    assert (out / "tiny-llm.mlpackage" / "Manifest.json").is_file()


def test_reject_output_inside_source(qtbot, window, model_dir):
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(model_dir / "out"))
    window.start_button.click()
    qtbot.wait(200)
    assert window._conv_worker is None
    assert "源目录" in window.status_label.text()


def test_reject_same_source_output(qtbot, window, model_dir):
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(model_dir))
    window.start_button.click()
    qtbot.wait(200)
    assert window._conv_worker is None
    assert "源目录" in window.status_label.text()


def test_pending_start_revalidates_output(qtbot, window, model_dir):
    """扫描期间输出框仍可编辑：由 _pending_start 触发时必须重新校验。

    否则用户在扫描过程中把输出改成源目录内的路径就能绕过检查，
    让输出写进正在被读取的源目录。
    """
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(model_dir / "nested-out"))
    window._pending_start = True
    window._begin_conversion()
    assert window._conv_worker is None
    assert "源目录" in window.status_label.text()


# --------------------------------------------------------------------- #
# 最大上下文长度输入
# --------------------------------------------------------------------- #

def test_max_seq_len_input_defaults_and_parsing(qtbot, window):
    """默认「自动（≤512）」；预设档位与自定义整数（含 k 后缀）生效；非法输入弹窗并返回 -1。

    默认选自动而非 8k：显式大长度会生成 GB 级 KV cache 且在 ANE 上大概率 -14，
    从而触发一长串自动降级重试（每次重做 ANE 编译，耗时极长），自动（≤512）最稳妥。
    """
    assert window.max_seq_combo is not None
    assert window.max_seq_combo.currentText() == "自动（≤512）"
    assert window._max_seq_len_from_input() is None  # 自动

    # 预设档位
    window.max_seq_combo.setCurrentText("8k")
    assert window._max_seq_len_from_input() == 8192
    window.max_seq_combo.setCurrentText("32k")
    assert window._max_seq_len_from_input() == 32768
    window.max_seq_combo.setCurrentText("自动（≤512）")
    assert window._max_seq_len_from_input() is None  # 自动

    # 自定义整数 / k 后缀
    window.max_seq_combo.setCurrentText("1024")
    assert window._max_seq_len_from_input() == 1024
    window.max_seq_combo.setCurrentText(" 512 ")
    assert window._max_seq_len_from_input() == 512
    window.max_seq_combo.setCurrentText("4k")
    assert window._max_seq_len_from_input() == 4096

    # 非法输入：弹窗提示并返回哨兵 -1
    window.max_seq_combo.setCurrentText("abc")
    assert window._max_seq_len_from_input() == -1
    assert "无效" in window.status_label.text()
    window.max_seq_combo.setCurrentText("-3")
    assert window._max_seq_len_from_input() == -1


def test_start_blocked_by_invalid_max_seq(qtbot, window, model_dir, tmp_path):
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(tmp_path / "out"))
    window.max_seq_combo.setCurrentText("abc")
    window.start_button.click()
    qtbot.wait(100)
    assert window._conv_worker is None
    assert "无效" in window.status_label.text()


def test_start_passes_max_seq_len_to_worker(qtbot, window, model_dir, tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    captured = {}

    def _spy_cls(source, output, files, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("modconv.main_window.ConversionWorker", _spy_cls)
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(tmp_path / "out"))
    window.max_seq_combo.setCurrentText("256")
    window.start_button.click()
    qtbot.wait(100)
    assert captured.get("max_seq_len") == 256


# --------------------------------------------------------------------- #
# 权重压缩下拉（fp16 / 4-bit / 6-bit / 8-bit）
# --------------------------------------------------------------------- #

def test_quantize_defaults_to_4bit_and_passes_to_worker(qtbot, window, model_dir,
                                                         tmp_path, monkeypatch):
    """默认做 4-bit 压缩；切到 fp16 / 6-bit / 8-bit 时向 worker 传对应 bit 数。"""
    from unittest.mock import MagicMock

    captured = []

    def _spy_cls(source, output, files, **kwargs):
        captured.append(kwargs)
        return MagicMock()

    monkeypatch.setattr("modconv.main_window.ConversionWorker", _spy_cls)
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(tmp_path / "out"))
    assert window.quant_combo.currentIndex() == 1  # 默认 4-bit

    def _click_and_reset():
        window.start_button.click()
        qtbot.wait(100)
        # 桩 worker 不会真正结束：手动复位运行状态，才能再点一次"开始"
        window._conv_worker = None
        window._conv_workers.clear()
        window._set_running(False)

    _click_and_reset()
    assert captured[-1].get("quantize_nbits") == 4

    # fp16（不压缩）
    window.quant_combo.setCurrentIndex(0)
    _click_and_reset()
    assert captured[-1].get("quantize_nbits") is None

    # 6-bit / 8-bit
    window.quant_combo.setCurrentIndex(2)
    _click_and_reset()
    assert captured[-1].get("quantize_nbits") == 6
    window.quant_combo.setCurrentIndex(3)
    _click_and_reset()
    assert captured[-1].get("quantize_nbits") == 8


def test_quantize_toggle_updates_estimate(qtbot, window, hf_model_dir, tmp_path):
    """切到 fp16 后体积预估应切换为未压缩口径（显著大于 4-bit 估算）。"""
    import re

    _prepare_scan(qtbot, window, hf_model_dir)
    window.output_edit.setText(str(tmp_path / "out"))
    before = window.estimate_stats_label.text()
    assert "4-bit 压缩估算" in before
    window.quant_combo.setCurrentIndex(0)  # fp16（不压缩）
    after = window.estimate_stats_label.text()
    assert "fp16 未压缩估算" in after

    def _bytes(text):
        m = re.search(r"([\d.]+)\s*(B|KB|MB|GB)", text)
        unit = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}[m.group(2)]
        return float(m.group(1)) * unit

    assert _bytes(after) > _bytes(before) * 2


def test_running_disables_quant_combo(window):
    """转换运行中压缩下拉应与其它配置一致被禁用。"""
    window._set_running(True)
    assert not window.quant_combo.isEnabled()
    window._set_running(False)
    assert window.quant_combo.isEnabled()


# --------------------------------------------------------------------- #
# 健壮性与体验
# --------------------------------------------------------------------- #

def test_running_disables_max_seq_combo(window):
    """转换运行中最大上下文长度下拉应与其它配置一致被禁用。"""
    window._set_running(True)
    assert not window.max_seq_combo.isEnabled()
    window._set_running(False)
    assert window.max_seq_combo.isEnabled()


def test_stage_progress_status_text(window):
    """真实转换的阶段进度不应显示为"当前文件"文案。"""
    window._on_file_progress("阶段 2/4", 2, 4, 2, 4)
    text = window.status_label.text()
    assert "阶段 2/4" in text
    assert "当前文件" not in text


def test_file_progress_status_text(window):
    """打包后端的单文件进度仍显示文件级文案。"""
    window._on_file_progress("model.safetensors", 50, 100, 50, 100)
    text = window.status_label.text()
    assert "model.safetensors" in text
    assert "当前文件" in text


def test_close_event_cancels_running_conversion(window):
    """转换运行中关闭窗口：先取消并等待线程退出，再接受关闭（防线程销毁崩溃）。"""
    from unittest.mock import MagicMock

    from PyQt6.QtGui import QCloseEvent

    worker = MagicMock()
    worker.wait.return_value = True
    window._conv_worker = worker
    event = QCloseEvent()
    window.closeEvent(event)
    worker.cancel.assert_called_once()
    worker.wait.assert_called_once()
    assert event.isAccepted()


def test_close_event_blocks_when_thread_unresponsive(window):
    """线程无法在时限内退出时应拒绝关闭，避免销毁运行中的 QThread。"""
    from unittest.mock import MagicMock

    from PyQt6.QtGui import QCloseEvent

    worker = MagicMock()
    worker.wait.return_value = False
    window._conv_worker = worker
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()


def test_close_event_without_worker(window):
    from PyQt6.QtGui import QCloseEvent

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_close_event_waits_for_worker_that_already_reported(window):
    """结果已发出、但线程还在收尾时关闭窗口，也不能漏掉它。

    conversion_finished 是在 run() 内部发出的，此时 _conv_worker 已置空，
    只看它会认为"没有转换在跑"而直接接受关闭 → 销毁运行中的 QThread。
    """
    from unittest.mock import MagicMock

    from PyQt6.QtGui import QCloseEvent

    worker = MagicMock()
    worker.wait.return_value = True
    window._conv_worker = None
    window._conv_workers = [worker]
    event = QCloseEvent()
    window.closeEvent(event)
    worker.cancel.assert_called_once()
    assert event.isAccepted()


def test_start_ignored_while_previous_thread_still_exiting(qtbot, window, model_dir, tmp_path):
    """上一轮线程还在收尾时不应再起一次转换（两个线程会同时写同一输出目录）。"""
    from unittest.mock import MagicMock

    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(tmp_path / "out"))
    window._conv_workers = [MagicMock()]  # 上一轮结果已到、线程未退出
    window.start_button.click()
    qtbot.wait(100)
    assert window._conv_worker is None


def test_worker_reference_released_after_thread_exit(qtbot, window, model_dir, tmp_path):
    """转换结束后不应继续持有 worker（它攥着整份 FileInfo 清单，大目录上是内存滞留）。"""
    out = tmp_path / "out"
    _prepare_scan(qtbot, window, model_dir)
    window.output_edit.setText(str(out))
    with qtbot.waitSignal(window.conversion_finished, timeout=20000):
        window.start_button.click()
    qtbot.waitUntil(lambda: not window._conv_workers, timeout=5000)
    assert window._conv_worker is None


def test_new_scan_cancels_stale_scan(qtbot, window, model_dir):
    """发起新扫描时，结果已作废的旧扫描应被取消（别再抢磁盘带宽）。"""
    from unittest.mock import MagicMock

    stale = MagicMock()
    stale.isRunning.return_value = True
    window._scan_workers = [stale]
    window.source_edit.setText(str(model_dir))
    window._refresh_scan()
    stale.cancel.assert_called_once()
    # 新扫描照常生效（旧扫描的结果不会被采纳）
    qtbot.waitUntil(lambda: window._scan_result is not None, timeout=5000)
    assert "5 个文件" in window.source_stats_label.text()
