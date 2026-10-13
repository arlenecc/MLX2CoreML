from pathlib import Path

from modconv.backend import BackendPlan
from modconv.converter import ConversionResult
from modconv.scanner import scan_directory
from modconv.workers import ConversionWorker, ScanWorker


class _FailingBackend:
    def __init__(self, source, output_dir, **kwargs):
        pass

    def run(self, hooks):
        raise RuntimeError("boom")


class _SuccessBackend:
    def __init__(self, source, output_dir, **kwargs):
        self._output_dir = Path(output_dir)

    def run(self, hooks):
        return ConversionResult(True, False, "real ok",
                                self._output_dir / "real.mlpackage", 1, 100, 0.5)


def test_scan_worker_emits_result(qtbot, model_dir):
    worker = ScanWorker(model_dir)
    with qtbot.waitSignal(worker.scan_finished, timeout=5000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert result.count == 5
    assert result.total_bytes == 20 + 21 + 500 + 64 + (1 << 20)


def test_scan_worker_reports_error(qtbot, tmp_path):
    worker = ScanWorker(tmp_path / "missing")
    with qtbot.waitSignal(worker.scan_error, timeout=5000) as blocker:
        worker.start()
    assert "missing" in blocker.args[0]


def test_scan_worker_cancel_stops_early(qtbot, model_dir):
    """取消后不应再发出 scan_finished（否则 UI 会接受已作废的扫描结果）。"""
    worker = ScanWorker(model_dir)
    finished = []
    worker.scan_finished.connect(lambda r: finished.append(r))
    worker.cancel()
    worker.start()
    assert worker.wait(5000)
    assert finished == []


def test_running_worker_survives_losing_external_reference(qtbot, model_dir):
    """运行中丢掉最后一个外部引用也不能让 QThread 被销毁。

    QThread 在运行期间被销毁会触发 Qt 的 qFatal（进程直接 abort，Python 层既
    捕获不到也无法降级）。只靠调用方"记得保留引用"不够——局部变量一出作用域
    就可能被 GC，而线程往往还要再跑几毫秒。基类必须在 start()/finished 之间
    自己持有一份引用，并在线程真正结束后释放（否则就是内存泄漏）。
    """
    import gc

    from modconv import workers as workers_mod

    worker = ScanWorker(model_dir)
    worker.start()
    assert worker in workers_mod._LIVE_WORKERS
    del worker  # 调用方彻底放手
    gc.collect()
    assert workers_mod._LIVE_WORKERS, "运行中的线程必须仍被登记保护"
    # 线程结束后登记要解除，不能一直挂着（Worker 还攥着整份文件清单）
    qtbot.waitUntil(lambda: not workers_mod._LIVE_WORKERS, timeout=5000)


def test_conversion_worker_success(qtbot, model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files)
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert result.success is True
    assert (out / "tiny-llm.mlpackage" / "Manifest.json").is_file()


def test_conversion_worker_cancel_before_start(qtbot, model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files)
    worker.cancel()
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000) as blocker:
        worker.start()
    assert blocker.args[0].cancelled is True


def test_conversion_worker_overall_progress_reaches_total(qtbot, model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files)
    totals = []
    worker.overall_progress.connect(lambda done, total: totals.append((done, total)))
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000):
        worker.start()
    assert totals
    assert totals[-1][0] == totals[-1][1] > 0


def test_worker_no_fallback_when_real_backend_fails(qtbot, model_dir, tmp_path, monkeypatch):
    """真实转换抛异常时直接报错——禁止自动降级回打包后端。"""
    monkeypatch.setattr(
        "modconv.workers.probe_backend",
        lambda source, requested="auto": BackendPlan("coremltools", "test", model_type="llama"))
    monkeypatch.setattr("modconv.workers.CoremltoolsBackend", _FailingBackend)
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files)
    logs = []
    worker.log_message.connect(logs.append)
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert result.success is False
    assert "boom" in result.message
    assert not (out / "tiny-llm.mlpackage" / "Manifest.json").is_file()
    assert not any("回退" in m for m in logs)


def test_worker_uses_real_backend_result(qtbot, model_dir, tmp_path, monkeypatch):
    """真实转换成功时直接采用其结果，不再执行打包。"""
    monkeypatch.setattr(
        "modconv.workers.probe_backend",
        lambda source, requested="auto": BackendPlan("coremltools", "test", model_type="llama"))
    monkeypatch.setattr("modconv.workers.CoremltoolsBackend", _SuccessBackend)
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files)
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000) as blocker:
        worker.start()
    assert blocker.args[0].message == "real ok"
    assert not (out / "tiny-llm.mlpackage").exists()


def test_worker_forced_package_skips_probe(qtbot, model_dir, tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("backend=package 时不应触发探测")

    monkeypatch.setattr("modconv.workers.probe_backend", _boom)
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files, backend="package")
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000) as blocker:
        worker.start()
    assert blocker.args[0].success is True


def test_worker_always_emits_finished_even_if_probe_raises(qtbot, model_dir, tmp_path,
                                                           monkeypatch):
    """探测阶段抛异常也必须发出 conversion_finished：否则 GUI 永久卡在"运行中"。

    probe_backend 属于探测阶段，之前位于 try 之外，异常会直接逃出 run()，
    UI 收不到结果 → 开始/浏览按钮永久禁用、取消按钮失效。
    """
    def _boom(*args, **kwargs):
        raise RuntimeError("probe boom")

    monkeypatch.setattr("modconv.workers.probe_backend", _boom)
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files)
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert result is not None
    assert result.success is False
    assert "probe boom" in result.message


def test_worker_passes_max_seq_len_to_real_backend(qtbot, model_dir, tmp_path, monkeypatch):
    """GUI 选择的最大上下文长度应透传给 coremltools 后端。"""
    captured = {}

    class _SpyBackend:
        def __init__(self, source, output_dir, max_seq_len=None, **kwargs):
            captured["max_seq_len"] = max_seq_len

        def run(self, hooks):
            return ConversionResult(True, False, "real ok",
                                    tmp_path / "real.mlpackage", 1, 100, 0.5)

    monkeypatch.setattr(
        "modconv.workers.probe_backend",
        lambda source, requested="auto": BackendPlan("coremltools", "test", model_type="llama"))
    monkeypatch.setattr("modconv.workers.CoremltoolsBackend", _SpyBackend)
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    worker = ConversionWorker(model_dir, out, scan.files, max_seq_len=1024)
    with qtbot.waitSignal(worker.conversion_finished, timeout=15000):
        worker.start()
    assert captured["max_seq_len"] == 1024
