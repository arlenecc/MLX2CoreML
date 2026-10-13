import json
from pathlib import Path

import pytest

from modconv.converter import ModelConverter
from modconv.scanner import FileInfo, scan_directory


class RecordingHooks:
    """记录转换过程回调的测试桩，可模拟在指定累计字节数后取消。"""

    def __init__(self, cancel_after_bytes=None):
        self.logs = []
        self.started = []
        self.progress = []
        self.finished = []
        self.cancel_after_bytes = cancel_after_bytes
        self._observed_bytes = 0

    def should_cancel(self):
        if self.cancel_after_bytes is None:
            return False
        return self._observed_bytes >= self.cancel_after_bytes

    def on_log(self, message):
        self.logs.append(message)

    def on_file_started(self, relpath, size, index, total_files):
        self.started.append((relpath, size, index, total_files))

    def on_file_progress(self, relpath, file_done, file_total, overall_done, overall_total):
        self.progress.append((relpath, file_done, file_total, overall_done, overall_total))
        self._observed_bytes = overall_done

    def on_file_finished(self, relpath, overall_done):
        self.finished.append((relpath, overall_done))
        self._observed_bytes = overall_done


def test_convert_copies_all_files(model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)

    result = ModelConverter(model_dir, out, scan.files).run(RecordingHooks())

    assert result.success is True
    assert result.cancelled is False
    pkg = out / "tiny-llm.mlpackage"
    assert result.output_path == pkg
    assert (pkg / "Manifest.json").is_file()
    # 权重文件 → Data/com.apple.CoreML/weights/<相对路径>
    weights = pkg / "Data" / "com.apple.CoreML" / "weights"
    assert (weights / "sub" / "model.safetensors").read_bytes() == \
        (model_dir / "sub" / "model.safetensors").read_bytes()
    # 其余文件 → Data/mlx/<相对路径>
    assert (pkg / "Data" / "mlx" / "config.json").read_bytes() == b"{}" * 10
    assert (pkg / "Data" / "mlx" / "tokenizer.json").read_bytes() == b"tok" * 7
    assert (pkg / "Data" / "mlx" / "notes.txt").read_bytes() == b"x" * 500
    assert (pkg / "Data" / "mlx" / "meta" / "params.json").read_bytes() == b"p" * 64


def test_manifest_content(model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    ModelConverter(model_dir, out, scan.files).run(RecordingHooks())
    manifest = json.loads((out / "tiny-llm.mlpackage" / "Manifest.json").read_text("utf-8"))
    assert manifest["fileIdentifier"] == "tiny-llm.mlpackage"
    assert isinstance(manifest["specificationVersion"], int)


def test_bytes_done_matches_total(model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    result = ModelConverter(model_dir, out, scan.files).run(RecordingHooks())
    manifest_len = (out / "tiny-llm.mlpackage" / "Manifest.json").stat().st_size
    assert result.bytes_done == scan.total_bytes + manifest_len


def test_hooks_receive_events(model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(model_dir)
    hooks = RecordingHooks()

    result = ModelConverter(model_dir, out, scan.files).run(hooks)

    assert {rel for rel, *_ in hooks.started} == {f.relpath for f in scan.files}
    assert sorted(idx for _, _, idx, _ in hooks.started) == list(range(1, scan.count + 1))
    assert all(total == scan.count for _, _, _, total in hooks.started)
    assert len(hooks.finished) == scan.count
    assert hooks.logs, "应有日志输出"
    # 最后一次进度回调：overall_done 为全部文件字节数，overall_total 含清单开销
    last = hooks.progress[-1]
    assert last[3] == scan.total_bytes
    assert last[4] == result.bytes_done


def test_progress_throttling_on_large_file(tmp_path):
    """大文件进度回调按字节间隔节流：32 MiB / 8 MiB 间隔 → 恰好 4 次。"""
    src = tmp_path / "big"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"\0" * (32 << 20))
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(src)

    hooks = RecordingHooks()
    conv = ModelConverter(src, out, scan.files,
                          chunk_size=1 << 20, emit_interval=8 << 20)
    conv.run(hooks)

    file_progress = [p for p in hooks.progress if p[0] == "model.safetensors"]
    assert [p[1] for p in file_progress] == [8 << 20, 16 << 20, 24 << 20, 32 << 20]


def test_zero_byte_file(tmp_path):
    src = tmp_path / "m"
    src.mkdir()
    (src / "empty.txt").write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(src)

    result = ModelConverter(src, out, scan.files).run(RecordingHooks())

    assert result.success is True
    assert (out / "m.mlpackage" / "Data" / "mlx" / "empty.txt").read_bytes() == b""


def test_cancel_midway(tmp_path):
    """文件 a 完成后、文件 b 拷贝途中触发取消。"""
    src = tmp_path / "m"
    src.mkdir()
    (src / "a.safetensors").write_bytes(b"\0" * (1 << 20))
    (src / "b.safetensors").write_bytes(b"\0" * (4 << 20))
    out = tmp_path / "out"
    out.mkdir()
    scan = scan_directory(src)

    hooks = RecordingHooks(cancel_after_bytes=int(1.5 * (1 << 20)))
    result = ModelConverter(src, out, scan.files,
                            chunk_size=256 << 10, emit_interval=256 << 10).run(hooks)

    assert result.cancelled is True
    assert result.success is False
    assert result.files_done == 1
    # 半成品包对调用方没有价值却可能占几 GB：取消即删除
    assert result.output_path is None
    assert not (out / "m.mlpackage").exists()
    b_dest = out / "m.mlpackage" / "Data" / "com.apple.CoreML" / "weights" / "b.safetensors"
    assert not b_dest.exists()


def test_rejects_relpath_escaping_package(tmp_path):
    """越出输出包的相对路径必须拒绝（库 API 可被外部传入自定义 FileInfo）。"""
    src = tmp_path / "m"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "w.safetensors").write_bytes(b"w")
    out = tmp_path / "out"
    out.mkdir()
    files = [
        FileInfo("../escape.safetensors", 1, True),
        FileInfo("/abs/escape.safetensors", 1, True),
        FileInfo("sub/w.safetensors", 1, True),
    ]
    hooks = RecordingHooks()
    result = ModelConverter(src, out, files).run(hooks)

    assert result.success is True
    assert result.files_done == 1
    assert not (tmp_path / "escape.safetensors").exists()
    assert not Path("/abs/escape.safetensors").exists()
    assert any("跳过非法相对路径" in m for m in hooks.logs)


def test_missing_source_file_is_skipped(tmp_path):
    """扫描后被删除的文件不应让整次转换失败。"""
    src = tmp_path / "m"
    src.mkdir()
    (src / "a.safetensors").write_bytes(b"a")
    out = tmp_path / "out"
    out.mkdir()
    files = [FileInfo("a.safetensors", 1, True),
             FileInfo("gone.safetensors", 1, True)]
    hooks = RecordingHooks()
    result = ModelConverter(src, out, files).run(hooks)

    assert result.success is True
    assert result.files_done == 1
    assert any("跳过无法读取" in m for m in hooks.logs)


def test_progress_reaches_total_when_size_mismatch(tmp_path):
    """文件大小与扫描时不一致时，末次进度仍要上报（否则进度条卡住）。"""
    src = tmp_path / "m"
    src.mkdir()
    (src / "a.safetensors").write_bytes(b"x" * 500)
    out = tmp_path / "out"
    out.mkdir()
    # 模拟扫描时 stat 失败把 size 记成 0
    files = [FileInfo("a.safetensors", 0, True)]
    hooks = RecordingHooks()
    result = ModelConverter(src, out, files).run(hooks)

    assert result.success is True
    assert result.bytes_done >= 500
    assert hooks.progress, "应至少上报一次进度"


def test_empty_file_list_fails(tmp_path):
    src = tmp_path / "m"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    result = ModelConverter(src, out, []).run(RecordingHooks())
    assert result.success is False
    assert not (out / "m.mlpackage").exists()


def test_missing_source_raises(tmp_path):
    with pytest.raises(ValueError):
        ModelConverter(tmp_path / "nope", tmp_path, [])


def test_overwrite_existing_package(model_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pkg = out / "tiny-llm.mlpackage"
    (pkg / "junk").mkdir(parents=True)
    (pkg / "junk" / "stale.bin").write_bytes(b"old")
    scan = scan_directory(model_dir)

    ModelConverter(model_dir, out, scan.files).run(RecordingHooks())

    assert not (pkg / "junk").exists()
    assert (pkg / "Manifest.json").is_file()
