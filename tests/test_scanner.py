import pytest

import modconv.scanner as scanner_mod
from modconv.scanner import is_weight_file, scan_directory


def test_empty_dir(tmp_path):
    result = scan_directory(tmp_path)
    assert result.count == 0
    assert result.total_bytes == 0
    assert result.files == []


def test_counts_and_sizes_nested(model_dir):
    result = scan_directory(model_dir)
    assert result.count == 5
    assert result.total_bytes == 20 + 21 + 500 + 64 + (1 << 20)


def test_relpaths_and_weight_flag(model_dir):
    result = scan_directory(model_dir)
    by_name = {f.relpath: f for f in result.files}
    assert by_name["sub/model.safetensors"].is_weight is True
    assert by_name["config.json"].is_weight is False
    assert by_name["meta/params.json"].is_weight is False


def test_weight_suffixes():
    """权重判定覆盖 safetensors / 旧版 .bin / npz / gguf，且与大小写无关。"""
    assert is_weight_file("model.safetensors") is True
    assert is_weight_file("pytorch_model.bin") is True
    assert is_weight_file("model.SAFETENSORS") is True
    assert is_weight_file("model.npz") is True
    assert is_weight_file("model.gguf") is True
    assert is_weight_file("config.json") is False
    assert is_weight_file("tokenizer.model") is False


def test_symlinks_are_skipped(tmp_path):
    real = tmp_path / "real.bin"
    real.write_bytes(b"123")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    result = scan_directory(tmp_path)
    assert result.count == 1
    assert result.files[0].relpath == "real.bin"


def test_nonexistent_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_directory(tmp_path / "missing")


def test_file_path_raises(tmp_path):
    # 单文件源现在受支持（如 ONNX→GGUF 的 .onnx 源）：返回仅含该文件的 ScanResult
    f = tmp_path / "f.txt"
    f.write_bytes(b"hi")
    result = scan_directory(f)
    assert result.count == 1 and result.files[0].size == 2
    assert result.files[0].is_weight is False  # .txt 不是权重后缀
    # 普通文件作为源也不再抛错；仍不存在的路径才抛 FileNotFoundError
    with pytest.raises(FileNotFoundError):
        scan_directory(tmp_path / "missing")


def test_progress_cb_reported_incrementally(model_dir, monkeypatch):
    """扫描进度回调按固定间隔上报，计数单调递增。"""
    monkeypatch.setattr(scanner_mod, "SCAN_PROGRESS_EVERY", 2)
    calls = []
    result = scanner_mod.scan_directory(
        model_dir, progress_cb=lambda count, size: calls.append((count, size))
    )
    assert calls, "至少应上报一次进度"
    assert calls[-1][0] >= result.count - 1
    assert all(calls[i][0] < calls[i + 1][0] for i in range(len(calls) - 1))
    assert all(calls[i][1] <= calls[i + 1][1] for i in range(len(calls) - 1))
