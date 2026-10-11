from modconv.estimator import (
    MANIFEST_EXTRA_BYTES,
    MANIFEST_FILE_COUNT,
    MLPACKAGE_FILE_COUNT,
    estimate_output,
)
from modconv.scanner import FileInfo, ScanResult


def _scan(files):
    return ScanResult(source=None, files=files)


def test_estimate_adds_manifest_overhead():
    files = [
        FileInfo("a.safetensors", 1000, True),
        FileInfo("config.json", 100, False),
    ]
    est = estimate_output(_scan(files))
    assert est.file_count == 2 + MANIFEST_FILE_COUNT
    assert est.total_bytes == 1100 + MANIFEST_EXTRA_BYTES


def test_estimate_empty_dir():
    est = estimate_output(_scan([]))
    assert est.file_count == MANIFEST_FILE_COUNT
    assert est.total_bytes == MANIFEST_EXTRA_BYTES


def test_estimate_coremltools_backend_compresses_weights():
    """真实转换口径：只计入权重（fp16 + 4-bit），辅助文件不进模型。"""
    files = [
        FileInfo("model.safetensors", 1_000_000, True),
        FileInfo("config.json", 100_000, False),
    ]
    est = estimate_output(_scan(files), backend="coremltools")
    # 4-bit 约为原始 fp32 权重的 1/8，非权重文件不计入
    assert est.total_bytes == 1_000_000 // 8
    assert est.file_count == MLPACKAGE_FILE_COUNT


def test_estimate_default_is_package_mode():
    """默认（打包）口径保持 1:1，避免影响既有调用方。"""
    files = [FileInfo("model.safetensors", 1000, True),
             FileInfo("config.json", 100, False)]
    assert estimate_output(_scan(files)).total_bytes == 1100 + MANIFEST_EXTRA_BYTES
