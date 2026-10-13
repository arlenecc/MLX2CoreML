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
    # 实测 0.8B 模型 fp16→4-bit 后约为源 bf16 的 0.23 倍，故 0.5×0.25 的理论
    # 系数偏低，按 0.5×0.35=0.175 估算；非权重文件不计入
    assert est.total_bytes == int(1_000_000 * 0.5 * 0.35)
    assert est.file_count == MLPACKAGE_FILE_COUNT


def test_estimate_coremltools_backend_without_compression():
    """关闭 4-bit 压缩时按 fp16（0.5 倍）估算。"""
    files = [FileInfo("model.safetensors", 1_000_000, True)]
    est = estimate_output(_scan(files), backend="coremltools", compressed=False)
    assert est.total_bytes == 500_000


def test_estimate_default_is_package_mode():
    """默认（打包）口径保持 1:1，避免影响既有调用方。"""
    files = [FileInfo("model.safetensors", 1000, True),
             FileInfo("config.json", 100, False)]
    assert estimate_output(_scan(files)).total_bytes == 1100 + MANIFEST_EXTRA_BYTES
