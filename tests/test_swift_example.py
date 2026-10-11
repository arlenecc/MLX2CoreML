"""生成的 Swift 调用示例必须真的能编译、能跑，且结果与 Python 侧一致。

只做类型检查（-typecheck）无法发现"能编译但一跑就崩"的问题（例如把元素总数
当成最后一维长度去索引 MLMultiArray.shape），因此这里额外做一次真实运行：
用 swiftc 编译"示例 + 驱动"，跑一遍 KV cache 自回归生成，并与 coremltools
Python 侧的贪心解码结果逐 token 比对。
"""
import importlib.util
import shutil
import subprocess

import pytest

from test_converter import RecordingHooks
from test_real_conversion_integration import _make_tiny_llama


def _heavy_available() -> bool:
    return all(importlib.util.find_spec(m) is not None
               for m in ("torch", "transformers", "coremltools", "numpy", "safetensors"))


pytestmark = pytest.mark.skipif(not _heavy_available(),
                                reason="需要 torch/transformers/coremltools/numpy/safetensors")


def _swiftc() -> list[str] | None:
    """可用的 Swift 编译命令；无工具链时返回 None（自动跳过）。"""
    if shutil.which("xcrun") is None:
        return None
    cmd = ["xcrun", "--sdk", "macosx", "swiftc"]
    try:
        probe = subprocess.run(cmd + ["--version"], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return cmd if probe.returncode == 0 else None


_SWIFT_DRIVER = '''
let modelURL = URL(fileURLWithPath: "{model_path}")
let generated = try generateWithConvertedModel(modelURL: modelURL,
                                               promptTokens: [1, 2, 3, 4],
                                               maxNewTokens: 4)
print("TOKENS:" + generated.map {{ String($0) }}.joined(separator: ","))
'''


def test_swift_example_typechecks_without_warnings(tmp_path):
    """示例文件应零警告通过 swiftc 类型检查（含 CoreML 框架）。"""
    cmd = _swiftc()
    if cmd is None:
        pytest.skip("未找到可用的 Swift 工具链（xcrun/swiftc）")
    from modconv.real_backend import _SWIFT_EXAMPLE_TEMPLATE

    src = tmp_path / "main.swift"
    src.write_text(_SWIFT_EXAMPLE_TEMPLATE.format(model_name="m", max_seq_len=512,
                                                  output_name="logits"),
                   encoding="utf-8")
    proc = subprocess.run(cmd + ["-typecheck", str(src)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"Swift 示例编译失败:\n{proc.stderr}"
    assert "warning" not in proc.stderr.lower(), f"Swift 示例存在编译警告:\n{proc.stderr}"


def test_swift_example_runs_and_matches_python(tmp_path):
    """编译并运行示例：Swift 端贪心生成结果须与 Python 侧逐 token 一致。"""
    cmd = _swiftc()
    if cmd is None:
        pytest.skip("未找到可用的 Swift 工具链（xcrun/swiftc）")
    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
    except Exception:
        pytest.skip("需要 coremltools 原生运行库做本地推理")

    import numpy as np

    import coremltools as ct
    from modconv.real_backend import CoremltoolsBackend

    src = _make_tiny_llama(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = CoremltoolsBackend(src, out, max_seq_len=16,
                                quantize_nbits=None).run(RecordingHooks())
    assert result.success is True, result.message

    swift_src = out / f"{src.name}-swift-example.swift"
    assert swift_src.is_file()

    # 示例 + 最小驱动 → 可执行文件
    main_swift = tmp_path / "main.swift"
    main_swift.write_text(
        swift_src.read_text(encoding="utf-8")
        + _SWIFT_DRIVER.format(model_path=str(result.output_path)),
        encoding="utf-8")
    binary = tmp_path / "swiftgen"
    proc = subprocess.run(cmd + ["-o", str(binary), str(main_swift)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"Swift 示例编译失败:\n{proc.stderr}"

    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, f"Swift 示例运行失败:\n{run.stdout}\n{run.stderr}"
    marker = [line for line in run.stdout.splitlines() if line.startswith("TOKENS:")]
    assert marker, f"未取到生成结果:\n{run.stdout}"
    swift_tokens = [int(t) for t in marker[0][len("TOKENS:"):].split(",") if t]
    assert len(swift_tokens) == 4, swift_tokens

    # Python 侧同一贪心解码（同一模型、同一 state 推进方式）
    mlmodel = ct.models.MLModel(str(result.output_path), compute_units=ct.ComputeUnit.ALL)
    state = mlmodel.make_state()

    def predict(ids, pos):
        o = mlmodel.predict({"input_ids": np.array(ids, dtype=np.int32),
                             "cache_position": np.array(pos, dtype=np.int32)}, state)
        return np.asarray(o["logits"], dtype=np.float32)

    logits = predict([[1, 2, 3, 4]], np.arange(4))
    assert np.isfinite(logits).all()
    py_tokens = []
    position = 4
    for _ in range(4):
        py_tokens.append(int(logits[0, -1].argmax()))
        if position >= 16:
            break
        logits = predict([[py_tokens[-1]]], [position])
        position += 1

    assert swift_tokens == py_tokens, (
        f"Swift 与 Python 贪心解码结果不一致: {swift_tokens} != {py_tokens}")
