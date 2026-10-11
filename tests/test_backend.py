"""后端探测与 coremltools 真实转换后端的单元测试（用桩依赖，不装重量级库也能跑）。"""
import json
import types
from pathlib import Path

import pytest

from test_converter import RecordingHooks

import modconv.real_backend as real_backend_mod
from modconv.backend import BackendPlan, detect_model_type, probe_backend
from modconv.real_backend import CoremltoolsBackend


def _write_config(tmp_path, model_type="llama", extra=None):
    src = tmp_path / "m"
    src.mkdir(exist_ok=True)
    payload = {"model_type": model_type}
    payload.update(extra or {})
    (src / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return src


class _StubModule:
    def eval(self):
        return self


class _StubModel:
    def eval(self):
        return self


class _StubExported:
    def run_decompositions(self, *args, **kwargs):
        return self


class _StubKVWrapper(_StubModule):
    """duck-typed KvCacheWrapper（单测隔离重量级依赖）。"""

    max_seq_len = 8
    num_layers = 2
    num_kv_heads = 2
    head_dim = 4

    def kv_state_specs(self):
        shape = (1, 2, 8, 4)
        return [("cache_0_k", shape, "float32"), ("cache_0_v", shape, "float32"),
                ("cache_1_k", shape, "float32"), ("cache_1_v", shape, "float32")]

    def state_buffers(self):
        return []


def _make_stub_spec():
    """duck-typed Core ML spec（用于断言元数据写入）。"""
    return types.SimpleNamespace(
        description=types.SimpleNamespace(
            metadata=types.SimpleNamespace(shortDescription=""),
            input=[types.SimpleNamespace(name="input_ids", shortDescription=""),
                   types.SimpleNamespace(name="cache_position", shortDescription="")],
            output=[types.SimpleNamespace(name="logits", shortDescription="")],
        )
    )


def _stub_libs():
    ct = types.SimpleNamespace(
        TensorType=lambda name=None, shape=None, dtype=None: {"name": name,
                                                              "shape": shape,
                                                              "dtype": dtype},
        RangeDim=lambda default=None, upper_bound=None: {"default": default,
                                                         "upper_bound": upper_bound},
        StateType=lambda wrapped_type, name: {"wrapped_type": wrapped_type, "name": name},
        precision=types.SimpleNamespace(FLOAT16="fp16"),
        target=types.SimpleNamespace(iOS18="ios18"),
    )

    class _FakeMLModel:
        def __init__(self, spec=None):
            self.spec = spec if spec is not None else _make_stub_spec()
            self.weights_dir = "stub-weights"

        def get_spec(self):
            return self.spec

        def save(self, path):
            from pathlib import Path
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "saved.marker").write_bytes(b"ok")

    ct.convert = lambda traced, **kwargs: _FakeMLModel()
    ct.models = types.SimpleNamespace(
        MLModel=lambda spec=None, weights_dir=None: _FakeMLModel(spec))
    ct.optimize = types.SimpleNamespace(coreml=types.SimpleNamespace(
        palettize_weights=lambda model, config=None: model,
        OptimizationConfig=lambda global_config=None, op_name_configs=None:
            {"global_config": global_config, "op_name_configs": op_name_configs},
        OpPalettizerConfig=lambda nbits=None, mode=None: {"nbits": nbits, "mode": mode}))

    torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        long="long",
        ones=lambda shape, dtype=None: ("ones", tuple(shape), dtype),
        arange=lambda n, **kw: ("arange", n),
        nn=types.SimpleNamespace(Module=_StubModule),
        export=types.SimpleNamespace(
            export=lambda model, args, **kw: _StubExported(),
            Dim=lambda name, min=None, max=None: ("dim", name, min, max)),
    )
    transformers = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(
            from_pretrained=lambda p: types.SimpleNamespace(use_cache=True, max_position_embeddings=64)),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda p, **kw: _StubModel()),
    )
    return types.SimpleNamespace(ct=ct, torch=torch, transformers=transformers,
                                 np=types.SimpleNamespace(int32="int32", float16="float16",
                                                          float32="float32"))


def _stub_run(src, out, monkeypatch, libs=None, **backend_kwargs):
    libs = libs or _stub_libs()
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: libs)
    monkeypatch.setattr(real_backend_mod, "_build_kv_wrapper",
                        lambda model, max_seq: _StubKVWrapper())
    return libs, CoremltoolsBackend(src, out, **backend_kwargs).run(RecordingHooks())


# --------------------------------------------------------------------- #
# 后端探测
# --------------------------------------------------------------------- #

def test_probe_selects_real_backend(tmp_path, monkeypatch):
    src = _write_config(tmp_path)
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: object())
    plan = probe_backend(src)
    assert plan.name == "coremltools"
    assert plan.model_type == "llama"


def test_probe_falls_back_when_heavy_deps_missing(tmp_path, monkeypatch):
    src = _write_config(tmp_path)

    def _raise():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(real_backend_mod, "_import_heavy", _raise)
    plan = probe_backend(src)
    assert plan.name == "package"
    assert "依赖" in plan.reason


def test_probe_falls_back_without_config(tmp_path):
    (tmp_path / "empty").mkdir()
    plan = probe_backend(tmp_path / "empty")
    assert plan.name == "package"


def test_probe_falls_back_on_broken_config(tmp_path):
    src = tmp_path / "m"
    src.mkdir()
    (src / "config.json").write_text("{not json", encoding="utf-8")
    plan = probe_backend(src)
    assert plan.name == "package"


def test_probe_falls_back_without_model_type(tmp_path):
    src = tmp_path / "m"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    plan = probe_backend(src)
    assert plan.name == "package"


def test_probe_falls_back_on_quantized_model_without_mlx(tmp_path, monkeypatch):
    """量化模型但 mlx 不可用 → 回退打包。"""
    src = _write_config(tmp_path, extra={"quantization": {"group_size": 64}})
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: object())
    monkeypatch.setattr("modconv.backend._mlx_available", lambda: False)
    plan = probe_backend(src)
    assert plan.name == "package"
    assert "量化" in plan.reason


def test_probe_selects_real_backend_for_quantized_model(tmp_path, monkeypatch):
    """4-bit 量化 + mlx/mlx-lm 可用 → 真实转换（先反量化）。"""
    src = _write_config(tmp_path, extra={"quantization": {"group_size": 64, "bits": 4}})
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: object())
    monkeypatch.setattr("modconv.backend._mlx_available", lambda: True)
    plan = probe_backend(src)
    assert plan.name == "coremltools"
    assert plan.model_type == "llama"
    assert "反量化" in plan.describe()


def test_probe_falls_back_for_quantized_unsupported_arch(tmp_path, monkeypatch):
    src = _write_config(tmp_path, model_type="bloom",
                        extra={"quantization": {"group_size": 64, "bits": 4}})
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: object())
    monkeypatch.setattr("modconv.backend._mlx_available", lambda: True)
    plan = probe_backend(src)
    assert plan.name == "package"
    assert "量化" not in plan.reason  # 先卡架构，不提量化


def test_probe_falls_back_on_unsupported_architecture(tmp_path, monkeypatch):
    src = _write_config(tmp_path, model_type="bloom")
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: object())
    plan = probe_backend(src)
    assert plan.name == "package"


def test_probe_selects_real_backend_for_qwen3_5(tmp_path, monkeypatch):
    """Qwen3.5 全系（含 text-only 检查点的 *_text model_type）走真实转换。"""
    for model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        src = _write_config(tmp_path, model_type=model_type,
                            extra={"quantization": {"group_size": 64, "bits": 4}})
        monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: object())
        monkeypatch.setattr("modconv.backend._mlx_available", lambda: True)
        plan = probe_backend(src)
        assert plan.name == "coremltools", model_type
        assert plan.model_type == model_type


def test_probe_forced_package(tmp_path, monkeypatch):
    src = _write_config(tmp_path)
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: object())
    plan = probe_backend(src, requested="package")
    assert plan.name == "package"


def test_detect_model_type_is_lightweight(tmp_path):
    """detect_model_type 只读 config.json，不导入重量级依赖（供 UI 线程调用）。"""
    src = _write_config(tmp_path, model_type="qwen3_5_moe")
    assert detect_model_type(src) == "qwen3_5_moe"
    assert detect_model_type(tmp_path / "missing") is None
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{not json", encoding="utf-8")
    assert detect_model_type(broken) is None
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / "config.json").write_text("[]", encoding="utf-8")
    assert detect_model_type(odd) is None


def test_backend_plan_describe():
    assert "coremltools" in BackendPlan("coremltools", model_type="llama").describe()
    assert ("反量化" in BackendPlan("coremltools", reason="量化模型先反量化",
                                    model_type="llama").describe())
    assert "打包" in BackendPlan("package", "原因X").describe()
    assert "原因X" in BackendPlan("package", "原因X").describe()


# --------------------------------------------------------------------- #
# 最大上下文长度
# --------------------------------------------------------------------- #

def test_pick_max_seq_len():
    cap = real_backend_mod.SEQ_LEN_CAP
    assert real_backend_mod._pick_max_seq_len(
        types.SimpleNamespace(max_position_embeddings=100000)) == cap
    assert real_backend_mod._pick_max_seq_len(
        types.SimpleNamespace(max_position_embeddings=64)) == 64
    assert real_backend_mod._pick_max_seq_len(
        types.SimpleNamespace(max_position_embeddings=None)) == cap
    assert real_backend_mod._pick_max_seq_len(
        types.SimpleNamespace(max_position_embeddings="bad")) == cap


def test_min_max_seq_len():
    """混合注意力（含 linear attention）架构要求 max_seq_len >= 2。"""
    ns = types.SimpleNamespace
    # 无 layer_types（纯注意力架构）→ 1
    assert real_backend_mod._min_max_seq_len(ns(layer_types=None)) == 1
    assert real_backend_mod._min_max_seq_len(ns(layer_types=["full_attention"] * 4,
                                                num_hidden_layers=4)) == 1
    assert real_backend_mod._min_max_seq_len(
        ns(layer_types=["sliding_attention", "full_attention"],
           num_hidden_layers=2)) == 1
    # 含 linear attention 且层数匹配 → 2
    assert real_backend_mod._min_max_seq_len(
        ns(layer_types=["linear_attention", "full_attention"] * 2,
           num_hidden_layers=4)) == 2
    # layer_types 与层数不一致（配置异常）→ 按非混合处理
    assert real_backend_mod._min_max_seq_len(ns(layer_types=["linear_attention"],
                                                num_hidden_layers=4)) == 1


def test_real_backend_bumps_max_seq_len_for_hybrid(tmp_path, monkeypatch):
    """用户指定的长度小于混合架构下限时自动上调并提示，而不是抛异常。"""
    src = _write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    libs = _stub_libs()
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: libs)
    monkeypatch.setattr(real_backend_mod, "_load_config", lambda s, l: types.SimpleNamespace(
        use_cache=True, max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"] * 2,
        num_hidden_layers=4))
    monkeypatch.setattr(real_backend_mod, "_build_kv_wrapper",
                        lambda model, max_seq: _StubKVWrapper())
    hooks = RecordingHooks()
    result = CoremltoolsBackend(src, out, max_seq_len=1).run(hooks)
    assert result.success is True, result.message
    assert any("最小值" in m for m in hooks.logs), hooks.logs


# --------------------------------------------------------------------- #
# torch 模型加载（dtype 参数兼容性）
# --------------------------------------------------------------------- #

def test_load_torch_model_prefers_dtype_param(tmp_path, monkeypatch):
    """transformers 5.x 用新参数名 dtype（torch_dtype 已弃用，不再打警告）。"""
    captured = []
    libs = types.SimpleNamespace(
        transformers=types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=lambda p, **kw: captured.append(kw) or _StubModel())),
        torch=types.SimpleNamespace(float32="float32", float16="float16"))
    config = types.SimpleNamespace(use_cache=True)
    real_backend_mod._load_torch_model(tmp_path, libs, config, dtype="float16")
    assert captured[0].get("dtype") == "float16"
    assert "torch_dtype" not in captured[0]
    assert captured[0].get("attn_implementation") == "eager"


def test_load_torch_model_falls_back_on_old_transformers(tmp_path):
    """旧版 transformers 不认识 dtype/attn_implementation 时回退 torch_dtype。"""
    calls = []

    def _from_pretrained(p, **kw):
        if "dtype" in kw or "attn_implementation" in kw:
            raise TypeError("unexpected keyword argument")
        calls.append(kw)
        return _StubModel()

    libs = types.SimpleNamespace(
        transformers=types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=_from_pretrained)),
        torch=types.SimpleNamespace(float32="float32", float16="float16"))
    config = types.SimpleNamespace(use_cache=True)
    real_backend_mod._load_torch_model(tmp_path, libs, config, dtype="float16")
    assert calls, "应回退后再次加载"
    assert calls[0].get("torch_dtype") == "float16"
    assert "dtype" not in calls[0]
    assert "attn_implementation" not in calls[0]


# --------------------------------------------------------------------- #
# coremltools 后端（桩依赖）
# --------------------------------------------------------------------- #

def test_real_backend_full_flow_stateful_mlprogram(tmp_path, monkeypatch):
    """转换主流程：动态长度输入 + KV cache states + 仅 mlprogram。"""
    src = _write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    libs = _stub_libs()
    convert_kwargs = {}

    class _ConvertedModel:
        weights_dir = "stub-weights"

        def get_spec(self):
            return _make_stub_spec()

        def save(self, path):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "saved.marker").write_bytes(b"ok")

    def _convert(traced, **kwargs):
        convert_kwargs.update(kwargs)
        return _ConvertedModel()

    libs.ct.convert = _convert
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: libs)
    monkeypatch.setattr(real_backend_mod, "_build_kv_wrapper",
                        lambda model, max_seq: _StubKVWrapper())

    hooks = RecordingHooks()
    result = CoremltoolsBackend(src, out, max_seq_len=8).run(hooks)

    assert result.success is True
    pkg = out / "m.mlpackage"
    assert result.output_path == pkg
    assert (pkg / "saved.marker").read_bytes() == b"ok"
    assert sum(m.startswith("[") for m in hooks.logs) == 4
    assert "coremltools" in result.message
    assert "KV cache" in result.message

    # 动态长度输入：input_ids (1, 1..8)
    assert convert_kwargs["convert_to"] == "mlprogram"
    assert convert_kwargs["minimum_deployment_target"] == "ios18"
    inputs = convert_kwargs["inputs"]
    assert [i["name"] for i in inputs] == ["input_ids", "cache_position"]
    assert inputs[0]["shape"] == (1, {"default": 1, "upper_bound": 8})
    assert inputs[0]["dtype"] == "int32"
    # KV cache states：数量 = 层数 × 2
    states = convert_kwargs["states"]
    assert [s["name"] for s in states] == [
        "cache_0_k", "cache_0_v", "cache_1_k", "cache_1_v"]
    assert all(s["wrapped_type"]["shape"] == (1, 2, 8, 4) for s in states)


def test_real_backend_cancel_before_start(tmp_path, monkeypatch):
    src = tmp_path / "m"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(real_backend_mod, "_import_heavy", _stub_libs)
    hooks = RecordingHooks(cancel_after_bytes=0)
    result = CoremltoolsBackend(src, out).run(hooks)
    assert result.cancelled is True
    assert result.success is False


def test_real_backend_error_propagates(tmp_path, monkeypatch):
    src = _write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    libs = _stub_libs()

    def _boom(*args, **kwargs):
        raise RuntimeError("convert failed")

    libs.ct.convert = _boom
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: libs)
    monkeypatch.setattr(real_backend_mod, "_build_kv_wrapper",
                        lambda model, max_seq: _StubKVWrapper())
    with pytest.raises(RuntimeError):
        CoremltoolsBackend(src, out, max_seq_len=4).run(RecordingHooks())


def test_real_backend_quantized_flow(tmp_path, monkeypatch):
    """量化模型：先反量化到临时目录，再走正常转换，结束后清理临时目录。"""
    src = _write_config(tmp_path, extra={"quantization": {"group_size": 64, "bits": 4}})
    out = tmp_path / "out"
    out.mkdir()
    calls = {}

    def _fake_dequantize(source, dest, log):
        dest = Path(dest)
        # 与 mlx_lm.convert 一致：目标目录不存在，由其自建
        dest.mkdir(parents=True)
        calls["source"] = Path(source)
        calls["dest"] = dest
        # 模拟 mlx-lm 写出的浮点模型目录
        (dest / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        (dest / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(real_backend_mod, "_dequantize_mlx_model", _fake_dequantize)
    libs = _stub_libs()
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: libs)
    monkeypatch.setattr(real_backend_mod, "_build_kv_wrapper",
                        lambda model, max_seq: _StubKVWrapper())
    hooks = RecordingHooks()
    result = CoremltoolsBackend(src, out).run(hooks)

    assert result.success is True
    assert calls["source"] == src
    assert calls["dest"].name.startswith("modconv-dequant-")
    # 量化模型多一个反量化阶段：阶段编号应为 1..5 且首阶段是反量化
    stages = [m for m in hooks.logs if m.startswith("[")]
    assert len(stages) == 5, stages
    assert "1/5" in stages[0] and "反量化" in stages[0], stages
    assert "5/5" in stages[-1], stages
    assert result.output_path == out / "m.mlpackage"
    # 临时反量化目录应被清理
    assert not calls["dest"].exists()


def test_real_backend_sets_metadata_and_swift_example(tmp_path, monkeypatch):
    """真实转换应把元数据写进落盘对象，并生成 stateful Swift 调用示例。"""
    src = _write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    captured = []
    saved_specs = []
    libs = _stub_libs()
    _orig_convert = libs.ct.convert

    def _convert(traced, **kwargs):
        model = _orig_convert(traced, **kwargs)
        captured.append(model)
        return model

    libs.ct.convert = _convert
    _orig_factory = libs.ct.models.MLModel

    def _factory(spec=None, weights_dir=None):
        saved_specs.append(spec)
        return _orig_factory(spec)

    libs.ct.models.MLModel = _factory
    monkeypatch.setattr(real_backend_mod, "_import_heavy", lambda: libs)
    monkeypatch.setattr(real_backend_mod, "_build_kv_wrapper",
                        lambda model, max_seq: _StubKVWrapper())

    result = CoremltoolsBackend(src, out, max_seq_len=8).run(RecordingHooks())

    assert result.success is True
    assert captured, "应捕获转换产物"
    desc = captured[0].get_spec().description
    assert "m" in desc.metadata.shortDescription
    assert desc.input[0].name == "input_ids"
    assert desc.input[0].shortDescription, "输入应有描述"
    assert desc.input[1].shortDescription, "cache_position 应有描述"
    assert desc.output[0].shortDescription, "输出应有描述"
    # 传给 save 的 MLModel 必须基于改过的 spec（get_spec 是深拷贝，需重建落盘）
    assert saved_specs, "应通过 ct.models.MLModel(spec) 重建落盘对象"
    assert "m" in saved_specs[0].description.metadata.shortDescription

    swift = out / "m-swift-example.swift"
    assert swift.is_file(), "应生成 Swift 调用示例"
    text = swift.read_text(encoding="utf-8")
    assert "import CoreML" in text
    assert "makeState" in text, "stateful 模型应演示 makeState"
    assert "input_ids" in text
    assert "cache_position" in text
    assert "1..8" in text  # 动态长度写进示例注释


def test_real_backend_quantizes_weights_mlprogram(tmp_path, monkeypatch):
    """mlprogram 输出应通过 palettize_weights 做 4-bit 量化。"""
    src = _write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = {}
    libs = _stub_libs()

    def _palettize(model, config=None):
        calls["config"] = config
        return model

    libs.ct.optimize.coreml.palettize_weights = _palettize
    _, result = _stub_run(src, out, monkeypatch, libs=libs, max_seq_len=8)

    assert result.success is True
    assert calls["config"] == {"global_config": {"nbits": 4, "mode": "kmeans"},
                               "op_name_configs": {}}
    assert "4-bit" in result.message


def test_real_backend_can_disable_quantization(tmp_path, monkeypatch):
    src = _write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = []
    libs = _stub_libs()

    def _palettize(model, config=None):
        calls.append(config)
        return model

    libs.ct.optimize.coreml.palettize_weights = _palettize
    _, result = _stub_run(src, out, monkeypatch, libs=libs, max_seq_len=8,
                          quantize_nbits=None)

    assert result.success is True
    assert calls == []
    assert "4-bit" not in result.message


def test_real_backend_quantization_failure_propagates(tmp_path, monkeypatch):
    """量化失败时异常向调用方传播——禁止降级为全精度输出。"""
    src = _write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    libs = _stub_libs()

    def _boom(model, config=None):
        raise RuntimeError("kmeans failed")

    libs.ct.optimize.coreml.palettize_weights = _boom
    with pytest.raises(RuntimeError, match="kmeans failed"):
        _stub_run(src, out, monkeypatch, libs=libs, max_seq_len=8)
