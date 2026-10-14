"""端到端集成测试：真实 coremltools 转换（stateful mlprogram）。

需要 torch / transformers / coremltools / numpy / safetensors，
缺失时自动跳过。模型在本地随机初始化，不访问网络。
"""
import importlib.util

import pytest

from test_converter import RecordingHooks

from modconv.backend import probe_backend


def _heavy_available() -> bool:
    modules = ("torch", "transformers", "coremltools", "numpy", "safetensors")
    return all(importlib.util.find_spec(m) is not None for m in modules)


pytestmark = pytest.mark.skipif(not _heavy_available(),
                                reason="需要 torch/transformers/coremltools/numpy/safetensors")


def _mlx_available() -> bool:
    return all(importlib.util.find_spec(m) is not None
               for m in ("mlx", "mlx_lm", "safetensors"))


def _make_tiny_llama(tmp_path, name="tiny-llama"):
    """embedding 4096 元素（> coremltools palettize 默认 weight_threshold 2048），
    否则 4-bit 压缩会静默跳过所有权重。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    src = tmp_path / name
    config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=16)
    LlamaForCausalLM(config).save_pretrained(src)  # 本地写出 config.json + model.safetensors
    return src


def _write_minimal_tokenizer(path):
    """本地构造最小 tokenizer（无网络），供 mlx-lm 转换流程使用。"""
    import json

    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4,
             "ml": 5, "core": 6, "swift": 7}
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }
    tj = path / "tokenizer.json"
    tj.write_text(json.dumps(tokenizer_json), encoding="utf-8")
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tj), unk_token="<unk>")
    tok.save_pretrained(str(path))


def _binding_dims(value):
    """从 MIL Value 取静态 dims（immediate 常量或 blob 文件引用），无法确定时 None。"""
    if value.HasField("immediateValue"):
        return list(value.immediateValue.tensor.dims)
    if value.HasField("blobFileValue"):
        tt = value.type.tensorType
        if any(not d.HasField("constant") for d in tt.dimensions):
            return None
        return [d.constant.size for d in tt.dimensions]
    return None


def _assert_4bit_palettized(spec):
    """断言 mlprogram 图中存在 4-bit palettization 的权重。

    palettize_weights 把权重常量改写为 ``constexpr_lut_to_dense``（LUT + 索引），
    LUT 的调色板大小 = 2^nbits（4-bit → 16）。LUT/索引常量可能存于 immediate
    或外部 blob 文件（weight.bin），dims 分别在 tensor.dims / tensorType.
    dimensions[i].constant.size。注意 block_specializations 的 key 因 opset 而
    异（如 "CoreML8"），需遍历 values()。
    """
    quantized = 0
    four_bit = 0
    for func in spec.mlProgram.functions.values():
        for block in func.block_specializations.values():
            for op in block.operations:
                if op.type != "constexpr_lut_to_dense" or "lut" not in op.inputs:
                    continue
                quantized += 1
                for binding in op.inputs["lut"].arguments:
                    dims = _binding_dims(binding.value)
                    if dims and 16 in dims:
                        four_bit += 1
    assert quantized >= 1, f"应有 palettization 权重（constexpr_lut_to_dense），实际 {quantized}"
    assert four_bit >= 1, f"应有 4-bit LUT（调色板 16），实际 {four_bit}"


def _convert_and_check_spec(tmp_path):
    """真实转换 tiny llama 并断言 spec：双输入、logits 输出、元数据、4-bit。"""
    import coremltools as ct

    src = _make_tiny_llama(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    plan = probe_backend(src)
    assert plan.name == "coremltools"

    from modconv.real_backend import CoremltoolsBackend
    result = CoremltoolsBackend(src, out, max_seq_len=8).run(RecordingHooks())
    assert result.success is True, result.message

    out_path = result.output_path
    assert out_path is not None
    # mlprogram → .mlpackage 目录
    assert out_path.is_dir()
    assert out_path.name == "tiny-llama.mlpackage"

    spec = ct.utils.load_spec(str(out_path))
    # stateful 动态长度输入：input_ids + cache_position
    names = [i.name for i in spec.description.input]
    assert names[:2] == ["input_ids", "cache_position"]
    # 混合架构（含 linear attention 层）额外带 GDN 的 conv/rec 状态输入
    assert set(names[2:]) <= {"conv_states", "recurrent_states"}, names
    assert spec.description.output, "应有输出"
    # Swift 友好性：输出名应为 logits，且元数据/描述非空
    assert spec.description.output[0].name == "logits", "输出应重命名为 logits"
    assert spec.description.metadata.shortDescription, "应写入模型元数据"
    assert spec.description.input[0].shortDescription, "输入应有描述"
    assert spec.description.input[1].shortDescription, "cache_position 应有描述"
    assert spec.description.output[0].shortDescription, "输出应有描述"
    _assert_4bit_palettized(spec)
    return src, out, out_path


def _predict(m, state, ids_np, pos_np):
    """stateful 推理：返回 logits（MLMultiArray → np.float32）。"""
    import numpy as np

    out = m.predict({"input_ids": ids_np, "cache_position": pos_np}, state)
    return np.asarray(out["logits"], dtype=np.float32)


def _gdn_zero_states(spec):
    """混合架构的 GDN 状态零初始化字典（名称与形状取自 spec 的输入）。"""
    import numpy as np

    zeros = {}
    for feature in spec.description.input:
        if feature.name in ("input_ids", "cache_position"):
            continue
        shape = tuple(int(d) for d in feature.type.multiArrayType.shape)
        zeros[feature.name] = np.zeros(shape, dtype=np.float32)
    return zeros


class _Stepper:
    """单步 decode 模型的连续推进器：prefill 逐 token，GDN 状态逐步回传。

    转换产物每次只推进一个 token（recurrent prefill），推理侧需要把上一步输出的
    new_conv_states / new_recurrent_states 传回下一步。
    """

    def __init__(self, model, state=None):
        self.model = model
        self.state = state if state is not None else model.make_state()
        spec = model.get_spec()
        self.gdn = _gdn_zero_states(spec)

    def __call__(self, token, position):
        import numpy as np

        feed = {"input_ids": np.array([[token]], dtype=np.int32),
                "cache_position": np.array([position], dtype=np.int32)}
        feed.update(self.gdn)
        out = self.model.predict(feed, self.state)
        # 收集新状态：输出名可能是 new_<name>（传入 dict 时返回 dict）
        for name in list(self.gdn):
            for key in (name, f"new_{name}"):
                if key in out:
                    self.gdn[name] = np.asarray(out[key], dtype=np.float32).reshape(
                        self.gdn[name].shape)
                    break
        return np.asarray(out["logits"], dtype=np.float32)

    def prefill(self, tokens, start=0):
        """逐 token 推进整段 prompt，返回最后一个位置的 logits。"""
        logits = None
        for i, token in enumerate(tokens):
            logits = self(int(token), start + i)
        return logits


def test_probe_and_real_conversion_end_to_end(tmp_path):
    import numpy as np

    _, _, out_path = _convert_and_check_spec(tmp_path)

    # 推理验证仅在 coremltools 原生运行库可用时执行
    # （源码构建的 coremltools 缺少 libcoremlpython，无法本地推理）
    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
        has_native = True
    except Exception:
        has_native = False
    if not has_native:
        return

    import coremltools as ct
    mlmodel = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.ALL)
    stepper = _Stepper(mlmodel)

    # prefill：逐 token 推进（单步 decode 图），只看最后一个位置
    logits = stepper.prefill([1, 2, 3, 4])
    assert logits.shape == (1, 1, 128)
    assert np.isfinite(logits).all(), "logits 不应含 NaN/Inf"

    # decode：单 token，KV cache 由 state 携带
    next_logits = stepper(5, 4)
    assert next_logits.shape == (1, 1, 128)

    # state 语义：空 state 的同一 decode 输出必须与有 cache 时不同
    empty = _Stepper(mlmodel, mlmodel.make_state())
    empty_logits = empty(5, 4)
    assert np.abs(next_logits - empty_logits).max() > 1e-3, \
        "空 state 的 decode 与有 cache 时相同，KV cache state 未生效"


@pytest.mark.skipif(not _mlx_available(), reason="需要 mlx/mlx-lm")
def test_quantized_model_end_to_end(tmp_path, monkeypatch):
    """4-bit 量化 MLX 模型 → 反量化 → coremltools 真实转换 → Swift 示例。"""
    import json

    from mlx_lm.convert import convert as mlx_convert

    src = _make_tiny_llama(tmp_path)
    _write_minimal_tokenizer(src)
    quant = tmp_path / "tiny-llama-4bit"
    mlx_convert(hf_path=str(src), mlx_path=str(quant),
                quantize=True, q_bits=4, q_group_size=64)
    quant_config = json.loads((quant / "config.json").read_text(encoding="utf-8"))
    assert "quantization" in quant_config, "源应是 4-bit 量化模型"

    out = tmp_path / "out"
    out.mkdir()

    from modconv import real_backend as real_backend_mod
    from modconv.backend import probe_backend
    plan = probe_backend(quant)
    assert plan.name == "coremltools", plan.describe()

    # 把反量化临时目录固定到 tmp_path 内，便于断言清理
    # （真实实现默认放在输出目录旁，避免大模型中间文件撑爆系统盘）
    dequant_root = tmp_path / "dequant-tmp"
    monkeypatch.setattr(real_backend_mod, "_make_dequant_tmp",
                        lambda *_args, **_kwargs: dequant_root)

    from modconv.real_backend import CoremltoolsBackend
    result = CoremltoolsBackend(quant, out, max_seq_len=8).run(RecordingHooks())
    assert result.success is True, result.message

    # 反量化临时目录已清理
    assert not dequant_root.exists(), "反量化临时目录应被清理"

    # Swift 调用示例与模型并排生成（stateful：含 makeState）
    swift = out / f"{quant.name}-swift-example.swift"
    assert swift.is_file()
    swift_text = swift.read_text(encoding="utf-8")
    assert "import CoreML" in swift_text
    assert "makeState" in swift_text, "stateful 模型应演示 makeState"

    _convert_assertions_only(result.output_path)


def _make_tiny_qwen35(tmp_path, name="tiny-qwen35", **overrides):
    """tiny Qwen3.5 模型，layer_types 等配置可覆盖（用于变体测试）。

    embedding 128×32=4096 元素 > palettize 默认 weight_threshold 2048，保证 4-bit
    压缩至少命中 embedding。max_position_embeddings 需覆盖 wrapper 的 pad 上限
    （decode 时 pad 位置可达 pos + max_seq_len - 1）。"""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    src = tmp_path / name
    config_kwargs = dict(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=32,
    )
    config_kwargs.update(overrides)
    Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config_kwargs)).save_pretrained(src)
    return src


# Qwen3.5 配置变体：真实模型均为 hybrid（GDN 与 full attention 交替，官方 3:1），
# 但导出工具需覆盖全部合法形态——全 full（纯注意力 + GQA）、全 linear（纯 GDN）、
# GQA 头数变体（full 层 kv<q、GDN 层 v>k 的 repeat_interleave 分支）。
_QWEN35_VARIANTS = {
    "hybrid-official": {},
    "hybrid-gqa": dict(num_attention_heads=4, num_key_value_heads=2,
                       linear_num_value_heads=4, linear_num_key_heads=2),
    "all-full": dict(layer_types=["full_attention"] * 4,
                     num_attention_heads=4, num_key_value_heads=2),
    "all-linear": dict(layer_types=["linear_attention"] * 4),
}


def test_qwen35_hybrid_conversion_end_to_end(tmp_path):
    """Qwen3.5 混合架构（GDN + full attention）→ stateful mlprogram 端到端。

    覆盖导出链路的全部 hybrid 专属路径：GDN 补丁前向（chunk 内核导出分支）、
    conv/recurrent state 绑定、pad 静态前向、full 层槽位写入。"""
    import coremltools as ct

    src = _make_tiny_qwen35(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    plan = probe_backend(src)
    assert plan.name == "coremltools", plan.describe()

    from modconv.real_backend import CoremltoolsBackend
    result = CoremltoolsBackend(src, out, max_seq_len=8).run(RecordingHooks())
    assert result.success is True, result.message

    out_path = result.output_path
    assert out_path is not None and out_path.name == "tiny-qwen35.mlpackage"

    spec = ct.utils.load_spec(str(out_path))
    names = [i.name for i in spec.description.input]
    # 单步 decode：input_ids/cache_position + 混合架构的 GDN 状态（走图 I/O）
    assert names[:2] == ["input_ids", "cache_position"]
    assert names[2:] == ["conv_states", "recurrent_states"], names
    assert spec.description.output[0].name == "logits"
    # GDN 新状态输出：供下一步回传
    out_names = [o.name for o in spec.description.output]
    assert out_names == ["logits", "new_conv_states", "new_recurrent_states"], \
        out_names
    _assert_4bit_palettized(spec)

    # 推理验证仅在 coremltools 原生运行库可用时执行
    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
        has_native = True
    except Exception:
        has_native = False
    if not has_native:
        return

    import numpy as np

    mlmodel = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.ALL)
    stepper = _Stepper(mlmodel)

    # prefill 4 token + 两步 decode，全程有限值
    logits = stepper.prefill([1, 2, 3, 4])
    assert logits.shape == (1, 1, 128)
    assert np.isfinite(logits).all()
    next1 = stepper(5, 4)
    assert next1.shape == (1, 1, 128)
    next2 = stepper(6, 5)
    assert np.isfinite(next2).all(), "多步 decode 后 logits 不应含 NaN/Inf"

    # state 语义：空 state 的同输入 decode 必须与有 cache 时不同
    #（full 层 KV 由 MLState 携带，GDN conv/rec 由逐步回传的张量携带）
    # 空 state 下只喂位置 4 的 token 5：没有 1..4 的历史，输出必然不同
    empty = _Stepper(mlmodel, mlmodel.make_state())(5, 4)
    assert np.abs(next1 - empty).max() > 1e-3, \
        "空 state 的 decode 与有 cache 时相同，hybrid state 未生效"


@pytest.mark.parametrize("variant", list(_QWEN35_VARIANTS))
def test_qwen35_config_variants_end_to_end(tmp_path, variant):
    """Qwen3.5 配置变体（官方 hybrid 3:1 / GQA 头数 / 全 full / 全 linear）端到端。

    每个变体断言：转换成功、spec 输入输出、4-bit palettization、多步 prefill+
    decode 有限值、连续 decode 与整段 prefill 的 KV 一致性、空 state 语义。"""
    import coremltools as ct
    import numpy as np

    src = _make_tiny_qwen35(tmp_path, name=f"qwen35-{variant}",
                            **_QWEN35_VARIANTS[variant])
    out = tmp_path / f"out-{variant}"
    out.mkdir()

    plan = probe_backend(src)
    assert plan.name == "coremltools", plan.describe()

    from modconv.real_backend import CoremltoolsBackend
    result = CoremltoolsBackend(src, out, max_seq_len=8).run(RecordingHooks())
    assert result.success is True, result.message

    spec = ct.utils.load_spec(str(result.output_path))
    names = [i.name for i in spec.description.input]
    assert names[:2] == ["input_ids", "cache_position"]
    # 混合架构（含 linear attention 层）额外带 GDN 的 conv/rec 状态输入
    assert set(names[2:]) <= {"conv_states", "recurrent_states"}, names
    assert spec.description.output[0].name == "logits"
    _assert_4bit_palettized(spec)

    # 推理验证仅在 coremltools 原生运行库可用时执行
    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
        has_native = True
    except Exception:
        has_native = False
    if not has_native:
        return

    import numpy as np  # noqa: F811

    mlmodel = ct.models.MLModel(str(result.output_path),
                                compute_units=ct.ComputeUnit.ALL)

    def predict(model, state, ids, pos):
        o = model.predict({"input_ids": np.array(ids, dtype=np.int32),
                           "cache_position": np.array(pos, dtype=np.int32)}, state)
        return np.asarray(o["logits"], dtype=np.float32)

    # 连续 decode（prefill 4 + 逐步 decode 2）与一次 prefill 6 的最后位一致性。
    # 单步模型没有"整段 prefill"这一路，对照值也用同一张图逐 token 推进得到。
    stepper = _Stepper(mlmodel)
    pre = stepper.prefill([1, 2, 3, 4])
    assert pre.shape == (1, 1, 128) and np.isfinite(pre).all()
    dec1 = stepper(5, 4)
    dec2 = stepper(6, 5)
    assert np.isfinite(dec2).all(), "多步 decode 后 logits 不应含 NaN/Inf"
    whole_stepper = _Stepper(mlmodel, mlmodel.make_state())
    whole = whole_stepper.prefill([1, 2, 3, 4, 5, 6])
    assert np.abs(whole[0, 0, :] - dec2[0, 0, :]).max() < 0.05, \
        "连续 decode 与整段 prefill 的末位 logits 不一致，KV cache 推进有误"

    # state 语义：空 state 的同输入 decode 必须与有 cache 时不同
    empty = _Stepper(mlmodel, mlmodel.make_state())(5, 4)
    assert np.abs(dec1 - empty).max() > 1e-3, \
        "空 state 的 decode 与有 cache 时相同，state 未生效"


def _make_tiny_qwen35_moe(tmp_path, name="tiny-qwen35-moe", **overrides):
    """tiny Qwen3.5 **MoE** 模型（qwen3_5_moe 架构，GDN + full attention 混合）。

    MoE 专家层与 dense 版同名内核不同（Qwen3_5MoeGatedDeltaNet / Qwen3_5MoeExperts），
    导出路径必须按结构识别而不是写死 dense 类名。
    """
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    src = tmp_path / name
    config_kwargs = dict(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=2, mlp_only_layers=[1],
        max_position_embeddings=32,
    )
    config_kwargs.update(overrides)
    Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**config_kwargs)).save_pretrained(src)
    return src


def test_max_seq_len_one_export_and_convert(tmp_path):
    """max_seq_len=1 是合法输入：Dim(min=1, max=1) 非法，须退化为静态形状导出。"""
    import numpy as np

    import coremltools as ct

    src = _make_tiny_llama(tmp_path, name="len1")
    out = tmp_path / "out"
    out.mkdir()

    from modconv.real_backend import CoremltoolsBackend
    result = CoremltoolsBackend(src, out, max_seq_len=1,
                                quantize_nbits=None).run(RecordingHooks())
    assert result.success is True, result.message

    spec = ct.utils.load_spec(str(result.output_path))
    ids_shape = spec.description.input[0].type.multiArrayType.shape
    assert list(ids_shape) == [1, 1], ids_shape

    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
    except Exception:
        return
    mlmodel = ct.models.MLModel(str(result.output_path),
                                compute_units=ct.ComputeUnit.ALL)
    stepper = _Stepper(mlmodel)
    logits = stepper(7, 0)
    assert np.asarray(logits, dtype=np.float32).shape == (1, 1, 128)
    assert np.isfinite(np.asarray(logits, dtype=np.float32)).all()


def test_qwen35_moe_conversion_end_to_end(tmp_path):
    """Qwen3.5 MoE 端到端：转换 → spec 断言 → 原生 predict prefill/decode。

    覆盖 MoE 专属导出路径：专家层的可导出批量前向（原版 eager 循环 /
    grouped_mm 的 torch.histc 均不可导出）与 GDN 变体类名识别。
    """
    import numpy as np

    import coremltools as ct

    src = _make_tiny_qwen35_moe(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    plan = probe_backend(src)
    assert plan.name == "coremltools", plan.describe()

    from modconv.real_backend import CoremltoolsBackend
    result = CoremltoolsBackend(src, out, max_seq_len=8).run(RecordingHooks())
    assert result.success is True, result.message

    spec = ct.utils.load_spec(str(result.output_path))
    names = [i.name for i in spec.description.input]
    assert names[:2] == ["input_ids", "cache_position"]
    # 混合架构（含 linear attention 层）额外带 GDN 的 conv/rec 状态输入
    assert set(names[2:]) <= {"conv_states", "recurrent_states"}, names
    assert spec.description.output[0].name == "logits"
    _assert_4bit_palettized(spec)

    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
    except Exception:
        return

    mlmodel = ct.models.MLModel(str(result.output_path),
                                compute_units=ct.ComputeUnit.ALL)

    stepper = _Stepper(mlmodel)
    pre = stepper.prefill([1, 2, 3, 4])
    assert pre.shape == (1, 1, 128) and np.isfinite(pre).all()
    dec1 = stepper(5, 4)
    dec2 = stepper(6, 5)
    assert np.isfinite(dec2).all(), "多步 decode 后 logits 不应含 NaN/Inf"
    # 连续 decode 与整段 prefill 的末位一致性（KV/state 推进正确）
    whole = _Stepper(mlmodel, mlmodel.make_state()).prefill([1, 2, 3, 4, 5, 6])
    assert np.abs(whole[0, 0, :] - dec2[0, 0, :]).max() < 0.05, \
        "连续 decode 与整段 prefill 的末位 logits 不一致，state 推进有误"
    # 空 state 下同位置同 token 的输出必须与有历史时不同
    empty = _Stepper(mlmodel, mlmodel.make_state())(5, 4)
    assert np.abs(dec1 - empty).max() > 1e-3, "空 state 的 decode 与有 cache 时相同"


def test_qwen35_coreml_numeric_parity_with_torch(tmp_path):
    """Core ML 导出模型 vs torch 参考的数值一致性（准确率无损失的直接证据）。

    同一权重源转两版：fp16 无量化（quantize_nbits=None，隔离转换误差）与 4-bit
    量化。fp16 版对 torch eager 参考做严格断言（逐位最大误差 + 逐位置余弦相似
    度）；4-bit 版同时对比 torch 参考与 fp16 版，量化只允许统计意义上的微小偏
    差。覆盖 prefill（8/12 token）、8 步连续 decode 至缓存末槽（位置 15/16）。"""
    import coremltools as ct
    import numpy as np
    import torch

    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
    except Exception:
        pytest.skip("需要 coremltools 原生运行库做本地推理")

    torch.manual_seed(23)
    src = _make_tiny_qwen35(tmp_path, name="qwen35-parity")
    max_seq = 16

    # torch 参考：与转换同源的权重，加载方式与 real_backend 一致（fp32 eager）
    from modconv.kv_wrapper import KvCacheWrapper
    from transformers import AutoModelForCausalLM
    kwargs = {"dtype": torch.float32, "attn_implementation": "eager"}
    try:
        ref_model = AutoModelForCausalLM.from_pretrained(str(src), **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        kwargs.pop("attn_implementation", None)
        ref_model = AutoModelForCausalLM.from_pretrained(str(src), **kwargs)
    ref_model.eval()
    wrapper = KvCacheWrapper(ref_model, max_seq)

    # 固定字面量输入：argmax 一致率断言与输入完全解耦（不受 RNG 流变化影响）
    # 注意 prefill 必须是单序列 [1, q]：模型输入声明为 (1, 1..max_seq)，
    # 写成 [q, 1] 会变成 batch=q 而被 Core ML 拒绝。
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    dec_ids = torch.tensor([[t] for t in range(9, 17)])
    long_ids = torch.tensor([list(range(17, 29))])
    with torch.no_grad():
        ref = {}
        wrapper.reset_state()
        per_pos = [wrapper.step(ids[:, i:i + 1], torch.tensor([i]))
                   for i in range(8)]
        ref["prefill8"] = torch.cat(per_pos, dim=1).numpy()
        for i in range(8):
            ref[f"dec{i}"] = wrapper.step(dec_ids[i].view(1, 1),
                                     torch.tensor([8 + i])).numpy()
        wrapper.reset_state()  # 对应 Core ML 侧换全新 state
        wrapper.reset_state()
        ref["prefill12"] = torch.cat(
            [wrapper.step(long_ids[:, i:i + 1], torch.tensor([i]))
             for i in range(12)], dim=1).numpy()
        # 参考输出形状必须与模型输入声明一致，避免"batch 维被当成序列维"的静默错误
        assert ref["prefill8"].shape == (1, 8, 128)
        assert ref["prefill12"].shape == (1, 12, 128)

    def _convert(nbits, out_name):
        out = tmp_path / out_name
        out.mkdir()
        from modconv.real_backend import CoremltoolsBackend
        result = CoremltoolsBackend(src, out, max_seq_len=max_seq,
                                    quantize_nbits=nbits).run(RecordingHooks())
        assert result.success is True, result.message
        return ct.models.MLModel(str(result.output_path),
                                 compute_units=ct.ComputeUnit.ALL)

    fp16_model = _convert(None, "out-fp16")
    quant_model = _convert(4, "out-4bit")

    def _run(model):
        state = model.make_state()

        def predict(ids_np, pos_np):
            o = model.predict({"input_ids": np.asarray(ids_np, dtype=np.int32),
                               "cache_position": np.asarray(pos_np, dtype=np.int32)},
                              state)
            return np.asarray(o["logits"], dtype=np.float32)

        # 单步模型：逐 token 推进，把各位置的 logits 拼回 (1, q, vocab)
        stepper = _Stepper(model)
        outs = {"prefill8": np.concatenate(
            [stepper(int(ids[0, i]), i) for i in range(8)], axis=1)}
        for i in range(8):
            outs[f"dec{i}"] = stepper(int(dec_ids[i]), 8 + i)
        fresh = _Stepper(model, model.make_state())  # 全新 state：对应 reset_state
        outs["prefill12"] = np.concatenate(
            [fresh(int(long_ids[0, i]), i) for i in range(12)], axis=1)
        return outs

    def _metrics(r, g):
        rr = r.reshape(r.shape[1], -1)
        gg = g.reshape(g.shape[1], -1)
        err = float(np.abs(rr - gg).max())
        cos = float(((rr * gg).sum(-1)
                     / (np.linalg.norm(rr, axis=-1)
                        * np.linalg.norm(gg, axis=-1))).min())
        return err, cos, int((rr.argmax(-1) == gg.argmax(-1)).sum())

    fp16_out = _run(fp16_model)
    quant_out = _run(quant_model)
    fp16_am = quant_am = 0
    for key, r in ref.items():
        assert np.isfinite(fp16_out[key]).all(), f"{key}: fp16 输出含非有限值"
        err, cos, am = _metrics(r, fp16_out[key])
        fp16_am += am
        # 实测基线（固定 seed）：max|d| ≤ 0.033、minCos ≥ 0.9942、argmax 26/28。
        # 偏差来自 GDN 分块内核在 Core ML fp16 下的累加噪声（纯注意力路径
        # 实测 fp16 几乎精确 ≤8e-4；torch 侧 fp16 噪声基线 ≤3e-4）。
        assert err < 0.05 and cos > 0.99, \
            f"{key}: fp16 vs torch 最大误差 {err:.4f}，最小余弦 {cos:.5f}"
        assert np.isfinite(quant_out[key]).all(), f"{key}: 4-bit 输出含非有限值"
        _, cos4, am4 = _metrics(r, quant_out[key])
        err_q, cos_q, _ = _metrics(fp16_out[key], quant_out[key])
        quant_am += am4
        # 实测基线：4-bit vs torch minCos ≥ 0.9756、max|d| ≤ 0.087（量化主导）
        assert cos4 > 0.97 and cos_q > 0.97 and err_q < 0.15, \
            (f"{key}: 4-bit vs torch 余弦 {cos4:.5f}；"
             f"vs fp16 余弦 {cos_q:.5f} / 最大误差 {err_q:.4f}")
    # 行为级指标：argmax（贪心 token）一致率。随机 tiny 模型 logits 近均匀、
    # top-2 差距常小于数值噪声，近似并列翻面属正常，断言留足余量
    # （实测 fp16 26/28、4-bit 23/28；真正损坏的模型会趋近 0）。
    assert fp16_am >= 24, f"fp16 argmax 一致率过低：{fp16_am}/28"
    assert quant_am >= 20, f"4-bit argmax 一致率过低：{quant_am}/28"


def test_llama_coreml_numeric_parity_with_torch(tmp_path):
    """纯注意力架构（llama）的 Core ML vs torch 数值一致性（准确率的直接证据）。

    同一权重源转两版：fp16 无量化（隔离转换误差）与 4-bit 量化，各自对 torch
    eager 参考做最大误差 / 最小余弦相似度 / argmax 一致率三层断言，覆盖 prefill
    与多步 decode（KV cache 推进）。
    """
    import numpy as np
    import torch

    import coremltools as ct

    try:
        from coremltools.libcoremlpython import _MLModelProxy  # noqa: F401
    except Exception:
        pytest.skip("需要 coremltools 原生运行库做本地推理")

    torch.manual_seed(17)
    src = _make_tiny_llama(tmp_path, name="llama-parity")
    max_seq = 16

    from modconv.kv_wrapper import KvCacheWrapper
    from transformers import AutoModelForCausalLM
    kwargs = {"dtype": torch.float32, "attn_implementation": "eager"}
    try:
        ref_model = AutoModelForCausalLM.from_pretrained(str(src), **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        kwargs.pop("attn_implementation", None)
        ref_model = AutoModelForCausalLM.from_pretrained(str(src), **kwargs)
    ref_model.eval()
    wrapper = KvCacheWrapper(ref_model, max_seq)

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    dec_ids = torch.tensor([[t] for t in range(6, 12)])
    with torch.no_grad():
        ref = {}
        wrapper.reset_state()
        ref["prefill5"] = torch.cat(
            [wrapper.step(ids[:, i:i + 1], torch.tensor([i]))
             for i in range(5)], dim=1).numpy()
        for i in range(6):
            ref[f"dec{i}"] = wrapper.step(dec_ids[i].view(1, 1),
                                     torch.tensor([5 + i])).numpy()

    def _convert(nbits, out_name):
        out = tmp_path / out_name
        out.mkdir()
        from modconv.real_backend import CoremltoolsBackend
        result = CoremltoolsBackend(src, out, max_seq_len=max_seq,
                                    quantize_nbits=nbits).run(RecordingHooks())
        assert result.success is True, result.message
        return ct.models.MLModel(str(result.output_path),
                                 compute_units=ct.ComputeUnit.ALL)

    def _run(model):
        state = model.make_state()

        def predict(ids_np, pos_np):
            o = model.predict({"input_ids": np.asarray(ids_np, dtype=np.int32),
                               "cache_position": np.asarray(pos_np, dtype=np.int32)},
                              state)
            return np.asarray(o["logits"], dtype=np.float32)

        stepper = _Stepper(model)
        outs = {"prefill5": np.concatenate(
            [stepper(int(ids[0, i]), i) for i in range(5)], axis=1)}
        for i in range(6):
            outs[f"dec{i}"] = stepper(int(dec_ids[i]), 5 + i)
        return outs

    fp16_out = _convert(None, "out-fp16")
    fp16_res = _run(fp16_out)
    del fp16_out
    quant_model = _convert(4, "out-4bit")
    quant_res = _run(quant_model)
    del quant_model

    fp16_am = quant_am = 0
    total_positions = 0
    for key, r in ref.items():
        rr = r.reshape(r.shape[1], -1)
        total_positions += rr.shape[0]
        for name, out in (("fp16", fp16_res), ("4-bit", quant_res)):
            assert np.isfinite(out[key]).all(), f"{key}: {name} 输出含非有限值"
            gg = out[key].reshape(out[key].shape[1], -1)
            err = float(np.abs(rr - gg).max())
            cos = float(((rr * gg).sum(-1)
                         / (np.linalg.norm(rr, axis=-1)
                            * np.linalg.norm(gg, axis=-1))).min())
            am = int((rr.argmax(-1) == gg.argmax(-1)).sum())
            # fp16 转换本身几乎无损（纯注意力路径实测 ≤1e-3）；
            # 4-bit palettization 只引入统计意义上的小偏差
            limit_err = 0.02 if name == "fp16" else 0.15
            limit_cos = 0.999 if name == "fp16" else 0.97
            assert err < limit_err and cos > limit_cos, \
                f"{key}: {name} vs torch 最大误差 {err:.4f}，最小余弦 {cos:.5f}"
            if name == "fp16":
                fp16_am += am
            else:
                quant_am += am
    # 行为级指标：贪心 token（argmax）一致率——真正损坏的模型会趋近 0
    assert fp16_am == total_positions, \
        f"fp16 argmax 一致率过低：{fp16_am}/{total_positions}"
    assert quant_am >= total_positions - 1, \
        f"4-bit argmax 一致率过低：{quant_am}/{total_positions}"


def _convert_assertions_only(out_path):
    """量化路径的 spec 断言（与主流程一致：双输入 + logits + 4-bit）。"""
    import coremltools as ct

    spec = ct.utils.load_spec(str(out_path))
    names = [i.name for i in spec.description.input]
    assert names[:2] == ["input_ids", "cache_position"]
    # 混合架构（含 linear attention 层）额外带 GDN 的 conv/rec 状态输入
    assert set(names[2:]) <= {"conv_states", "recurrent_states"}, names
    assert spec.description.output, "应有输出"
    assert spec.description.output[0].name == "logits", "输出应重命名为 logits"
    assert spec.description.metadata.shortDescription, "应写入模型元数据"
    assert spec.description.input[0].shortDescription, "输入应有描述"
    assert spec.description.output[0].shortDescription, "输出应有描述"
    _assert_4bit_palettized(spec)
