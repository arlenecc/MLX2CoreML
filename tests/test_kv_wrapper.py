"""KvCacheWrapper 的数值 parity 测试：与 HF DynamicCache 参考实现对比。"""
import importlib.util

import pytest


def _heavy_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


pytestmark = pytest.mark.skipif(not _heavy_available(), reason="需要 torch/transformers")

MAX_SEQ = 16


def _tiny_model():
    import torch

    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=32)
    return LlamaForCausalLM(config).eval()


def test_wrapper_state_specs():
    """全部层的 K/V 合并为单个 state（每层独立 state 会撞上 ANE 的数量上限）。"""
    from modconv.kv_wrapper import KvCacheWrapper
    wrapper = KvCacheWrapper(_tiny_model(), MAX_SEQ)
    names = [name for name, _, _ in wrapper.kv_state_specs()]
    assert names == ["kv_all"]
    # (full 层数, K/V, kv 头数, max_seq_len, head_dim)
    for name, shape, dtype in wrapper.kv_state_specs():
        assert tuple(shape) == (2, 2, 2, MAX_SEQ, 8)
        assert dtype == "float16"
    for buf in wrapper.state_buffers():
        assert tuple(buf.shape) == (2, 2, 2, MAX_SEQ, 8)


def test_prefill_and_decode_parity_with_hf():
    import torch

    from transformers.cache_utils import DynamicCache
    from modconv.kv_wrapper import KvCacheWrapper
    model = _tiny_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (1, 5))
    steps = torch.randint(0, 64, (2, 1))

    # HF 参考：DynamicCache 逐步推进（decode 的 input_ids 必须是 (1,1)）
    with torch.no_grad():
        cache = DynamicCache()
        ref = [model(input_ids=ids, cache_position=torch.arange(5),
                     past_key_values=cache, use_cache=True).logits]
        pos = 5
        for step_ids in steps:
            ref.append(model(input_ids=step_ids.reshape(1, 1),
                             cache_position=torch.tensor([pos]),
                             past_key_values=cache, use_cache=True).logits)
            pos += 1

    wrapper = KvCacheWrapper(model, MAX_SEQ)
    with torch.no_grad():
        got = [wrapper(ids, torch.arange(5))]
        pos = 5
        for step_ids in steps:
            got.append(wrapper(step_ids.reshape(1, 1), torch.tensor([pos])))
            pos += 1

    for i, (r, g) in enumerate(zip(ref, got)):
        assert torch.allclose(r, g, atol=1e-4), \
            f"第 {i} 步 logits 不一致（最大误差 {(r - g).abs().max().item()}）"


def test_reset_state_clears_buffers():
    import torch

    from modconv.kv_wrapper import KvCacheWrapper
    wrapper = KvCacheWrapper(_tiny_model(), MAX_SEQ)
    with torch.no_grad():
        wrapper(torch.randint(0, 64, (1, 3)), torch.arange(3))
    assert any(b.abs().sum() > 0 for b in wrapper.state_buffers())
    wrapper.reset_state()
    assert all(torch.count_nonzero(b) == 0 for b in wrapper.state_buffers())


# --------------------------------------------------------------------- #
# 混合注意力架构（Qwen3.5：GDN linear attention + full attention）
# --------------------------------------------------------------------- #

def _tiny_qwen35_config():
    from transformers import Qwen3_5TextConfig
    return Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
    )


def _tiny_qwen35_model():
    import torch

    from transformers import Qwen3_5ForCausalLM
    torch.manual_seed(11)
    return Qwen3_5ForCausalLM(_tiny_qwen35_config()).eval()


def test_qwen35_hybrid_state_specs():
    """混合架构同样合并 state：full 层 K/V 一个、GDN 层 conv/rec 各一个。"""
    from modconv.kv_wrapper import KvCacheWrapper
    wrapper = KvCacheWrapper(_tiny_qwen35_model(), MAX_SEQ)
    specs = wrapper.kv_state_specs()
    names = [name for name, _, _ in specs]
    assert names == ["kv_all", "conv_all", "rec_all"]
    shapes = {name: tuple(shape) for name, shape, _ in specs}
    # 4 层里 2 层 full attention（层 1/3）→ kv_all 第 0 维为 2
    assert shapes["kv_all"] == (2, 2, 2, MAX_SEQ, 8)
    # 2 层 GDN（层 0/2）→ conv/rec 的第 0 维为 2；conv 末维是卷积核大小
    assert shapes["conv_all"][0] == 2 and len(shapes["conv_all"]) == 3
    assert shapes["rec_all"][0] == 2
    for _, _, dtype in specs:
        assert dtype == "float16"
    assert wrapper.state_buffers()


def test_qwen35_hybrid_prefill_decode_parity_with_hf():
    import torch

    from transformers.cache_utils import DynamicCache
    from modconv.kv_wrapper import KvCacheWrapper
    model = _tiny_qwen35_model()
    torch.manual_seed(3)
    ids = torch.randint(0, 64, (1, 5))
    steps = torch.randint(0, 64, (2, 1))

    # HF 参考：DynamicCache（hybrid：linear 层 conv/rec + full 层动态 KV）逐步推进
    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        ref = [model(input_ids=ids, cache_position=torch.arange(5),
                     past_key_values=cache, use_cache=True).logits]
        pos = 5
        for step_ids in steps:
            ref.append(model(input_ids=step_ids.reshape(1, 1),
                             cache_position=torch.tensor([pos]),
                             past_key_values=cache, use_cache=True).logits)
            pos += 1

    wrapper = KvCacheWrapper(model, MAX_SEQ)
    with torch.no_grad():
        got = [wrapper(ids, torch.arange(5))]
        pos = 5
        for step_ids in steps:
            got.append(wrapper(step_ids.reshape(1, 1), torch.tensor([pos])))
            pos += 1

    for i, (r, g) in enumerate(zip(ref, got)):
        assert torch.allclose(r, g, atol=1e-4), \
            f"第 {i} 步 logits 不一致（最大误差 {(r - g).abs().max().item()}）"


def _tiny_qwen35_moe_model():
    import torch

    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    torch.manual_seed(11)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=2, mlp_only_layers=[1],
        max_position_embeddings=64,
    )
    return Qwen3_5MoeForCausalLM(config).eval()


def test_qwen35_moe_prefill_decode_parity_with_hf():
    """Qwen3.5 MoE 与 HF DynamicCache 参考逐步一致（含专家层与 GDN 变体）。"""
    import torch

    from transformers.cache_utils import DynamicCache
    from modconv.kv_wrapper import KvCacheWrapper
    model = _tiny_qwen35_moe_model()
    torch.manual_seed(3)
    ids = torch.randint(0, 64, (1, 5))
    steps = torch.randint(0, 64, (2, 1))

    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        ref = [model(input_ids=ids, cache_position=torch.arange(5),
                     past_key_values=cache, use_cache=True).logits]
        pos = 5
        for step_ids in steps:
            ref.append(model(input_ids=step_ids.reshape(1, 1),
                             cache_position=torch.tensor([pos]),
                             past_key_values=cache, use_cache=True).logits)
            pos += 1

    wrapper = KvCacheWrapper(model, MAX_SEQ)
    with torch.no_grad():
        got = [wrapper(ids, torch.arange(5))]
        pos = 5
        for step_ids in steps:
            got.append(wrapper(step_ids.reshape(1, 1), torch.tensor([pos])))
            pos += 1

    for i, (r, g) in enumerate(zip(ref, got)):
        assert torch.allclose(r, g, atol=1e-4), \
            f"第 {i} 步 logits 不一致（最大误差 {(r - g).abs().max().item()}）"


def test_moe_experts_forward_matches_reference():
    """MoE 专家层的导出友好前向与原版 eager 前向逐位等价（准确率无损失）。"""
    import torch

    from modconv.kv_wrapper import KvCacheWrapper, _moe_experts_forward

    model = _tiny_qwen35_moe_model()
    KvCacheWrapper(model, MAX_SEQ)  # 触发补丁（实例级 forward 已被替换）
    experts = next(m for m in model.modules()
                   if type(m).__name__.endswith("Experts")
                   and hasattr(m, "gate_up_proj"))
    # 参考实现用 eager 专家路径：融合内核（grouped_mm/batched_mm）需要
    # torch.histc 等算子，且数值上应与 eager 一致但依赖可选内核是否可用
    experts.config._experts_implementation = "eager"
    ref_forward = type(experts).forward

    torch.manual_seed(5)
    num_tokens = 7
    top_k = int(experts.config.num_experts_per_tok)
    x = torch.randn(num_tokens, model.config.hidden_size)
    routing = torch.softmax(torch.randn(num_tokens, experts.num_experts), dim=-1)
    weights, index = torch.topk(routing, top_k, dim=-1)
    with torch.no_grad():
        got = _moe_experts_forward(experts, x, index, weights)
        ref = ref_forward(experts, x, index, weights)
    assert torch.allclose(got, ref, atol=1e-5), \
        f"专家层前向不等价（最大误差 {(got - ref).abs().max().item()}）"


def test_qwen35_hybrid_rejects_max_seq_len_1():
    from modconv.kv_wrapper import KvCacheWrapper
    with pytest.raises(ValueError, match="至少为 2"):
        KvCacheWrapper(_tiny_qwen35_model(), 1)


# --------------------------------------------------------------------- #
# Qwen3.5 rotary 导出补丁的数值等价性（导出精度无损失的证据）
# --------------------------------------------------------------------- #

def test_qwen35_text_rope_simplification_equivalence():
    """纯文本（T/H/W 三组 position_ids 相同）时 TextRotary 化简实现与原版逐位一致。

    wrapper 把 Qwen3_5TextRotaryEmbedding.forward 替换为 _qwen35_text_rope_forward
    （绕开 coremltools 前端对 expand(3,-1,-1) 的符号化缺陷）。此处用未 patch 的
    原版模块独立验证替换无精度损失，覆盖 decode（q=1）与 prefill（q>1）。"""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    from modconv.kv_wrapper import _qwen35_text_rope_forward
    rot = Qwen3_5TextRotaryEmbedding(_tiny_qwen35_config())
    for q in (1, 5):
        x = torch.randn(1, q, 8)
        pos3 = torch.arange(q).view(1, 1, -1).expand(3, 1, -1)
        with torch.no_grad():
            cos_ref, sin_ref = rot(x, pos3)  # 原版 forward（未 patch）
            cos_new, sin_new = _qwen35_text_rope_forward(rot, x, pos3)
        assert torch.allclose(cos_ref, cos_new, rtol=0, atol=1e-6), \
            f"q={q} cos 不一致（最大误差 {(cos_ref - cos_new).abs().max().item()}）"
        assert torch.allclose(sin_ref, sin_new, rtol=0, atol=1e-6), \
            f"q={q} sin 不一致（最大误差 {(sin_ref - sin_new).abs().max().item()}）"


def test_qwen35_rope_recomposition_patch_equivalence():
    """_qwen35_rope_recomposition 与原版 in-place 切片重排逐位一致。

    原版用 slice 赋值重排 mrope 的 T/H/W 三组频率（导出后 slice_scatter 过不了
    coremltools 前端校验），wrapper 替换为静态掩码 where 链。此处构造 T/H/W 互不
    相同的多模态式输入，并让 head_dim 大到使 H/W 各覆盖多个交错下标（非平凡
    区间），验证重排语义完全等价——不依赖"三组相同"的文本特例。"""
    import torch
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    from modconv.kv_wrapper import _qwen35_rope_recomposition
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, num_hidden_layers=1,
                               num_attention_heads=4, head_dim=64)
    # head_dim=64 × 默认 partial_rotary_factor 0.25 → dim=16 → half=8；
    # mrope_section [2,2,2] 使 H 覆盖 {1,4}、W 覆盖 {2,5}，均为多下标交错区间
    config.rope_parameters["mrope_section"] = [2, 2, 2]
    rot = Qwen3_5TextRotaryEmbedding(config)
    assert rot.mrope_section == [2, 2, 2]

    q = 5
    pos3 = torch.stack([torch.arange(q), torch.arange(q) + 7,
                        torch.arange(q) + 13]).view(3, 1, q)
    freqs = pos3[..., None].float() * rot.inv_freq.float()  # (3, 1, q, 8)
    cos = freqs.cos() * rot.attention_scaling

    new = _qwen35_rope_recomposition(rot, cos)  # 纯函数，不改输入
    ref = rot.recomposition_frequencies(cos.clone())  # 原版会 in-place 改写输入
    assert torch.allclose(ref, new, rtol=0, atol=1e-6), \
        f"重排结果不一致（最大误差 {(ref - new).abs().max().item()}）"
