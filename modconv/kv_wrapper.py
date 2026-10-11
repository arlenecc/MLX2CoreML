"""带 KV cache 状态的 torch 包装模块（供 torch.export 导出为 Core ML stateful 模型）。

设计：
- 每层 K/V 注册为 float32 buffer（形状 (1, n_kv_heads, max_seq, head_dim)），导出后成为
  Core ML State（coremltools 按 named_buffers 名字匹配 StateType）。buffer 必须固定
  fp32：前端对 read_state 无条件插入 fp32 cast，in-place 写（scatter）的 data 因此
  恒为 fp32，updates 也必须 fp32；写回 state 时前端自动 cast 回 fp16。
- transformers StaticCache 面向 torch.export 设计：early_initialization 预分配后，
  把 layers[i].keys/values 指向上述 buffer，update() 的 scatter 原地写即落 buffer。
- 每次前向把 layer.cumulative_length 重置为 cache_position 起点（从输入推导，
  避免导出图里 bake 常量 0 导致 decode 步写错槽位）。
- 注意力掩码由 wrapper 内部按 cache_position 自算静态形状 4D 因果掩码（单序列无
  padding，Swift 无需传 mask），绕开 HF 动态掩码机器（其运行时 tile 会破坏 ANE）。
- 混合注意力架构（Qwen3.5：linear_attention/GDN 与 full_attention 交替）：
  * linear 层的 cache 是 conv/recurrent state（非 K/V），保留 transformers 原生
    LinearAttentionLayer，把其 state 槽位重绑定为 wrapper buffer（update 为 copy_
    原地写，即 Core ML State 写入）；state 形状与序列长度无关，直接由 config 推出，
    无需预热。
  * GDN 分块内核的 chunk 循环次数取决于序列长度（动态值无法导出），因此 wrapper
    把输入 pad 到固定 max_seq_len 再前向——追踪期 seq_len 恒为常量，HF 前向里
    所有 seq_len==1 的 decode 快速路径随之静态消除，chunk 循环次数也为常量。
  * pad 语义：pad 位置的 KV/conv 输入是无效值，但只会写入"未处理"槽位（生成按
    位置递增推进，位置 >= start 的槽位还没有真实数据），且注意力掩码按位置因果
    屏蔽、GDN 投影按真实区间置零，不影响结果；这些槽位会被后续真实 token 逐步
    覆写。输出切回真实长度。

注意：本模块顶层 import torch/transformers（重量级），只应从真实转换流程懒加载。
"""
from __future__ import annotations

import importlib
from types import MethodType

import torch
from torch import nn
from transformers.cache_utils import (
    LinearAttentionCacheLayerMixin,
    StaticCache,
    StaticLayer,
)


class _SlotStaticLayer(StaticLayer):
    """StaticLayer 变体：写入槽位 = arange(kv_length) + cumulative_length，
    但不 in-place 累加游标。

    原版 update 会执行 ``cumulative_length.add_(kv_length)``，而 wrapper 每次
    forward 都把游标重绑定为 cache_position 起点（输入派生张量），torch.export
    对这种重绑定后的张量做 in-place 累加会报
    "cannot mutate tensors with frozen storage"。由于游标每步都被重置，
    累加本来就不需要；图中唯一保留的 mutation 是对 KV buffer 的 scatter_ 写入
    （导出后对应 Core ML State 写入，属合法的 buffer mutation）。

    混合架构下 kv_length 是静态的 max_seq_len（pad 到固定长度前向），写入索引
    可能超出缓冲（start+max-1 > max-1），用 clamp 收敛到最后一个槽位：被 clamp
    的都是 pad 的无效数据，且其掩码位置在真实游标之后、永远不会被注意得到。
    """

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        kv_length = key_states.shape[-2]
        cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
        cache_position = cache_position.clamp(0, self.max_cache_len - 1)
        # 用 rank-4 索引的 scatter_ 写入（而非 index_copy_）：coremltools 前端把
        # scatter 映射为 scatter_along_axis（支持动态长度），而 index_put 的
        # 整片切片路径遇到动态 1-D 索引会因 squeeze 后 rank 不匹配而崩溃。
        # 索引扩维用 unsqueeze + zeros_like 加法广播，而非 expand：expand 在前端
        # 会退化为运行时 shape 张量，输出全符号形状（丢失静态维度导致断言失败）。
        idx = cache_position.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        idx = torch.zeros_like(key_states, dtype=idx.dtype) + idx
        kv_dtype = self.keys.dtype
        try:
            self.keys.scatter_(2, idx, key_states.to(kv_dtype))
            self.values.scatter_(2, idx, value_states.to(kv_dtype))
        except NotImplementedError:
            # MPS 等后端 scatter_/index_copy_ 不可用时的回退
            self.keys[:, :, cache_position] = key_states.to(kv_dtype)
            self.values[:, :, cache_position] = value_states.to(kv_dtype)
        # 注意力消费方与 key_states 同精度（eager fp32 / 导出后统一 fp16）
        return self.keys.to(key_states.dtype), self.values.to(value_states.dtype)


def _qwen35_gdn_forward(self, hidden_states, cache_params, attention_mask=None, **kwargs):
    """Qwen3.5 GatedDeltaNet 导出适配前向。

    改自 transformers/models/qwen3_5/modeling_qwen3_5.py 的
    Qwen3_5GatedDeltaNet.forward（Apache-2.0）。wrapper 已把输入 pad 到静态
    max_seq_len（seq_len>=2），原版的 seq_len==1 decode 快速路径在追踪期被静态
    消除，故整个前向只保留 prefill 路径。与原版的实质差异：

    1. conv state 改为"滚动收集"更新——wrapper 的输入带 pad，无法像原版那样
       cat(state, new) 后取末尾 kernel 个（会把 pad 收进 state）。改为用纯张量
       索引从 cat(prev_state, padded_stream) 中收集位置 [end-k+1, end] 的原始
       qkv（end=cache_position 末元素），索引越界部分置零，结果 copy_ 回 state
       buffer（导出后即 Core ML State 写入）。
    2. causal conv 输出与衰减 g 按真实区间 [start, end] 置零——pad 位置的投影含
       偏置项、g 由 a 投影 + dt_bias 计算均非零，会污染 recurrent state；真实
       token 的 qkv 全部来自投影输出，置零 pad 即可保证 state 只由真实 token
       推进。decode 时 [start, end] 只含当前 token，不会重复计入历史。

    内核函数由 `_prepare_hybrid` 在打补丁时解析并挂到 `self._modconv_kernels`：
    dense 与 MoE 变体各自持有一份由 modular 生成的同名内核（实现相同），且
    `importlib.import_module` 这类动态导入无法被 torch.export 追踪，必须在追踪
    前把函数对象固定下来。
    """
    import torch.nn.functional as F

    (apply_mask_to_padding_states, causal_conv1d_fn,
     torch_chunk_gated_delta_rule) = self._modconv_kernels

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
        self.layer_idx, state_idx=0
    )
    cache_pos = kwargs.get("cache_position")
    if cache_pos is not None:
        cache_pos = cache_pos.reshape(-1)
        start, end = cache_pos[0], cache_pos[-1]
        # 真实 token 区间 [start, end]；pad 位置为 0。注意 padded 序列索引 i 对应
        # 位置 start+i，因此索引 i 真实 ⟺ i <= end-start（即 i < 真实 token 数）。
        # 不能拿 arange(seq_len) 直接与 start/end 比较——decode 时会把前面的真实
        # token 置零、把后面的 pad 当真实，prefill（start=0）恰好不出错。
        real_mask = (torch.arange(seq_len, device=hidden_states.device) <= (end - start))
        real_mask = real_mask.to(hidden_states.dtype)  # (seq_len,)，使用处再 reshape
    else:
        start = end = None
        real_mask = None

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    kernel = self.conv_kernel_size
    if cache_params is not None:
        layer_cache = cache_params.layers[self.layer_idx]
        conv_buf = layer_cache.conv_states[0]  # (1, conv_dim, kernel)，fp32 buffer
        stream = mixed_qkv.to(conv_buf.dtype)
        cat_in = torch.cat([conv_buf, stream], dim=2)  # (1, conv_dim, kernel+seq)
        # conv 输出：位置 start+j 对应 cat 索引 kernel+j（prev_state 覆盖
        # [start-kernel, start-1]，cat 索引 = 位置 - start + kernel）
        conv_out = causal_conv1d_fn(
            cat_in.to(mixed_qkv.dtype),
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
            **kwargs,
        )
        mixed_qkv = conv_out[:, :, kernel:]
        # conv state 滚动更新：收集位置 [end-kernel+1, end] 的原始 qkv
        #（cat 索引 = 位置 - start + kernel = end-start+1+j），越界（早期位置）
        # 收集结果置零，等价于首步用零左上下文。
        idx = ((end - start + 1) + torch.arange(kernel, device=conv_buf.device)).long()
        valid = (idx >= 0).to(cat_in.dtype).view(1, 1, -1)
        gathered = cat_in[:, :, idx.clamp(0, cat_in.shape[-1] - 1)]
        conv_buf.copy_(gathered * valid)
    else:
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
            **kwargs,
        )
        mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    if real_mask is not None:
        # (1, seq, conv_dim) × (1, seq, 1)：置零 pad 位置的 qkv 投影
        mixed_qkv = mixed_qkv * real_mask.view(1, seq_len, 1)
    query, key, value = torch.split(
        mixed_qkv,
        [
            self.key_dim,
            self.key_dim,
            self.value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if real_mask is not None:
        g = g * real_mask.to(g.dtype).view(1, seq_len, -1)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
        **kwargs,
    )

    # Update cache
    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


def _qwen35_text_rope_forward(self, x, position_ids):
    """Qwen3_5TextRotaryEmbedding.forward 的纯文本等价实现。

    原版（transformers/models/qwen3_5/modeling_qwen3_5.py，Apache-2.0）：TextModel
    传入 (3, bs, q) 的 T/H/W 三组 position_ids，forward 里 ``expand(3, -1, -1)``
    （对已是 3 份的输入是 no-op）后算 freqs (3, bs, q, dim//2)，再用
    recomposition_frequencies 按 mrope_section 重排三组频率。纯文本时三组
    position_ids 完全相同，重排输出的每一段都等于 freq[0]，因此整个流程恒等于
    ``cat(freqs[0].cos(), freqs[0].cos())``，可安全省去 3 倍广播与重排。

    这同时绕开 coremltools 前端的两个缺陷：expand(3, -1, -1) 在动态输入下类型
    推断会把静态维也符号化并沿广播链污染整图（KV scatter 的 data/indices batch
    维不匹配断言失败）；recomposition 的 where 链输出在广播中产生不一致符号。
    仅适用于文本导出（本工具不导出多模态）。position_ids 取 [0] 对 3D/2D 输入
    均取到第一组（文本语义）。
    """
    freqs = position_ids[0][..., None].float() * self.inv_freq.to(
        device=x.device, dtype=torch.float)
    cos = freqs.cos() * self.attention_scaling
    sin = freqs.sin() * self.attention_scaling
    return (torch.cat((cos, cos), dim=-1).to(dtype=x.dtype),
            torch.cat((sin, sin), dim=-1).to(dtype=x.dtype))


def _qwen35_rope_recomposition(self, freq):
    """Qwen3_5RotaryEmbedding.recomposition_frequencies 的导出友好实现。

    改自 transformers/models/qwen3_5/modeling_qwen3_5.py（Apache-2.0）。原版用
    in-place 切片赋值（freqs_thw[..., idx] = freq[dim][..., idx]）重排 mrope 的
    T/H/W 三组频率，导出后成为 slice_scatter，coremltools 前端对其静态切片形状
    校验失败（"update tensor should have shape [..., 0]"）。三个赋值区间按
    mod 3 余数（0/1/2）天然不重叠，改用静态掩码 where 链等价重写，多模态与
    纯文本语义均与原版一致。
    """
    half = freq.shape[-1]
    cols = torch.arange(half, device=freq.device)
    out = freq[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = self.mrope_section[dim] * 3
        in_slice = ((cols >= offset) & (cols < length)
                    & ((cols - offset) % 3 == 0))
        out = torch.where(in_slice.view(1, 1, -1), freq[dim], out)
    return torch.cat((out, out), dim=-1)


def _moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Qwen3.5 MoE 专家层的导出友好前向（数值与原版逐专家累加完全等价）。

    transformers 提供的两条专家路径都不可导出：
    - eager：在 Python 层遍历数据相关的 ``expert_hit``（``for expert_idx in
      expert_hit``），dynamo 无法静态展开；
    - grouped_mm / batched_mm：用 ``torch.histc`` 统计每张专家的 token 数，
      coremltools 前端不支持该算子（"Unsupported fx node histc"）。

    改为静态批量路径：把每个 (token, top-k 槽位) 对展平成 S = T*k 行，按 expert
    id gather 出对应专家的权重，做 batched matmul，再 reshape 回 (T, k, H) 按
    槽位求和。未选中的专家权重恒为 0（router 已把非 top-k 置零），因此"跳过未命
    中专家"与原版等价，不必依赖数据相关的控制流。计算量也与原版相同——只对
    top-k 命中的专家做前向，而非全专家。

    注意：权重是 gather 出来的非常量，Core ML 会用 matmul（而非 linear/gemm）
    执行，无法走 ANE 的常量权重快路径，但仍比"全专家计算"快 k 倍。
    """
    num_tokens = hidden_states.shape[0]
    top_k = int(top_k_index.shape[-1])
    flat_idx = top_k_index.reshape(-1)                                   # (S,)
    flat_w = top_k_weights.reshape(-1, 1)                                # (S, 1)
    # 每个 token 复制 top_k 份，与 flat_idx 一一对应
    x = hidden_states.repeat_interleave(top_k, dim=0)                    # (S, H)

    proj = torch.matmul(self.gate_up_proj[flat_idx], x.unsqueeze(-1)).squeeze(-1)
    gate, up = proj.chunk(2, dim=-1)
    act = self.act_fn(gate) * up                                         # (S, I)
    out = torch.matmul(self.down_proj[flat_idx], act.unsqueeze(-1)).squeeze(-1)
    out = out * flat_w                                                   # (S, H)
    return out.reshape(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


class KvCacheWrapper(nn.Module):
    """把 AutoModelForCausalLM 包装为 (input_ids, cache_position) → logits 的
    状态模块，prefill（整段）与 decode（单 token）共用同一图。"""

    def __init__(self, model, max_seq_len: int):
        super().__init__()
        self.inner = model
        self.max_seq_len = int(max_seq_len)

        config = model.config.get_text_config(decoder=True)
        # Qwen3.5 的 mrope 频率重排用 in-place 切片赋值实现，导出后 slice_scatter
        # 过不了 coremltools 前端校验，替换为静态掩码实现（见函数文档）。
        # 另将 TextRotary 的 forward 换成纯文本等价实现（省去 expand(3,-1,-1)）：
        # 非 hybrid 路径 position_ids 由模型内部从动态 cache_position 推导，
        # 原版 expand(3,-1,-1) 会让前端把静态维也符号化并污染整图（见函数文档）。
        # 按结构特征匹配而非写死类名：Qwen3.5 全系（dense 的 Qwen3_5* 与 MoE 的
        # Qwen3_5Moe*）由同一份 modular 源码生成，类名随变体而变。
        for module in model.modules():
            cls_name = type(module).__name__
            if not cls_name.startswith("Qwen3_5"):
                continue
            # MoE 专家层（Qwen3_5MoeExperts）：换成可导出的批量前向
            if (cls_name.endswith("Experts") and hasattr(module, "gate_up_proj")
                    and hasattr(module, "down_proj")):
                module.forward = MethodType(_moe_experts_forward, module)
                continue
            if not hasattr(module, "recomposition_frequencies"):
                continue
            module.recomposition_frequencies = MethodType(
                _qwen35_rope_recomposition, module)
            if cls_name.endswith("TextRotaryEmbedding"):
                module.forward = MethodType(_qwen35_text_rope_forward, module)
        self.num_layers = int(config.num_hidden_layers)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", None)
                                or config.num_attention_heads)
        self.head_dim = int(getattr(config, "head_dim", None)
                            or config.hidden_size // config.num_attention_heads)
        # state buffer 名单（按注册顺序，与 named_buffers 一致）
        self._state_names: list[str] = []

        # 混合架构检测：layer_types 中非 full/sliding/chunked 的层视为 linear attention
        layer_types = list(getattr(config, "layer_types", None) or [])
        linear_idx = {i for i, t in enumerate(layer_types)
                      if t not in ("full_attention", "sliding_attention", "chunked_attention")}
        self._hybrid = bool(linear_idx) and len(layer_types) == self.num_layers
        if self._hybrid and self.max_seq_len < 2:
            raise ValueError("混合注意力模型（linear attention）的 max_seq_len 至少为 2")
        # Qwen3.5 TextModel 需要显式 4 行 mrope position_ids（见 forward 非 hybrid
        # 分支注释）；其他架构不接受 3D 位置，绝不能传。
        self._qwen35_mrope_positions = str(getattr(config, "model_type", "")).startswith("qwen3_5")

        param = next(model.parameters())
        # buffer 固定 fp32：coremltools 前端把 state 读取无条件 cast 成 fp32 参与图
        # 计算（converter.py 对 read_state 插入 fp32 cast），in-place 写的 data 恒为
        # fp32，updates 也必须 fp32，否则前端校验报 "updates must have the same
        # data type as data"。写回 state 时前端自动 cast 回 fp16（Core ML State
        # 仅支持 fp16），因此 buffer 精度与模型无关。
        self.buffer_dtype = torch.float32
        # 掩码精度跟随模型原精度：fp16 模型的 eager attention 里 attn_weights + mask
        # 之后还要与 value_states（fp16）做 matmul，掩码若是 fp32 会把 attn_weights
        # 提升成 fp32 导致 matmul dtype 不匹配。
        self.kv_dtype = param.dtype

        cache = StaticCache(config=config, max_cache_len=self.max_seq_len)
        # full attention 层换成槽位写入层；linear attention 层保留原生
        # LinearAttentionLayer（state 槽位稍后重绑定为 wrapper buffer）
        new_layers = []
        for i, layer in enumerate(cache.layers):
            if i in linear_idx and isinstance(layer, LinearAttentionCacheLayerMixin):
                new_layers.append(layer)
            else:
                new_layers.append(_SlotStaticLayer(max_cache_len=self.max_seq_len))
                kv_shape = (1, self.num_kv_heads, self.max_seq_len, self.head_dim)
                for suffix in ("k", "v"):
                    name = f"cache_{i}_{suffix}"
                    self.register_buffer(name, torch.zeros(kv_shape, dtype=self.buffer_dtype,
                                                           device=param.device))
                    self._state_names.append(name)
        cache.layers = new_layers
        cache.early_initialization(batch_size=1, num_heads=self.num_kv_heads,
                                   head_dim=self.head_dim, dtype=param.dtype,
                                   device=param.device)
        for i, layer in enumerate(cache.layers):
            if isinstance(layer, _SlotStaticLayer):
                layer.keys = getattr(self, f"cache_{i}_k")
                layer.values = getattr(self, f"cache_{i}_v")
        self._cache = cache

        if self._hybrid:
            self._prepare_hybrid(linear_idx, param)

    # ---------------------------------------------------------------- #
    # 混合架构（linear attention）准备
    # ---------------------------------------------------------------- #
    def _prepare_hybrid(self, linear_idx: set[int], param: torch.Tensor):
        """给 GDN 层打导出适配补丁，把 conv/recurrent state 重绑定为 wrapper
        buffer（导出后即 Core ML State）。state 形状与序列长度无关，直接取自
        模型模块属性，无需预热前向。"""
        base = self.inner.model if hasattr(self.inner, "model") else self.inner
        layers = list(getattr(base, "layers", []) or [])
        mixers = {}
        for i in sorted(linear_idx):
            mixer = getattr(layers[i], "linear_attn", None) if i < len(layers) else None
            # GDN 变体类名随模型族而变（dense: Qwen3_5GatedDeltaNet；
            # MoE: Qwen3_5MoeGatedDeltaNet），按后缀识别
            if mixer is None or not type(mixer).__name__.endswith("GatedDeltaNet"):
                raise RuntimeError(
                    f"第 {i} 层未找到 GatedDeltaNet 模块（实际: "
                    f"{type(mixer).__name__ if mixer is not None else 'None'}）")
            mixers[i] = mixer
            # 内核函数从 mixer 所属模型模块解析（dense / MoE 各一份），追踪前固定
            kernels_mod = importlib.import_module(type(mixer).__module__)
            mixer._modconv_kernels = (kernels_mod.apply_mask_to_padding_states,
                                      kernels_mod.causal_conv1d_fn,
                                      kernels_mod.torch_chunk_gated_delta_rule)
            mixer.forward = MethodType(_qwen35_gdn_forward, mixer)

        for i, mixer in mixers.items():
            kernel = int(mixer.conv_kernel_size)
            conv_dim = int(mixer.key_dim) * 2 + int(mixer.value_dim)
            rec_shape = (1, int(mixer.num_v_heads),
                         int(mixer.head_k_dim), int(mixer.head_v_dim))
            layer = self._cache.layers[i]
            for suffix, shape in (("conv", (1, conv_dim, kernel)), ("rec", rec_shape)):
                name = f"cache_{i}_{suffix}"
                self.register_buffer(name, torch.zeros(shape, dtype=self.buffer_dtype,
                                                       device=param.device))
                self._state_names.append(name)
            # state 槽位重绑定为 buffer，并标记已初始化/有历史（追踪期走既定分支，
            # 不触发懒初始化；buffer 全零 = 全新会话语义）
            layer.conv_states[0] = getattr(self, f"cache_{i}_conv")
            layer.recurrent_states[0] = getattr(self, f"cache_{i}_rec")
            layer.is_conv_states_initialized[0] = True
            layer.is_recurrent_states_initialized[0] = True
            layer.has_previous_state[0] = True
            layer.conv_kernel_size[0] = kernel

    # ---------------------------------------------------------------- #
    # 状态元信息（供 coremltools StateType 声明使用）
    # ---------------------------------------------------------------- #
    def kv_state_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """返回 [(state 名, 形状, dtype 字符串)]，顺序与 named_buffers 一致。

        Core ML State 后端仅支持 fp16，因此 state 声明固定 fp16（torch buffer
        本身可以是 fp32，前端会自动插入双向 cast）。
        """
        return [(name, tuple(getattr(self, name).shape), "float16")
                for name in self._state_names]

    def state_buffers(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name, _, _ in self.kv_state_specs()]

    def reset_state(self) -> None:
        for buf in self.state_buffers():
            buf.zero_()

    # ---------------------------------------------------------------- #
    # 前向：input_ids (1, q) + cache_position (q,) → logits (1, q, vocab)
    # ---------------------------------------------------------------- #
    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        # 每层游标重绑定为 cache_position 起点（输入派生）：_SlotStaticLayer.update
        # 用 arange(kv) + 游标定位写入槽位；不做 in-place 累加（见 _SlotStaticLayer
        # 文档），因此导出图中的槽位始终由 cache_position 推导，decode 不会写错槽。
        start = cache_position.reshape(-1)[0]
        for layer in self._cache.layers:
            if isinstance(layer, StaticLayer):
                layer.cumulative_length = start.clone()

        if not self._hybrid:
            # 静态形状因果掩码 (1, 1, q, L)：查询 i 允许看见槽位 0..cache_position[i]。
            # 传入已是 4D 的 attention_mask 时，HF create_causal_mask 会 early-exit 原样
            # 返回（masking_utils._preprocess_mask_arguments），完全绕开其动态掩码机器——
            # 该机器在导出图中产生"运行时 reps 的 tile"（全符号 bool 输出），是 ANE
            # 执行计划构建失败（-14）的根因。arange/比较的广播形状均可静态推断。
            # 掩码值用"布尔 cast 后乘 min"构造而不用 torch.where：coremltools 的 MIL
            # select 在 a/b 为标量时 type_inference 只广播 a/b、忽略 cond 的形状，输出
            # 退化为标量，运行时掩码会被整个丢掉（add 变成加 0）。
            # min 取 fp16 的 -65504 而非本 dtype 的 finfo.min：fp32 的 -3.4e38 在 fp16
            # 计算里溢出为 -inf，Core ML softmax 会产生 NaN；-65504 在两种精度下均为
            # 有限值且 exp(-65504) 下溢为 0，掩码语义不变。
            # position_ids（仅 Qwen3.5）传 (4, 1, q)：TextModel 对 3D/shape[0]==4
            # 的输入只做 slice 取 text/rope 份；若不传（内部 arange）或传 2D，其
            # 内部的 ``view(1,1,-1).expand(4, bs, -1)`` 在 q 动态时会让 coremltools
            # 前端把 expand 输出的静态 4/1 维也符号化，沿广播链污染整图（KV
            # scatter 的 data/indices batch 维不匹配断言失败）。cat 复制 4 份即
            # 等价于该 expand 的语义（text + T/H/W 四份，纯文本下全相同）。
            # 其他架构既无此缺陷也不接受 3D 位置（Llama 会把 (4,1,q) 广播进
            # rotary，使 K/V 多出一个 mrope 维），必须保持不传。
            kwargs = {}
            if self._qwen35_mrope_positions:
                pos3d = cache_position.reshape(1, 1, -1)
                kwargs["position_ids"] = torch.cat(
                    [pos3d, pos3d, pos3d, pos3d], dim=0)
            slots = torch.arange(self.max_seq_len, device=input_ids.device)
            blocked = slots[None, None, None, :] > cache_position.reshape(-1)[None, None, :, None]
            mask = blocked.to(self.kv_dtype) * torch.finfo(torch.float16).min
            out = self.inner(input_ids=input_ids, cache_position=cache_position,
                             attention_mask=mask, past_key_values=self._cache,
                             use_cache=True, **kwargs)
            return out.logits

        # ---- 混合架构：pad 到固定 max_seq_len 静态前向 ----
        # GDN 分块内核的 chunk 循环次数取决于序列长度（动态值无法导出），pad 成
        # 常量后所有 seq_len 相关分支在追踪期静态消除。KV 写入索引由
        # _SlotStaticLayer clamp 到缓冲范围内（见其文档）。
        q = input_ids.shape[1]
        dev = input_ids.device
        # 用 scatter 把真实 token/位置写进静态 (1, max_seq_len) 缓冲的前 q 个槽位
        # （其余槽位为 0），得到静态形状的 ids/positions。不能用
        # cat([input_ids, pad 切片]) 拼接：动态长度切片的输出在 MIL 前端是符号
        # 长度 (max - q)，concat 无法静态折叠 q + (max - q) = max，符号长度沿
        # 整个计算图泄漏到 rope 与 K/V，scatter 写 state 时与静态缓冲形状断言
        # 冲突。也不能用 torch.zeros((1, pad_len)) 直接构造 pad：dynamo 对工厂
        # 函数（zeros/full/empty）的符号尺寸有"非零"假设，会生成 pad_len >= 1
        # 守卫，使 q = max_seq_len 的完整 prefill 无法导出。scatter 写入的索引
        # 长度沿 axis 可与缓冲不同，与非 hybrid 路径的 KV state 写入同构。
        slot = torch.arange(q, device=dev).unsqueeze(0)  # (1, q) 动态长度索引
        ids = torch.zeros((1, self.max_seq_len), dtype=input_ids.dtype,
                          device=dev).scatter(1, slot, input_ids)
        # pad 槽位位置补 0：pad query 行的输出会被切掉；pad K/V 写入的槽位在
        # 真实 query 读到之前必被真实 token 覆写（生成按位置递增推进）。
        positions = torch.zeros((1, self.max_seq_len), dtype=torch.long,
                                device=dev).scatter(
            1, slot, cache_position.reshape(1, -1).long())
        # 掩码 key 轴必须是 KV 缓冲的"槽位索引"（buffer 按位置索引槽位，槽位 j 存
        # 位置 j 的 K/V）：查询 i（位置 positions[i]）允许看见槽位 0..positions[i]。
        # 不能用 positions 做 key 轴——decode 时 positions=[5..20] 与槽位号错开，
        # 会让真实 query 只看到槽位 0；prefill（start=0）两者恰好重合所以不易暴露。
        # pad query 行的输出会被 logits[:, :q] 切掉，其掩码行取值无影响。
        slots = torch.arange(self.max_seq_len, device=dev)
        blocked = slots[None, None, None, :] > positions.reshape(-1)[None, None, :, None]
        mask = blocked.to(self.kv_dtype) * torch.finfo(torch.float16).min
        # Qwen3.5 的 TextModel 忽略 cache_position、用 get_seq_length() 推位置
        #（混合 cache 下不可靠），必须显式传 position_ids；llama 系模型收到
        # 一致的 position_ids 也无副作用。
        out = self.inner(input_ids=ids, position_ids=positions, cache_position=cache_position,
                         attention_mask=mask, past_key_values=self._cache, use_cache=True)
        # 输出切回真实长度。不能直接 logits[:, :q]：动态 end 的 slice 在 end 可
        # 等于维度上限时（q = max_seq_len 合法输入）会生成 "end != size" 守卫，
        # 违反 1..max 动态范围声明，导出报错。先 cat 一列 dummy 使维度上限变为
        # max+1，切片 end 恒小于上限，守卫静态成立、不会加入图约束。
        logits = out.logits
        dummy = torch.zeros((1, 1, logits.shape[-1]), dtype=logits.dtype, device=dev)
        return torch.cat([logits, dummy], dim=1)[:, :q]
