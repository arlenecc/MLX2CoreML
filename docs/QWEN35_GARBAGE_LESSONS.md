# Qwen3.5 MLX→HF 转换乱码：根因与标准处理方式（经验证）

> **这篇的目的**：下次改 modconv 的 MLX→HF 加载/归一化路径，或再遇"转换成功但输出
> 乱码"时，先看这篇，**不要从头排查**。第 0 节是速查卡；第 3 节是验证过的代码级
> 处理细节；第 4 节是验证命令与实测判据。照做即可。

---

## 0. 30 秒速查卡

| | |
|---|---|
| **症状** | 转换"成功"、权重零 MISSING、shape 全对，加载后输出乱码 |
| **根因** | MLX convert 产物的 RMSNorm 权重已预折叠 `1+gain`；HF 的 `Qwen3_5RMSNorm` 又加一次 `1.0+weight` → **双重 +1（`2+gain`）**，归一化尺度翻倍 |
| **判定信号** | `transposed` 含 conv1d（MLX 布局）⟺ convert 消过毒 ⟺ norm 存的是 `1+gain` |
| **修复位置** | `modconv/real_backend.py`：`_needs_norm_unfolding()`（L500）+ `_write_text_checkpoint(..., unfold_norms, ...)`（L516，还原逻辑 L548-551） |
| **绝对不动** | GDN 内部 `linear_attn.norm`（`Qwen3_5RMSNormGated`，HF 侧不加 +1，两侧约定相同） |
| **验证** | `python3 -m pytest tests/ -q` → `PYTHONPATH=. python3 examples/swift-gui/_verify_e2e.py` |
| **通过判据** | HF generate 连贯；wrapper vs HF logits cos-sim > 0.99、top-5 逐位一致（实测 1.000002） |

---

## 1. 根因：完整逻辑链

### 1.1 三方约定（全部事实，直接引用，不必再翻源码）

| 方 | 行为 |
|---|---|
| **mlx-lm** `mlx_lm/models/qwen3_5.py` `sanitize()` | convert 时对 norm 权重 `+1.0` 折叠 + conv1d `moveaxis(2,1)`；再次加载时以 `has_unsanitized_conv1d`（conv1d 是否 PyTorch 布局）为信号防重复叠加 |
| **transformers** `modeling_qwen3_5.py` `Qwen3_5RMSNorm.forward` | 零中心约定：`output * (1.0 + weight)`（weight 初值 0） |
| **checkpoint 实际存储**（mlx-community/Qwen3.5-0.8B-bf16） | conv1d 为 MLX 布局 `(6144,4,1)`；norm 权重均值 ≈1.24（即 `1+gain`，不是原始 gain） |

mlx-lm 的 sanitize 折叠名单（权威清单，五个后缀，一个都不能少也不能多）：

```python
norm_keys = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
```

### 1.2 推理链

```
checkpoint 的 conv1d 是 MLX 布局 (out, kernel, in)
  ⟺ 它是 mlx-lm convert 的产物（convert 时 sanitize 已消过毒）
  ⟺ norm 权重存的是 1+gain（折叠值）
  ⟹ 用 HF 加载时，Qwen3_5RMSNorm 会再算一次 1.0+weight → 2+gain → 尺度翻倍 → 乱码
  ⟹ 写回 HF 布局时必须减 1.0，把折叠还原成原始 gain
```

**为什么这个信号可靠**：它与 mlx-lm 自己的防重叠加机制是同一个信号（conv1d 布局），
只是方向相反——mlx-lm 看到 PyTorch 布局才 +1.0，modconv 看到 MLX 布局才 -1.0。
原始 HF checkpoint（conv1d 本就是 `(out,in,kernel)`）信号为假，不做任何减法。

**为什么 GDN 的 norm 不动**：`linear_attn.norm` 是 `Qwen3_5RMSNormGated`，
transformers 侧前向本就不加 +1，mlx-lm 侧也不加——两侧约定本来就一致，动它才会错。

**为什么最阴险**：不加这次修正时，不报错、shape 全对、320/320 权重"加载成功"，
只有输出是乱码。所有基于"加载是否成功"的检查都会漏掉它。

---

## 2. 已排除假设（都有数据，不要重走）

| # | 假设 | 结论 | 证据 |
|---|------|------|------|
| 1 | 权重损坏 | 排除 | embed std=0.0197（训练态）；FA 层权重与 checkpoint maxdiff=0.0 |
| 2 | conv1d 转置错误 | 排除 | MLX/PyTorch 卷积数值等价，diff 1.2e-7 |
| 3 | 前缀剥离失效 | 排除 | 320/320 权重无 MISSING |
| 4 | RoPE 配置缺失 | 排除 | inv_freq 与 rope_theta=1e7 完全匹配 |
| 5 | GDN 数学错误 | 排除 | 与 CoreML-LLM 参考等价；**GDN 置零仍乱码** |
| 6 | embedding 不一致 | 排除 | 与 mlx-lm cos-sim 1.0 |
| 7 | FA 层权重加载错误 | 排除 | layer3 各权重 maxdiff=0.0 |
| 8 | config 维度错配 | 排除 | config 与权重形状一致 |
| 9 | attention q/gate 拆分约定不同 | 排除 | mlx-lm `qwen3_next.py` 与 transformers `Qwen3_5Attention` 逐行一致（q_proj 拆 query+gate、sigmoid 门控、scale=head_dim^-0.5） |

三个关键隔离实验（复现成本低、信息量大，是定位的胜负手）：

1. **mlx-lm 原生加载同 checkpoint 作 oracle**：产出完全连贯文本 → 权重完好，
   问题在 HF 加载路径而非权重；
2. **full-attn-only（GDN 输出置零）仍乱码**（cos-sim 甚至 -0.27）→ 问题在公共路径
   （FA/RMSNorm/MLP），与 GDN 无关；
3. **单 token、position 0（RoPE 为单位阵）仍乱码** → 与 RoPE/递归/cache 无关，
   只剩静态计算（norm/投影/MLP）。

最后命中实验：patch `Qwen3_5RMSNorm.forward` 去掉 `1.0+`，HF top5 立刻变为
`Paris, the, located, :, \n`，与 mlx-lm 完全一致，cos-sim 0.99997，生成连贯。

---

## 3. 标准处理方式（代码级细节，已验证）

### 3.1 常量（`modconv/real_backend.py` L148-155）

```python
_FOLDED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
_FOLDED_NORM_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
```

后缀名单与 mlx-lm sanitize 逐条一致；`model_type` 白名单防止误伤其他架构
（其他模型即使有 MLX conv1d 也不猜——不共用零中心 RMSNorm 约定）。

### 3.2 判定函数（L500-513）

```python
def _needs_norm_unfolding(config, transposed: set) -> bool:
    if not any("conv1d" in name for name in transposed):
        return False
    model_type = str(getattr(_text_config(config), "model_type", "") or "")
    return model_type in _FOLDED_NORM_MODEL_TYPES
```

两个条件同时满足才还原：**conv1d 需要转置**（MLX 布局信号）+ **qwen3_5 系**
（`_text_config` 自动解包多模态配置的 `text_config`，两种 model_type 都认）。

### 3.3 还原逻辑（`_write_text_checkpoint` L548-551）

```python
if unfold_norms and key.endswith(_FOLDED_NORM_SUFFIXES):
    # 折叠值 1+gain → 原始 gain；经 fp32 中间量计算，避免
    # bf16 存储下 (1.2x - 1.0) 的二次舍入误差
    tensor = (tensor.float() - 1.0).to(tensor.dtype)
```

细节要点：

- 与 conv1d 转置（L546-547）在同一个循环内顺带完成，无额外 I/O；
- `str.endswith(tuple)` 精确匹配五类后缀，`linear_attn.norm.weight` 不在列；
- fp32 中间量：folded 值 ≈1.24 存 bf16 时直接减会丢有效位；
- `.to(tensor.dtype)` 保持原存储精度，不改变加载时的 dtype 契约。

### 3.4 调用方（L1036 起）

```python
unfold_norms = _needs_norm_unfolding(config, transposed)
...
_write_text_checkpoint(torch_source, norm_dir, prefix, transposed,
                       unfold_norms, hooks.on_log)
```

转换日志会明示：`checkpoint 为 文本权重带 'language_model.' 前缀（多模态布局）、
18 个 conv1d 权重为 MLX 布局（out, kernel, in）、RMSNorm 权重已预折叠 1+gain，将还原为
HF 的零中心约定，正在另存为 HF 布局…`——**下次看到这行日志，就知道三条归一化都在生效**。

### 3.5 决策表：遇到各类 checkpoint 该做什么

| checkpoint 类型 | conv1d 布局 | `transposed` | `unfold_norms` | 动作 |
|---|---|---|---|---|
| mlx-lm convert 产物（本文案例） | MLX `(out,k,in)` | 非空 | **True** | 转置 conv1d + norm 减 1.0 |
| 原始 HF checkpoint（含多模态前缀） | PyTorch `(out,in,k)` | 空 | False | 只剥前缀，norm 不动 |
| 量化 MLX 模型 | 反量化后仍 MLX | 非空 | True | 先反量化再走上一行（同一信号） |
| 非 qwen3_5 架构 | — | — | False | 不猜，保持原行为 |

MLX→HF 布局差异共三类，全部在 `_write_text_checkpoint` 归一化：**前缀剥离、
conv1d 轴序、RMSNorm 折叠**。新增第四类差异时，按同一模式加常量 + 判定 + 写入即可。

---

## 4. 验证方式（命令 + 判据 + 实测数字）

每次动这条路径后**必跑**：

```bash
# 1) 单元测试（含 norm 折叠还原的夹具，0.9s）
/Users/arlenecc/.pyenv/versions/3.13.16/bin/python3 -m pytest tests/ -q
#   关键用例：test_needs_norm_unfolding_only_for_mlx_sanitized_qwen35
#             test_write_text_checkpoint_unfolds_qwen35_norms
#   实测：144 passed

# 2) 端到端数值对齐（需挂载 /Volumes/500G 源 checkpoint）
cd /Users/arlenecc/codebase/modconv
PYTHONPATH=. /Users/arlenecc/.pyenv/versions/3.13.16/bin/python3 \
    examples/swift-gui/_verify_e2e.py
```

`_verify_e2e.py` 的通过判据（全部满足才算修好）：

1. 经 modconv 归一化路径加载的模型，原始 HF `generate` 输出**连贯文本**；
2. wrapper 单步循环 logits 与 HF prefill logits **cos-sim > 0.99**、**top-5 逐位一致**；
3. top-1 为语义正确 token。

修复后的实测输出（prompt `"The capital of France is"`，5 token）：

```
detected prefix: 'language_model.' transposed: 18 unfold_norms: True
归一化权重分片 model.safetensors（320 个张量，转置 18 个 conv1d 权重，还原 RMSNorm 权重折叠）…
HF   top5: [(' Paris', 14.856), (' the', 14.595), (' located', 14.268), (':', 14.073), ('\n', 14.015)]
wrap top5: [(' Paris', 14.856), (' the', 14.595), (' located', 14.268), (':', 14.073), ('\n', 14.015)]
cos-sim(wrapper, HF) = 1.000002   max|diff| = 3.72e-05
PASS: wrapper 与 HF 完全对齐
```

另跑 `_oracle_fullpipe.py` 冒烟（对照组同时验证）：HF `generate` →
`"The capital of France is Paris."`；wrapper → `" Paris.\n..."` 连贯开头
（后续重复是该 0.8B 模型贪心解码固有行为，HF 对照组同样重复，非 bug）。

辅助诊断脚本（`examples/swift-gui/_oracle_*.py`，一次性实验，留档）：

- `_oracle_fullpipe.py`：完整管线 + 三条 prompt 生成对照（含 raw HF 对照组）；
- `_oracle_diff.py` / `_oracle_fix.py`：分量隔离对照（置零 GDN、full-attn-only）；
- `_oracle_conv.py`：conv1d 转置数值等价验证。

---

## 5. 排查套路（下次遇到新的"转换成功但乱码"）

1. **找 oracle**：mlx-lm 原生加载同一 checkpoint，能出连贯文本就证明权重完好，
   问题在加载/前向路径，不在权重本身；
2. **隔离公共路径**：置零可疑模块、单 token、position 0，把 RoPE/递归/cache 变量
   逐个消掉，把范围压缩到静态计算（norm/投影/MLP）；
3. **比实现不比 shape**：逐行对照 mlx-lm 与 transformers 的同名模块（norm 公式、
   缩放、门控、split 顺序），shape 全对不代表语义对；
4. **警惕静默约定差**：MLX 生态 convert 会在权重里**预折叠计算**（+1.0、轴序），
   HF 侧再算一次就是双重计算。以 mlx-lm 的 `sanitize` 为权威清单，逐个核对。

---

## 6. 权威参照（确切路径）

| 内容 | 路径 |
|---|---|
| mlx-lm Qwen3.5 sanitize（折叠名单+信号） | `~/.pyenv/versions/3.13.16/lib/python3.13/site-packages/mlx_lm/models/qwen3_5.py`（`sanitize`，L346-368） |
| mlx-lm Qwen3-Next sanitize（同约定） | 同目录 `mlx_lm/models/qwen3_next.py`（L480-516） |
| mlx-lm Attention（q/gate 拆分） | 同目录 `mlx_lm/models/qwen3_next.py`（`Qwen3NextAttention`，L74-151） |
| HF Qwen3.5 建模 | 同 site-packages `transformers/models/qwen3_5/modeling_qwen3_5.py`（`Qwen3_5RMSNorm` L831、`Qwen3_5Attention` L739、`Qwen3_5RMSNormGated` L208） |
| CoreML-LLM 转换参考 | `CoreML-LLM/conversion/qwen3_5_decode_layer_ane.py` |
| CoreML-LLM 可运行 decode 栈 | `CoreML-LLM/conversion/test_qwen3_5_full_decode_trace.py` |
| modconv 加载/归一化 | `modconv/real_backend.py`（L148-155、L500-513、L516-566、L1036） |
| modconv 回归测试 | `tests/test_backend.py`（`test_write_text_checkpoint_unfolds_qwen35_norms` 等） |
| modconv 端到端验证 | `examples/swift-gui/_verify_e2e.py` |
| 设计文档（布局三类差异） | `docs/design.md` §3.1 |
