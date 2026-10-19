# 模型转换工具集（tools/）

本目录保存会话期间做过的**模型转换（最终验证可用版）** 与**模型测试 / 修复 / 验证**相关代码，便于后续复用。

- 转换工具（Python）：`convert_jina_mlx_to_int4.py` / `convert_onnx_reranker_to_mlx.py`（ONNX reranker → MLX int4/int8）/ `convert_mlx_rekey_quantize.py`
- 一键校验：`verify_all_deliverables.py`
- 模型加载 / 测试 / 验证运行时（Swift，归档）：`model-runtime/`

所有工具均已在真实模型上跑通并验证。依赖：`mlx`、`onnx`、`onnxruntime`、`torch`、`transformers`、`safetensors`、`tokenizers`、`numpy`；Swift 运行时另需 `mlx-swift-lm`、`OnnxRuntimeBindings`、`tokenizers` 等（见 `model-runtime/Package.swift`）。

---

## 1. `convert_jina_mlx_to_int4.py` — jina v5 风格 MLX → MLX int4

**场景**：自定义 MLX Python 实现（Qwen3 backbone + 每任务 LoRA adapter，如
`jina-embeddings-v5-text-small-mlx`）转 int4。

**要点 / 坑**（已验证）：
- 把 6 类投影 `nn.Linear`（q/k/v/o_proj、gate/up/down_proj）改为
  `nn.QuantizedLinear(in, out, False, 64, 4, "affine")`；embedding 保留 fp16（保检索质量）。
- `mx.load` 加载量化 `safetensors` 返回的是**裸 uint32**，重载时模型必须用 `QuantizedLinear`。
- LoRA 合并（`_apply_adapter` 的 `current + scale*(B@A)`）对量化权重会崩：必须
  **反量化 → 合并 → 重新量化**，并**直接给 `ql.weight/scales/biases` 赋值**
  （`model.load_weights` 一次性传入数百个量化键会把它误存成 dict，导致前向崩）。
- 保存：`dict(model.parameters())` 含 `config`（字典标量）与模块标量属性
  （`scale/head_dim` 等），会让 `save_safetensors` 报 `bad_cast`；用递归 `collect_arrays`
  只取 `mx.array` 叶子并兼容 `list`（layers）。
- `JinaEmbeddingModel.__init__` 需容错忽略 `quantization` 等额外 config 字段。

**用法**：
```bash
python3 convert_jina_mlx_to_int4.py --src SRC_MLX_DIR --dst DST_DIR \
    [--bits 4 --group-size 64] [--no-verify] [--verify-only]
```
转换会：改 model.py/utils.py（int4 版）→ 量化权重 → 写 safetensors → 拷 tokenizer/adapters/config。
`--verify` 加载 int4 目录、切换 4 个任务、与 fp16 基线比余弦（典型 0.95–0.97）。

---

## 2. `convert_onnx_reranker_to_mlx.py` — ONNX reranker → MLX int4/int8

**场景**：xlm-roberta/bert 类 reranker 的 ONNX 导出版（如 `bge-reranker-base-ONNX`）
转 MLX 量化版（int4 或 int8，MLXEmbedders roberta 命名），供 onnx-runtime-tester 的 MLXModelRuntime 加载。
`--bits 4`（默认）或 `--bits 8`。

**要点 / 坑**（已验证，fp16 无损对比 onnxruntime logits 误差 2e-4）：
- 编码器 72 个 2D Linear 权重名是通用 `onnx::MatMul_*`，但**节点名是语义的**（如
  `/roberta/encoder/layer.0/attention/self/query/MatMul`），由节点名解析层号+角色，无需按 bias 追踪。
- **朝向**（最易错）：编码器 MatMul 权重无 Transpose 前置 → 存 `[in,out]` → **转置为 `[out,in]`**；
  分类头 Gemm `transB=1` → 已是 `[out,in]` 不转置；嵌入表 Gather 消费 → 不转置。
- 量化陪衬键是**兄弟键**：`{prefix}.weight` / `{prefix}.scales` / `{prefix}.biases`
  （切勿写成 `{prefix}.weight.scales`——那会让 Swift 加载器 `verify:[.all]` 找不到陪衬键而崩溃）。
  bits=4/8 的 packed 权重均为 uint32，scales/biases 为 float16。
- 初始化器名用**点号**（`roberta.encoder.layer.0...bias`），正则须兼容 `[\./]`。
- config.json = 源 config（含 `model_type=xlm-roberta`）+ `quantization` + `num_labels:1`。
  roberta 有 `embeddings.LayerNorm`，会被捕获并写成 `embeddings.norm`（真实值，非占位）。

**用法**：
```bash
python3 convert_onnx_reranker_to_mlx.py --src SRC_ONNX_DIR --dst DST_DIR \
    [--onnx PATH] [--bits 4|8 --group-size 64] [--no-verify] [--verify-only]
```
`--verify`：onnxruntime 跑样本得 logits，再把转换权重反量化 `load_state_dict` 进
HF `XLMRobertaForSequenceClassification` 跑同一输入对比（int4 下绝对差 ~1.0，排序无损；fp16 下 2e-4）。

---

## 3. `convert_mlx_rekey_quantize.py` — 通用 MLX 重命名 + 量化

**场景**：把「HF roberta/bert 命名」的 MLX 权重（如 `bge-reranker-v2-m3-mlx-fp16`，
键名 `roberta.encoder.layer.N...`）重命名为 MLXEmbedders roberta 命名并可选量化。
已**精确复现** `bge-reranker-v2-m3-mlx-fp16-fixed` / `-int8-fixed` / `-int4`（键集合 100% 一致）。

**重命名映射**：剥离 `roberta.`；`encoder.layer.→encoder.layers.`；
`attention.self.{query,key,value}.→attention.{q,k,v}_proj.`；
`attention.output.dense.→attention.out_proj.`；`attention.output.LayerNorm.→ln1.`；
`output.LayerNorm.→ln2.`；`intermediate.dense.→linear1.`；`output.dense.→linear2.`；
`embeddings.LayerNorm.→embeddings.norm.`；丢弃 `pooler.*`。
（注意替换顺序：更具体的子串先替换，避免 `output.LayerNorm` 误伤 `attention.output.LayerNorm`。）

**用法**：
```bash
python3 convert_mlx_rekey_quantize.py --src SRC_MLX_DIR --dst DST_DIR \
    [--bits 4 --group-size 64] [--no-quantize] [--ref REF_DIR]
```
`--no-quantize` 仅重命名（fp16-fixed）；`--ref` 指定参考交付物目录做键名对比验证。
量化时 2D 线性/嵌入权重量化、1D 偏置/LayerNorm 保留 fp16；若源已是量化(uint32)则仅重命名。

---

## 通用陷阱小结
- `save_safetensors` 只接受 `mx.array` 叶子，非数组标量（dict/int/float）会 `bad_cast`。
- int4 权重落盘为 `uint32` 打包；推理端必须用 `QuantizedLinear`，且合并 LoRA 要反量化/重量化后直接赋值。
- ONNX→MLX 的**权重朝向**是头号坑：永远用 HF `load_state_dict` + onnxruntime 对比 logits 验证，
  不要凭直觉。
- 命名映射务必用真实交付物的键集合做 `missing/extra` diff 验证。

---

## 4. `verify_all_deliverables.py` — 一键校验全部交付物

从源模型重新转换到临时目录，对各交付物做端到端校验（ONNX 工具对比 onnxruntime 原 logits；
重命名工具做键集合 `missing/extra` diff；jina 工具与 fp16 基线比余弦）。

```bash
python3 verify_all_deliverables.py --base "/Volumes/SSD/embedding&reranker" [--tmp TMP_DIR]
```

---

## 5. `model-runtime/` — 模型加载 / 测试 / 验证运行时（Swift，归档）

转换工具产出的 MLX 模型目录，由这里的 Swift 运行时真正加载运行做测速与效果验证。
`MLXModelRuntime.swift` 即为让 MLX 版 reranker / embedding 模型能被加载运行所做的**修复**
（基于 MLXEmbedders / MLXRerankers 的 `EmbedderModelFactory` / `RerankerModelFactory`）；
`TestCorpus.swift` 是验证所用的内置测试语料（含「西湖」等排序样例）。
详见 `model-runtime/README.md`。

---

## 6. `convert_onnx_fp16_to_int8.py` — ONNX fp16 embedding → ONNX int8（weight-only）

**场景**：把 fp16 的 ONNX embedding 模型（如 `nomic-embed-text-v1.5-onnx-fp16`）权重量化到 int8，
供 onnx-runtime-tester 的 ONNX/CoreML 后端加载。**已验证产出 `nomic-embed-text-v1.5-onnx-int8`**。

**要点 / 坑（已验证）**：
- 模型只输出 `last_hidden_state`（token 级），embedding 需外部做「attention_mask 平均池化 + L2 归一化」。
- onnxruntime 的 `quantize_dynamic` 对 fp16 源模型**不会真正量化 fp16 权重**，且动态量化会把
  **激活也量化**（embedding 对激活量化敏感，余弦只到 0.985）。故改为**手写 weight-only int8**：
  先把 fp16→fp32（`fp16_to_fp32` 必须同时改 Cast 的 `to` 属性，否则破坏图内 Cast 节点），再对
  MatMul/Gemm 权重按输出维 per-channel 量化为 int8，插入 `DequantizeLinear` 还原 fp32 再计算，
  激活全程 fp32。
- 权重朝向：ONNX 线性层权重存为 `[in, out]`（如 `(768, 2304)`），量化轴取输出维（MatMul→axis=1，
  Gemm transB=1→axis=0）。

**用法**：
```bash
python3 convert_onnx_fp16_to_int8.py --src SRC_ONNX_DIR --dst DST_DIR \
    [--no-per-channel] [--no-verify] [--verify-only] [--max-length 512]
```
`--verify`：onnxruntime CPU 跑 fp16 源与 int8，比较 embedding 余弦（weight-only 典型 >0.997）
并做检索式相关性检查（相关文档相似度应高于无关文档）。

---

## 7. `convert_onnx_jina_reranker_v2_to_mlx_int4.py` — jina-reranker-v2 ONNX → MLX int4

**场景**：把 `jina-reranker-v2-base-multilingual-onnx`（xlm-roberta 系，**fp16 ONNX**，含 `onnx/model_fp16.onnx`）
转换为 MLX int4（affine / group_size=64），供 MLXEmbedders `XLMRobertaForSequenceClassification` 加载。
**已验证产出 `jina-reranker-v2-base-multilingual-mlx-int4`**（156MB，源 fp16 约 556MB）。

**要点 / 坑（已验证）**：
- 与 §2 的 bge 工具架构相同（xlm-roberta），但命名不同：
  - 节点命名 `mixer/Wqkv`、`mixer/out_proj`、`mlp/fc1`、`mlp/fc2`；**QKV 融合**（`Wqkv` 形状 `[in,3H]`），需拆成
    `query/key/value_proj`。
  - ONNX 仅 `input_ids`/`attention_mask` 两个输入（无 `token_type_ids`）；无 embeddings.LayerNorm。
  - 归一化命名 `norm1`/`norm2` → MLX `ln1`/`ln2`。
- **关键坑**：ONNX 的 MatMul **权重**初始化器名是 `onnx::MatMul_NNNN`（不是 `roberta.*`），故必须按**节点名**
  （`/roberta/encoder/layers.N/mixer/Wqkv/MatMul`）定位角色，再从 `n.input` 取权重 —— 按初始化器名遍历会漏掉全部权重。
  而 bias/norm/embedding 初始化器保留 `roberta.*` 名，按名匹配。
- 权重朝向（对照 HF `model.safetensors` 确认）：编码器 MatMul 无 Transpose 前置 → `[in,out]` 转置为 `[out,in]`，
  再按行拆 q/k/v；分类头 Gemm `transB=1` → 权重已是 `[out,in]`，不转置。
- **致命坑（曾导致加载失败）**：量化权重的 `scales`/`biases` 必须是**兄弟键**——去掉 `.weight` 后追加
  `.scales`/`.biases`，即 `query_proj.weight` → `query_proj.scales` / `query_proj.biases`（参考 bge 系 MLX 模型的
  实际键名）。**绝不能**写成 `query_proj.weight.scales`。写错时 Swift 加载器（`verify:[.all]` 的 `allModelKeysSet`）
  找不到陪衬键直接抛错加载失败；而 Python 自检若用同一错误约定会漏检，须警惕。存储用
  `prefix.rsplit(".weight",1)[0]+".scales"` / `".biases"`。
- **验证**：以 HF 模型为基准做逐权重对比（fp16 无损版最大误差 0.00000，证明朝向/命名/拆分完全正确；
  int4 版平均余弦 0.998、最差单权重余弦 0.988，余量为纯量化噪声）。另跑 onnxruntime fp16 基线确认 logits 有限。

**用法**：
```bash
python3 convert_onnx_jina_reranker_v2_to_mlx_int4.py --src SRC_ONNX_DIR --dst DST_DIR \
    [--onnx PATH] [--bits 4] [--group-size 64] [--no-quantize] [--verify-only] [--max-length 512]
```
`--no-quantize`：仅转 fp16（无损，用于隔离验证朝向）；`--verify-only`：对已有 DST 重新验证。

---

## 8. `convert_onnx_jina_reranker_v2_to_mlx_int8.py` — jina-reranker-v2 ONNX → MLX int8

**场景**：同一源 `jina-reranker-v2-base-multilingual-onnx` 的 **MLX int8** 版（affine / group_size=64）。
与 §7 的唯一区别：默认 `bits=8`，量化更精细。**已验证产出 `jina-reranker-v2-base-multilingual-mlx-int8`**（296MB，
介于 fp16 556MB 与 int4 156MB 之间）。

- 架构映射 / 朝向 / 融合 QKV 拆分 / 验证逻辑与 §7 完全一致（int8 仅改变量化比特）。
- **验证质量（int8 远优于 int4）**：逐权重对比 HF 基准，最大绝对误差 **0.01024**、最小余弦 **0.999632**、
  平均余弦 **0.999989**（int4 版为 0.988 / 0.998）；onnxruntime fp16 基线 logits 有限。

- **加载失败修复（已落到两个工具）**：jina 源 `config.json` 只有 `auto_map` 没有 `model_type`，
  Swift/MLXEmbedders 的 `RerankerDescriptor`/`BertConfiguration` 用 `try decode(model_type)` 且
  `loadWeights(verify:[.all])` 会强制所有模型参数齐备 —— 故必须：(1) 写出时显式补 `model_type="xlm-roberta"`；
  (2) 提供 `embeddings.norm.weight/bias`（ones/zeros）占位。后者是因为 jina 参考实现 embeddings 无 LayerNorm，
  但 MLXEmbedders 的 `BertEmbedding` 总会创建并应用 `embeddings.norm`，缺失该权重会让 `loadWeights` 抛
  `updateMissing`。ones/zeros 的归一化经第一层 `ln1` 抵消，不会破坏 reranker。`bge-*` 交付物自带
  `model_type` 且含 embeddings.LayerNorm，故不受影响。

**用法**：
```bash
python3 convert_onnx_jina_reranker_v2_to_mlx_int8.py --src SRC_ONNX_DIR --dst DST_DIR \
    [--onnx PATH] [--bits 8] [--group-size 64] [--no-quantize] [--verify-only] [--max-length 512]
```

