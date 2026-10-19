# MLX2CoreML — 多目标模型转换工具

把 MLX/HF 格式的模型目录（`config.json` + `*.safetensors`）转换为多种目标格式：
**Core ML `.mlpackage`**（iOS / macOS / visionOS 直接推理）、**ONNX**、**MLX（Apple）**、
**GGUF（llama.cpp）**。除 Core ML 外，也支持 **HF→ONNX**、**ONNX→MLX**、**HF→MLX**、**HF→GGUF**
四条链路，覆盖 LLM（causal）与 encoder 类（embedding / reranker）模型。

> Python 包名/命令行仍为 `modconv`，macOS 应用与仓库名为 **MLX2CoreML**。

## 功能特性

- **Stateful KV cache 模型**：KV cache 以 Core ML State 存储（iOS 18+ 目标），
  prefill 与 decode 共用同一模型，Swift 端无需重算前缀，自回归生成每步只前向 1 个 token。
- **动态序列长度**：输入 `input_ids [1, q]` + `cache_position [q]`，`q ∈ 1..max_seq_len`，
  一次转换覆盖任意上下文长度，无需按固定长度多次导出。
- **最大上下文长度可配**：GUI 下拉框指定（默认「自动（≤512）」）。留空/选「自动」时
  按 `min(模型位置编码上限, 512)` 选取（`SEQ_LEN_CAP=512` 是 ANE 能稳定构建执行计划的上限）；
  也可选预设档位或输入自定义整数（支持 `k`/`m` 后缀，如 `8k`）。显式大于 512 的值会被
  采纳，但可能生成 GB 级的 KV cache，且在 ANE 上大概率 `-14`（执行计划构建失败），
  届时工具会自动逐级降长度直到可运行——因此默认走「自动」最稳妥。超过模型位置编码上限时自动截断。
- **4-bit 权重压缩**：默认对权重做 4-bit palettization（kmeans LUT），显著减小体积、加速端侧推理；
  压缩失败直接报错（不静默降级为全精度，避免产出与预期不符的模型）。
- **MLX 量化模型支持**：含 `quantization` 的 MLX 模型先用 mlx-lm 反量化再转换。
- **Swift 调用示例**：转换成功后在输出目录生成 `<模型名>-swift-example.swift`，
  含 `makeState()` + 自回归生成循环，可直接拷入 Xcode 使用。示例会**自动编译**
  传入的 `.mlpackage`（Core ML 只能加载编译后的模型）。
- **混合后端兜底**：依赖缺失或架构不支持时自动回退到打包后端（权重 1:1 拷贝），工具始终可用；
  但真实转换**过程**中失败会直接报错，不回退到打包产物（打包产物不是可推理的 Core ML 模型）。
- **全程后台线程**：目录扫描与转换均在 QThread 中执行，大模型转换时界面不卡顿；
  支持随时取消（扫描也可取消，关闭窗口时先取消再等待线程退出）；
  切换到新源目录时会取消上一次已作废的扫描，不浪费 CPU/IO。
- **输入健壮性**：源目录输入框防抖扫描；转换前校验输出目录不位于源目录内；
  单文件读取失败跳过而非让整次转换失败；扫描后立即提示模型架构是否走真实
  转换（不支持/无法识别时提前告知，而不是等到转换阶段才发现走了兜底路径）。

## 分发包（macOS .app / .dmg）

不想折腾 Python 环境的话，用分发包：

1. 从 [Releases](https://github.com/arlenecc/MLX2CoreML/releases) 或本地
   `packaging/build_macos.py` 产物拿到 `MLX2CoreML-<版本>.dmg`，双击挂载；
2. 把 **MLX2CoreML.app** 拖进「应用程序」；
3. 双击镜像里的 **安装依赖.command**（仅首次）：它在
   `~/Library/Application Support/MLX2CoreML/venv` 创建独立虚拟环境并安装依赖，
   会询问是否一并装上模型转换依赖（torch / transformers / coremltools，约 2–4 GB）；
4. 双击 **MLX2CoreML** 启动。未签名应用若被 Gatekeeper 拦截，
   在 Finder 里按住 Control 点击 App →「打开」。

### 为什么是「启动器形态」的 .app

转换功能依赖 torch / transformers / coremltools，三者合计 2–4 GB，且与 Python 版本
强绑定。把它们全部打进 `.app` 会得到一个数 GB 的产物，且换 Python 版本即失效。
因此 `.app` **内嵌图标与源码**（`Contents/Resources/app`），运行时复用宿主 Python：
优先用一键安装脚本创建的独立虚拟环境，其次用 PATH 上已装好 PyQt6 的 `python3`，
都不可用时弹出图形提示并引导安装。日志写入 `~/Library/Logs/MLX2CoreML.log`。

用环境变量可指定解释器（须已装 PyQt6）：

```bash
MLX2COREML_PYTHON=/path/to/python3 open -a MLX2CoreML
```

### 自行构建分发包

```bash
python3 packaging/make_icon.py       # 生成 packaging/AppIcon.icns（需 Pillow + numpy）
python3 packaging/build_macos.py     # 产出 dist/MLX2CoreML.app 与 dist/MLX2CoreML-<版本>.dmg
```

`build_macos.py` 会在图标缺失时自动调用 `make_icon.py`；`--skip-dmg` 可只出 `.app`。
图标设计为蓝紫渐变圆角底 + 白色转换箭头，脚本可复现（见 `packaging/make_icon.py`）。

## 环境要求

- macOS（Apple Silicon 推荐；MLX 反量化仅支持 Apple Silicon）
- Python ≥ 3.10（**推荐 3.13**：coremltools 官方轮子在 3.13 上自带原生序列化库，
  可正常产出 `.mlpackage`；更低版本可能触发降级）
- GUI：PyQt6；真实转换：torch / transformers / coremltools / safetensors / numpy
- 新增目标（按需安装）：ONNX（`onnx` / `onnxruntime`）、MLX（`mlx-lm` / `mlx-embeddings` / `mlx-vlm`）、
  GGUF（`gguf`，并需 `transformers` 导出分词器）

## 安装

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,convert]"
```

## 使用

```bash
modconv          # 或 python -m modconv
```

1. 选择 **MLX 模型目录**（自动扫描并预估输出体积）；
2. 选择 **Core ML 输出目录**；
3. 可选选择/填写 **最大上下文长度**（默认「自动（≤512）」；可选手动档位或自定义整数，如 `8k`/`4096`）；
4. 点击 **开始转换**。转换阶段与进度实时显示在状态栏与日志区，可随时 **取消**。

转换完成后输出目录包含：

```
<输出目录>/
├── <模型名>.mlpackage          # stateful Core ML 模型
└── <模型名>-swift-example.swift # Swift 调用示例
```

### Swift 侧调用（概要）

```swift
let model = try MLModel(contentsOf: modelURL)
let state = try model.makeState()
// prefill：整段 prompt 一次前向，KV cache 写入 state
// decode：每步传 1 个 token + 起始位置，state 自动携带 KV cache
```

完整可运行示例见生成的 `<模型名>-swift-example.swift`（`generateWithConvertedModel`）。

## 支持的模型架构

`llama` / `mistral` / `qwen2` / `qwen3` / `qwen3_5` / `qwen3_5_moe` / `gemma` / `gemma2` / `phi3`
（MLX 量化模型需 `config.json` 含 `quantization` 字段且本机装有 mlx / mlx-lm）。

Qwen3.5 全系（dense 与 MoE）均支持其混合注意力架构：GDN（linear attention）与
full attention 交替的层，其 conv/recurrent 状态同样以 Core ML State 承载。

## 多目标格式转换（HF→ONNX / ONNX→MLX / HF→MLX / HF→GGUF）

除 Core ML 外，工具在「目标格式」下拉中提供 **ONNX / MLX / GGUF** 三个新目标，
按模型类型（causal / embedding / reranker / vlm）与源格式自动选择转换链路与所用库：

| 目标 | 源 | 实现（库） | 量化 |
| --- | --- | --- | --- |
| ONNX | HF | `torch.onnx.export` + `onnxruntime` | int8 / int4（best-effort，不支持时回退 int8） |
| ONNX | ONNX | weight-only int8（复刻 tools 验证逻辑） | int8（激活保持 fp32） |
| ONNX | MLX | 归一化为 HF 后 `torch.onnx.export`（causal 直接反量化；encoder 反向映射命名） | fp16 / int8 / int4 |
| ONNX | GGUF | GGUF 反量化 + 重建 config/tokenizer 后导出 | fp16 / int8 / int4 |
| MLX | HF | `mlx-lm`(causal) / `mlx-embeddings`(embedding/reranker) / `mlx-vlm`(vlm) | int4 / int8（由库原生支持） |
| MLX | MLX | 重命名为 MLXEmbedders 命名 + 量化（纯 MLX，复刻 tools 验证逻辑） | int4 / int8 / fp16 |
| MLX | ONNX | 内置 ONNX→MLX 权重映射（仅 encoder：BERT/RoBERTa/XLM-R） | int4 / int8（MLX 量化） |
| MLX | GGUF | GGUF 归一化为 HF 后走 `mlx-lm` | int4 / int8 / fp16 |
| GGUF | HF / MLX / ONNX | `gguf`（Llama 布局：llama/qwen2/qwen3/mistral） | int4 / int8 / fp16 |

要点与限制：

- **转换完成后自动验证**：每次转换成功都会在产物上自动做可用性验证（详见
  `modconv/verify.py`），结果随转换日志与状态一并展示：
  - 用目标原生运行时加载产物（ONNX→onnxruntime、MLX→mlx、GGUF→llama_cpp、CoreML→coremltools）；
  - 跑一次前向/生成，检查输出有限值、形状正常；
  - 能力保真用**参考余弦**衡量：相同输入下转换产物与源（源 ONNX / 源 HF）输出/权重的
    余弦（fp16≈1.0、int8>0.99、int4 嵌入>0.95），低于 0.9 视为「有额外损失」并告警；
  - 依赖缺失或无法构建参考时**跳过验证而非失败**（不阻塞转换结果），日志中注明原因。
- **ONNX→MLX（自定义）**：仅支持 encoder 类 embedding/reranker（BERT/RoBERTa/XLM-R），
  需原始 HF 的 `config.json` 与 ONNX 同目录（提供结构信息）；
  decoder（LLM）不支持，会给出明确报错（LLM 请用 HF→MLX）。权重经 ONNX 图遍历恢复
  （含内联为 Constant 的 bias 与 LayerNorm），输出 `model.safetensors` + `config.json`。
- **embedding / reranker（encoder 家族：bge / jina / nomic 等）格式间转换**：
  - HF→ONNX：按模型类型导出（embedding→last_hidden_state，reranker→logits）。
  - ONNX→MLX：按**节点名**解析角色（兼容 bge 的 `attention/self/query` 与 jina 的
    `mixer/Wqkv` **融合 QKV 自动拆分**两种方言），朝向按 Transpose 前置/transB 判定
    （无 Transpose 的 `[in,out]` 自动转置为 HF `[out,in]`），输出 **MLXEmbedders 命名**
    （`encoder.layers.N.attention.query_proj` 等），量化陪衬为**兄弟键**；config 补
    `model_type`/`num_labels`，缺 embeddings.LayerNorm 时补 ones/zeros 占位。
  - MLX→MLX：HF roberta/bert 命名 → MLXEmbedders 命名 + 可选量化（纯 MLX 实现，
    复刻 tools/convert_mlx_rekey_quantize.py，已精确复现 bge-reranker-v2-m3 交付物）；
    源已是量化(uint32) 时仅重命名并沿用源量化配置。
  - ONNX→ONNX / ONNX 目标 int8：embedding/reranker 用 **weight-only 量化**（只量化
    MatMul/Gemm 权重、激活保持 fp32；onnxruntime 动态量化会把激活也量化，对 embedding
    掉精度；fp16 源须先整体转 fp32）——nomic-embed-text-v1.5 已验证余弦 >0.997。
- **HF→ONNX**：仅导出 `input_ids`（不带 attention_mask），以保证 ONNX→MLX 链路可无损转回权重。
- **GGUF 目标支持三种源**：HF 目录、MLX 目录（含 int4/int8 量化权重，自动反量化）
  与单个 `.onnx` 文件（界面有「选 ONNX 文件」按钮；权重取自 .onnx 且需保留 HF 风格
  权重名，`config.json` / `tokenizer.json` 取自其同目录）。仅覆盖 Llama 布局的
  BPE causal 模型；MoE、SentencePiece 特有模型暂不支持。
- **跨格式互转（新增）**：`source_normalize.py` 把 MLX / GGUF 源**归一化为临时 HF
  目录**（MLX 量化权重反量化、MLXEmbedders↔HF 命名互转、GGUF 张量反量化与形状还原、
  config/tokenizer 由 GGUF 元数据与词表重建——分词 ID 实测一致），再复用既有
  HF→{ONNX,MLX,GGUF} 后端。因此新增：
  - MLX→ONNX（causal 与 encoder 均可；与 HF→ONNX 的 logits/embedding 余弦≈1）；
  - GGUF→MLX、GGUF→ONNX（llama 布局 causal）；
  - embedding/reranker 在 ONNX ↔ MLX 间互转（含量化版本）。
  注意：GGUF→GGUF（重量化）暂不支持；**embedding 的 GGUF 腿不可用**——llama.cpp
  只承载 llama 布局 causal 模型，BERT 家族 encoder（bge/jina/nomic）无法写入 GGUF，
  转换时会给出明确报错（decoder 型 embedding 如 Qwen3-Embedding 可走 GGUF）。
- 两个已修复的兼容性问题：transformers 5.x 的 `masking_utils` 用 `torch.diff` 做
  packed-sequence 检测，ONNX 无对应算子导致 causal 导出报 `aten::diff`——导出期间
  屏蔽该检测（只传 input_ids、无 packed sequence，数值无影响，余弦 1.0）；
  MLX 量化权重在维度不整除 group_size 时会回退到更小的 group_size，反量化改为
  **按 scales 形状推断**而非信任 config 字段。
- 目标相关的依赖缺失时，界面会明确提示（不静默降级）。体积预估按各目标口径动态调整。

量化档位随目标自适应：Core ML 为 fp16/4-bit/6-bit/8-bit（palettization）；
ONNX/MLX/GGUF 为 fp16/int8/int4。

## 开发与测试

```bash
.venv/bin/python3.13 -m pytest tests/ -q          # 全量测试（含真实转换集成测试）
QT_QPA_PLATFORM=offscreen .venv/bin/python3.13 -m modconv   # 无显示环境冒烟
```

单元测试以桩依赖隔离重量级库；集成测试用内置 tiny 模型走完整转换链路
（导出 → 转换 → 4-bit 压缩 → 原生 predict parity 校验）；另有 Swift 集成测试
用 `swiftc` 真实编译并运行生成的调用示例，与 Python 侧贪心解码逐 token 比对。

## 项目结构

```
modconv/             Python 包（GUI + 转换核心）
  main_window.py     PyQt6 主窗口
  workers.py         QThread 后台线程（扫描 / 转换）
  backend.py         后端探测与架构白名单（4 目标：coreml/onnx/mlx/gguf）
  real_backend.py    torch.export + coremltools 真实转换（HF→Core ML）
  onnx_backend.py    HF→ONNX（torch.onnx.export + onnxruntime 量化）
  mlx_backend.py     ONNX→MLX（encoder 权重映射：节点名角色/朝向/融合QKV/MLXEmbedders命名）
  mlx_rekey_backend.py MLX→MLX（HF 命名→MLXEmbedders 命名 + 量化，纯 MLX）
  mlx_libs_backend.py HF→MLX（mlx-lm / mlx-embeddings / mlx-vlm）
  gguf_backend.py    HF/MLX/ONNX→GGUF（gguf 写 Llama 布局）
  gguf_to_mlx_backend.py GGUF→MLX（归一化为 HF 后走 mlx-lm）
  source_normalize.py MLX/GGUF 源 → 临时 HF 目录（反量化/命名互转/config/分词器重建）
  kv_wrapper.py      KV cache / GDN 状态的 Core ML State 模块
  converter.py       打包后端（1:1 拷贝兜底）
  scanner.py         目录扫描     estimator.py  输出体积预估
  utils.py           通用工具（体积格式化 / 混合注意力层类型名单）
packaging/           macOS 分发包构建
  make_icon.py       生成 AppIcon.icns
  build_macos.py     构建 .app 与 .dmg
tests/               单元测试 + 真实转换/Swift 集成测试
docs/design.md       设计文档
```

## 文档

- 软件设计文档：[docs/design.md](docs/design.md)

## 许可证

GPL-3.0，见 [LICENSE](LICENSE)。
