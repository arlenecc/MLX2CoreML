# MLX2CoreML — MLX → Core ML 模型转换工具

把 MLX/HF 格式的大语言模型目录（`config.json` + `*.safetensors`）转换为 **Core ML `.mlpackage`**，
供 iOS / macOS / visionOS 应用通过 Core ML 框架直接推理。

> Python 包名/命令行仍为 `modconv`，macOS 应用与仓库名为 **MLX2CoreML**。

## 功能特性

- **Stateful KV cache 模型**：KV cache 以 Core ML State 存储（iOS 18+ 目标），
  prefill 与 decode 共用同一模型，Swift 端无需重算前缀，自回归生成每步只前向 1 个 token。
- **动态序列长度**：输入 `input_ids [1, q]` + `cache_position [q]`，`q ∈ 1..max_seq_len`，
  一次转换覆盖任意上下文长度，无需按固定长度多次导出。
- **最大上下文长度可配**：GUI 输入框指定；留空自动选取（模型位置编码上限与 512 取小）。
  超过模型位置编码上限时自动截断。
- **4-bit 权重压缩**：默认对权重做 4-bit palettization（kmeans LUT），显著减小体积、加速端侧推理；
  压缩失败直接报错（不静默降级为全精度，避免产出与预期不符的模型）。
- **MLX 量化模型支持**：含 `quantization` 的 MLX 模型先用 mlx-lm 反量化再转换。
- **Swift 调用示例**：转换成功后在输出目录生成 `<模型名>-swift-example.swift`，
  含 `makeState()` + 自回归生成循环，可直接拷入 Xcode 使用。示例会**自动编译**
  传入的 `.mlpackage`（Core ML 只能加载编译后的模型）。
- **混合后端兜底**：依赖缺失或架构不支持时自动回退到打包后端（权重 1:1 拷贝），工具始终可用；
  但真实转换**过程**中失败会直接报错，不回退到打包产物（打包产物不是可推理的 Core ML 模型）。
- **全程后台线程**：目录扫描与转换均在 QThread 中执行，大模型转换时界面不卡顿；
  支持随时取消（扫描也可取消，关闭窗口时先取消再等待线程退出）。
- **输入健壮性**：源目录输入框防抖扫描；转换前校验输出目录不位于源目录内；
  单文件读取失败跳过而非让整次转换失败。

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
3. 可选填写 **最大上下文长度**（正整数，token 数；留空自动选取）；
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
  backend.py         后端探测与架构白名单
  real_backend.py    torch.export + coremltools 真实转换
  kv_wrapper.py      KV cache / GDN 状态的 Core ML State 模块
  converter.py       打包后端（1:1 拷贝兜底）
  scanner.py         目录扫描     estimator.py  输出体积预估
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
