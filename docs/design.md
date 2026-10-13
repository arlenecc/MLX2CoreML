# MLX2CoreML 软件设计文档

版本：0.1.0 ｜ 更新日期：2026-10-11

把 MLX/HF 格式大语言模型转换为带 KV cache 状态的 Core ML `.mlpackage`（PyQt6 GUI）。
Python 包与命令行名为 `modconv`，macOS 应用与仓库名为 **MLX2CoreML**。

## 1. 系统概览

```
┌────────────────────────── MainWindow (PyQt6) ──────────────────────────┐
│  配置区：模型目录 / 输出目录 / 最大上下文长度                              │
│  显示区：统计行 + 开始/取消 + 日志；状态栏：总进度                        │
└──────────────┬─────────────────────────────┬───────────────────────────┘
               │ QThread                     │ QThread
        ┌──────▼──────┐              ┌───────▼─────────┐
        │  ScanWorker │              │ ConversionWorker │
        └──────┬──────┘              └───────┬─────────┘
               │ scan_directory              │ probe_backend → 混合选择
               ▼                             ▼
        ┌─────────────┐      ┌──────────────────────────┐
        │  scanner.py │      │ CoremltoolsBackend (真实) │→ 失败即报错（不回退）
        └─────────────┘      │  ModelConverter (打包)    │
                             └──────────────────────────┘
```

分发包（§9）不改动上述架构：`.app` 只是一个带图标的启动器，
负责定位 Python 解释器并把 `Contents/Resources/app` 加入 `PYTHONPATH` 后
执行 `python -m modconv`。

## 2. 模块职责

| 模块 | 职责 |
|---|---|
| `app.py` / `__main__.py` | 应用入口（QApplication + MainWindow） |
| `main_window.py` | 界面构建、输入校验、信号接线、线程生命周期（关闭保护） |
| `workers.py` | `ScanWorker` / `ConversionWorker`（QThread），回调 → Qt 信号桥接 |
| `backend.py` | 后端探测：架构白名单 + 依赖探测 + MLX 量化检测，产出 `BackendPlan` |
| `real_backend.py` | 真实转换后端：权重布局归一化 → torch.export → coremltools（stateful mlprogram）→ 4-bit 压缩 → 元数据/Swift 示例 |
| `kv_wrapper.py` | `KvCacheWrapper`：把 HF 模型包装为 (input_ids, cache_position) → logits 的状态模块 |
| `converter.py` | 打包后端：流式拷贝权重 1:1 + Manifest（兜底，不可直接推理）；相对路径做越界校验，单文件读取失败跳过而非整体失败 |
| `scanner.py` / `estimator.py` / `utils.py` | 目录扫描、输出预估（打包/真实两种口径）、格式化工具 |
| `packaging/make_icon.py` | 构建期：用 Pillow+numpy 绘制图标并 `iconutil` 打包为 `AppIcon.icns`（可复现） |
| `packaging/build_macos.py` | 构建期：组装带图标的 `MLX2CoreML.app`（内嵌源码 + Info.plist + 启动器）并 `hdiutil` 打成 `.dmg` |

## 3. 真实转换流程（CoremltoolsBackend.run）

5 个阶段（MLX 量化模型多一步反量化，共 5；非量化 4）：

0. **权重布局归一化**（按需，见 §3.1）：checkpoint 的权重名/形状与架构期望不符时，
   另存一份纯 HF 布局的临时副本再加载；中间产物同样写在输出目录旁，`finally` 中清理；
1. **反量化**（可选）：`mlx_lm.convert(dequantize=True)` 输出到临时目录，`finally` 中清理；
2. **加载配置与权重**：`AutoConfig` + `AutoModelForCausalLM`（eager 注意力 + eager
   专家实现）。最大上下文长度 = 用户输入（先截断到 `max_position_embeddings`、
   再抬到架构下限，混合注意力架构为 2）或自动选取
   （`min(max_position_embeddings, 512)`）。MoE 架构必须指定 `experts_implementation="eager"`：
   融合内核用 `torch.histc` 统计每张专家的 token 数，coremltools 前端不支持该算子；
   量化模型的反量化中间产物写在**输出目录旁**（而非系统临时目录），避免数十 GB
   中间文件撑爆启动盘；
3. **torch.export**：`KvCacheWrapper` 包装后以动态维度导出（详见 §4）。
   `max_seq_len == 1` 时 `Dim(min=1, max=1)` 非法，退化为静态形状导出；
4. **coremltools 转换**：`convert_to="mlprogram"`，`minimum_deployment_target=iOS18`，
   `compute_precision=FLOAT16`，`states=` 每层 K/V（及 GDN 的 conv/rec）的 `StateType`。
   转换完成后立即 `del` torch 侧大对象（model/exported/wrapper）并 `gc.collect()`，
   降低量化与保存阶段的峰值内存；
5. **后处理与保存**：4-bit palettization（`OpPalettizerConfig(nbits=4, mode="kmeans")`，
   失败直接抛异常）→ spec 重命名输出为 `logits` + 写元数据 → `MLModel(spec, weights_dir=…)`
   重建（mlprogram 不能仅从 spec 重建；重建后旧模型连同其临时权重目录一起释放）
   → 保存 `.mlpackage` + 生成 Swift 示例。

**执行计划构建失败（-14）的自动收敛（阶段 3–5 整体可重试）**：ANE 能否构建执行
计划既无文档也无先验公式，只能实测。Qwen3.5-0.8B 在本机的实测结论：

| 配置 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|
| 转换（单独编译） | 通过 | 通过 | 通过 | — |
| 4-bit 量化后 | -14 | -14 | -14 | **通过** |
| 未量化 | 通过 | — | — | — |

即：**4-bit palettization 才是主导因素**（量化后的图在 512/256/128 都构建不了计划，
未量化时这些长度都能过；fp16 加载把权重内存减半也救不了量化场景）。因此"追踪 →
转换 → 量化 → 重建"整段都可重试，降级候选按实测影响力排序（`_fallback_plans`）：

1. 用户配置原样（量化 + 原长度）；
2. 量化 + 长度减半——给"缩短上下文就能过"的机型/模型一次机会；
3. **取消量化并回到原长度**：实测 4-bit 图才是 -14 的主因，而上下文砍到 64 基本
   没法用，所以宁可输出体积更大的未量化模型；
4. 仍未过，再逐级降长度（下限 64 且不低于架构下限），之后才按失败处理（§3.1）。

所有降级都会写进日志，并附在结果消息里（`已由 512 自动降到 …` / `已改为未量化
输出`），不会静默发生。注意这与"量化失败直接抛异常、禁止降级为全精度"并不冲突：
那条针对的是 palettization 本身报错（说明配置有问题），而这里量化已成功，只是产物
在 ANE 上跑不起来——交付一个能跑的大模型，严格优于交付一个跑不起来的小模型。

**每构造一个 `MLModel` 都会构建一次执行计划**（大模型十几分钟），因此：
- `ct.convert` 返回的模型默认就是 ANE 编译过的——它本身就是"这个长度的图能不能跑"
  的判据，不需要再验一次；
- 量化后 `palettize_weights` 内部也会重建并编译，同样作为判据（压缩后的图与转换
  输出不同，必须重新判）；
- 而写元数据后的"重建"**只为让改过的 spec 落盘**，预测能力毫无必要，故显式传
  `compute_units=CPU_ONLY` 跳过 ANE 编译——省掉十几分钟，也避免一次纯粹由内存
  紧张引起的误报（落盘内容不含计算单元信息，产物完全相同）。

**临时包必须显式删除**（`_drop_package`）：`ct.convert` 与 `MLModel(spec, weights_dir=…)`
都会在系统临时目录下生成一份完整包（大模型上 GB 级），而 coremltools 只在**进程正常
退出**时（atexit）清理。GUI 是长驻进程：连续转换会一直堆积，-14 重试时甚至同时存在
数份——既吃磁盘也干扰后续编译。因此重建完成、重试切换、保存落盘后都立即删除旧包
（只删 coremltools 自己标记为 `is_temp_package` 的包，用户传入的目录不动）。

任何阶段失败抛异常，`ConversionWorker` 捕获后**直接向用户报错**——不回退打包后端
（打包产物不是可推理的 Core ML 模型，静默降级会掩盖失败）。只有在 `probe_backend`
阶段就判定"依赖缺失或架构不支持"时才走打包后端。

同理，coremltools 遇到**编译不过的模型并不抛异常**，只发一条
`RuntimeWarning: You will not be able to run predict() ... error code: -14` 就照常
返回对象。若不拦住，界面会显示"转换完成"，交付的却是一个跑不起来的模型。因此
重建 `MLModel` 时用 `warnings.catch_warnings` 捕获该告警：命中即**判定失败**
（产物照常落盘便于排查，但明确告知不可用，并建议减小上下文长度或关闭量化）。

转换期 coremltools 还会为每个 int64 → int32 的常量降级发一条
`might lose precision` 告警（torch 的 scatter/gather 索引必须是 int64，ANE 只吃
int32），大模型可达数万条、把日志刷满。这些常量都是远小于 2³¹ 的索引，降级无损，
故在 `coremltools` logger 上挂过滤规则丢弃。

### 3.1 权重布局归一化（加载前的强制校验）

MLX 生态的 checkpoint 未必是 transformers 能直接加载的布局，两类差异必须修正：

1. **多模态（复合）前缀**：MLX 的多模态布局把文本模型放在 `language_model.` 之下
   （并带有 `vision_tower.*`），而 `AutoModelForCausalLM` 期望 `model.layers.*`；
2. **Conv1d 权重轴序**：MLX 的 `Conv1d` 权重是 `(out, kernel, in)`，PyTorch 是
   `(out, in, kernel)`。GDN / linear attention 的 conv1d 权重因此要么形状对不上而
   直接报错，要么在两维恰好相等时被**静默当成正确权重**加载。mlx-lm 的
   `convert`（含量化模型的反量化）不做这一步转换。

关键在于：**名字对不上时 transformers 不会报错**，它把所有权重"新建初始化"，
于是转换"成功"、产出的却是一个随机权重的模型。因此加载前必须先校验：

- `_expected_state_shapes`：在 **meta 设备**上搭骨架取期望的名字与形状（0.8B 约 0.2s，
  不分配真实内存）；
- `_checkpoint_tensor_shapes`：只读 safetensors 头部取 checkpoint 的名字与形状；
- `_resolve_weight_layout`：选出匹配的前缀，并按"期望形状 = checkpoint 形状交换
  后两维"判定需要转置的张量；重合比例低于阈值时返回 `None` → **中止转换并报错**；
- 需要归一化时 `_write_text_checkpoint` 逐分片另存（内存约等于一个分片），
  只保留文本权重，`vision_tower.*` 一并丢弃。

布局已经匹配时这一整套检查为零成本（不读权重数据、不落盘），只有真的不匹配才
产生一份临时副本。

## 4. Stateful KV cache 设计（kv_wrapper.py）

- **State 载体**：每层 K/V 注册为 named buffer `cache_{i}_k/v`，形状
  `(1, n_kv_heads, max_seq_len, head_dim)`；coremltools 按 buffer 名匹配 `StateType`。
- **buffer 固定 fp32**：coremltools 前端对 `read_state` 无条件插入 fp32 cast，
  in-place scatter 的 data 恒为 fp32，updates 也必须 fp32；写回 state 时前端自动
  cast 回 fp16（Core ML State 仅支持 fp16）。掩码精度独立跟随模型原精度，
  避免 fp32 掩码提升 `attn_weights` 破坏 fp16 matmul。
- **写入槽位**：`_SlotStaticLayer.update` 用 `arange(kv_length) + cumulative_length`
  定位写入槽位，游标每次 forward 重绑定为 `cache_position[0]`（输入派生），
  不做 in-place 累加；导出图中唯一的 mutation 是对 KV buffer 的 rank-4 `scatter_`
  （前端映射为 `scatter_along_axis`，支持动态长度）。
- **注意力掩码**：wrapper 内部按 `cache_position` 自算静态形状 4D 因果掩码
  （`slots > cache_position` 广播比较 → 布尔 cast 乘 fp16 min -65504），
  绕开 HF 动态掩码机器（其运行时 tile 会破坏 ANE）；掩码值取 fp16 有限最小值，
  避免 fp32 min 在 fp16 计算中溢出为 -inf 产生 NaN。
- **Qwen3.5 全系（dense / MoE）**：模块补丁按**结构特征**识别而非写死类名——
  transformers 由同一份 modular 源码为各变体生成具名类（dense 的
  `Qwen3_5GatedDeltaNet` / MoE 的 `Qwen3_5MoeGatedDeltaNet`、
  `Qwen3_5TextRotaryEmbedding` / `Qwen3_5MoeTextRotaryEmbedding`），写死类名会让
  MoE 架构直接失败。GDN 前向所需的三个内核函数在打补丁时从 mixer 所属模块解析并
  挂到实例上（`importlib.import_module` 之类动态导入无法被 torch.export 追踪）。
- **MoE 专家层**：transformers 的两条专家路径都不可导出（eager 在 Python 层遍历
  数据相关的 `expert_hit`；`grouped_mm`/`batched_mm` 用 `torch.histc`）。wrapper
  替换为静态批量前向：把 (token, top-k 槽位) 展平成 S = T×k 行 → 按 expert id
  gather 权重 → batched matmul → reshape 回 (T, k, H) 求和。与原版逐专家累加
  **逐位等价**（未选中专家的路由权重恒为 0），且只对 top-k 命中的专家计算。
- **混合架构先切 hidden 再算 lm_head**：pad 到 `max_seq_len` 只为静态化 GDN 分块
  循环，输出只取前 q 个位置。若先算 `(1, max_seq_len, vocab)` 再切片，会为 pad
  位置也算一整份 logits（词表 248320、长度 512 时 fp32 下 508MB），是全图最大的
  中间结果，既浪费算力也容易让 ANE 执行计划构建失败（-14）。`lm_head` 逐位置独立，
  改为先切 hidden（`H` 维而非词表维）再算 logits，与原实现**数值等价**。
- **导出要点**：
  - 示例长度 ≥2（`min(max_seq_len, 4)`）：torch.export 会把尺寸为 1 的维度特化为常量；
  - 导出包在 `torch.no_grad()`：参数 requires_grad 时 buffer 的 in-place 变异会被
    aot_export 拒绝；
  - 分解表去掉 `aten.matmul` 分解并保留 `alias` 恒等：matmul 分解产生以 sym_size
    为目标形状的 expand，coremltools 无法静态证明 tile reps=1，导致 ANE 执行计划
    构建失败（-14）。

## 5. 线程模型与生命周期

- 所有耗时操作（扫描 / 转换）在 QThread 中执行，通过信号跨线程更新 UI；
- `ConversionWorker` 默认 `backend="auto"`：`probe_backend` 决定真实/打包后端，
  真实转换失败**直接报错**（不回退）；`max_seq_len` 独立参数透传给
  `CoremltoolsBackend`（不混入打包后端的 `**kwargs`）。`run()` 整体包在
  try/finally 里，`conversion_finished` **无条件发出**——探测阶段抛异常若漏掉它，
  主线程永远不会调用 `_set_running(False)`，GUI 会永久卡在禁用态；
- `ScanWorker` 支持 `cancel()`（从扫描回调抛内部异常提前退出，不发
  `scan_finished`）：关闭窗口时先取消再等待，否则大目录扫描远超等待预算，
  接受关闭后线程会在对象销毁后继续运行而崩溃；
- 扫描线程结束后立即从 `MainWindow._scan_workers` 移除（避免长期持有 QThread）；
  源目录输入框 `editingFinished` 经 400ms 防抖后才触发扫描（连续输入不会各起
  一次无法取消的全目录扫描）；任何"放弃扫描"的分支都递增 `_scan_gen`，避免
  在飞行的旧扫描结果覆盖已失效的状态；
- `_begin_conversion()` 自身会重新校验输出目录（不位于源目录内、可创建）：
  由 `_pending_start` 触发时绕过了 `_on_start_clicked` 的检查，而扫描期间输出框
  仍可编辑；
- 关闭等待用 `_wait_for_thread`（小步 `wait` + `processEvents`）替代单次
  `wait(30s)`，避免主线程整段阻塞、窗口无法重绘；
- 取消：`cancel()` 置标记，转换器在块间/阶段间检查（深度阶段如 mlx 反量化、
  `ct.convert` 内部不可中断，取消在阶段边界生效）；量化阶段结束后还会再查一次，
  否则在最长阶段里点的取消会被忽略、一路走到保存；
- **关闭保护**（`closeEvent`）：转换运行中先 `cancel()` 并等待至多 30s，超时则拒绝
  关闭（避免销毁运行中的 QThread 导致进程崩溃）；扫描线程先 `cancel()` 再等待至多 3s，
  同样检查返回值；线程判活用 `_thread_alive()` 兼容 C++ 对象已被 `deleteLater` 销毁的情况；
- **线程引用模型**：`ScanWorker` / `ConversionWorker` 继承 `_TrackedThread`，
  `start()` 时登记到模块级集合、线程真正结束后（`finished` 已在主线程派发）注销。
  QThread 在运行中被销毁会触发 Qt 的 `qFatal` → 进程直接 abort，Python 层无法捕获，
  因此不能把"调用方记得保留引用"当作前提（局部变量出作用域即可被 GC）；
- 转换线程分两层持有：`_conv_workers` 保存**所有尚未退出**的线程（在 `finished`
  时移除），`_conv_worker` 只指向"当前这次"。`conversion_finished` 是在线程 `run()`
  内部发出的，随后线程还要再跑几毫秒——只看 `_conv_worker` 会漏掉这段窗口期，
  导致关闭窗口时销毁仍在运行的线程、或重复启动两个转换线程。不再用退役列表长期
  保留已结束的 worker（它们攥着整份 `FileInfo` 清单，大目录上是实打实的内存滞留）；
- 新扫描开始前先取消在飞的旧扫描（结果已因 `_scan_gen` 递增而作废，继续跑只会
  抢磁盘带宽）。

## 6. 内存与性能

- 扫描/打包拷贝均为流式（扫描只 stat；打包 4 MiB 块），内存占用与文件大小无关；
- 扫描用 `os.scandir` + 显式栈遍历：`DirEntry` 自带 d_type，判定符号链接/目录通常
  不需要系统调用，每个文件只剩一次 stat（原 `os.walk` + `Path` 写法是两次 stat
  再加一次 `relative_to` 字符串运算）；文件名排序保证结果与文件系统返回顺序无关；
- 打包进度按 16 MiB 字节间隔节流，扫描每 256 个文件回报，避免事件洪泛；
- 真实转换：量化模型反量化后以 fp16 加载（内存减半）；一律带 `low_cpu_mem_usage`
  就地载入权重（省掉"先随机初始化再整体覆盖"的一倍峰值内存，老版本 transformers
  不认该参数时逐级降级）；追踪结束后先取出 state 元信息再**丢弃 wrapper + HF 模型**
  （`wrapper.inner` 就是 model，只 del 局部变量释放不掉），`ct.convert` 完成后立即
  释放导出图（大模型数十 GB）再进入量化/保存阶段；4-bit 压缩后再释放压缩前的
  模型对象及其临时权重目录；
- 4-bit palettization 受 coremltools `weight_threshold`（2048 元素）约束，
  小权重自动跳过；
- 真实转换的阶段进度以 `file_progress` 信号上报（relpath = "阶段 i/total"），
  GUI 侧识别该前缀显示阶段文案并同步进度条。

## 7. 测试策略

- **单元测试**（桩依赖隔离 torch/coremltools）：后端探测、最大上下文选取、
  dtype 参数兼容、转换主流程 kwargs 断言、量化/取消/失败报错（不回退）、
  GUI 交互（输入校验、阶段文案、closeEvent 保护、信号透传）；
- **集成测试**（真实依赖，内置 tiny 模型，不联网）：KV wrapper 与 HF eager
  parity、完整转换链路产出 `.mlpackage`、4-bit LUT proto 断言、原生
  `predict` prefill/decode parity 与空 state 差异校验、反量化临时目录清理；
  Qwen3.5 dense（4 种配置变体）与 MoE 端到端；Core ML 与 torch 参考的数值一致性
  （最大误差 / 余弦相似度 / argmax 一致率三层断言）；
- **Swift 集成测试**（需 `xcrun swiftc`）：生成的调用示例零警告通过类型检查；
  编译并真实运行自回归生成，与 Python 侧贪心解码逐 token 比对（这是"导出模型
  能被 Swift 正常调用"的直接证据）；
- 边界与失败路径：探测阶段抛异常也必须发出 `conversion_finished`（否则 GUI 永久
  卡在禁用态）、`max_seq_len=1` 的完整转换、越界的相对路径被拒绝、缺失源文件
  被跳过、扫描取消不发结果、由 `_pending_start` 触发时输出目录仍要校验；
- GUI 测试在 `QT_QPA_PLATFORM=offscreen` 下运行。

## 8. 已知限制

- 支持架构白名单：llama / mistral / qwen2 / qwen3 / qwen3_5 / qwen3_5_moe /
  gemma / gemma2 / phi3；
- `SEQ_LEN_CAP = 512`：自动选取的上限（KV buffer 体积随长度线性增长，
  fp32 buffer 转 fp16 state 后约 `2 × layers × n_kv_heads × L × head_dim × 2B`）；
  用户可显式输入更大值（仍受模型位置编码上限截断）；
- 非量化模型以 fp32 加载（转换目标 fp16，保证单次舍入）；
- 打包后端输出为权重 1:1 拷贝（含 Manifest），不能直接用 Core ML 推理，
  仅作为依赖缺失时的兜底产物；
- 输出体积预估（GUI "预计 Core ML 输出"）按后端口径分别估算：架构受支持时按
  fp16 + 4-bit 折算（未计入 LUT / Manifest 等固定开销，系统性偏低），否则按打包
  1:1，仅供粗略参考；
- 多模态（VL）checkpoint 只转换其**文本 LM**部分：`vision_tower.*` 权重被丢弃，
  产出的是纯文本模型；
- 权重布局不匹配时（见 §3.1）需要额外写出一份与源权重等大的临时副本，
  占用同等磁盘空间（仍写在输出目录旁）。
- 分发包是**启动器形态**的 `.app`，不内嵌 torch / transformers / coremltools
  （合计 2–4 GB 且与 Python 版本强绑定），需要宿主 Python 环境（见 §9）。

## 9. 分发包构建

```
packaging/make_icon.py      1024×1024 绘制（Pillow + numpy）→ 各尺寸 PNG → iconutil → AppIcon.icns
packaging/build_macos.py    pyproject 取版本 → 组装 .app → ditto/hdiutil → .dmg
```

`.app` 结构（`build_macos.py` 无第三方依赖，仅用标准库 + `ditto`/`hdiutil`）：

```
MLX2CoreML.app/Contents/
├── Info.plist                        CFBundleIconFile=AppIcon、BundleID、最低 12.0
├── PkgInfo
├── MacOS/MLX2CoreML                  启动器（bash，可执行）
└── Resources/
    ├── AppIcon.icns                  图标
    ├── install_dependencies.command  一键装依赖（App 内 + DMG 根目录各一份）
    └── app/                          内嵌源码：modconv/ + pyproject.toml + README + LICENSE
```

启动器解析顺序：`$MLX2COREML_PYTHON` → `~/Library/Application Support/MLX2CoreML/venv`
→ PATH 上已装 PyQt6 的 `python3`（3.13→3.10 依次尝试）。全部不可用时弹
`osascript` 对话框引导运行安装脚本。启动前把 `Contents/Resources/app` 前置到
`PYTHONPATH`，`cd "$HOME"` 后执行 `python -m modconv`，输出重定向到
`~/Library/Logs/MLX2CoreML.log`；非零退出码时弹错误框并 `open -R` 日志文件。

安装脚本在与 App 同级、App 内两种位置都能定位源码（按候选路径探测），
在独立虚拟环境里 `pip install "<源码目录>[convert]"`（extras 可选）。

DMG 内容：`MLX2CoreML.app`、`安装依赖.command`、`使用说明.txt`、`README.md`、
`LICENSE` 与指向 `/Applications` 的软链接；`hdiutil ... -fs HFS+ -format UDZO -imagekey zlib-level=9`。

构建产物（`dist/`、`*.dmg`、`icon-preview-512.png`）已被 `.gitignore` 排除。
