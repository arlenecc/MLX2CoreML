# model-runtime/ — 模型加载 / 测试 / 验证运行时（Swift）

本目录归档了 `onnx-runtime-tester` 中**用于实际加载、测试、验证转换后模型**的 Swift 代码，
与 `tools/` 下的三个 Python 转换工具配合使用：转换工具产出 MLX 模型目录后，
由这里的运行时把它们真正跑起来做测速与效果验证。

> 这是「测试 / 修复 / 验证」相关代码（而非转换脚本）。`MLXModelRuntime.swift` 即为
> 让 MLX 版 reranker / embedding 模型能被加载运行所做的修复（基于 MLXEmbedders /
> MLXRerankers 的 `EmbedderModelFactory` / `RerankerModelFactory`）。

## 目录内容
- `Sources/ONNXRuntimeTester/Services/MLXModelRuntime.swift` —— **核心修复**：用
  MLXEmbedders 加载 embedding、MLXRerankers 加载 reranker（本地目录 + `allowUnverifiedModel`），
  提供 `embed(...)` / `rerank(...)`。
- `Sources/ONNXRuntimeTester/Services/ORTEngine.swift` —— ONNX Runtime 后端（`.onnx`）。
- `Sources/ONNXRuntimeTester/Services/CoreMLModelRuntime.swift` —— Core ML 后端。
- `Sources/ONNXRuntimeTester/Services/ONNXBenchmarkEngine.swift` —— 基准/验证引擎：
  加载模型、生成测试数据、**跑 embedding 相似度 / reranker 排序并输出效果**。
- `Sources/ONNXRuntimeTester/Models/TestCorpus.swift` —— **内置测试语料**（中英文混合句子、
  reranker 的 query/文档对、相似度正负样本），即验证模型效果所用的「西湖」等样例数据。
- `Sources/ONNXRuntimeTester/Models/ModelTypes.swift` —— 模型种类、Benchmark 配置与结果类型。
- `Sources/ONNXRuntimeTester/CLI.swift` —— 无界面基准模式（测速 + 效果验证）：
  ```
  swift run ONNXRuntimeTester --embedding <模型目录> --embedding-backend mlx \
      --reranker <模型目录> --reranker-backend mlx [--batch 16] [--count 128] [--max-length 128]
  ```
- 其余 Views / ViewModels / Tokenizer / StatFormatter 为 GUI 配套，随包归档。
- `Package.swift` / `build-app.sh` —— 构建配置（依赖 mlx-swift-lm、OnnxRuntimeBindings 等）。

## 用法
```bash
cd tools/model-runtime
# 仅测速 + 效果验证（无 GUI）
swift run ONNXRuntimeTester \
    --embedding "/Volumes/SSD/embedding&reranker/bge-reranker-v2-m3-mlx-int4" --embedding-backend mlx \
    --reranker  "/Volumes/SSD/embedding&reranker/bge-reranker-v2-m3-mlx-int4" --reranker-backend mlx
# 不加任何参数则启动图形界面
```

## 与转换工具的配合
1. 用 `tools/convert_jina_mlx_to_int4.py` / `convert_onnx_reranker_to_mlx_int4.py` /
   `convert_mlx_rekey_quantize.py` 生成 MLX 模型目录（含 `config.json`、权重、tokenizer）。
2. 用本运行时加载该目录，确认能正常 `embed` / `rerank` 且效果（相似度 / 排序）合理。
