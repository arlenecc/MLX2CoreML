// swift-tools-version: 6.2
//
// ONNXRuntimeTester —— 基于 onnxruntime-swift（ONNX Runtime 官方 SPM 包）的
// Embedding / Reranker 模型加载与推理测速工具。
//
// 功能：选择本地 ONNX 模型（embedding / reranker 各一个），开启 CoreML
// 执行提供程序加载，自动生成一批测试数据，测量两种模型的推理速度
// （句/s、tokens/s、平均延迟）并展示效果样例。
//
// 运行方式：
//   1. GUI 应用：      swift run ONNXRuntimeTester
//   2. 命令行基准：     swift run ONNXRuntimeTester --embedding <模型.onnx> --reranker <模型.onnx>
//   3. 打包 .app：     ./scripts/build-app.sh
//
// 构建环境：macOS 14+，需要完整版 Xcode（ONNX Runtime 二进制框架 +
// CoreML 执行提供程序）。
//
import PackageDescription

let package = Package(
    name: "ONNXRuntimeTester",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        // ONNX Runtime 官方 Swift Package Manager 支持（产品名 onnxruntime）
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.16.0"),
        // Hugging Face tokenizers（读取 tokenizer.json）
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // mlx-swift-lm：提供 MLXEmbedders / MLXRerankers 以加载并运行 MLX 格式的
        // embedding 与 reranker 模型（BERT / Qwen3 / NomicBert / BGE-M3 等）
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.32.3"),
    ],
    targets: [
        .executableTarget(
            name: "ONNXRuntimeTester",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXRerankers", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            ],
            path: "Sources/ONNXRuntimeTester"
        )
    ]
)
