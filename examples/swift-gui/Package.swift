// swift-tools-version:6.0
//
// swift-gui —— ModConv 模型体验 App（支持 CoreML / MLX / ONNX 三后端）
//
// 独立 SwiftPM 包（不依赖 tools/model-runtime）。依赖：
//   - onnxruntime-swift：运行通用 ONNX 自回归 decode 图
//   - swift-transformers：Swift Tokenizers 读 tokenizer.json（替代 Python 桥接）
//   - mlx-swift-lm：加载并运行 MLX checkpoint（Qwen3.5 等）
//
// 构建：  swift build
// 运行：  swift run swift-gui
// 打包：  见 build.sh（swift build 后打成 .app）

import PackageDescription

let package = Package(
    name: "swift-gui",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.16.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.32.3"),
    ],
    targets: [
        .executableTarget(
            name: "swift-gui",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXRerankers", package: "mlx-swift-lm"),
            ],
            path: "Sources/swift-gui",
            // AppKit 主线程 UI 模式 + Swift 5 语言模式（与原 swiftc 行为一致，
            // 避免 Swift 6 严格并发把 main-actor/AppKit 调用判为错误）。
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
