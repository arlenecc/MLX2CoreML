import Foundation

/// 模型类型
enum ModelKind: String, CaseIterable, Identifiable {
    case embedding = "Embedding"
    case reranker = "Reranker"

    var id: String { rawValue }
}

/// 推理后端：同一模型可来自不同格式/运行时，用于横向对比测速
enum BackendKind: String, CaseIterable, Identifiable {
    case onnx = "ONNX"
    case mlx = "MLX"
    case coreml = "CoreML"

    var id: String { rawValue }

    /// 选择面板里展示的中文说明
    var displayName: String { rawValue }
}

/// GUI 中的一个模型配置槽：角色（Embedding/Reranker）+ 后端 + 路径
struct ModelSlot: Identifiable {
    let id = UUID()
    var role: ModelKind = .embedding
    var backend: BackendKind = .onnx
    var path: String = ""
}

/// CoreML 执行提供程序配置
struct CoreMLOptions {
    /// 是否启用 CoreML EP（关闭则纯 CPU 运行，便于对比）
    var enabled = true
    /// 仅使用 CPU（不占用 GPU/ANE）
    var cpuOnly = false
    /// 允许 CoreML 接管子图（模型部分算子不支持时仍可运行）
    var enableOnSubgraphs = true
    /// 生成 MLProgram 格式（Core ML 5+，通常比 NeuralNetwork 更快）
    var createMLProgram = true
}

/// 已加载模型的信息
struct LoadedModel: Identifiable, Equatable {
    let id = UUID()
    let kind: ModelKind
    let backend: BackendKind
    let name: String
    var path: String
    let inputNames: [String]
    let outputNames: [String]
    /// CoreML EP 是否实际启用成功（仅 ONNX / CoreML 后端有意义）
    let coreMLEnabled: Bool
    let loadDuration: TimeInterval
    let fileBytes: Int64

    /// 统一的状态描述（含后端）
    var statusDescription: String {
        switch backend {
        case .onnx:
            return coreMLEnabled ? "ONNX · CoreML 已启用" : "ONNX · 纯 CPU"
        case .coreml:
            return "CoreML"
        case .mlx:
            return "MLX"
        }
    }
}

/// 一次基准测试的性能指标
struct BenchmarkMetrics: Equatable {
    /// 每批条目数（句子数或 query-doc 对数）
    var batchSize: Int = 0
    /// 计时迭代轮数
    var iterations: Int = 0
    /// 平均每批延迟（毫秒）
    var avgLatencyMs: Double = 0
    /// 延迟中位数（毫秒）
    var p50LatencyMs: Double = 0
    /// 延迟 P90（毫秒）
    var p90LatencyMs: Double = 0
    /// 吞吐：句/s 或 对/s
    var itemsPerSecond: Double = 0
    /// 吞吐：tokens/s（不含 padding）
    var tokensPerSecond: Double = 0
    /// 总条目数
    var totalItems: Int = 0
    /// 总 token 数（不含 padding）
    var totalTokens: Int = 0
    /// 总耗时（秒）
    var totalSeconds: Double = 0
}

/// Embedding 基准测试结果
struct EmbeddingBenchmark: Equatable {
    var metrics: BenchmarkMetrics = .init()
    /// 向量维度
    var dimension: Int = 0
    /// 效果样例：几组文本的余弦相似度
    var similaritySamples: [SimilaritySample] = []
    /// 区分多个模型的标签（如「模型 1 · ONNX」）
    var modelLabel: String = ""
}

struct SimilaritySample: Equatable {
    var label: String
    var textA: String
    var textB: String
    var similarity: Float
}

/// Reranker 基准测试结果
struct RerankerBenchmark: Equatable {
    var metrics: BenchmarkMetrics = .init()
    /// 效果样例：query 对若干文档的排序得分
    var rankedSamples: [RankedSample] = []
    /// 区分多个模型的标签（如「模型 1 · ONNX」）
    var modelLabel: String = ""
}

struct RankedSample: Equatable {
    var query: String
    var document: String
    var score: Float
}

/// 日志条目
struct LogEntry: Identifiable, Equatable {
    let id = UUID()
    let time = Date()
    let message: String
    var level: Level

    enum Level: Equatable {
        case info
        case success
        case warning
        case error
    }

    var timeString: String {
        time.formatted(.dateTime.hour().minute().second())
    }
}

/// 基准测试过程中的事件（通过 AsyncStream 推送）
enum BenchmarkEvent: Sendable {
    case progress(String)
    case embeddingFinished(slot: Int, EmbeddingBenchmark)
    case rerankerFinished(slot: Int, RerankerBenchmark)
    case failed(String)
}

/// 加载结果报告
struct LoadReport: Equatable {
    /// 每个模型槽的已加载信息（按槽位顺序排列）
    var models: [LoadedModel]
    var tokenizerDescription: String
}
