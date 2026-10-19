import Foundation

/// 统一的模型运行时抽象：屏蔽 ONNX / MLX / CoreML 三种后端的差异，
/// 引擎只需调用 `embed` / `rerank` 即可完成推理与计时，无需关心底层实现。
///
/// 设计要点：
/// - 分词与池化等前置/后置处理由各后端在内部完成（ONNX/CoreML 共用 swift-transformers 分词器，MLX 用自带分词器）。
/// - `embed` / `rerank` 接收**原始文本**，返回（向量 / 得分）与真实 token 数，便于引擎统一计算吞吐。
protocol ModelRuntime: Sendable {
    var kind: ModelKind { get }
    var backend: BackendKind { get }
    var name: String { get }
    var inputNames: [String] { get }
    var outputNames: [String] { get }
    /// 统一状态描述（含后端与具体执行方式）
    var statusDescription: String { get }
    var loadDuration: TimeInterval { get }
    var fileBytes: Int64 { get }

    /// 加载信息（用于界面展示；path 由引擎在加载后补全）
    var info: LoadedModel { get }

    /// Embedding：对一批文本编码并返回句向量与真实 token 数（不含 padding）
    func embed(texts: [String], maxLength: Int) async throws -> (vectors: [[Float]], tokens: Int)

    /// Reranker：对一批 (query, document) 对打分，返回（与输入顺序一致的得分, 真实 token 数）
    func rerank(pairs: [(query: String, document: String)], maxLength: Int) async throws -> (scores: [Float], tokens: Int)
}

/// 共享的句向量池化逻辑（ONNX / CoreML 后端复用）：
/// - 若输出已是 [batch, hidden] 二维，直接归一化返回；
/// - 若为 [batch, seq, hidden] 三维，按 attention mask 做平均池化后 L2 归一化。
func poolEmbeddingOutput(
    _ floats: [Float],
    shape: [Int],
    masks: [[Int]]
) -> [[Float]] {
    if shape.count == 2 {
        let batch = shape[0]
        let hidden = shape[1]
        var vectors: [[Float]] = []
        for b in 0..<batch {
            let start = b * hidden
            vectors.append(Array(floats[start..<(start + hidden)]))
        }
        return vectors.map(normalizeVector)
    }

    guard shape.count == 3 else { return [] }

    let batch = shape[0]
    let seq = shape[1]
    let hidden = shape[2]
    var vectors: [[Float]] = []
    for b in 0..<batch {
        var sum = [Float](repeating: 0, count: hidden)
        var count: Float = 0
        let mask = b < masks.count ? masks[b] : []
        for s in 0..<seq {
            let maskValue = s < mask.count ? Float(mask[s]) : 1
            if maskValue == 0 { continue }
            let offset = (b * seq + s) * hidden
            for h in 0..<hidden {
                sum[h] += floats[offset + h] * maskValue
            }
            count += maskValue
        }
        if count > 0 {
            for h in 0..<hidden { sum[h] /= count }
        }
        vectors.append(normalizeVector(sum))
    }
    return vectors
}

/// L2 归一化（模长为 0 时原样返回）
func normalizeVector(_ vector: [Float]) -> [Float] {
    var sum: Float = 0
    for value in vector { sum += value * value }
    let norm = sum.squareRoot()
    guard norm > 0 else { return vector }
    return vector.map { $0 / norm }
}
