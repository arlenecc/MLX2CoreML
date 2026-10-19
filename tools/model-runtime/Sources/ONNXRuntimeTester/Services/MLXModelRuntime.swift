import Foundation
import MLX
import MLXEmbedders
import MLXRerankers
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

/// MLX 后端：使用 mlx-swift-lm 的 MLXEmbedders / MLXRerankers 加载并运行
/// MLX 格式的 embedding 与 reranker 模型（BERT / Qwen3 / NomicBert / BGE-M3 等，
/// 由库自动按 config.json 识别架构）。
///
/// MLX 自带分词器与池化/打分逻辑，无需 swift-transformers，也无需 ONNX Runtime。
final class MLXModelRuntime: @unchecked Sendable {
    let kind: ModelKind
    let name: String
    let loadDuration: TimeInterval
    let fileBytes: Int64
    let modelDirectory: URL

    private let embeddingContainer: EmbedderModelContainer?
    private let rerankerContainer: RerankerContainer?

    private init(
        kind: ModelKind,
        name: String,
        loadDuration: TimeInterval,
        fileBytes: Int64,
        modelDirectory: URL,
        embeddingContainer: EmbedderModelContainer?,
        rerankerContainer: RerankerContainer?
    ) {
        self.kind = kind
        self.name = name
        self.loadDuration = loadDuration
        self.fileBytes = fileBytes
        self.modelDirectory = modelDirectory
        self.embeddingContainer = embeddingContainer
        self.rerankerContainer = rerankerContainer
    }

    /// 加载 MLX 模型：embedding 用 EmbedderModelFactory，reranker 用 RerankerModelFactory。
    /// 两者都从本地目录加载（目录需包含 config.json、权重与 tokenizer 文件）。
    static func load(directory: URL, kind: ModelKind) async throws -> MLXModelRuntime {
        let start = ContinuousClock.now
        let loader = #huggingFaceTokenizerLoader()

        let (embeddingContainer, rerankerContainer): (EmbedderModelContainer?, RerankerContainer?)
        do {
            if kind == .embedding {
                embeddingContainer = try await EmbedderModelFactory.shared.loadContainer(
                    from: directory, using: loader)
                rerankerContainer = nil
            } else {
                embeddingContainer = nil
                rerankerContainer = try await RerankerModelFactory.shared.loadContainer(
                    from: directory, using: loader, allowUnverifiedModel: true)
            }
        } catch {
            throw MLXRuntimeError.loadFailed("\(error.localizedDescription)")
        }

        let elapsed = start.duration(to: .now)
        let loadDuration = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let fileBytes = modelFileBytes(directory)

        return MLXModelRuntime(
            kind: kind,
            name: directory.lastPathComponent,
            loadDuration: loadDuration,
            fileBytes: fileBytes,
            modelDirectory: directory,
            embeddingContainer: embeddingContainer,
            rerankerContainer: rerankerContainer
        )
    }

    var info: LoadedModel {
        LoadedModel(
            kind: kind,
            backend: .mlx,
            name: name,
            path: modelDirectory.path,
            inputNames: embeddingContainer != nil ? ["input_ids", "attention_mask", "token_type_ids"] : [],
            outputNames: embeddingContainer != nil ? ["pooled_output"] : ["logits"],
            coreMLEnabled: false,
            loadDuration: loadDuration,
            fileBytes: fileBytes
        )
    }
}

enum MLXRuntimeError: LocalizedError {
    case loadFailed(String)
    case runFailed(String)
    case unsupported(String)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let m): return "MLX 模型加载失败：\(m)"
        case .runFailed(let m): return "MLX 推理失败：\(m)"
        case .unsupported(let m): return m
        }
    }
}

extension MLXModelRuntime: ModelRuntime {
    var backend: BackendKind { .mlx }
    var statusDescription: String { "MLX" }
    var inputNames: [String] {
        embeddingContainer != nil ? ["input_ids", "attention_mask", "token_type_ids"] : []
    }
    var outputNames: [String] {
        embeddingContainer != nil ? ["pooled_output"] : ["logits"]
    }

    func embed(texts: [String], maxLength: Int) async throws -> (vectors: [[Float]], tokens: Int) {
        guard let container = embeddingContainer else {
            throw MLXRuntimeError.unsupported("该 MLX 模型不是 embedding 模型")
        }
        let (vectors, tokens) = await container.perform { ctx in
            let encoded = texts.map { ctx.tokenizer.encode(text: $0, addSpecialTokens: true) }
            let trimmed = encoded.map { Array($0.prefix(maxLength)) }
            let seq = trimmed.map { $0.count }.max() ?? 1
            let batch = trimmed.count
            var idFlat: [Int32] = []
            var maskFlat: [Float] = []
            for e in trimmed {
                for i in 0..<seq {
                    if i < e.count {
                        idFlat.append(Int32(e[i]))
                        maskFlat.append(1)
                    } else {
                        idFlat.append(0)
                        maskFlat.append(0)
                    }
                }
            }
            let ids = MLXArray(idFlat).reshaped([batch, seq])
            let mask = MLXArray(maskFlat).reshaped([batch, seq])
            let types = MLXArray([Int32](repeating: 0, count: batch * seq)).reshaped([batch, seq])

            let output = ctx.model(ids, positionIds: nil, tokenTypeIds: types, attentionMask: mask)
            let pooled = ctx.pooling(output, mask: mask, normalize: true, applyLayerNorm: true)
            MLX.eval(pooled)
            let flat = pooled.asArray(Float.self)
            let tokenCount = trimmed.reduce(0) { $0 + $1.count }
            return (flat, tokenCount)
        }

        let hidden = batchSize(for: vectors.count, batch: texts.count)
        var result: [[Float]] = []
        for b in 0..<texts.count {
            result.append(Array(vectors[b * hidden..<(b + 1) * hidden]))
        }
        return (result, tokens)
    }

    func rerank(pairs: [(query: String, document: String)], maxLength: Int) async throws -> (scores: [Float], tokens: Int) {
        guard let container = rerankerContainer else {
            throw MLXRuntimeError.unsupported("该 MLX 模型不是 reranker 模型")
        }
        var scores: [Float] = []
        for pair in pairs {
            let response = try await container.scores(
                query: pair.query,
                documents: [pair.document],
                instruction: nil
            )
            let score = response.results.first?.score ?? 0
            scores.append(Float(score))
        }
        // MLX reranker 容器不对外暴露分词器，token 数无法精确统计，记为 0（itemsPerSecond 仍有效）
        return (scores, 0)
    }

    private func batchSize(for flatCount: Int, batch: Int) -> Int {
        batch > 0 ? max(1, flatCount / batch) : 1
    }
}

/// 计算目录（MLX 模型权重所在）的总字节数
func modelFileBytes(_ url: URL) -> Int64 {
    let fileManager = FileManager.default
    var isDir: ObjCBool = false
    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir) else { return 0 }
    if !isDir.boolValue {
        return (try? fileManager.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
    }
    var total: Int64 = 0
    if let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
    }
    return total
}
