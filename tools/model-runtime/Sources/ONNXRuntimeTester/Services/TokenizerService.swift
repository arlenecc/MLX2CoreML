import Foundation
import Tokenizers

/// 一段文本的编码结果（已包含特殊 token 与截断）
struct EncodedText {
    var ids: [Int]
    var attentionMask: [Int]
    var tokenTypeIds: [Int]

    init(ids: [Int], attentionMask: [Int]? = nil, tokenTypeIds: [Int]? = nil) {
        self.ids = ids
        self.attentionMask = attentionMask ?? [Int](repeating: 1, count: ids.count)
        self.tokenTypeIds = tokenTypeIds ?? [Int](repeating: 0, count: ids.count)
    }

    /// 真实 token 数（不含 padding）
    var tokenCount: Int { ids.count }
}

/// 分词器抽象：单句编码与 query-document 对编码
protocol TextTokenizer: Sendable {
    var description: String { get }
    /// 编码单句（含 [CLS]/[SEP]）
    func encode(_ text: String, maxLength: Int) -> EncodedText
    /// 编码 query-document 对（BERT 风格：[CLS] query [SEP] doc [SEP]）
    func encodePair(query: String, document: String, maxLength: Int) -> EncodedText
}

/// 分词器加载错误
enum TokenizerLoadError: LocalizedError {
    case noTokenizerFiles(String)
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .noTokenizerFiles(let directory):
            return "目录中未找到 tokenizer.json 或 vocab.txt：\(directory)"
        case .loadFailed(let message):
            return "分词器加载失败：\(message)"
        }
    }
}

/// 分词器加载：优先 tokenizer.json（swift-transformers AutoTokenizer），
/// 失败或缺失时回退到 vocab.txt（内置 WordPiece 实现）
enum TokenizerService {

    static func load(fromDirectory directory: URL) async throws -> TextTokenizer {
        let fileManager = FileManager.default
        let tokenizerJSON = directory.appendingPathComponent("tokenizer.json")
        let vocabTXT = directory.appendingPathComponent("vocab.txt")
        let hasJSON = fileManager.fileExists(atPath: tokenizerJSON.path)
        let hasVocab = fileManager.fileExists(atPath: vocabTXT.path)

        if hasJSON {
            do {
                let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
                return HFTextTokenizer(tokenizer: tokenizer)
            } catch {
                if hasVocab {
                    return try WordPieceTokenizer(directory: directory)
                }
                throw TokenizerLoadError.loadFailed(error.localizedDescription)
            }
        }

        if hasVocab {
            return try WordPieceTokenizer(directory: directory)
        }

        throw TokenizerLoadError.noTokenizerFiles(directory.path)
    }

    /// 依次在多个候选目录中查找分词器，返回第一个成功加载的及所用目录
    static func load(fromDirectories directories: [URL]) async throws -> (tokenizer: TextTokenizer, directory: URL) {
        for directory in directories {
            if let loaded = try? await load(fromDirectory: directory) {
                return (loaded, directory)
            }
        }
        let paths = directories.map { $0.path }.joined(separator: ", ")
        throw TokenizerLoadError.noTokenizerFiles(paths)
    }
}

/// 基于 swift-transformers AutoTokenizer 的分词器（tokenizer.json）
final class HFTextTokenizer: TextTokenizer, @unchecked Sendable {
    private let tokenizer: Tokenizers.Tokenizer
    private let clsId: Int
    private let sepId: Int

    var description: String { "tokenizer.json（AutoTokenizer）" }

    init(tokenizer: Tokenizers.Tokenizer) {
        self.tokenizer = tokenizer
        self.clsId = tokenizer.convertTokenToId("[CLS]") ?? 101
        self.sepId = tokenizer.convertTokenToId("[SEP]") ?? 102
    }

    func encode(_ text: String, maxLength: Int) -> EncodedText {
        var ids = tokenizer.encode(text: text, addSpecialTokens: true)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        return EncodedText(ids: ids)
    }

    func encodePair(query: String, document: String, maxLength: Int) -> EncodedText {
        let queryIds = tokenizer.encode(text: query, addSpecialTokens: false)
        let documentIds = tokenizer.encode(text: document, addSpecialTokens: false)
        return Self.assemblePair(
            queryIds: queryIds,
            documentIds: documentIds,
            clsId: clsId,
            sepId: sepId,
            maxLength: maxLength
        )
    }

    /// 按 BERT 对编码规范拼接：[CLS] query [SEP] doc [SEP]，
    /// 超长时从较长的一侧依次丢弃（longest-first 截断）
    static func assemblePair(
        queryIds: [Int],
        documentIds: [Int],
        clsId: Int,
        sepId: Int,
        maxLength: Int
    ) -> EncodedText {
        var a = queryIds
        var b = documentIds
        let budget = max(1, maxLength - 3)
        while a.count + b.count > budget, !a.isEmpty, !b.isEmpty {
            if a.count >= b.count {
                a.removeLast()
            } else {
                b.removeLast()
            }
        }

        var ids = [clsId]
        ids.append(contentsOf: a)
        ids.append(sepId)
        ids.append(contentsOf: b)
        ids.append(sepId)

        var typeIds = [Int](repeating: 0, count: a.count + 2)
        typeIds.append(contentsOf: [Int](repeating: 1, count: b.count + 1))

        return EncodedText(ids: ids, attentionMask: nil, tokenTypeIds: typeIds)
    }
}
