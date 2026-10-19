import Foundation

/// 内置 BERT WordPiece 分词器（vocab.txt 回退方案）
///
/// 实现 Hugging Face BertTokenizer 的核心流程：
/// BasicTokenizer（清洗、CJK 单字拆分、标点拆分、小写、去重音）
/// + WordPieceTokenizer（贪心最长匹配）。
final class WordPieceTokenizer: TextTokenizer, @unchecked Sendable {

    private let vocab: [String: Int]
    private let doLowerCase: Bool
    private let clsId: Int
    private let sepId: Int
    private let unkId: Int
    private let padId: Int
    private let maxInputCharsPerWord = 100

    var description: String { "vocab.txt（内置 WordPiece，共 \(vocab.count) 词）" }

    init(directory: URL) throws {
        let vocabURL = directory.appendingPathComponent("vocab.txt")
        guard let content = try? String(contentsOf: vocabURL, encoding: .utf8) else {
            throw TokenizerLoadError.noTokenizerFiles(directory.path)
        }

        var vocab: [String: Int] = [:]
        let lines = content.components(separatedBy: "\n")
        for (index, line) in lines.enumerated() {
            let token = line.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
            if !token.isEmpty && vocab[token] == nil {
                vocab[token] = index
            }
        }
        guard !vocab.isEmpty else {
            throw TokenizerLoadError.loadFailed("vocab.txt 为空")
        }
        self.vocab = vocab

        self.clsId = vocab["[CLS]"] ?? 101
        self.sepId = vocab["[SEP]"] ?? 102
        self.unkId = vocab["[UNK]"] ?? 100
        self.padId = vocab["[PAD]"] ?? 0

        // 读取 do_lower_case（默认 true，与 BERT 一致）
        var lower = true
        if let data = FileManager.default.contents(atPath: directory.appendingPathComponent("tokenizer_config.json").path),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let value = json["do_lower_case"] as? Bool
        {
            lower = value
        }
        self.doLowerCase = lower
    }

    var padTokenId: Int { padId }

    // MARK: - TextTokenizer

    func encode(_ text: String, maxLength: Int) -> EncodedText {
        var tokens = tokenizeToTokens(text)
        let limit = max(1, maxLength - 2)
        if tokens.count > limit {
            tokens = Array(tokens.prefix(limit))
        }
        var ids = [clsId]
        ids.append(contentsOf: tokens.map { vocab[$0] ?? unkId })
        ids.append(sepId)
        return EncodedText(ids: ids)
    }

    func encodePair(query: String, document: String, maxLength: Int) -> EncodedText {
        var a = tokenizeToTokens(query).map { vocab[$0] ?? unkId }
        var b = tokenizeToTokens(document).map { vocab[$0] ?? unkId }
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

    // MARK: - BasicTokenizer

    /// 完整基础分词流程：清洗、CJK 拆分、空白切分、（小写、去重音）、标点拆分、WordPiece
    func tokenizeToTokens(_ text: String) -> [String] {
        var cleaned = ""
        for character in text {
            if isControl(character) {
                continue
            }
            if character.unicodeScalars.allSatisfy({ CharacterSet.whitespacesAndNewlines.contains($0) }) {
                cleaned.append(" ")
            } else {
                cleaned.append(character)
            }
        }

        var tokens: [String] = []
        for token in splitOnCJK(cleaned).split(separator: " ", omittingEmptySubsequences: true) {
            var piece = String(token)
            if doLowerCase {
                piece = piece.lowercased()
                piece = stripAccents(piece)
            }
            for sub in splitOnPunctuation(piece) {
                tokens.append(contentsOf: wordPiece(sub))
            }
        }
        return tokens
    }

    /// 在 CJK 字符两侧补空格，使汉字被单独切分
    private func splitOnCJK(_ text: String) -> String {
        var output = ""
        for character in text {
            if isCJK(character) {
                output.append(" ")
                output.append(character)
                output.append(" ")
            } else {
                output.append(character)
            }
        }
        return output
    }

    /// 标点切分：ASCII 标点与 Unicode 标点类别均独立成词
    private func splitOnPunctuation(_ text: String) -> [String] {
        var output: [String] = []
        var current = ""
        for character in text {
            if isPunctuation(character) {
                if !current.isEmpty {
                    output.append(current)
                    current = ""
                }
                output.append(String(character))
            } else {
                current.append(character)
            }
        }
        if !current.isEmpty {
            output.append(current)
        }
        return output
    }

    /// 去除变音符号（NFD 后过滤组合标记）
    private func stripAccents(_ text: String) -> String {
        let decomposed = text.decomposedStringWithCanonicalMapping
        var result = ""
        for scalar in decomposed.unicodeScalars {
            if (0x0300...0x036F).contains(scalar.value) { continue }
            result.unicodeScalars.append(scalar)
        }
        return result
    }

    // MARK: - WordPieceTokenizer

    /// 贪心最长匹配：未登录子词回退为 [UNK]
    private func wordPiece(_ token: String) -> [String] {
        if token.count > maxInputCharsPerWord {
            return ["[UNK]"]
        }
        let characters = Array(token)
        var output: [String] = []
        var start = characters.startIndex
        let end = characters.endIndex

        while start < end {
            var currentEnd = end
            var matched: String?
            while start < currentEnd {
                var candidate = String(characters[start..<currentEnd])
                if start != characters.startIndex {
                    candidate = "##" + candidate
                }
                if vocab[candidate] != nil {
                    matched = candidate
                    break
                }
                currentEnd = characters.index(before: currentEnd)
            }
            if let matched {
                output.append(matched)
                start = currentEnd
            } else {
                output.append("[UNK]")
                start = characters.index(after: start)
            }
        }
        return output
    }

    // MARK: - 字符类别判断

    private func isControl(_ character: Character) -> Bool {
        for scalar in character.unicodeScalars {
            if scalar.value == 0 || scalar.value == 0xFFFD { return true }
            if CharacterSet.controlCharacters.contains(scalar) { return true }
        }
        return false
    }

    private func isCJK(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first, character.unicodeScalars.count == 1 else {
            return false
        }
        let code = scalar.value
        return (0x4E00...0x9FFF).contains(code)
            || (0x3400...0x4DBF).contains(code)
            || (0x20000...0x2A6DF).contains(code)
            || (0x2A700...0x2B73F).contains(code)
            || (0x2B740...0x2B81F).contains(code)
            || (0x2B820...0x2CEAF).contains(code)
            || (0xF900...0xFAFF).contains(code)
            || (0x2F800...0x2FA1F).contains(code)
    }

    private func isPunctuation(_ character: Character) -> Bool {
        for scalar in character.unicodeScalars {
            if (33...47).contains(scalar.value) { return true }
            if (58...64).contains(scalar.value) { return true }
            if (91...96).contains(scalar.value) { return true }
            if (123...126).contains(scalar.value) { return true }
            if CharacterSet.punctuationCharacters.contains(scalar) { return true }
        }
        return false
    }
}
