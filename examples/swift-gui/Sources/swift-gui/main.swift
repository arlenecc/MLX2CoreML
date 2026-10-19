// swift-gui —— ModConv 模型体验 App
//
// 支持三种后端加载并运行 Qwen3.5 等对话模型：
//   - CoreML：加载 modconv 转出的 .mlpackage / .mlmodelc（ANE 加速）
//   - MLX   ：通过 mlx-swift-lm 直接跑 MLX checkpoint（需 mlx-swift-lm）
//   - ONNX  ：加载通用自回归 decode 图（约定 input_ids/position/logits/
//             past_key_values.{l}.{k} → present.{l}.{k}，需 onnxruntime-swift）
//
// 分词统一用 swift-transformers 的 Swift Tokenizers（读 tokenizer.json），
// 替代原先的 Python 桥接（tok_helper.py）。MLX 后端由 mlx-swift-lm 内部用
// 同一套 Tokenizers 适配器处理。

import Cocoa
import CoreML
import Dispatch
import Foundation
import Combine
import UniformTypeIdentifiers
import Tokenizers           // swift-transformers
import OnnxRuntimeBindings  // ORTSession / ORTEnv / ORTValue
import MLXLMCommon          // ModelContainer / UserInput / GenerateParameters / Tokenizer
import MLXEmbedders         // mlx-swift-lm 内置：BertModel / BertRerankerModel 等 encoder
import MLX                  // MLXArray / Pooling 张量运算

// MARK: - 常量

let MAX_SEQ: Int32 = 8192

// 重复惩罚：对提示 + 已生成 token 降权，抑制贪心解码下的循环退化。
let REPETITION_PENALTY: Float = 1.15

// Qwen3.5 词表边界：id >= 248044 均为特殊/填充 token（含 <|im_end|>=248046）。
let TOKENIZER_VOCAB: Int = 248044
let TOK_IM_END: Int32 = 248046   // <|im_end|>：多轮时用于关闭上一轮助手回复

/// 轻量诊断日志：追加到 /tmp/modconv_gui.log，定位「发送无反应」类问题。
func glog(_ msg: String) {
    let line = "[\(Date())] \(msg)\n"
    let path = "/tmp/modconv_gui.log"
    if let h = FileHandle(forWritingAtPath: path) {
        h.seekToEndOfFile()
        if let d = line.data(using: .utf8) { h.write(d) }
        try? h.close()
    } else {
        try? line.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

// MARK: - Qwen3.5 聊天模板（CoreML / ONNX 后端手动套用；MLX 由自身 processor 处理）

/// 把单轮用户文本包成 Qwen3.5 的 chat 模板（含 <|im_start|>/<|im_end|> 特殊 token）。
/// 返回的字符串交给分词器时务必 addSpecialTokens:false，避免特殊 token 被重复加。
func qwenChatTemplate(_ userText: String) -> String {
    return "<|im_start|>user\n\(userText)<|im_end|>\n<|im_start|>assistant\n"
}

// MARK: - 分词封装（Swift Tokenizers，替代 Python 桥接 tok_helper.py）

/// 统一封装 swift-transformers 的 Swift Tokenizers。
/// CoreML / ONNX 后端经此类编解码；编码 addSpecialTokens:false（Qwen3.5 模板已含
/// 特殊 token，避免重复），解码 skipSpecialTokens:true。MLX 后端用其内部适配器。
final class SwiftTokenizer {
    private let inner: Tokenizers.Tokenizer

    init(_ inner: Tokenizers.Tokenizer) { self.inner = inner }

    /// 从含 tokenizer.json 的文件夹加载（swift-transformers AutoTokenizer）。
    static func load(folder: URL) async throws -> SwiftTokenizer {
        SwiftTokenizer(try await Tokenizers.AutoTokenizer.from(modelFolder: folder))
    }

    func encode(_ text: String) throws -> [Int32] {
        inner.encode(text: text, addSpecialTokens: false).map { Int32($0) }
    }

    /// 按分词器默认模板编码（附加 CLS/SEP 等特殊 token）。
    func encodeWithSpecialTokens(_ text: String) throws -> [Int32] {
        inner.encode(text: text, addSpecialTokens: true).map { Int32($0) }
    }

    /// 查特殊 token id（如 [CLS]/[SEP]），查不到用 fallback。
    func specialId(_ token: String, fallback: Int32) -> Int32 {
        if let id = inner.convertTokenToId(token) { return Int32(id) }
        return fallback
    }

    /// pad token id（XLM-R <pad>=1，BERT [PAD]=0），查不到返回 nil。
    var padTokenId: Int32? {
        if let id = inner.convertTokenToId("[PAD]") { return Int32(id) }
        if let id = inner.convertTokenToId("<pad>") { return Int32(id) }
        return nil
    }

    func decode(_ ids: [Int32]) throws -> String {
        inner.decode(tokens: ids.map { Int($0) }, skipSpecialTokens: true)
    }
}

// MARK: - 后端枚举与协议

enum BackendType: CaseIterable {
    case coreml, mlx, onnx
    var title: String {
        switch self {
        case .coreml: return "CoreML"
        case .mlx: return "MLX"
        case .onnx: return "ONNX"
        }
    }
}

enum BackendError: Error { case noModel, notLoaded, unsupportedKind(String) }

// MARK: - 模型类型（对话 / 向量 / 重排）

enum ModelKind: String {
    case chat, embedding, reranker
    var title: String {
        switch self {
        case .chat: return "对话"
        case .embedding: return "向量"
        case .reranker: return "重排"
        }
    }
}

/// 依据 MLX checkpoint 的 config.json 判定模型类型。
/// 与 MLXEmbedders 的 EmbedderTypeRegistry 一致：bert/xlm-roberta 等纯 encoder 视为
/// embedding/reranker（architectures 含 ForSequenceClassification → reranker），
/// 其余（qwen3_5/gemma3 等）仍按对话模型处理。
func detectModelKind(configDir: URL) -> ModelKind {
    guard let data = try? Data(contentsOf: configDir.appendingPathComponent("config.json")),
          let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        return .chat
    }
    let modelType = obj["model_type"] as? String ?? ""
    let encoderTypes: Set<String> = ["bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert"]
    guard encoderTypes.contains(modelType) else { return .chat }
    let archs = (obj["architectures"] as? [String]) ?? []
    if archs.contains(where: { $0.contains("ForSequenceClassification") }) { return .reranker }
    return .embedding
}

/// 各后端的统一接口：ChatController 持有其一并驱动对话。
protocol ModelBackend {
    var backendName: String { get }
    var modelURL: URL? { get set }
    var tokenizerURL: URL? { get set }       // CoreML / ONNX 使用；MLX 用模型内 tokenizer
    var computeUnit: MLComputeUnits { get set } // 仅 CoreML 有意义
    func load() async throws
    func resetKV()
    /// 生成回复；返回 (回复文本, 生成 token 数)。
    func generate(userText: String, maxNew: Int, penalty: Float) throws -> (String, Int)
    /// 加载完成后模型类型（chat/embedding/reranker），ChatController 依此切换 UI 与路由。
    var kind: ModelKind { get }
    /// 计算文本向量（L2 归一化）。仅 embedding 模型实现。
    func embed(_ text: String) throws -> [Float]
    /// query-doc 相关性打分（原始 logit）。仅 reranker 模型实现。
    func score(query: String, doc: String) throws -> Double
}

extension ModelBackend {
    var kind: ModelKind { .chat }
    func embed(_ text: String) throws -> [Float] {
        throw BackendError.unsupportedKind("当前模型不是向量模型")
    }
    func score(query: String, doc: String) throws -> Double {
        throw BackendError.unsupportedKind("当前模型不是重排模型")
    }
}

// MARK: - CoreML 后端

final class CoreMLBackend: ModelBackend {
    let backendName = "CoreML"
    var modelURL: URL?
    var tokenizerURL: URL?
    var computeUnit: MLComputeUnits = .all

    private var model: MLModel?
    private var state: MLState?
    private var gdnStates: [String: MLMultiArray] = [:]
    private var vocab: Int = 0
    private var compiledURL: URL?
    private var tokenizer: SwiftTokenizer?
    private var pos: Int32 = 0   // 跨轮持续累加（KV 不重置）
    /// 非 nil 时为整段序列 encoder（embedding/reranker）模式。
    private var seqCore: CoreMLSeqCore?

    /// tokenizerURL 可能是 tokenizer.json 文件（GUI 选择）或目录（无头测试）。
    static func tokenizerFolder(from tokenizerURL: URL?, modelURL: URL) -> URL {
        guard let t = tokenizerURL else { return modelURL.deletingLastPathComponent() }
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: t.path, isDirectory: &isDir), isDir.boolValue {
            return t
        }
        return t.deletingLastPathComponent()
    }

    /// 从模型输入描述构造初始（全零）GDN 状态。
    private static func makeInitialStates(_ m: MLModel) -> [String: MLMultiArray] {
        var gs: [String: MLMultiArray] = [:]
        for (name, constraint) in m.modelDescription.inputDescriptionsByName {
            guard let shape = constraint.multiArrayConstraint?.shape,
                  name != "input_ids", name != "cache_position" else { continue }
            if let arr = try? MLMultiArray(shape: shape, dataType: .float32) { gs[name] = arr }
        }
        return gs
    }

    func load() async throws {
        guard let url = modelURL else { throw BackendError.noModel }
        let m = try loadConvertedModel(at: url, units: computeUnit)
        // encoder 检测：带 attention_mask/token_type_ids 输入，或 input_ids 为 [1, L>1]
        // （对话模型是 [1,1] 逐 token + GDN 状态；柔性形状可能读出 [1,1]，故看输入名）
        let ins = m.modelDescription.inputDescriptionsByName
        let idsShape = ins["input_ids"]?.multiArrayConstraint?.shape
        let isEncoder = ins["attention_mask"] != nil || ins["token_type_ids"] != nil
            || (idsShape.map { $0.count >= 2 && $0[1].intValue > 1 } == true)
        if isEncoder {
            let core = try await CoreMLSeqCore(
                model: m, tokenizerFolder: Self.tokenizerFolder(from: tokenizerURL, modelURL: url))
            seqCore = core
            tokenizer = core.swiftTokenizer
            glog("coreml encoder mode: kind=\(core.kind) seqLen=\(core.seqLen) inputs=\(core.inputNames.sorted())")
            return
        }
        seqCore = nil
        model = m; vocab = _readVocab(m)
        state = m.makeState()
        gdnStates = Self.makeInitialStates(m)
        pos = 0
        // 预热：触发 ANE 特化（首次可能 1–2 分钟），放在加载期完成，
        // 避免用户首次发送消息时长时间无响应。
        // 注意：必须用一次性 MLState——GDN 是循环状态，喂 token 会推进状态，
        // 用真状态预热会污染后续对话的初始 KV。
        if !gdnStates.isEmpty {
            _ = try forwardWithConvertedModel(m, m.makeState(), token: 0, position: 0,
                                              gdnStates: gdnStates)
        }
        // tokenizer：优先显式选的 tokenizer.json（或其目录），否则用模型文件夹。
        let tokFolder = Self.tokenizerFolder(from: tokenizerURL, modelURL: url)
        tokenizer = try await SwiftTokenizer.load(folder: tokFolder)
    }

    func resetKV() {
        guard let m = model else { return }
        state = m.makeState()
        gdnStates = Self.makeInitialStates(m)
        pos = 0
    }

    func generate(userText: String, maxNew: Int, penalty: Float) throws -> (String, Int) {
        guard let model = model, let tok = tokenizer, let st = state else { throw BackendError.notLoaded }
        let ids = try tok.encode(qwenChatTemplate(userText))
        glog("coreml generate: promptTok=\(ids.count) pos=\(pos) vocab=\(vocab) states=\(gdnStates.count)")
        guard !ids.isEmpty else { return ("（分词为空，请检查 prompt）", 0) }
        var gs = gdnStates
        var last: MLMultiArray?
        // 1) 多轮：先补一个 <|im_end|> 关闭上一轮助手回复，保证对话格式良好
        if pos > 0 {
            let o = try forwardWithConvertedModel(model, st, token: TOK_IM_END, position: pos, gdnStates: gs)
            gs = o.newStates; last = o.logits; pos += 1
        }
        // 2) prefill：把（模板化后的）用户输入逐 token 喂入（KV 累加）
        for t in ids {
            let o = try forwardWithConvertedModel(model, st, token: t, position: pos, gdnStates: gs)
            gs = o.newStates; last = o.logits; pos += 1
        }
        gdnStates = gs
        // 3) 自回归生成（decode）；id >= TOKENIZER_VOCAB 均为特殊/填充 token（含 <|im_end|>）即停
        var gen: [Int32] = []
        var ctx: [Int32] = ids
        for _ in 0..<maxNew {
            guard let cur = last else { break }
            let penalized = Set(ctx)
            let nxt = argmaxFromFloat(mlmultiArrayToFloatArray(cur), vocab: vocab,
                                       penalty: penalty, penalized: penalized)
            if Int(nxt) >= TOKENIZER_VOCAB { break }
            gen.append(nxt)
            let o = try forwardWithConvertedModel(model, st, token: nxt, position: pos, gdnStates: gdnStates)
            gdnStates = o.newStates; last = o.logits; pos += 1
            ctx.append(nxt)
        }
        let text = try tok.decode(gen)
        glog("coreml decode: genTok=\(gen.count) replyLen=\(text.count) firstIds=\(gen.prefix(5))")
        return (text, gen.count)
    }

    var kind: ModelKind { seqCore?.kind ?? .chat }

    func embed(_ text: String) throws -> [Float] {
        guard let core = seqCore else {
            throw BackendError.unsupportedKind("该 CoreML 模型是对话模型，不支持向量计算")
        }
        return try core.embed(text)
    }

    func score(query: String, doc: String) throws -> Double {
        guard let core = seqCore else {
            throw BackendError.unsupportedKind("该 CoreML 模型是对话模型，不支持重排打分")
        }
        return try core.score(query: query, doc: doc)
    }
}

// MARK: - ONNX 后端（onnxruntime-swift）

/// ONNX decode 图的 I/O 命名约定；如导出图不同，改这里即可。
struct OnnxGraphSpec {
    var inputIds = "input_ids"
    var position = "position"
    var logits = "logits"
    var pastPrefix = "past_key_values"   // 输入名： "\(pastPrefix).\(layer).\(k)"，k ∈ {key,value}
    var presentPrefix = "present"        // 输出名："\(presentPrefix).\(layer).\(k)"
    // 首步 past KV 张量形状；按你的导出图调整（Qwen0.8B：8 KV 头、128 头维、0 已缓存）。
    var initialKVShape: [Int] = [1, 8, 0, 128]
}

final class OnnxBackend: ModelBackend {
    let backendName = "ONNX"
    var modelURL: URL?
    var tokenizerURL: URL?
    var computeUnit: MLComputeUnits = .all

    private var env: ORTEnv?
    private var session: ORTSession?
    private var tokenizer: SwiftTokenizer?
    private var spec = OnnxGraphSpec()
    /// (past 输入名, present 输出名) 配对；layer 交给命名约定。
    private var kvPairs: [(past: String, present: String)] = []
    private var pos: Int32 = 0
    /// 非 nil 时为整段序列 encoder（embedding/reranker）模式。
    private var encoder: OnnxEncoderCore?

    func load() async throws {
        guard let url = modelURL else { throw BackendError.noModel }
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        session = try ORTSession(env: env!, modelPath: url.path, sessionOptions: opts)
        configureKVNames()
        let tokFolder = CoreMLBackend.tokenizerFolder(from: tokenizerURL, modelURL: url.deletingLastPathComponent())
        tokenizer = try await SwiftTokenizer.load(folder: tokFolder)
        // 无 KV past 输入 → 整段 encoder 图（embedding/reranker）
        if kvPairs.isEmpty {
            encoder = try OnnxEncoderCore(session: session!, tokenizer: tokenizer!)
            glog("onnx encoder mode: kind=\(encoder!.kind) inputs=\(encoder!.inputNames.sorted()) outputs=\(encoder!.outputNames)")
        }
    }

    var kind: ModelKind { encoder?.kind ?? .chat }

    func embed(_ text: String) throws -> [Float] {
        guard let enc = encoder else {
            throw BackendError.unsupportedKind("该 ONNX 模型是对话模型，不支持向量计算")
        }
        return try enc.embed(text)
    }

    func score(query: String, doc: String) throws -> Double {
        guard let enc = encoder else {
            throw BackendError.unsupportedKind("该 ONNX 模型是对话模型，不支持重排打分")
        }
        return try enc.score(query: query, doc: doc)
    }

    /// 扫描 session 输入/输出名，按 spec 前缀配对每层 key/value 张量。
    private func configureKVNames() {
        kvPairs.removeAll()
        guard let session = session else { return }
        let inputs = (try? session.inputNames()) ?? []
        let outputs = (try? session.outputNames()) ?? []
        for name in inputs where name.hasPrefix(spec.pastPrefix) {
            let comps = name.dropFirst(spec.pastPrefix.count).components(separatedBy: ".")
            guard comps.count == 2, let k = comps.last, ["key", "value"].contains(k) else { continue }
            let layer = comps[0]
            let present = "\(spec.presentPrefix).\(layer).\(k)"
            guard outputs.contains(present) else { continue }
            kvPairs.append((name, present))
        }
    }

    func resetKV() { pos = 0 }

    func generate(userText: String, maxNew: Int, penalty: Float) throws -> (String, Int) {
        guard let session = session, let tok = tokenizer else { throw BackendError.notLoaded }
        let ids = try tok.encode(qwenChatTemplate(userText))
        var past: [String: (NSMutableData, [NSNumber])] = [:]
        var lastLogits: [Float]?
        var lastPresent: [String: (NSMutableData, [NSNumber])] = [:]
        var p: Int32 = 0
        // prefill
        for t in ids {
            let r = try step(token: t, position: p, past: &past)
            lastLogits = r.logits; lastPresent = r.present; p += 1
        }
        var gen: [Int32] = []
        var ctx: [Int32] = ids
        for _ in 0..<maxNew {
            guard let cur = lastLogits else { break }
            let penalized = Set(ctx)
            let nxt = argmaxFromFloat(cur, vocab: cur.count, penalty: penalty, penalized: penalized)
            if Int(nxt) >= TOKENIZER_VOCAB { break }
            gen.append(nxt)
            let r = try step(token: nxt, position: p, past: &lastPresent)
            lastLogits = r.logits; lastPresent = r.present; p += 1
            ctx.append(nxt)
        }
        let text = try tok.decode(gen)
        return (text, gen.count)
    }

    /// 单步：喂 token/position + 过往 KV，返回 logits 与本步 present KV。
    private func step(token: Int32, position: Int32,
                      past: inout [String: (NSMutableData, [NSNumber])]) throws
        -> (logits: [Float], present: [String: (NSMutableData, [NSNumber])]) {
        guard let session = session else { throw BackendError.notLoaded }
        var inputs: [String: ORTValue] = [:]
        inputs[spec.inputIds] = try ortInt64(shape: [1, 1], values: [Int64(token)])
        inputs[spec.position] = try ortInt64(shape: [1], values: [Int64(position)])
        for (pastName, presentName) in kvPairs {
            if let (d, sh) = past[presentName] {
                inputs[pastName] = try ORTValue(tensorData: NSMutableData(data: d as Data),
                                                elementType: .float, shape: sh)
            } else {
                // 首步无过往 KV：用约定初始形状建零张量（形状按导出图调整）。
                let sh = spec.initialKVShape.map { NSNumber(value: $0) }
                inputs[pastName] = try ORTValue(tensorData: NSMutableData(length: shapeByteCount(sh))!,
                                                elementType: .float, shape: sh)
            }
        }
        let outputNames = Set([spec.logits] + kvPairs.map { $0.present })
        let outputs = try session.run(withInputs: inputs, outputNames: outputNames, runOptions: nil)

        guard let logitsValue = outputs[spec.logits] else { throw BackendError.notLoaded }
        let logits = floatArray(from: try logitsValue.tensorData())

        var present: [String: (NSMutableData, [NSNumber])] = [:]
        for (_, presentName) in kvPairs {
            guard let v = outputs[presentName] else { continue }
            let md = try v.tensorData()
            let sh = (try? v.tensorTypeAndShapeInfo().shape) ?? []
            present[presentName] = (md, sh)
        }
        return (logits, present)
    }

    private func ortInt64(shape: [NSNumber], values: [Int64]) throws -> ORTValue {
        let data = values.withUnsafeBytes { Data($0) }
        return try ORTValue(tensorData: NSMutableData(data: data), elementType: .int64, shape: shape)
    }

    /// 把 float32 的 NSMutableData 转成 [Float]（memcpy，避免桥转开销）。
    private func floatArray(from data: NSMutableData) -> [Float] {
        let count = data.length / MemoryLayout<Float>.stride
        return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            memcpy(buffer.baseAddress, data.bytes, data.length)
            initialized = count
        }
    }

    private func shapeByteCount(_ shape: [NSNumber]) -> Int {
        let elems = shape.reduce(1) { $0 * max($1.intValue, 1) }
        return elems * 4
    }
}

// MARK: - ONNX 整段 encoder 核心（embedding/reranker）

/// 通用 ONNX encoder 推理：input_ids/attention_mask（可选 token_type_ids），
/// 输出 logits → reranker；sentence_embedding 或 [1,L,H] hidden → mean-pool 向量。
final class OnnxEncoderCore {
    let kind: ModelKind
    let inputNames: Set<String>
    let outputNames: [String]
    private let session: ORTSession
    private let tokenizer: SwiftTokenizer
    private let logitsOut: String?
    private let sentEmbOut: String?
    private let hiddenOut: String?

    init(session: ORTSession, tokenizer: SwiftTokenizer) throws {
        self.session = session
        self.tokenizer = tokenizer
        let ins = Set((try? session.inputNames()) ?? [])
        let outs = (try? session.outputNames()) ?? []
        inputNames = ins
        outputNames = outs
        let logits = outs.first { $0.lowercased().contains("logit") }
        let sent = outs.first { $0.lowercased().contains("sentence_embedding") }
        let hidden = outs.first {
            let n = $0.lowercased()
            return n.contains("hidden_state") || n.contains("token_embeddings") || n.contains("last_hidden")
        } ?? outs.first { $0 != logits }
        logitsOut = logits
        sentEmbOut = sent
        hiddenOut = hidden
        kind = logits != nil ? .reranker : .embedding
    }

    private func int64Value(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        let nsShape = shape.map { NSNumber(value: $0) }
        let data = values.withUnsafeBytes { Data($0) }
        return try ORTValue(tensorData: NSMutableData(data: data),
                            elementType: .int64, shape: nsShape)
    }

    /// 整段推理，返回 float 输出（含形状）。
    private func run(ids: [Int32], tt: [Int32]?) throws -> (values: [Float], shape: [Int]) {
        let L = ids.count
        var inputs: [String: ORTValue] = [
            "input_ids": try int64Value(ids.map { Int64($0) }, shape: [1, L]),
            "attention_mask": try int64Value([Int64](repeating: 1, count: L), shape: [1, L]),
        ]
        if inputNames.contains("token_type_ids") {
            // 图强制要求该输入；缺省补全零
            let values = tt ?? [Int32](repeating: 0, count: L)
            inputs["token_type_ids"] = try int64Value(values.map { Int64($0) }, shape: [1, L])
        }
        let names = Set([logitsOut, sentEmbOut, hiddenOut].compactMap { $0 })
        let outs = try session.run(withInputs: inputs, outputNames: names, runOptions: nil)
        let name = logitsOut ?? sentEmbOut ?? hiddenOut ?? ""
        guard let v = outs[name] else { throw BackendError.unsupportedKind("缺少输出 \(name)") }
        let shape = ((try? v.tensorTypeAndShapeInfo().shape) ?? []).map { $0.intValue }
        let md = try v.tensorData()
        let count = md.length / MemoryLayout<Float>.stride
        let arr = [Float](unsafeUninitializedCapacity: count) { buf, ok in
            memcpy(buf.baseAddress, md.bytes, md.length); ok = count
        }
        return (arr, shape)
    }

    func embed(_ text: String) throws -> [Float] {
        let ids = try tokenizer.encodeWithSpecialTokens(text)
        let (vals, shape) = try run(ids: ids, tt: nil)
        if shape.count == 2 {
            var v = vals
            l2Normalize(&v)
            return v
        }
        // [1, L, H] mean pooling
        let L = shape.count >= 2 ? shape[1] : 0
        let H = shape.count >= 3 ? shape[2] : vals.count / max(L, 1)
        let realLen = min(ids.count, L)
        var acc = [Float](repeating: 0, count: H)
        for l in 0..<realLen { for h in 0..<H { acc[h] += vals[l * H + h] } }
        if realLen > 0 { for h in 0..<H { acc[h] /= Float(realLen) } }
        l2Normalize(&acc)
        return acc
    }

    /// query-doc 打分：BERT 段对格式 [CLS] q [SEP] d [SEP] + token_type_ids 0/1。
    func score(query: String, doc: String) throws -> Double {
        guard let logitsOut = logitsOut else {
            throw BackendError.unsupportedKind("ONNX 图没有 logits 输出，无法打分")
        }
        let q = try tokenizer.encode(query)
        let d = try tokenizer.encode(doc)
        let cls = tokenizer.specialId("[CLS]", fallback: 101)
        let sep = tokenizer.specialId("[SEP]", fallback: 102)
        var toks: [Int32] = [cls] + q + [sep] + d + [sep]
        var tt: [Int32] = [Int32](repeating: 0, count: q.count + 2)
            + [Int32](repeating: 1, count: d.count + 1)
        if toks.count > 512 { toks = Array(toks.prefix(512)); tt = Array(tt.prefix(512)) }
        let L = toks.count
        var inputs: [String: ORTValue] = [
            "input_ids": try int64Value(toks.map { Int64($0) }, shape: [1, L]),
            "attention_mask": try int64Value([Int64](repeating: 1, count: L), shape: [1, L]),
        ]
        if inputNames.contains("token_type_ids") {
            inputs["token_type_ids"] = try int64Value(tt.map { Int64($0) }, shape: [1, L])
        }
        let outs = try session.run(withInputs: inputs, outputNames: [logitsOut], runOptions: nil)
        guard let v = outs[logitsOut] else { throw BackendError.unsupportedKind("缺少输出 \(logitsOut)") }
        let md = try v.tensorData()
        let count = md.length / 4
        let arr = [Float](unsafeUninitializedCapacity: count) { buf, initialized in
            memcpy(buf.baseAddress, md.bytes, md.length); initialized = count
        }
        guard let first = arr.first else { throw BackendError.unsupportedKind("logits 为空") }
        return Double(first)
    }

    private func l2Normalize(_ v: inout [Float]) {
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
        if n > 0 { for i in v.indices { v[i] /= n } }
    }
}

// MARK: - MLX 后端（mlx-swift-lm）

/// 把 swift-transformers 的分词器适配为 mlx-swift-lm 的 Tokenizer 协议。
struct TokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: Tokenizers.Tokenizer
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(messages: messages, tools: tools,
                                                  additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

/// mlx-swift-lm 的 TokenizerLoader：用 swift-transformers 从模型文件夹加载 tokenizer。
struct TransformersLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream: upstream)
    }
}

final class MLXBackend: ModelBackend {
    let backendName = "MLX"
    var modelURL: URL?
    var tokenizerURL: URL?
    var computeUnit: MLComputeUnits = .all

    private var container: ModelContainer?

    func load() async throws {
        guard let dir = modelURL else { throw BackendError.noModel }
        container = try await loadModelContainer(from: dir, using: TransformersLoader())
    }

    func resetKV() {}

    func generate(userText: String, maxNew: Int, penalty: Float) throws -> (String, Int) {
        guard let c = container else { throw BackendError.notLoaded }
        var text = ""
        var err: Error?
        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                let params = GenerateParameters(maxTokens: maxNew, repetitionPenalty: penalty)
                let input = try await c.prepare(input: UserInput(prompt: userText))
                let stream = try await c.generate(input: input, parameters: params)
                for await gen in stream {
                    if case .chunk(let t) = gen { text += t }
                }
            } catch { err = error }
            sem.signal()
        }
        sem.wait()
        if let e = err { throw e }
        return (text, 0)
    }
}

// MARK: - 共享 argmax（含重复惩罚）

func argmaxFromFloat(_ logits: [Float], vocab: Int, penalty: Float,
                     penalized: Set<Int32>?) -> Int32 {
    let v = min(vocab, logits.count)
    var best = 0
    var bestVal = -Double.greatestFiniteMagnitude
    let p = Double(penalty)
    for i in 0..<v {
        var val = Double(logits[i])
        if p != 1.0, let s = penalized, s.contains(Int32(i)) {
            val = val > 0 ? val / p : val * p
        }
        if val > bestVal { bestVal = val; best = i }
    }
    return Int32(best)
}

// MARK: - MLX Embedder/Reranker 后端（mlx-swift-lm 内置 MLXEmbedders）

/// 走 MLXEmbedders 的 encoder 模型：bert/xlm-roberta（BertModel/BertRerankerModel）、
/// nomic_bert、qwen3、gemma3、lfm2 等。embedding 输出 L2 归一化向量；
/// reranker 对 query-doc 段对打分（原始 logit）。
final class MLXEmbedBackend: ModelBackend {
    let backendName = "MLX"
    var modelURL: URL?
    var tokenizerURL: URL?
    var computeUnit: MLComputeUnits = .all

    private(set) var kind: ModelKind = .embedding
    private var container: EmbedderModelContainer?
    private var modelType = ""
    private var maxLen = 512

    func load() async throws {
        guard let dir = modelURL else { throw BackendError.noModel }
        // 类型识别 + 截断长度
        kind = detectModelKind(configDir: dir)
        if let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
           let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            modelType = obj["model_type"] as? String ?? ""
            if let mpe = obj["max_position_embeddings"] as? Int { maxLen = min(max(mpe, 8), 8192) }
        }
        glog("mlx embed load: kind=\(kind) modelType=\(modelType) maxLen=\(maxLen)")
        container = try await EmbedderModelFactory.shared.loadContainer(
            from: dir, using: TransformersLoader())
    }

    func resetKV() {}

    func generate(userText: String, maxNew: Int, penalty: Float) throws -> (String, Int) {
        throw BackendError.unsupportedKind("该 MLX 模型是 encoder（embedding/reranker），不支持对话生成")
    }

    func embed(_ text: String) throws -> [Float] {
        guard let c = container else { throw BackendError.notLoaded }
        var out: [Float] = []
        var err: Error?
        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                out = try await c.perform { ctx -> [Float] in
                    var toks = ctx.tokenizer.encode(text: text, addSpecialTokens: true)
                    toks = Array(toks.prefix(self.maxLen))
                    if toks.isEmpty { toks = [0] }
                    let input = MLXArray(toks.map { Int32($0) }).reshaped([1, toks.count])
                    let mask = MLXArray.ones([1, toks.count])
                    let o = ctx.model(input, positionIds: nil, tokenTypeIds: nil,
                                      attentionMask: mask)
                    // 模型目录缺 1_Pooling/config.json 时 Pooling 默认 .none；bge 系用 CLS
                    var strategy = ctx.pooling.strategy
                    if strategy == .none { strategy = .cls }
                    let vec = Pooling(strategy: strategy)(o, mask: mask, normalize: true)
                    return vec.asArray(Float.self)
                }
            } catch { err = error }
            sem.signal()
        }
        sem.wait()
        if let e = err { throw e }
        return out
    }

    func score(query: String, doc: String) throws -> Double {
        guard let c = container else { throw BackendError.notLoaded }
        var out = 0.0
        var err: Error?
        let sem = DispatchSemaphore(value: 0)
        let mt = modelType
        Task {
            do {
                out = try await c.perform { ctx -> Double in
                    let tok = ctx.tokenizer
                    let q = tok.encode(text: query, addSpecialTokens: false)
                    let d = tok.encode(text: doc, addSpecialTokens: false)
                    let input: MLXArray
                    let tt: MLXArray?
                    if mt == "bert" {
                        // [CLS] q [SEP] d [SEP]，token_type_ids 0/1
                        let cls = Self.specialId("[CLS]", 101, tok)
                        let sep = Self.specialId("[SEP]", 102, tok)
                        let (qq, dd) = Self.truncatePair(q, d, budget: self.maxLen - 3)
                        let ids = [cls] + qq + [sep] + dd + [sep]
                        tt = MLXArray(([Int32](repeating: 0, count: qq.count + 2)
                            + [Int32](repeating: 1, count: dd.count + 1)).map { $0 })
                            .reshaped([1, ids.count])
                        input = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
                    } else {
                        // xlm-roberta/roberta：<s> q </s></s> d </s>
                        let bos = Self.specialId("<s>", 0, tok)
                        let eos = Self.specialId("</s>", 2, tok)
                        let (qq, dd) = Self.truncatePair(q, d, budget: self.maxLen - 4)
                        let ids = [bos] + qq + [eos, eos] + dd + [eos]
                        tt = nil
                        input = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
                    }
                    let mask = MLXArray.ones([1, input.shape[1]])
                    let model = ctx.model
                    if let rm = model as? BertRerankerModel {
                        let logits = rm.score(input, positionIds: nil, tokenTypeIds: tt,
                                              attentionMask: mask)
                        return Double(logits[0, 0].item(Float.self))
                    }
                    if let rm = model as? BertSequenceClassificationRerankerModel {
                        let logits = rm.score(input, positionIds: nil, tokenTypeIds: tt,
                                              attentionMask: mask)
                        return Double(logits[0, 0].item(Float.self))
                    }
                    throw BackendError.unsupportedKind(
                        "模型类型 \(mt) 不支持打分（无 sequence-classification 头）")
                }
            } catch { err = error }
            sem.signal()
        }
        sem.wait()
        if let e = err { throw e }
        return out
    }

    private static func specialId(_ token: String, _ fallback: Int,
                                  _ tok: any MLXLMCommon.Tokenizer) -> Int {
        if let id = tok.convertTokenToId(token) { return id }
        let ids = tok.encode(text: token, addSpecialTokens: false)
        if ids.count == 1, let id = ids.first { return id }
        return fallback
    }

    /// 段对截断：优先保 query，剩余预算给 doc。
    private static func truncatePair(_ q: [Int], _ d: [Int], budget: Int)
        -> (first: [Int], second: [Int]) {
        let b = max(budget, 8)
        if q.count + d.count <= b { return (q, d) }
        let qq = Array(q.prefix(max(b / 2, 4)))
        let dd = Array(d.prefix(max(b - qq.count, 4)))
        return (qq, dd)
    }
}

// MARK: - ChatController

final class ChatController: ObservableObject {
    @Published var status = "未加载"
    @Published var loaded = false
    @Published var modelName = ""
    @Published var loadedName = ""
    @Published var tokenizerName = ""
    @Published var transcript = "对话将在模型加载后开始。\n在下方输入内容并点「发送」即可体验多轮对话。\n"
    @Published var running = false
    @Published var kind: ModelKind = .chat   // 加载成功后由后端回填，驱动 UI 切换

    var backendType: BackendType = .coreml
    var computeUnit: MLComputeUnits = .all

    private var backend: ModelBackend?
    private var selectedModelURL: URL?
    private var selectedTokenizerURL: URL?

    func selectModel(url: URL) {
        selectedModelURL = url
        modelName = url.lastPathComponent
        status = "已选择：\(url.lastPathComponent)｜点「加载模型」载入"
    }

    func selectTokenizer(url: URL) {
        selectedTokenizerURL = url
        tokenizerName = url.lastPathComponent
        status = "已选分词器：\(url.lastPathComponent)（与所选模型配合）"
    }

    // MARK: 加载

    func loadSelected() {
        guard let url = selectedModelURL else {
            status = "请先点「选择模型」选取模型文件/文件夹"
            return
        }
        glog("load start: backend=\(backendType) model=\(url.path) tokenizer=\(selectedTokenizerURL?.path ?? "nil") cu=\(computeUnit)")
        var backend: ModelBackend
        switch backendType {
        case .coreml: backend = CoreMLBackend()
        case .mlx:
            // encoder（embedding/reranker）走 MLXEmbedders 后端；对话模型走原 LLM 后端
            backend = detectModelKind(configDir: url) == .chat ? MLXBackend() : MLXEmbedBackend()
        case .onnx: backend = OnnxBackend()
        }
        backend.modelURL = url
        backend.tokenizerURL = selectedTokenizerURL
        backend.computeUnit = computeUnit
        self.backend = backend

        Task.detached(priority: .userInitiated) { [self] in
            do {
                try await backend.load()
                glog("load OK: backend=\(backend.backendName) kind=\(backend.kind)")
                DispatchQueue.main.async {
                    self.loaded = true
                    self.loadedName = url.lastPathComponent
                    self.kind = backend.kind
                    self.transcript = self.introText(for: backend.kind, backend: backend.backendName)
                    self.status = "已加载（\(backend.backendName)·\(backend.kind.title)）：\(url.lastPathComponent)"
                    // embedding/reranker 加载成功后自动跑 3 条示例并显示结果
                    if backend.kind != .chat {
                        self.runAutoSamples(backend: backend)
                    }
                }
            } catch {
                glog("load FAIL: \(error)")
                DispatchQueue.main.async {
                    self.status = "加载失败：\(error.localizedDescription)"
                }
            }
        }
    }

    func reset() {
        backend?.resetKV()
        transcript = kind == .chat ? "（对话已清空）\n" : "（已清空）\n"
    }

    /// 按模型类型生成加载后的提示文案。
    private func introText(for k: ModelKind, backend name: String) -> String {
        switch k {
        case .chat:
            return "对话模型已加载（\(name)）。\n请在下方输入内容并点「发送」来测试多轮对话。\n"
        case .embedding:
            return "向量模型已加载（\(name)）。正在用 3 条示例文本自动测试…\n完成后可在下方输入自己的文本点「计算向量」。\n"
        case .reranker:
            return "重排模型已加载（\(name)）。正在用 3 组查询-文档对自动测试…\n完成后可在上方输入查询、下方输入文档，点「打分」测试。\n"
        }
    }

    // MARK: 示例自测（embedding/reranker 加载后自动执行）

    /// 加载成功后自动跑 3 条合适示例并逐条追加到对话区；期间锁定发送按钮。
    func runAutoSamples(backend: ModelBackend) {
        DispatchQueue.main.async { self.running = true }
        DispatchQueue.global(qos: .userInitiated).async {
            let t0 = Date()
            func emit(_ line: String) {
                glog("auto: \(line.trimmingCharacters(in: .whitespacesAndNewlines))")
                DispatchQueue.main.async { self.transcript += line }
            }
            emit("\n—— 自动测试（3 条示例）——\n")

            switch self.kind {
            case .embedding:
                let samples = [
                    "机器学习是人工智能的一个分支。",
                    "今天天气很好，适合去公园散步。",
                    "工程师正在调试服务器集群。",
                ]
                var vecs: [[Float]] = []
                var failed = false
                for (i, s) in samples.enumerated() {
                    do {
                        let t = Date()
                        let v = try backend.embed(s)
                        vecs.append(v)
                        let dt = Date().timeIntervalSince(t)
                        let norm = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
                        emit(String(format: "%d. 「%@」→ 维度 %d，L2 %.4f，%.0f ms\n",
                                    i + 1, s, v.count, norm, dt * 1000))
                    } catch {
                        emit("\(i + 1). 「\(s)」→ 出错：\(error)\n")
                        failed = true
                        break
                    }
                }
                // 语义相似度演示：第 1 条与其余两条的余弦相似度
                if !failed, vecs.count == samples.count {
                    func cosine(_ a: [Float], _ b: [Float]) -> Double {
                        guard a.count == b.count, !a.isEmpty else { return 0 }
                        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
                        let na = (a.reduce(0) { $0 + $1 * $1 }).squareRoot()
                        let nb = (b.reduce(0) { $0 + $1 * $1 }).squareRoot()
                        return na > 0 && nb > 0 ? Double(dot / (na * nb)) : 0
                    }
                    let s12 = cosine(vecs[0], vecs[1])
                    let s13 = cosine(vecs[0], vecs[2])
                    emit(String(format: "   余弦相似度：文本1↔文本2 = %.4f，文本1↔文本3 = %.4f\n", s12, s13))
                }
            case .reranker:
                let pairs: [(query: String, doc: String)] = [
                    ("什么是机器学习", "机器学习是人工智能的一个分支，通过数据训练模型。"),
                    ("什么是机器学习", "这家餐厅的红烧肉非常有名，每天都排长队。"),
                    ("如何烤蛋糕", "烤蛋糕需要准备面粉、鸡蛋、糖和烤箱。"),
                ]
                var failed = false
                for (i, p) in pairs.enumerated() {
                    do {
                        let t = Date()
                        let s = try backend.score(query: p.query, doc: p.doc)
                        let dt = Date().timeIntervalSince(t)
                        let sig = 1.0 / (1.0 + exp(-s))
                        emit(String(format: "%d. 查询「%@」↔ 文档「%@」→ logit %.4f，sigmoid %.4f，%.0f ms\n",
                                    i + 1, p.query, p.doc, s, sig, dt * 1000))
                    } catch {
                        emit("\(i + 1). 查询「\(p.query)」→ 出错：\(error)\n")
                        failed = true
                        break
                    }
                }
                if !failed {
                    emit("   参考：sigmoid 越接近 1 越相关，接近 0 为无关。\n")
                }
            case .chat:
                break
            }

            let dt = Date().timeIntervalSince(t0)
            emit(String(format: "—— 自动测试完成（%.2f s）——\n\n", dt))
            DispatchQueue.main.async { self.running = false }
        }
    }

    // MARK: 生成 / 推理

    func send(_ text: String, doc: String = "") {
        let userText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let docText = doc.trimmingCharacters(in: .whitespacesAndNewlines)
        glog("send called: kind=\(kind) textLen=\(userText.count) docLen=\(docText.count) backend?=\(backend != nil) loaded=\(loaded) running=\(running)")
        guard !userText.isEmpty, let backend = backend, loaded, !running else {
            glog("send REJECTED at guard")
            return
        }
        if kind == .reranker, docText.isEmpty {
            status = "重排模型需要同时填写查询与文档"
            return
        }
        running = true
        var convo = transcript
        switch kind {
        case .chat:     convo += "\n你: \(userText)\n助手: "
        case .embedding: convo += "\n文本: \(userText)\n"
        case .reranker: convo += "\n查询: \(userText)\n文档: \(docText)\n"
        }
        transcript = convo
        let t0 = Date()

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result: String
                var genTokens = 0
                switch self.kind {
                case .chat:
                    let (reply, gen) = try backend.generate(userText: userText,
                                                            maxNew: 256, penalty: REPETITION_PENALTY)
                    genTokens = gen
                    glog("generate done: gen=\(gen) replyLen=\(reply.count)")
                    glog("reply=\(reply)")
                    result = reply
                case .embedding:
                    let v = try backend.embed(userText)
                    let dt = Date().timeIntervalSince(t0)
                    let norm = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
                    let head = v.prefix(8).map { String(format: "%.4f", $0) }
                        .joined(separator: ", ")
                    glog("embed done: dim=\(v.count) norm=\(norm)")
                    result = "维度 \(v.count)｜L2 范数 \(String(format: "%.4f", norm))\n"
                        + "前 8 维: [\(head)]\n"
                        + "⏱ \(String(format: "%.2f", dt)) s"
                case .reranker:
                    let s = try backend.score(query: userText, doc: docText)
                    let dt = Date().timeIntervalSince(t0)
                    let sig = 1.0 / (1.0 + exp(-s))
                    glog("score done: logit=\(s) sigmoid=\(sig)")
                    result = "logit \(String(format: "%.4f", s))｜sigmoid \(String(format: "%.4f", sig))\n"
                        + "⏱ \(String(format: "%.2f", dt)) s"
                }
                let dt = Date().timeIntervalSince(t0)
                var out = convo + result
                if genTokens > 0 {
                    let tps = Double(genTokens) / dt
                    out += "\n⏱ \(genTokens) tok / \(String(format: "%.1f", dt)) s (\(String(format: "%.1f", tps)) t/s)"
                }
                DispatchQueue.main.async {
                    self.transcript = out + "\n"
                    self.running = false
                }
            } catch {
                // 推理错误写入日志便于诊断；对话区同时给出完整信息
                let detail = "\(error)"
                try? detail.write(toFile: "/tmp/modconv_gui_infer_error.log",
                                  atomically: true, encoding: .utf8)
                DispatchQueue.main.async {
                    self.transcript = convo + "（推理出错：\(detail)）\n"
                    self.running = false
                }
            }
        }
    }
}

// MARK: - AppDelegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var backendSeg: NSSegmentedControl!
    private var selectButton: NSButton!
    private var loadButton: NSButton!
    private var selectedPathLabel: NSTextField!
    private var selectTokButton: NSButton!
    private var tokenizerLabel: NSTextField!
    private var computeUnitPopUp: NSPopUpButton!
    private var maxNewStepper: NSStepper!
    private var maxNewLabel: NSTextField!
    private var promptField: NSTextField!
    private var docField: NSTextField!
    private var sendButton: NSButton!
    private var resetButton: NSButton!
    private var transcriptView: NSTextView!
    private var statusLabel: NSTextField!

    private let controller = ChatController()
    private var cancellables = Set<AnyCancellable>()

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        buildUI()
        bindController()
        installSignalCleanup()
    }

    // MARK: UI 构建

    private func buildUI() {
        let win = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 620),
                           styleMask: [.titled, .closable, .resizable],
                           backing: .buffered, defer: false)
        win.title = "ModConv 模型体验"
        win.isReleasedWhenClosed = false
        window = win

        let col = NSStackView(); col.orientation = .vertical; col.spacing = 10
        col.edgeInsets = NSEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)

        // 后端选择
        let rowB = NSStackView(); rowB.orientation = .horizontal; rowB.spacing = 8
        let backendLabel = NSTextField(labelWithString: "后端：")
        backendSeg = NSSegmentedControl(labels: BackendType.allCases.map { $0.title },
                                        trackingMode: .selectOne, target: self,
                                        action: #selector(backendChanged(_:)))
        backendSeg.selectedSegment = 0
        rowB.addArrangedSubview(backendLabel)
        rowB.addArrangedSubview(backendSeg)

        // 模型选择行
        let row1 = NSStackView(); row1.orientation = .horizontal; row1.spacing = 8
        selectButton = NSButton(title: "选择模型", target: self, action: #selector(selectTapped(_:)))
        selectedPathLabel = NSTextField(labelWithString: "（未选择模型）")
        selectedPathLabel.font = NSFont.systemFont(ofSize: 11)
        selectedPathLabel.textColor = .secondaryLabelColor
        selectedPathLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        loadButton = NSButton(title: "加载模型", target: self, action: #selector(loadTapped(_:)))
        resetButton = NSButton(title: "清空对话", target: self, action: #selector(resetTapped(_:)))
        row1.addArrangedSubview(selectButton)
        row1.addArrangedSubview(selectedPathLabel)
        row1.addArrangedSubview(loadButton)
        row1.addArrangedSubview(resetButton)

        // 分词器选择行
        let tokRow = NSStackView(); tokRow.orientation = .horizontal; tokRow.spacing = 8
        selectTokButton = NSButton(title: "选择分词器", target: self, action: #selector(selectTokTapped(_:)))
        tokenizerLabel = NSTextField(labelWithString: "（默认 tokenizer）")
        tokenizerLabel.font = NSFont.systemFont(ofSize: 11)
        tokenizerLabel.textColor = .secondaryLabelColor
        tokenizerLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        tokRow.addArrangedSubview(selectTokButton)
        tokRow.addArrangedSubview(tokenizerLabel)

        // 计算单元（仅 CoreML）
        let rowC = NSStackView(); rowC.orientation = .horizontal; rowC.spacing = 8
        let cuLabel = NSTextField(labelWithString: "计算单元：")
        computeUnitPopUp = NSPopUpButton()
        for t in ["全部", "仅 CPU", "仅 GPU", "仅 ANE", "CPU+GPU", "CPU+ANE"] {
            computeUnitPopUp.addItem(withTitle: t)
        }
        computeUnitPopUp.target = self; computeUnitPopUp.action = #selector(computeUnitChanged(_:))
        rowC.addArrangedSubview(cuLabel)
        rowC.addArrangedSubview(computeUnitPopUp)

        // 每轮生成上限
        let row2 = NSStackView(); row2.orientation = .horizontal; row2.spacing = 8
        maxNewStepper = NSStepper(); maxNewStepper.minValue = 16; maxNewStepper.maxValue = 1024
        maxNewStepper.integerValue = 256
        maxNewLabel = NSTextField(labelWithString: "每轮最多 256 token")
        maxNewLabel.font = NSFont.systemFont(ofSize: 11)
        row2.addArrangedSubview(NSTextField(labelWithString: "生成长度："))
        row2.addArrangedSubview(maxNewStepper)
        row2.addArrangedSubview(maxNewLabel)

        // 输入 + （reranker 的）文档输入 + 发送
        let row3 = NSStackView(); row3.orientation = .horizontal; row3.spacing = 8
        promptField = NSTextField(); promptField.placeholderString = "输入内容，回车或点「发送」"
        promptField.target = self; promptField.action = #selector(sendTapped(_:))   // 回车即发送
        promptField.setContentHuggingPriority(.defaultLow, for: .horizontal)
        docField = NSTextField(); docField.placeholderString = "文档（重排模型）"
        docField.isHidden = true
        docField.setContentHuggingPriority(.defaultLow, for: .horizontal)
        sendButton = NSButton(title: "发送", target: self, action: #selector(sendTapped(_:)))
        row3.addArrangedSubview(promptField)
        row3.addArrangedSubview(docField)
        row3.addArrangedSubview(sendButton)

        transcriptView = NSTextView()
        transcriptView.isEditable = false
        transcriptView.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        // 关键：作为 documentView 的 NSTextView 必须设置纵向可伸展 + 宽度自适应 + 非零初始 frame，
        // 否则 NSScrollView 不会撑开它，视图保持零尺寸 → 文字完全不可见
        transcriptView.isVerticallyResizable = true
        transcriptView.isHorizontallyResizable = false
        transcriptView.autoresizingMask = [.width]
        transcriptView.textContainer?.widthTracksTextView = true
        transcriptView.frame = NSRect(x: 0, y: 0, width: 700, height: 400)
        let scroll = NSScrollView(); scroll.hasVerticalScroller = true
        scroll.documentView = transcriptView
        scroll.translatesAutoresizingMaskIntoConstraints = false

        statusLabel = NSTextField(labelWithString: "未加载")
        statusLabel.font = NSFont.systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor

        let rows: [NSView] = [rowB, row1, tokRow, rowC, row2, row3, scroll, statusLabel]
        for v in rows { col.addArrangedSubview(v) }
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 200).isActive = true

        win.contentView = col
        win.center()
        win.makeKeyAndOrderFront(nil)
        updateUI()
        applyBackendVisibility()
    }

    private func bindController() {
        controller.$status.receive(on: RunLoop.main)
            .sink { [weak self] s in self?.statusLabel.stringValue = s }.store(in: &cancellables)
        controller.$transcript.receive(on: RunLoop.main)
            .sink { [weak self] t in
                self?.transcriptView.string = t
                self?.transcriptView.scrollToEndOfDocument(nil)
            }.store(in: &cancellables)
        controller.$modelName.receive(on: RunLoop.main).sink { [weak self] n in
            self?.selectedPathLabel.stringValue = n.isEmpty ? "（未选择模型）" : "已选: \(n)"
            self?.updateUI()   // 选中/切换模型后刷新「加载模型」等按钮可用态
        }.store(in: &cancellables)
        controller.$tokenizerName.receive(on: RunLoop.main).sink { [weak self] n in
            self?.tokenizerLabel.stringValue = n.isEmpty ? "（默认 tokenizer）" : "分词器: \(n)"
        }.store(in: &cancellables)
        controller.$loaded.receive(on: RunLoop.main).sink { [weak self] _ in self?.updateUI() }.store(in: &cancellables)
        controller.$running.receive(on: RunLoop.main).sink { [weak self] _ in self?.updateUI() }.store(in: &cancellables)
        controller.$kind.receive(on: RunLoop.main).sink { [weak self] k in
            self?.applyKindUI(k)
        }.store(in: &cancellables)
    }

    /// 按模型类型切换：按钮文案、占位符、reranker 的文档输入框显隐。
    private func applyKindUI(_ k: ModelKind) {
        docField.isHidden = k != .reranker
        switch k {
        case .chat:
            sendButton.title = "发送"
            promptField.placeholderString = "输入内容，回车或点「发送」"
        case .embedding:
            sendButton.title = "计算向量"
            promptField.placeholderString = "输入文本计算向量，回车或点「计算向量」"
        case .reranker:
            sendButton.title = "打分"
            promptField.placeholderString = "查询（query），回车或点「打分」"
        }
        updateUI()
    }

    private func updateUI() {
        let hasSel = !controller.modelName.isEmpty
        loadButton.isEnabled = hasSel && !controller.running
        sendButton.isEnabled = controller.loaded && !controller.running
        resetButton.isEnabled = controller.loaded && !controller.running
        sendButton.title = controller.running ? "生成中…" : "发送"
        maxNewLabel.stringValue = "每轮最多 \(maxNewStepper.integerValue) token"
    }

    /// 按后端切换：分词器/计算单元仅 CoreML、ONNX 需要；MLX 用模型内 tokenizer。
    private func applyBackendVisibility() {
        let isCoreML = controller.backendType == .coreml
        let usesExternalTok = isCoreML || controller.backendType == .onnx
        selectTokButton.isEnabled = usesExternalTok
        computeUnitPopUp.isEnabled = isCoreML
    }

    // MARK: 动作

    @objc private func backendChanged(_ sender: Any) {
        controller.backendType = BackendType.allCases[backendSeg.selectedSegment]
        applyBackendVisibility()
        selectedPathLabel.stringValue = "（未选择模型）"
    }

    @objc private func computeUnitChanged(_ sender: Any) {
        // MLComputeUnits 仅提供 all / cpuOnly / cpuAndGPU / cpuAndNeuralEngine
        let order = ["全部", "仅 CPU", "CPU+GPU", "CPU+ANE"]
        let units: [MLComputeUnits] = [.all, .cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine]
        let title = computeUnitPopUp.titleOfSelectedItem ?? "全部"
        if let i = order.firstIndex(of: title) { controller.computeUnit = units[i] }
    }

    @objc private func selectTapped(_ sender: Any) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.canCreateDirectories = false
        panel.allowsMultipleSelection = false
        switch controller.backendType {
        case .coreml:
            panel.allowedContentTypes = [.init(filenameExtension: "mlpackage")!,
                                         .init(filenameExtension: "mlmodelc")!]
            panel.message = "选择 modconv 转换产物 .mlpackage 或 .mlmodelc"
        case .mlx:
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.message = "选择 MLX checkpoint 文件夹（含 safetensors + config.json + tokenizer.json）"
        case .onnx:
            panel.allowedContentTypes = [.init(filenameExtension: "onnx")!]
            panel.message = "选择通用 ONNX 自回归 decode 图 .onnx"
        }
        panel.beginSheetModal(for: self.window) { [weak self] resp in
            guard resp == .OK, let url = panel.url else { return }
            self?.controller.selectModel(url: url)
        }
    }

    @objc private func loadTapped(_ sender: Any) { controller.loadSelected() }

    @objc private func selectTokTapped(_ sender: Any) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.canCreateDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        panel.message = "选择 tokenizer.json（须与所选模型同族）"
        panel.beginSheetModal(for: self.window) { [weak self] resp in
            guard resp == .OK, let url = panel.url else { return }
            self?.controller.selectTokenizer(url: url)
        }
    }

    @objc private func sendTapped(_ sender: Any) {
        let text = promptField.stringValue
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let doc = controller.kind == .reranker ? docField.stringValue : ""
        promptField.stringValue = ""
        controller.send(text, doc: doc)
    }

    @objc private func resetTapped(_ sender: Any) { controller.reset() }
}

// MARK: - CoreML 辅助（保持与原实现一致）

/// CoreML 整段序列 encoder（embedding/reranker）通用推理核心。
/// 适配约定输入 input_ids/attention_mask（可选 token_type_ids/position_ids），
/// 输出 logits → reranker 得分；sentence_embedding 或 last_hidden_state → 向量。
final class CoreMLSeqCore {
    let model: MLModel
    let swiftTokenizer: SwiftTokenizer
    let seqLen: Int
    let kind: ModelKind
    let inputNames: Set<String>
    private let logitsOut: String?
    private let sentEmbOut: String?
    private let hiddenOut: String?
    private let padId: Int32

    init(model: MLModel, tokenizerFolder: URL) async throws {
        self.model = model
        let desc = model.modelDescription
        inputNames = Set(desc.inputDescriptionsByName.keys)

        guard let idsShape = desc.inputDescriptionsByName["input_ids"]?.multiArrayConstraint?.shape,
              idsShape.count >= 2 else {
            throw BackendError.unsupportedKind("模型缺少 [1, L] 形状的 input_ids 输入")
        }
        seqLen = idsShape[1].intValue

        let outs = desc.outputDescriptionsByName
        // reranker logits：名字含 logit，或唯一输出末维 ≤16（如 var_xxx [1,1]）
        var logits = outs.keys.first { $0.lowercased().contains("logit") }
        if logits == nil {
            let small = outs.filter { ($0.value.multiArrayConstraint?.shape.last?.intValue ?? 999) <= 16 }
            if small.count == 1, outs.count == 1 { logits = small.keys.first }
        }
        let sent = outs.keys.first { $0.lowercased().contains("sentence_embedding") }
        let hidden = outs.keys.first {
            let n = $0.lowercased()
            return n.contains("hidden_state") || n.contains("token_embeddings") || n.contains("last_hidden")
        } ?? outs.keys.first { $0 != logits }
        logitsOut = logits
        sentEmbOut = sent
        hiddenOut = hidden
        kind = logits != nil ? .reranker : .embedding

        swiftTokenizer = try await SwiftTokenizer.load(folder: tokenizerFolder)
        // pad id：XLM-R 是 <pad>=1，BERT 是 [PAD]=0；取不到时按词表族兜底
        padId = swiftTokenizer.padTokenId ?? 1
    }

    /// 按输入描述的数据类型构造 [1, L] 整型张量。
    private func intInput(_ name: String, _ values: [Int32]) throws -> MLFeatureValue {
        let desc = model.modelDescription.inputDescriptionsByName[name]
        let dt = desc?.multiArrayConstraint?.dataType ?? .int32
        let arr = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: dt)
        for i in 0..<values.count { arr[i] = NSNumber(value: values[i]) }
        return MLFeatureValue(multiArray: arr)
    }

    /// 单段整段推理（embedding 用）：特殊 token 由分词器添加，超长截断，右侧 pad。
    private func run(_ ids: [Int32], tokenTypeIds: [Int32]? = nil,
                     outputName: String? = nil) throws -> MLFeatureValue? {
        var toks = Array(ids.prefix(seqLen))
        if toks.isEmpty { toks = [padId] }
        var mask = [Int32](repeating: 1, count: toks.count)
        mask += [Int32](repeating: 0, count: seqLen - toks.count)
        toks += [Int32](repeating: padId, count: seqLen - toks.count)

        var dict: [String: MLFeatureValue] = [
            "input_ids": try intInput("input_ids", toks),
            "attention_mask": try intInput("attention_mask", mask),
        ]
        if inputNames.contains("token_type_ids") {
            let tt = tokenTypeIds.map { Array($0.prefix(seqLen)) }
                ?? [Int32](repeating: 0, count: seqLen)
            dict["token_type_ids"] = try intInput("token_type_ids", tt)
        }
        if inputNames.contains("position_ids") {
            dict["position_ids"] = try intInput(
                "position_ids", (0..<Int32(seqLen)).map { $0 })
        }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
        return out.featureValue(for: outputName ?? sentEmbOut ?? hiddenOut ?? "")
    }

    func embed(_ text: String) throws -> [Float] {
        // embedding 需要 CLS/SEP 特殊 token：用分词器默认模板编码
        let full = try swiftTokenizer.encodeWithSpecialTokens(text)
        let out = try run(full)
        guard let mv = out?.multiArrayValue, mv.count > 0 else {
            throw BackendError.unsupportedKind("模型没有可用输出")
        }
        let shape = mv.shape.map { $0.intValue }
        if shape.count == 2 {
            // 直接是句向量 [1, H]
            var v = mlmultiArrayToFloatArray(mv)
            l2Normalize(&v)
            return v
        }
        // [1, L, H]：mean pooling（截断后真实长度，padding 排除在外）
        let L = shape.count >= 2 ? shape[1] : 0
        let H = shape.count >= 3 ? shape[2] : mv.count / max(L, 1)
        let realLen = min(full.count, seqLen)
        var acc = [Float](repeating: 0, count: H)
        let base = mv
        for l in 0..<realLen {
            for h in 0..<H { acc[h] += Float(base[l * H + h].doubleValue) }
        }
        if realLen > 0 { for h in 0..<H { acc[h] /= Float(realLen) } }
        l2Normalize(&acc)
        return acc
    }

    /// query-doc 打分：BERT 段对格式 [CLS] q [SEP] d [SEP]，token_type_ids 0/1。
    func score(query: String, doc: String) throws -> Double {
        let q = try swiftTokenizer.encode(query)
        let d = try swiftTokenizer.encode(doc)
        let cls = swiftTokenizer.specialId("[CLS]", fallback: 101)
        let sep = swiftTokenizer.specialId("[SEP]", fallback: 102)
        let budget = max(seqLen - 3, 8)
        var qq = q, dd = d
        if q.count + d.count > budget {
            let half = budget / 2
            qq = Array(q.prefix(max(half, 4)))
            dd = Array(d.prefix(max(budget - qq.count, 4)))
        }
        var toks: [Int32] = [cls] + qq + [sep] + dd + [sep]
        var tt: [Int32] = [Int32](repeating: 0, count: qq.count + 2)
            + [Int32](repeating: 1, count: dd.count + 1)
        if toks.count > seqLen { toks = Array(toks.prefix(seqLen)); tt = Array(tt.prefix(seqLen)) }
        let padN = seqLen - toks.count
        toks += [Int32](repeating: padId, count: padN)
        tt += [Int32](repeating: 0, count: padN)

        var dict: [String: MLFeatureValue] = [
            "input_ids": try intInput("input_ids", toks),
            "attention_mask": try intInput(
                "attention_mask", [Int32](repeating: 1, count: toks.count - padN)
                    + [Int32](repeating: 0, count: padN)),
        ]
        if inputNames.contains("token_type_ids") {
            dict["token_type_ids"] = try intInput("token_type_ids", tt)
        }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
        guard let name = logitsOut, let mv = out.featureValue(for: name)?.multiArrayValue,
              mv.count > 0 else {
            throw BackendError.unsupportedKind("模型没有 logits 输出，无法打分")
        }
        return mv[0].doubleValue
    }

    private func l2Normalize(_ v: inout [Float]) {
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
        if n > 0 { for i in v.indices { v[i] /= n } }
    }
}

nonisolated(unsafe) private var _lastCompiledURL: URL?

func loadConvertedModel(at url: URL, units: MLComputeUnits) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = units
    let compiled = try compiledModelCached(url)
    return try MLModel(contentsOf: compiled, configuration: config)
}

/// 编译结果持久缓存：按 文件名+大小+mtime 去重，避免每次加载重复编译
/// 并在系统临时目录堆积 GB 级 .mlmodelc 副本（曾把系统盘塞满）。
func compiledModelCached(_ url: URL) throws -> URL {
    let fm = FileManager.default
    let cacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("ModConvDemo/CoreML", isDirectory: true)
    try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)

    // 缓存超 30GB 时整体清空（防模型反复重导出后无限增长）
    if let items = try? fm.contentsOfDirectory(atPath: cacheDir.path) {
        let total = items.reduce(Int64(0)) { acc, it in
            let p = cacheDir.appendingPathComponent(it)
            let s = (try? fm.attributesOfItem(atPath: p.path)[.size] as? Int64) ?? 0
            return acc + s
        }
        if total > 30 * 1024 * 1024 * 1024 {
            for it in items { try? fm.removeItem(at: cacheDir.appendingPathComponent(it)) }
        }
    }

    let attrs = try? fm.attributesOfItem(atPath: url.path)
    let size = (attrs?[.size] as? Int64) ?? 0
    let mtime = ((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
    let cached = cacheDir.appendingPathComponent("\(url.lastPathComponent)-\(size)-\(Int(mtime)).mlmodelc")
    if fm.fileExists(atPath: cached.path) { return cached }

    let compiled = try MLModel.compileModel(at: url)
    try? fm.removeItem(at: cached)
    try? fm.moveItem(at: compiled, to: cached)
    _lastCompiledURL = cached
    return fm.fileExists(atPath: cached.path) ? cached : compiled
}

func _readVocab(_ model: MLModel) -> Int {
    for (_, d) in model.modelDescription.outputDescriptionsByName {
        if let t = d.multiArrayConstraint, d.name.lowercased().contains("logit") {
            return t.shape.last?.intValue ?? 0
        }
    }
    return 0
}

/// 单步前向：推进一个 token，返回该步的 logits 与下一步要回传的 GDN 状态。
func forwardWithConvertedModel(
    _ model: MLModel, _ state: MLState,
    token: Int32, position: Int32,
    gdnStates: [String: MLMultiArray]
) throws -> (logits: MLMultiArray, newStates: [String: MLMultiArray]) {
    var dict: [String: MLFeatureValue] = [
        "input_ids": MLFeatureValue(multiArray: MLMultiArray(
            MLShapedArray<Int32>(scalars: [token], shape: [1, 1]))),
        "cache_position": MLFeatureValue(multiArray: MLMultiArray(
            MLShapedArray<Int32>(scalars: [position], shape: [1]))),
    ]
    for (name, value) in gdnStates { dict[name] = MLFeatureValue(multiArray: value) }
    let output = try model.prediction(
        from: try MLDictionaryFeatureProvider(dictionary: dict), using: state)
    guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
          logits.count > 0 else {
        throw BackendError.notLoaded
    }
    var newStates: [String: MLMultiArray] = [:]
    for name in gdnStates.keys {
        if let v = output.featureValue(for: name)?.multiArrayValue {
            newStates[name] = v
        } else if let v = output.featureValue(for: "new_" + name)?.multiArrayValue {
            newStates[name] = v
        }
    }
    return (logits, newStates)
}

func mlmultiArrayToFloatArray(_ a: MLMultiArray) -> [Float] {
    let cnt = a.count
    var out = [Float](repeating: 0, count: cnt)
    for i in 0..<cnt { out[i] = Float(a[i].doubleValue) }
    return out
}

// MARK: - 退出清理（CoreML 编译缓存）

private func signalHandler(_ sig: Int32) {
    cleanupCompiledCache()
    exit(0)
}

private func installSignalCleanup() {
    signal(SIGTERM, signalHandler)
    signal(SIGINT, signalHandler)
}

private func cleanupCompiledCache() {
    let fm = FileManager.default
    let base = (ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp") as NSString
    let dir = base.expandingTildeInPath
    for prefix in ["CoreML", "com.apple.CoreML"] {
        guard let items = try? fm.contentsOfDirectory(atPath: dir) else { continue }
        for it in items where it.hasPrefix(prefix) {
            try? fm.removeItem(atPath: (dir as NSString).appendingPathComponent(it))
        }
    }
}

// MARK: - 入口（含无头自测模式）

/// 无头自测：
///   swift-gui --embed <MLX模型目录> <文本>
///   swift-gui --score <MLX模型目录> <query> <doc>
///   swift-gui --onnx-embed <model.onnx> [tokenizer.json] <文本>
///   swift-gui --onnx-score <model.onnx> [tokenizer.json] <query> <doc>
///   swift-gui --coreml-embed <model.mlpackage> [tokenizer.json] <文本>
///   swift-gui --coreml-score <model.mlpackage> [tokenizer.json] <query> <doc>
func headlessTest() -> Bool {
    let args = CommandLine.arguments
    guard args.count >= 4, args[1].hasPrefix("--") else { return false }

    enum Op { case embed, score }
    let op: Op
    var backend: ModelBackend
    var textIdx: Int
    switch args[1] {
    case "--embed":
        op = .embed; backend = MLXEmbedBackend(); backend.modelURL = URL(fileURLWithPath: args[2]); textIdx = 3
    case "--score":
        op = .score; backend = MLXEmbedBackend(); backend.modelURL = URL(fileURLWithPath: args[2]); textIdx = 3
    case "--onnx-embed", "--onnx-score":
        op = args[1].hasSuffix("embed") ? .embed : .score
        let b = OnnxBackend()
        b.modelURL = URL(fileURLWithPath: args[2])
        if isTokenizerArg(args[3]) { b.tokenizerURL = URL(fileURLWithPath: args[3]); textIdx = 4 } else { textIdx = 3 }
        backend = b
    case "--coreml-embed", "--coreml-score":
        op = args[1].hasSuffix("embed") ? .embed : .score
        let b = CoreMLBackend()
        b.modelURL = URL(fileURLWithPath: args[2])
        if isTokenizerArg(args[3]) { b.tokenizerURL = URL(fileURLWithPath: args[3]); textIdx = 4 } else { textIdx = 3 }
        backend = b
    default:
        return false
    }
    guard textIdx < args.count else { return false }
    let sem = DispatchSemaphore(value: 0)
    Task {
        do {
            try await backend.load()
            glog("headless kind=\(backend.kind)")
            switch op {
            case .embed:
                let v = try backend.embed(args[textIdx])
                let norm = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
                print("dim=\(v.count) norm=\(norm) head=\(v.prefix(4).map { String(format: "%.4f", $0) })")
            case .score:
                let doc = textIdx + 1 < args.count ? args[textIdx + 1] : ""
                let s = try backend.score(query: args[textIdx], doc: doc)
                print("logit=\(s) sigmoid=\(1.0 / (1.0 + exp(-s)))")
            }
        } catch {
            print("FAIL: \(error)")
        }
        sem.signal()
    }
    sem.wait()
    return true
}

/// 判断无头参数是否为分词器（.json 文件或目录）。
private func isTokenizerArg(_ path: String) -> Bool {
    if path.hasSuffix(".json") { return true }
    var isDir: ObjCBool = false
    return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue
}

if headlessTest() { exit(0) }

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
