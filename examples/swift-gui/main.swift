// ModConv · CoreML 模型体验（对话版，macOS / Apple Silicon）
//
// 这是一个最小可运行的 Cocoa 示例，用来在 macOS 上直接体验 modconv 转换出的
// Core ML 模型：加载 .mlpackage → 用 Core ML 原生 API 做单步 decode 自回归生成。
// 推理全部在 Swift + CoreML 中完成；只有「文字 ↔ token id」的桥接借用了本地
// tokenizers 库的 Python 小脚本（tok_helper.py）。KV cache 在对话过程中持续保留，
// 因此多轮对话是连贯的。
//
// 构建见同目录 build.sh。

import Cocoa
import CoreML
import Dispatch
import Foundation
import Combine
import UniformTypeIdentifiers

// 模型改为运行时选取：点「选择模型」弹出文件夹选择框，选中 .mlpackage（或已编译的
// .mlmodelc）后点「加载模型」即载入。不再内置固定路径下拉。


// 分词脚本与 Python：优先用本 App 包内的 tok_helper.py，其次回退到脚本同目录/默认路径。
let PYTHON: String = {
    let p = "/Users/arlenecc/.pyenv/versions/3.13.16/bin/python3"
    return FileManager.default.isExecutableFile(atPath: p) ? p : "python3"
}()
let TOK_HELPER: String = {
    if let p = Bundle.main.path(forResource: "tok_helper", ofType: "py") { return p }
    return (ProcessInfo.processInfo.environment["MODCONV_TOK_HELPER"]) ?? "/tmp/swift_gui/tok_helper.py"
}()

let MAX_SEQ: Int32 = 8192

// 重复惩罚：对提示 + 已生成 token 降权，抑制贪心解码下 "- **Ask**:" 这类循环退化。
// 1.0 = 关闭；1.1~1.3 常用，1.15 在连贯与多样间较平衡。
let REPETITION_PENALTY: Float = 1.15

// Qwen3.5 特殊 token 与基础词表大小（用于 decode 停止 / 避开填充区）
let TOK_IM_START: Int32 = 248045
let TOK_IM_END:   Int32 = 248046
let TOKENIZER_VOCAB: Int = 248044   // 基础词表（0..248043）；id >= 此值均为特殊/填充 token

// MARK: - 模型 IO 辅助（与 modconv 生成的 *_swift-example.swift 同源）
enum ConvertedModelError: Error { case missingOutput(String) }

/// 载入转换产物。Core ML 只能加载**已编译**的模型：传入未编译的 .mlpackage 时
/// 先调用 compileModel 生成 .mlmodelc（把模型加入 Xcode 工程时，Xcode 会在构建期
/// 自动完成这一步，此时直接传 Bundle 里的 URL 即可，本函数会原样使用）。
/// 返回 (模型, 编译缓存URL)；该缓存须在任务结束后清理以释放磁盘。
func loadConvertedModel(_ modelURL: URL, computeUnits: MLComputeUnits = .all) throws -> (model: MLModel, compiled: URL?) {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let compiled: URL?
    let url: URL
    if modelURL.pathExtension == "mlpackage" {
        let c = try MLModel.compileModel(at: modelURL)
        compiled = c; url = c
    } else {
        compiled = nil; url = modelURL
    }
    let model = try MLModel(contentsOf: url, configuration: configuration)
    return (model, compiled)
}

/// 单步前向：推进一个 token，返回该步的 logits 与下一步要回传的 GDN 状态。
func forwardWithConvertedModel(
    _ model: MLModel, _ state: MLState,
    token: Int32, position: Int32,
    gdnStates: [String: MLMultiArray]
) throws -> (logits: MLMultiArray, newStates: [String: MLMultiArray]) {
    var dict: [String: Any] = [
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
        throw ConvertedModelError.missingOutput("logits")
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

func argmaxFromLogits(_ logits: MLMultiArray, vocab: Int,
                       penalty: Float = 1.0, penalized: Set<Int32>? = nil) -> Int32 {
    let v = min(vocab, logits.shape.last?.intValue ?? 0)
    guard v > 0 else { return 0 }
    var best = 0, bestVal = -Double.greatestFiniteMagnitude
    let base = logits.count - v
    let p = Double(penalty)
    for i in 0..<v {
        var val = logits[base + i].doubleValue
        // 重复惩罚（Fan et al.）：对上下文已出现过的 token 降权，抑制循环生成。
        // 正值除以 p、负值乘以 p，与 transformers 的 repetition_penalty 一致。
        if p != 1.0, let set = penalized, set.contains(Int32(i)) {
            val = val > 0 ? val / p : val * p
        }
        if val > bestVal { bestVal = val; best = i }
    }
    return Int32(best)
}

// MARK: - 分词器常驻桥接
/// 把 tok_helper.py 跑成常驻进程（--server）：启动一次、复用已加载的分词器，
/// 避免原 runPython 每条消息都 spawn 新 Python + 重新加载分词器带来的固定延迟。
final class TokenizerBridge {
    static let shared = TokenizerBridge()

    private var process: Process?
    private var stdinHandle: FileHandle?
    private var stdoutHandle: FileHandle?
    private let queue = DispatchQueue(label: "modconv.tokbridge")
    private var tokenizerURL: URL?
    private var launchedWith: URL?

    /// 设置分词器路径；若与已启动进程所用的不同，则下次请求时重建进程。
    func configure(tokenizer: URL?) {
        tokenizerURL = tokenizer
        if let t = tokenizer, t != launchedWith { process = nil }
    }

    private func ensureStarted() throws {
        if let p = process, p.isRunning { return }
        process = nil
        let p = Process()
        p.executableURL = URL(fileURLWithPath: PYTHON)
        p.arguments = [TOK_HELPER, "--server"]
        if let tok = tokenizerURL {
            var env = ProcessInfo.processInfo.environment
            env["MODCONV_TOKENIZER"] = tok.path
            p.environment = env
        }
        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe
        try p.run()
        process = p
        stdinHandle = inPipe.fileHandleForWriting
        stdoutHandle = outPipe.fileHandleForReading
        launchedWith = tokenizerURL
        DispatchQueue.global(qos: .utility).async {
            let d = errPipe.fileHandleForReading.readDataToEndOfFile()
            if let s = String(data: d, encoding: .utf8), !s.isEmpty {
                NSLog("tok_helper stderr: \(s)")
            }
        }
    }

    /// 阻塞读取直到遇到一个换行符（常驻进程每行回一条响应）。
    private func readLineBlocking() throws -> String {
        guard let out = stdoutHandle else {
            throw NSError(domain: "tok", code: 3, userInfo: [NSLocalizedDescriptionKey: "分词器未启动"])
        }
        var buf = Data()
        while true {
            let chunk = out.availableData
            if chunk.isEmpty {
                throw NSError(domain: "tok", code: 4, userInfo: [NSLocalizedDescriptionKey: "分词器进程已退出"])
            }
            buf.append(chunk)
            if let last = buf.last, last == UInt8(ascii: "\n") { break }
        }
        buf.removeLast()
        guard let s = String(data: buf, encoding: .utf8) else {
            throw NSError(domain: "tok", code: 5, userInfo: [NSLocalizedDescriptionKey: "分词器返回非 UTF-8"])
        }
        return s
    }

    private func request(_ payload: [String: Any]) throws -> [String: Any] {
        return try queue.sync {
            try ensureStarted()
            guard let stdin = stdinHandle else {
                throw NSError(domain: "tok", code: 3, userInfo: [NSLocalizedDescriptionKey: "分词器未启动"])
            }
            let data = try JSONSerialization.data(withJSONObject: payload)
            stdin.write(data)
            stdin.write(Data([UInt8(ascii: "\n")]))
            let resp = try readLineBlocking()
            guard let obj = try JSONSerialization.jsonObject(with: Data(resp.utf8)) as? [String: Any] else {
                throw NSError(domain: "tok", code: 6, userInfo: [NSLocalizedDescriptionKey: "分词器返回非法 JSON: \(resp)"])
            }
            if let err = obj["error"] as? String {
                throw NSError(domain: "tok", code: 7, userInfo: [NSLocalizedDescriptionKey: "分词器错误: \(err)"])
            }
            return obj
        }
    }

    func encode(_ text: String) throws -> [Int32] {
        let obj = try request(["mode": "encode", "arg": text])
        let data = (obj["data"] as? [Int]) ?? []
        return data.map { Int32($0) }
    }

    func decode(_ ids: [Int32]) throws -> String {
        let obj = try request(["mode": "decode", "arg": ids.map { Int($0) }])
        return (obj["data"] as? String) ?? ""
    }
}

// MARK: - 对话控制器
struct ChatResult {
    let reply: String
    let genTokens: Int
    let prefillTokens: Int
    let prefillMs: Double
    let decodeMs: Double
}

final class ChatController: ObservableObject {
    @Published var status = "未加载"
    @Published var loaded = false
    @Published var modelName = ""   // 当前选中（待加载）模型名（取自文件夹名）
    @Published var loadedName = ""  // 已加载模型名
    @Published var transcript = "对话将在模型加载后开始。\n在下方输入内容并点「发送」即可体验多轮对话。\n"
    @Published var running = false
    @Published var tokenizerName = ""   // 已选分词器文件名（空 = 用 tok_helper.py 默认）

    private var model: MLModel?
    private var state: MLState?
    private var gdnStates: [String: MLMultiArray] = [:]
    private var vocab: Int = 0
    private var compiledURL: URL?   // 最近一次 compileModel 生成的 .mlmodelc，退出时清理
    private var selectedURL: URL?   // 用户通过文件选择框选中的模型路径
    private var selectedTokenizerURL: URL?   // 用户通过文件选择框选中的 tokenizer.json 路径
    private var pos: Int32 = 0   // 当前序列长度（跨轮持续累加，KV cache 不重置）
    var selectedComputeUnit: MLComputeUnits = .all   // 加载时使用的计算单元（UI 下拉选择，便于对比 ANE/GPU/CPU 速度）

    /// 用户在文件夹选择框选中模型后调用：仅记录路径并更新显示，不立即加载。
    func select(url: URL) {
        selectedURL = url
        modelName = url.lastPathComponent
        status = "已选择：\(url.lastPathComponent)｜点「加载模型」载入"
    }

    /// 用户选中 tokenizer.json 后调用：记录路径，encode/decode 时通过
    /// MODCONV_TOKENIZER 环境变量传给 tok_helper.py，与所选模型配合。
    func selectTokenizer(url: URL) {
        selectedTokenizerURL = url
        tokenizerName = url.lastPathComponent
        status = "已选分词器：\(url.lastPathComponent)（与所选模型配合）"
    }

    /// 点「加载模型」时调用：加载当前选中的模型文件夹。
    func loadSelected() {
        guard let url = selectedURL else {
            status = "请先点「选择模型」选取 .mlpackage 模型文件"
            return
        }
        load(url: url)
    }

    func load(url: URL) {
        modelName = url.lastPathComponent
        let path = url.path
        let cu = self.selectedComputeUnit   // 主线程捕获，避免后台读取竞态
        guard !path.isEmpty else { return }
        let devHint = self.computeUnitLabel(cu)
        DispatchQueue.main.async { self.status = "加载中（首次编译/特化可能需 1–2 分钟，device=\(devHint)）…" }
        DispatchQueue.global(qos: .userInitiated).async {
            self.clearPreviousCompiledCache()   // 切换/重载前先删掉上一模型的编译缓存
            let fileURL = URL(fileURLWithPath: path)
            // 加载 + 零初始化 GDN 状态 + 首次预热，封装成闭包以便失败后重试。
            let attempt: (MLComputeUnits) throws -> (MLModel, URL?, [String: MLMultiArray], Int, String) = { unit in
                let (m, compiled) = try loadConvertedModel(fileURL, computeUnits: unit)
                var gs: [String: MLMultiArray] = [:]
                for (n, d) in m.modelDescription.inputDescriptionsByName {
                    if n.hasSuffix("_states"), let shape = d.multiArrayConstraint?.shape {
                        let arr = try MLMultiArray(shape: shape, dataType: .float32)
                        memset(arr.dataPointer, 0, arr.count * MemoryLayout<Float>.size)
                        gs[n] = arr
                    }
                }
                var vocab = 151936
                if let sh = m.modelDescription.outputDescriptionsByName["logits"]?
                    .multiArrayConstraint?.shape, let last = sh.last {
                    vocab = last.intValue
                }
                // 预热：用一次性 state 触发 ANE 程序特化，不污染正式对话的 KV cache
                let _ = try forwardWithConvertedModel(m, m.makeState(), token: 0, position: 0, gdnStates: gs)
                return (m, compiled, gs, vocab, self.computeUnitLabel(unit))
            }
            var m: MLModel?; var compiled: URL?; var gs: [String: MLMultiArray] = [:]
            var vocab = 0; var devLabel = ""
            do {
                do {
                    (m, compiled, gs, vocab, devLabel) = try attempt(cu)
                } catch {
                    // 选「自动(.all)」时若 ANE 构建执行计划失败（-14，常见于大上下文模型），
                    // 自动退到「仅 GPU+CPU」再试一次；显式选了其他设备则不回退。
                    if cu == .all {
                        (m, compiled, gs, vocab, devLabel) = try attempt(.cpuAndGPU)
                        devLabel += "（ANE 计划失败，已自动回退仅GPU+CPU）"
                    } else {
                        throw error
                    }
                }
                let model = m!, c = compiled, g = gs, v = vocab, d = devLabel
                DispatchQueue.main.async {
                    self.model = model
                    self.state = model.makeState()   // 干净的 KV
                    self.gdnStates = g
                    self.vocab = v
                    self.pos = 0
                    self.compiledURL = c   // 记录编译缓存路径，便于退出时清理
                    self.loaded = true
                    self.loadedName = fileURL.lastPathComponent
                    self.status = "已加载：\(fileURL.lastPathComponent)｜vocab=\(v)｜device=\(d)"
                }
            } catch {
                // 把完整错误同时打到可滚动对话区与日志文件，便于定位（尤其是 ANE -14）。
                let detail = "\(error)"
                let msg = "加载失败：\(detail)"
                try? detail.write(toFile: "/tmp/modconv_gui_load_error.log",
                                 atomically: true, encoding: .utf8)
                DispatchQueue.main.async {
                    self.status = msg
                    self.loaded = false
                    self.append(msg + "\n（完整错误已写入 /tmp/modconv_gui_load_error.log）\n")
                }
            }
        }
    }

    /// 删除上一模型记录的编译缓存（切换/重载前调用），避免多个 .mlmodelc 在磁盘堆积。
    private func clearPreviousCompiledCache() {
        if let u = compiledURL {
            try? FileManager.default.removeItem(at: u)
            compiledURL = nil
        }
    }

    /// 任务结束（App 退出）时彻底清理 CoreML 编译产物：
    /// 1) 当前记录的编译缓存；2) TMPDIR 下所有遗留 .mlmodelc；3) 持久缓存目录 C/com.arlenecc.modconvdemo。
    func cleanupCompiledCache() {
        clearPreviousCompiledCache()
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        if let items = try? FileManager.default.contentsOfDirectory(at: tmp,
                                                                    includingPropertiesForKeys: nil) {
            for it in items where it.pathExtension == "mlmodelc" {
                try? FileManager.default.removeItem(at: it)
            }
        }
        let cCache = tmp.deletingLastPathComponent()
            .appendingPathComponent("C")
            .appendingPathComponent("com.arlenecc.modconvdemo")
        try? FileManager.default.removeItem(at: cCache)
    }

    /// 多轮对话一步：把本轮用户输入追加进 KV，再自回归生成回复。
    /// 不重放历史——KV cache 持续保留，故多轮连贯。
    /// 同时统计 prefill（prompt 处理）与 decode（生成）的耗时与速度。
    func chat(userText: String, maxNew: Int) -> ChatResult {
        guard let m = model, let st = state else {
            return ChatResult(reply: "（模型未加载）", genTokens: 0, prefillTokens: 0,
                              prefillMs: 0, decodeMs: 0)
        }
        let trimmed = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ChatResult(reply: "", genTokens: 0, prefillTokens: 0,
                              prefillMs: 0, decodeMs: 0)
        }
        do {
            // Qwen3.5 是指令模型：必须套 chat template，否则裸文本会输出乱码。
            // 生成前缀按官方模板给“非思考”形式：<|im_start|>assistant\n<think>\n\n</think>\n\n
            let userTurn = "<|im_start|>user\n" + trimmed + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            let ids = try pyEncode(userTurn)
            guard !ids.isEmpty else {
                return ChatResult(reply: "（分词为空，请检查 prompt）", genTokens: 0,
                                  prefillTokens: 0, prefillMs: 0, decodeMs: 0)
            }
            // KV 容量检查：接近上限则自动重置上下文
            if pos + Int32(ids.count) + Int32(maxNew) >= MAX_SEQ {
                resetConversationState()
                append("（上下文接近上限，已自动重置对话记忆）\n")
            }
            // 1) 多轮：先补一个 <|im_end|> 关闭上一轮助手回复，保证对话格式良好
            let t0 = Date()
            var gs = gdnStates
            var lastLogits: MLMultiArray? = nil
            if pos > 0 {
                let out = try forwardWithConvertedModel(m, st, token: TOK_IM_END, position: pos, gdnStates: gs)
                gs = out.newStates; lastLogits = out.logits; pos += 1
            }
            // 2) 把（模板化后的）用户输入喂入模型（prefill，KV 累加）
            for t in ids {
                let out = try forwardWithConvertedModel(m, st, token: t, position: pos, gdnStates: gs)
                gs = out.newStates; lastLogits = out.logits; pos += 1
            }
            gdnStates = gs
            let t1 = Date()
            let prefillMs = t1.timeIntervalSince(t0) * 1000

            // 3) 自回归生成回复（decode）；遇到 eos/特殊 token/填充区即停止
            var gen: [Int32] = []
            var ctx: [Int32] = ids   // 提示 + 已生成 token，作为重复惩罚上下文
            for _ in 0..<maxNew {
                guard let cur = lastLogits else { break }
                let penalized = Set(ctx)
                let nxt = argmaxFromLogits(cur, vocab: vocab,
                                           penalty: REPETITION_PENALTY, penalized: penalized)
                // id >= 基础词表(248044) 均为特殊/填充 token（含 <|im_end|>=248046、<|endoftext|>=248044）
                if Int(nxt) >= TOKENIZER_VOCAB { break }
                gen.append(nxt)
                let out = try forwardWithConvertedModel(m, st, token: nxt, position: pos, gdnStates: gdnStates)
                gdnStates = out.newStates; lastLogits = out.logits; pos += 1
                ctx.append(nxt)
            }
            let t2 = Date()
            let decodeMs = t2.timeIntervalSince(t1) * 1000
            let reply = try pyDecode(gen)
            return ChatResult(reply: reply, genTokens: gen.count, prefillTokens: ids.count,
                              prefillMs: prefillMs, decodeMs: decodeMs)
        } catch {
            // 推理错误（如所选设备缺少某算子的 GPU/CPU 内核）写到日志，便于诊断；
            // 对话区同时给出完整信息，避免「加载成功却不能用」却看不到原因。
            let detail = "\(error)"
            let msg = "推理出错：\(detail)"
            try? detail.write(toFile: "/tmp/modconv_gui_infer_error.log",
                             atomically: true, encoding: .utf8)
            return ChatResult(reply: msg + "\n（详情见 /tmp/modconv_gui_infer_error.log）",
                              genTokens: 0, prefillTokens: 0, prefillMs: 0, decodeMs: 0)
        }
    }

    func chatAndPublish(userText: String, maxNew: Int) {
        guard loaded, modelName == loadedName else {
            append("（请先加载所选模型）\n"); return
        }
        let text = userText
        DispatchQueue.main.async { self.running = true; self.append("你: \(text)\n") }
        DispatchQueue.global(qos: .userInitiated).async {
            let r = self.chat(userText: text, maxNew: maxNew)
            DispatchQueue.main.async {
                if !r.reply.isEmpty { self.append("助手: \(r.reply)\n") }
                if r.genTokens > 0 {
                    let prefillTps = r.prefillMs > 0
                        ? Double(r.prefillTokens) / (r.prefillMs / 1000) : 0
                    let decodeTps = r.decodeMs > 0
                        ? Double(r.genTokens) / (r.decodeMs / 1000) : 0
                    let line = String(format:
                        "⏱ prompt %d tok / %.0f ms (%.0f t/s) · 解码 %d tok / %.1f s (%.1f t/s)\n\n",
                        r.prefillTokens, r.prefillMs, prefillTps,
                        r.genTokens, r.decodeMs / 1000, decodeTps)
                    self.append(line)
                }
                self.running = false
            }
        }
    }

    /// 仅重置 KV cache 与位置、重新零初始化 GDN 状态（不动 transcript）。
    private func resetConversationState() {
        guard let m = model else { return }
        state = m.makeState()
        pos = 0
        var gs: [String: MLMultiArray] = [:]
        for (n, d) in m.modelDescription.inputDescriptionsByName {
            if n.hasSuffix("_states"), let shape = d.multiArrayConstraint?.shape {
                if let arr = try? MLMultiArray(shape: shape, dataType: .float32) {
                    memset(arr.dataPointer, 0, arr.count * MemoryLayout<Float>.size)
                    gs[n] = arr
                }
            }
        }
        gdnStates = gs
    }

    /// 清空对话记忆（按钮用）：重置状态并把 transcript 也清空。
    func resetConversation() {
        resetConversationState()
        transcript = "（对话已清空，记忆已重置）\n"
    }

    /// 计算单元枚举 → 中文标签，用于状态栏展示当前加载所用设备。
    private func computeUnitLabel(_ u: MLComputeUnits) -> String {
        switch u {
        case .all:                 return "自动(ANE+GPU+CPU)"
        case .cpuAndGPU:           return "仅GPU+CPU"
        case .cpuAndNeuralEngine:  return "仅ANE+CPU"
        case .cpuOnly:             return "仅CPU"
        @unknown default:          return "未知"
        }
    }

    private func append(_ s: String) {
        DispatchQueue.main.async { self.transcript += s }
    }

    private func pyEncode(_ text: String) throws -> [Int32] {
        TokenizerBridge.shared.configure(tokenizer: selectedTokenizerURL)
        return try TokenizerBridge.shared.encode(text)
    }
    private func pyDecode(_ ids: [Int32]) throws -> String {
        TokenizerBridge.shared.configure(tokenizer: selectedTokenizerURL)
        return try TokenizerBridge.shared.decode(ids)
    }
}

// MARK: - Cocoa UI
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var selectButton: NSButton!
    private var loadButton: NSButton!
    private var selectedPathLabel: NSTextField!
    private var selectTokButton: NSButton!
    private var tokenizerLabel: NSTextField!
    private var statusLabel: NSTextField!
    private var promptField: NSTextField!
    private var maxNewStepper: NSStepper!
    private var computeUnitPopup: NSPopUpButton!
    private var maxNewLabel: NSTextField!
    private var sendButton: NSButton!
    private var resetButton: NSButton!
    private var transcriptView: NSTextView!
    private let controller = ChatController()
    private var cancellables = Set<AnyCancellable>()
    private var sigtermSource: DispatchSourceSignal?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 720, height: 600),
                              styleMask: [.titled, .closable, .resizable, .miniaturizable],
                              backing: .buffered, defer: false)
        window.title = "ModConv · CoreML 对话体验"
        self.window = window

        let root = NSStackView()
        root.orientation = .vertical
        root.spacing = 10
        root.edgeInsets = NSEdgeInsets(top: 14, left: 14, bottom: 14, right: 14)
        window.contentView = root

        let row1 = NSStackView(); row1.orientation = .horizontal; row1.spacing = 8
        selectButton = NSButton(title: "选择模型", target: self, action: #selector(selectTapped(_:)))
        selectedPathLabel = NSTextField(labelWithString: "（未选择模型文件）")
        selectedPathLabel.font = NSFont.systemFont(ofSize: 11)
        selectedPathLabel.textColor = .secondaryLabelColor
        selectedPathLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        loadButton = NSButton(title: "加载模型", target: self, action: #selector(loadTapped(_:)))
        resetButton = NSButton(title: "清空对话", target: self, action: #selector(resetTapped(_:)))
        row1.addArrangedSubview(selectButton)
        row1.addArrangedSubview(selectedPathLabel)
        row1.addArrangedSubview(loadButton)
        row1.addArrangedSubview(resetButton)

        let tokRow = NSStackView(); tokRow.orientation = .horizontal; tokRow.spacing = 8
        selectTokButton = NSButton(title: "选择分词器", target: self, action: #selector(selectTokTapped(_:)))
        tokenizerLabel = NSTextField(labelWithString: "（默认 tokenizer）")
        tokenizerLabel.font = NSFont.systemFont(ofSize: 11)
        tokenizerLabel.textColor = .secondaryLabelColor
        tokenizerLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        tokRow.addArrangedSubview(selectTokButton)
        tokRow.addArrangedSubview(tokenizerLabel)

        // 计算设备下拉：加载前选择 ANE/GPU/CPU 组合，便于直观对比不同后端的速度。
        // 默认「自动」，与 modconv 的 ANE 取向一致；也可选「仅 GPU+CPU」排除 ANE。
        let computeRow = NSStackView(); computeRow.orientation = .horizontal; computeRow.spacing = 8
        let computeLabel = NSTextField(labelWithString: "计算设备:")
        computeLabel.font = NSFont.systemFont(ofSize: 11)
        computeUnitPopup = NSPopUpButton()
        computeUnitPopup.addItems(withTitles: ["自动 (ANE+GPU+CPU)", "仅 GPU+CPU", "仅 ANE+CPU", "仅 CPU"])
        computeUnitPopup.target = self
        computeUnitPopup.action = #selector(computeUnitChanged(_:))
        computeRow.addArrangedSubview(computeLabel)
        computeRow.addArrangedSubview(computeUnitPopup)

        let row2 = NSStackView(); row2.orientation = .horizontal; row2.spacing = 8
        maxNewLabel = NSTextField(labelWithString: "每轮生成上限: 64")
        maxNewStepper = NSStepper()
        maxNewStepper.minValue = 16; maxNewStepper.maxValue = 256; maxNewStepper.increment = 16
        maxNewStepper.integerValue = 64
        maxNewStepper.target = self; maxNewStepper.action = #selector(stepperChanged(_:))
        row2.addArrangedSubview(maxNewLabel)
        row2.addArrangedSubview(maxNewStepper)

        statusLabel = NSTextField(labelWithString: "未加载")
        statusLabel.font = NSFont.systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor

        transcriptView = NSTextView()
        transcriptView.isEditable = false
        transcriptView.isSelectable = true
        transcriptView.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        let scroll = NSScrollView()
        scroll.documentView = transcriptView
        scroll.hasVerticalScroller = true
        scroll.autoresizingMask = [.width, .height]
        scroll.setContentHuggingPriority(.defaultLow, for: .vertical)
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 240).isActive = true

        let inputRow = NSStackView(); inputRow.orientation = .horizontal; inputRow.spacing = 8
        promptField = NSTextField(string: "")
        promptField.placeholderString = "输入消息，回车或点「发送」…"
        promptField.font = NSFont.systemFont(ofSize: 13)
        promptField.target = self; promptField.action = #selector(sendTapped(_:))
        promptField.setContentHuggingPriority(.defaultLow, for: .horizontal)
        sendButton = NSButton(title: "发送", target: self, action: #selector(sendTapped(_:)))
        inputRow.addArrangedSubview(promptField)
        inputRow.addArrangedSubview(sendButton)

        root.addArrangedSubview(row1)
        root.addArrangedSubview(tokRow)
        root.addArrangedSubview(computeRow)
        root.addArrangedSubview(row2)
        root.addArrangedSubview(statusLabel)
        root.addArrangedSubview(scroll)
        root.addArrangedSubview(inputRow)

        bindController()
        updateUI()

        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        installSignalCleanup()   // 捕获 SIGTERM（pkill/注销），退出前清理 CoreML 缓存
    }

    /// 监听 SIGTERM（pkill / 系统注销 / kill 默认信号）：触发后清理编译缓存再退出，
    /// 弥补 applicationWillTerminate 在信号终止下不触发的问题。
    private func installSignalCleanup() {
        signal(SIGTERM, SIG_IGN)   // 屏蔽默认终止行为，交由下方 DispatchSource 派发
        let src = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        src.setEventHandler { [weak self] in
            self?.controller.cleanupCompiledCache()
            src.cancel()
            NSApp.terminate(nil)
        }
        src.resume()
        sigtermSource = src
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        controller.cleanupCompiledCache()   // 退出即清理 CoreML 编译缓存，释放磁盘
    }

    private func bindController() {
        controller.$status.receive(on: RunLoop.main)
            .sink { [weak self] s in self?.statusLabel.stringValue = s }.store(in: &cancellables)
        controller.$transcript.receive(on: RunLoop.main)
            .sink { [weak self] t in self?.transcriptView.string = t
                    self?.transcriptView.scrollToEndOfDocument(nil) }.store(in: &cancellables)
        controller.$modelName.receive(on: RunLoop.main).sink { [weak self] n in
            self?.selectedPathLabel.stringValue = n.isEmpty ? "（未选择模型文件）" : "已选: \(n)"
        }.store(in: &cancellables)
        controller.$tokenizerName.receive(on: RunLoop.main).sink { [weak self] n in
            self?.tokenizerLabel.stringValue = n.isEmpty ? "（默认 tokenizer）" : "分词器: \(n)"
        }.store(in: &cancellables)
        controller.$loaded.receive(on: RunLoop.main).sink { [weak self] _ in self?.updateUI() }.store(in: &cancellables)
        controller.$running.receive(on: RunLoop.main).sink { [weak self] _ in self?.updateUI() }.store(in: &cancellables)
        controller.$modelName.receive(on: RunLoop.main).sink { [weak self] _ in self?.updateUI() }.store(in: &cancellables)
    }

    private func updateUI() {
        let hasSelection = !controller.modelName.isEmpty
        let ok = controller.loaded && controller.modelName == controller.loadedName && !controller.running
        loadButton.isEnabled = hasSelection && !controller.running
        sendButton.isEnabled = ok
        resetButton.isEnabled = controller.loaded && !controller.running
        sendButton.title = controller.running ? "生成中…" : "发送"
    }

    /// 弹出文件选择框：选中转换产物 .mlpackage（或已编译 .mlmodelc）文件。
    @objc private func selectTapped(_ sender: Any) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false  // 选文件，不选目录（目录无法直接加载）
        panel.canChooseFiles = true          // .mlpackage 在 Finder 中作为包（文件）呈现，直接选该文件
        panel.canCreateDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "选择 modconv 转换产物：.mlpackage 或 .mlmodelc 文件"
        panel.beginSheetModal(for: self.window) { [weak self] resp in
            guard resp == .OK, let url = panel.url else { return }
            self?.controller.select(url: url)
        }
    }
    @objc private func loadTapped(_ sender: Any) {
        controller.loadSelected()
    }
    /// 弹出文件选择框：选中与所选模型同族的 tokenizer.json。
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
    @objc private func resetTapped(_ sender: Any) {
        controller.resetConversation()
    }
    @objc private func stepperChanged(_ sender: Any) {
        maxNewLabel.stringValue = "每轮生成上限: \(maxNewStepper.integerValue)"
    }
    /// 计算设备下拉：把选中项映射到 MLComputeUnits，供下次「加载模型」使用。
    @objc private func computeUnitChanged(_ sender: Any) {
        let units: [MLComputeUnits] = [.all, .cpuAndGPU, .cpuAndNeuralEngine, .cpuOnly]
        let idx = computeUnitPopup.indexOfSelectedItem
        guard idx >= 0, idx < units.count else { return }
        controller.selectedComputeUnit = units[idx]
    }
    @objc private func sendTapped(_ sender: Any) {
        let text = promptField.stringValue
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        promptField.stringValue = ""
        controller.chatAndPublish(userText: text, maxNew: maxNewStepper.integerValue)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
