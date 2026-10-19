import CoreML
import Foundation
import Tokenizers

/// CoreML 后端：直接用 Core ML 框架加载 .mlpackage / .mlmodelc 模型并推理。
///
/// 与 ONNX 后端共用 swift-transformers 分词器（引擎在加载后注入），
/// 区别仅在于推理走 Core ML 而非 ONNX Runtime。
final class CoreMLModelRuntime: @unchecked Sendable {
    let model: MLModel
    let kind: ModelKind
    let name: String
    let inputNames: [String]
    let outputNames: [String]
    let loadDuration: TimeInterval
    let fileBytes: Int64
    let coreMLCPUOnly: Bool

    /// 分词器（引擎注入）
    var tokenizer: (any TextTokenizer)?

    /// 从模型描述中识别出的输入/输出名
    private let idInputName: String
    private let maskInputName: String
    private let typeInputName: String?
    private let outputName: String
    /// 模型固定的输入形状（部分 HF→CoreML 转换为固定 [batch, seq]，如 [1, 512]）；
    /// 为 nil 表示支持动态/灵活形状。
    private let fixedBatch: Int?
    private let fixedSeq: Int?

    /// 首次 embedding 推理后缓存的向量维度
    private var cachedDimension: Int = 0

    private init(
        model: MLModel,
        kind: ModelKind,
        name: String,
        coreMLCPUOnly: Bool,
        loadDuration: TimeInterval,
        fileBytes: Int64,
        idInputName: String,
        maskInputName: String,
        typeInputName: String?,
        outputName: String,
        fixedBatch: Int?,
        fixedSeq: Int?
    ) {
        self.model = model
        self.kind = kind
        self.name = name
        self.coreMLCPUOnly = coreMLCPUOnly
        self.loadDuration = loadDuration
        self.fileBytes = fileBytes
        self.idInputName = idInputName
        self.maskInputName = maskInputName
        self.typeInputName = typeInputName
        self.outputName = outputName
        self.fixedBatch = fixedBatch
        self.fixedSeq = fixedSeq
        self.inputNames = model.modelDescription.inputDescriptionsByName.keys.sorted()
        self.outputNames = model.modelDescription.outputDescriptionsByName.keys.sorted()
    }

    static func load(url: URL, kind: ModelKind, cpuOnly: Bool) throws -> CoreMLModelRuntime {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = cpuOnly ? .cpuOnly : .all

        // .mlpackage 是未编译的源格式，MLModel(contentsOf:) 无法直接加载；
        // 先用 MLModel.compileModel 编译为 .mlmodelc（首次较慢，之后走磁盘缓存）。
        let modelURL = try Self.loadableCoreMLURL(for: url)

        let start = ContinuousClock.now
        let model: MLModel
        do {
            model = try MLModel(contentsOf: modelURL, configuration: configuration)
        } catch {
            throw CoreMLRuntimeError.loadFailed("\(error.localizedDescription)")
        }
        let elapsed = start.duration(to: .now)
        let loadDuration = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18

        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName

        let idInputName = inputs.keys.first(where: { $0.lowercased().contains("input_ids") || $0.lowercased() == "ids" })
            ?? inputs.keys.first ?? "input_ids"
        let maskInputName = inputs.keys.first(where: { $0.lowercased().contains("attention_mask") || $0.lowercased().contains("mask") })
            ?? "attention_mask"
        let typeInputName = inputs.keys.first(where: { $0.lowercased().contains("token_type_ids") || $0.lowercased().contains("segment_ids") })

        let outputName: String
        if kind == .embedding {
            outputName = outputs.keys.first(where: {
                let l = $0.lowercased()
                return l.contains("hidden") || l.contains("embedding") || l.contains("pool") || l.contains("last_hidden")
            }) ?? outputs.keys.first ?? "output"
        } else {
            outputName = outputs.keys.first(where: {
                let l = $0.lowercased()
                return l.contains("logit") || l.contains("score") || l.contains("output")
            }) ?? outputs.keys.first ?? "output"
        }

        let fileBytes = directorySize(url)

        // 读取模型固定的输入形状（如 HF→CoreML 转换的 [1, 512]），用于按固定 batch/seq 构造输入
        let idShape = Self.fixedShape(inputs[idInputName])
        let fixedBatch = (idShape?.count == 2) ? idShape?[0] : nil
        let fixedSeq = (idShape?.count == 2) ? idShape?[1] : nil

        return CoreMLModelRuntime(
            model: model,
            kind: kind,
            name: url.lastPathComponent,
            coreMLCPUOnly: cpuOnly,
            loadDuration: loadDuration,
            fileBytes: fileBytes,
            idInputName: idInputName,
            maskInputName: maskInputName,
            typeInputName: typeInputName,
            outputName: outputName,
            fixedBatch: fixedBatch,
            fixedSeq: fixedSeq
        )
    }

    /// 从输入特征描述中提取固定形状（[batch, seq]）。
    /// CoreML 用 `MLMultiArrayShapeConstraint.enumeratedShapes` 表示固定形状（单个枚举值）；
    /// 无法确定固定形状时返回 nil（按动态/灵活形状处理）。
    private static func fixedShape(_ desc: MLFeatureDescription?) -> [Int]? {
        guard let constraint = desc?.multiArrayConstraint?.shapeConstraint else { return nil }
        if let first = constraint.enumeratedShapes.first, first.count == 2 {
            return first.map { $0.intValue }
        }
        return nil
    }

    /// 返回可直接被 `MLModel(contentsOf:)` 加载的 URL：
    /// - `.mlmodelc`：已编译，直接使用；
    /// - `.mlpackage` / `.mlmodel`：先编译为 `.mlmodelc`，并缓存到
    ///   `~/Library/Caches/ONNXRuntimeTester/CoreML/<hash>.mlmodelc`，避免每次加载重复编译。
    private static func loadableCoreMLURL(for url: URL) throws -> URL {
        if url.pathExtension.lowercased() == "mlmodelc" {
            return url
        }
        // 先校验是否为真正的 CoreML 模型包，避免把畸形包交给 MLModel.compileModel
        // （某些"转换器"只把权重/源码塞进 .mlpackage 目录，Manifest 无 fileFormatVersion，
        //   compileModel 解析 Manifest 会抛未捕获的 C++ 异常直接崩溃）。
        try Self.validateCoreMLPackage(url)

        // 命中缓存直接复用（键含路径与修改时间，模型更新后会自动重编）
        if let cached = cachedCompiledURL(for: url), FileManager.default.fileExists(atPath: cached.path) {
            return cached
        }

        let compiled: URL
        do {
            compiled = try MLModel.compileModel(at: url)
        } catch {
            throw CoreMLRuntimeError.loadFailed("编译 CoreML 模型失败：\(error.localizedDescription)")
        }

        // 尝试把编译结果搬到缓存目录；搬失败就退回 compileModel 返回的临时路径
        if let cached = cachedCompiledURL(for: url) {
            do {
                try FileManager.default.createDirectory(at: cached.deletingLastPathComponent(), withIntermediateDirectories: true)
                if FileManager.default.fileExists(atPath: cached.path) {
                    try FileManager.default.removeItem(at: cached)
                }
                try FileManager.default.moveItem(at: compiled, to: cached)
                return cached
            } catch {
                return compiled
            }
        }
        return compiled
    }

    /// 校验是否为可加载的 CoreML 模型包：必须包含模型程序 `Data/com.apple.CoreML/*.mlmodel`。
    /// 只打包了权重/源码、没有生成 CoreML 计算图的伪 .mlpackage 会在此被拦下并给出可读错误，
    /// 而不是让 `MLModel.compileModel` 解析畸形 Manifest 时崩溃。
    private static func validateCoreMLPackage(_ url: URL) throws {
        let coreMLDir = url.appendingPathComponent("Data/com.apple.CoreML")
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: coreMLDir.path, isDirectory: &isDir), isDir.boolValue else {
            throw CoreMLRuntimeError.loadFailed("不是有效的 CoreML 模型包：缺少 Data/com.apple.CoreML 目录（\(url.lastPathComponent)）")
        }
        let hasProgram = (try? FileManager.default.contentsOfDirectory(atPath: coreMLDir.path))?
            .contains { $0.hasSuffix(".mlmodel") } ?? false
        guard hasProgram else {
            throw CoreMLRuntimeError.loadFailed(
                "CoreML 模型包缺少模型程序（Data/com.apple.CoreML/*.mlmodel）——这可能是只打包了权重/源码、并未生成 CoreML 计算图的伪 .mlpackage，无法加载。请用 coremltools 真正转换为 Core ML，或改用 ONNX 版。"
            )
        }
    }

    /// 编译缓存位置（稳定哈希，跨进程一致）
    private static func cachedCompiledURL(for url: URL) -> URL? {
        guard let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            return nil
        }
        let mtime = (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)?
            .timeIntervalSince1970 ?? 0
        let key = stableHash("\(url.standardizedFileURL.path)|\(Int(mtime))")
        return base.appendingPathComponent("ONNXRuntimeTester/CoreML/\(key).mlmodelc")
    }

    /// FNV-1a 稳定哈希（不依赖进程随机种子，可作持久缓存键）
    private static func stableHash(_ string: String) -> String {
        var hash: UInt64 = 1469598103934665603
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return String(hash, radix: 16)
    }

    var info: LoadedModel {
        LoadedModel(
            kind: kind,
            backend: .coreml,
            name: name,
            path: "",
            inputNames: inputNames,
            outputNames: outputNames,
            coreMLEnabled: true,
            loadDuration: loadDuration,
            fileBytes: fileBytes
        )
    }

    /// 把一组编码结果整理为模型需要的形状：每条截断/补齐到 `length`，
    /// 行数补齐到 `groupSize`（固定 batch 模型的最后一批用全 0 行占位，输出时丢弃）。
    /// 返回的 realCount 为真实条目数（不含占位行）。
    private func padRows(
        _ batch: [EncodedText],
        length: Int,
        groupSize: Int
    ) -> (ids: [[Int]], masks: [[Int]], types: [[Int]], realCount: Int) {
        func fit(_ row: [Int]) -> [Int] {
            if row.count > length { return Array(row.prefix(length)) }
            if row.count < length { return row + [Int](repeating: 0, count: length - row.count) }
            return row
        }
        var ids: [[Int]] = []
        var masks: [[Int]] = []
        var types: [[Int]] = []
        for item in batch {
            ids.append(fit(item.ids))
            masks.append(fit(item.attentionMask))
            types.append(fit(item.tokenTypeIds))
        }
        let zero = [Int](repeating: 0, count: length)
        while ids.count < groupSize {
            ids.append(zero)
            masks.append(zero)
            types.append(zero)
        }
        return (ids, masks, types, batch.count)
    }

    /// 用 CoreML 模型推理一个批次，返回原始输出 MLMultiArray
    private func runBatch(ids: [[Int]], masks: [[Int]], types: [[Int]]) throws -> MLMultiArray {
        let batch = ids.count
        let seq = ids[0].count

        func makeArray(_ rows: [[Int]]) throws -> MLMultiArray {
            let array = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: seq)], dataType: .int32)
            let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: batch * seq)
            var offset = 0
            for row in rows {
                for v in row {
                    pointer[offset] = Int32(v)
                    offset += 1
                }
            }
            return array
        }

        var features: [String: MLFeatureValue] = [:]
        features[idInputName] = try MLFeatureValue(multiArray: makeArray(ids))
        features[maskInputName] = try MLFeatureValue(multiArray: makeArray(masks))
        if let typeInputName {
            features[typeInputName] = try MLFeatureValue(multiArray: makeArray(types))
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw CoreMLRuntimeError.runFailed("\(error.localizedDescription)")
        }

        guard let value = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw CoreMLRuntimeError.outputReadFailed("未找到输出特征：\(outputName)")
        }
        return value
    }

    private func readFloatArray(_ array: MLMultiArray) -> (floats: [Float], shape: [Int]) {
        let shape = array.shape.map { $0.intValue }
        let count = shape.reduce(1, *)
        var floats = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { floats[i] = pointer[i] }
        case .float16:
            let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { floats[i] = Float(Float16(bitPattern: pointer[i])) }
        case .double:
            let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { floats[i] = Float(pointer[i]) }
        default:
            break
        }
        return (floats, shape)
    }
}

enum CoreMLRuntimeError: LocalizedError {
    case loadFailed(String)
    case runFailed(String)
    case outputReadFailed(String)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let m): return "CoreML 模型加载失败：\(m)"
        case .runFailed(let m): return "CoreML 推理失败：\(m)"
        case .outputReadFailed(let m): return "CoreML 读取输出失败：\(m)"
        }
    }
}

extension CoreMLModelRuntime: ModelRuntime {
    var backend: BackendKind { .coreml }

    var statusDescription: String { "CoreML\(coreMLCPUOnly ? " · 仅 CPU" : "")" }

    func embed(texts: [String], maxLength: Int) async throws -> (vectors: [[Float]], tokens: Int) {
        guard let tokenizer else {
            throw CoreMLRuntimeError.runFailed("分词器未加载，无法构造输入")
        }
        let encoded = texts.map { tokenizer.encode($0, maxLength: maxLength) }
        // 固定 batch 的模型（如 [1, 512]）逐条/按固定批量推理；灵活形状则一次整批
        let groupSize = fixedBatch ?? max(1, encoded.count)
        let targetLen = fixedSeq ?? (encoded.map { $0.ids.count }.max() ?? 1)

        var vectors: [[Float]] = []
        var totalTokens = 0
        var start = 0
        while start < encoded.count {
            let group = Array(encoded[start..<min(start + groupSize, encoded.count)])
            let (ids, masks, types, real) = padRows(group, length: targetLen, groupSize: groupSize)
            let output = try runBatch(ids: ids, masks: masks, types: types)
            let (floats, shape) = readFloatArray(output)
            let pooled = poolEmbeddingOutput(floats, shape: shape, masks: masks)
            for i in 0..<real {
                if i < pooled.count { vectors.append(pooled[i]) }
                totalTokens += masks[i].reduce(0, +)
            }
            start += groupSize
        }
        if let first = vectors.first { cachedDimension = first.count }
        return (vectors, totalTokens)
    }

    func rerank(pairs: [(query: String, document: String)], maxLength: Int) async throws -> (scores: [Float], tokens: Int) {
        guard let tokenizer else {
            throw CoreMLRuntimeError.runFailed("分词器未加载，无法构造输入")
        }
        let encoded = pairs.map { tokenizer.encodePair(query: $0.query, document: $0.document, maxLength: maxLength) }
        let groupSize = fixedBatch ?? max(1, encoded.count)
        let targetLen = fixedSeq ?? (encoded.map { $0.ids.count }.max() ?? 1)

        var scores: [Float] = []
        var totalTokens = 0
        var start = 0
        while start < encoded.count {
            let group = Array(encoded[start..<min(start + groupSize, encoded.count)])
            let (ids, masks, types, real) = padRows(group, length: targetLen, groupSize: groupSize)
            let output = try runBatch(ids: ids, masks: masks, types: types)
            let (floats, _) = readFloatArray(output)
            scores.append(contentsOf: floats.prefix(real))
            for i in 0..<real { totalTokens += masks[i].reduce(0, +) }
            start += groupSize
        }
        return (scores, totalTokens)
    }
}

/// 计算文件或目录（.mlpackage / .mlmodelc）的总字节数
private func directorySize(_ url: URL) -> Int64 {
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
