import AppKit
import Foundation
import OnnxRuntimeBindings
import SwiftUI
import UniformTypeIdentifiers

/// 主界面状态与交互逻辑
@MainActor
final class ONNXTesterViewModel: ObservableObject {

    // 两个模型配置槽：每个槽独立选择角色（Embedding/Reranker）与后端
    @Published var slots: [ModelSlot] = [
        ModelSlot(role: .embedding, backend: .onnx, path: ""),
        ModelSlot(role: .reranker, backend: .onnx, path: ""),
    ]

    // CoreML 配置
    @Published var coreMLEnabled = true
    @Published var coreMLCPUOnly = false
    @Published var isCoreMLAvailable = false
    @Published var ortVersion = ""

    // 基准测试参数
    @Published var batchSize = 16
    @Published var testCount = 128
    @Published var maxLength = 128

    // 状态
    @Published var isLoading = false
    @Published var isBenchmarking = false
    @Published var benchmarkProgress = ""
    @Published var loadReport: LoadReport?
    @Published var embeddingResults: [Int: EmbeddingBenchmark] = [:]
    @Published var rerankerResults: [Int: RerankerBenchmark] = [:]
    @Published var logs: [LogEntry] = []

    private var engine: BenchmarkEngine?

    init() {
        isCoreMLAvailable = ORTIsCoreMLExecutionProviderAvailable()
        ortVersion = ORTVersion() ?? "未知"
        log("ONNX Runtime \(ortVersion) 已就绪", level: .info)
        log(
            isCoreMLAvailable
                ? "CoreML 执行提供程序可用"
                : "CoreML 执行提供程序不可用，将使用 CPU", level: isCoreMLAvailable ? .success : .warning)
    }

    // MARK: - 模型选择

    /// 为第 index 个模型槽选择模型文件（角色与后端取该槽当前配置）
    func chooseModel(for index: Int) {
        guard slots.indices.contains(index) else { return }
        let slot = slots[index]
        chooseModel(title: "选择模型（\(slot.role.rawValue) · \(slot.backend.displayName)）") { path in
            self.slots[index].path = path
        }
    }

    private func chooseModel(title: String, completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.title = title
        panel.prompt = "选择"
        // 允许选择 .onnx / .mlpackage 文件，也允许选择模型目录（MLX / .mlmodelc 为目录）
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        let onnxType = UTType(filenameExtension: "onnx") ?? .data
        let mlpackageType = UTType(filenameExtension: "mlpackage") ?? .data
        panel.allowedContentTypes = [onnxType, mlpackageType]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        completion(url.path)
    }

    // MARK: - 加载与测试

    /// 点击「加载并测试」：加载模型，成功后自动生成测试数据并测速
    func loadAndBenchmark() {
        guard !isLoading, !isBenchmarking else { return }
        guard slots.contains(where: { !$0.path.isEmpty }) else {
            log("请先选择至少一个模型文件", level: .error)
            return
        }

        isLoading = true
        embeddingResults = [:]
        rerankerResults = [:]

        Task { [weak self] in
            guard let self else { return }
            do {
                let engine = try BenchmarkEngine()
                self.engine = engine
                let coreMLDesc = self.coreMLCPUOnly
                    ? "仅 CPU（纯 CPU）"
                    : (self.coreMLEnabled ? "CoreML 开启" : "CoreML 关闭（纯 CPU）")
                self.log("正在加载模型（\(coreMLDesc)）…", level: .info)

                let requests: [ModelLoadRequest?] = self.slots.map { slot in
                    slot.path.isEmpty
                        ? nil
                        : ModelLoadRequest(path: slot.path, backend: slot.backend, kind: slot.role)
                }

                let report = try await engine.load(requests, coreML: self.currentCoreMLOptions())
                self.loadReport = report

                for (index, model) in report.models.enumerated() {
                    self.log(
                        "模型 \(index + 1)（\(model.kind.rawValue) · \(model.backend.rawValue)）加载完成：\(model.name)（\(String(format: "%.2f", model.loadDuration))s，\(model.statusDescription)）",
                        level: .success)
                    self.log("  输入：\(model.inputNames.joined(separator: ", "))", level: .info)
                    self.log("  输出：\(model.outputNames.joined(separator: ", "))", level: .info)
                }
                self.log("分词器：\(report.tokenizerDescription)", level: .info)

                self.isLoading = false
                // 加载成功后自动开始测速
                self.runBenchmark()
            } catch {
                self.isLoading = false
                let message = error.localizedDescription
                self.log("加载失败：\(message)\(coreMLCompatibilityHint(message))", level: .error)
            }
        }
    }

    /// 用当前参数重新测速
    func runBenchmark() {
        guard !isBenchmarking, !isLoading else { return }
        guard let engine else {
            log("请先加载模型", level: .error)
            return
        }

        isBenchmarking = true
        benchmarkProgress = "准备中…"
        embeddingResults = [:]
        rerankerResults = [:]

        Task { [weak self] in
            guard let self else { return }
            let config = BenchmarkConfig(
                batchSize: self.batchSize,
                testCount: self.testCount,
                maxLength: self.maxLength
            )

            for await event in engine.runBenchmarks(config: config) {
                switch event {
                case .progress(let message):
                    self.benchmarkProgress = message
                    self.log(message, level: .info)
                case .embeddingFinished(let slot, let result):
                    self.embeddingResults[slot] = result
                    self.log(
                        "模型 \(slot + 1) Embedding 测速完成：\(String(format: "%.1f", result.metrics.itemsPerSecond)) 句/s · \(String(format: "%.1f", result.metrics.tokensPerSecond)) tokens/s · 平均延迟 \(String(format: "%.1f", result.metrics.avgLatencyMs))ms",
                        level: .success)
                case .rerankerFinished(let slot, let result):
                    self.rerankerResults[slot] = result
                    self.log(
                        "模型 \(slot + 1) Reranker 测速完成：\(String(format: "%.1f", result.metrics.itemsPerSecond)) 对/s · \(String(format: "%.1f", result.metrics.tokensPerSecond)) tokens/s · 平均延迟 \(String(format: "%.1f", result.metrics.avgLatencyMs))ms",
                        level: .success)
                case .failed(let message):
                    self.log("测速失败：\(message)", level: .error)
                }
            }

            self.isBenchmarking = false
            self.benchmarkProgress = ""
        }
    }

    func unload() {
        guard !isBenchmarking, !isLoading else { return }
        let engine = self.engine
        self.engine = nil
        loadReport = nil
        embeddingResults = [:]
        rerankerResults = [:]
        log("已卸载模型", level: .info)
        Task {
            await engine?.unload()
        }
    }

    // MARK: - 工具

    private func currentCoreMLOptions() -> CoreMLOptions {
        // 「仅使用 CPU」为独立开关：打开时直接不追加 CoreML EP（纯 CPU 运行），
        // 不再只是 CoreML 的 CPU-only 子选项，避免语义混乱
        CoreMLOptions(enabled: coreMLEnabled && !coreMLCPUOnly, cpuOnly: coreMLCPUOnly)
    }

    func log(_ message: String, level: LogEntry.Level = .info) {
        logs.append(LogEntry(message: message, level: level))
        if logs.count > 200 {
            logs.removeFirst(logs.count - 200)
        }
    }

    var canStart: Bool {
        slots.contains { !$0.path.isEmpty } && !isLoading && !isBenchmarking
    }
}
