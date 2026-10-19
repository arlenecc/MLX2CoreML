import SwiftUI

/// 顶部模型加载区：模型选择、CoreML 开关、加载按钮与状态
struct ModelLoadView: View {
    @ObservedObject var viewModel: ONNXTesterViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // 标题行
            HStack(spacing: 8) {
                Image(systemName: "cube.box.fill")
                    .font(.title3)
                    .foregroundStyle(Color.accentColor)
                Text("模型测速对比工具")
                    .font(.headline)
                Text("ONNX / MLX / CoreML · Embedding / Reranker 推理测速")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("ORT \(viewModel.ortVersion)")
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(Capsule())
                Label(
                    viewModel.isCoreMLAvailable ? "CoreML 可用" : "CoreML 不可用",
                    systemImage: viewModel.isCoreMLAvailable ? "checkmark.circle.fill" : "xmark.circle.fill"
                )
                .font(.caption)
                .foregroundStyle(viewModel.isCoreMLAvailable ? .green : .secondary)
            }

            // 模型 1 / 模型 2：每个槽独立选择角色（Embedding/Reranker）与后端
            ForEach(Array(viewModel.slots.enumerated()), id: \.element.id) { index, _ in
                modelRow(
                    slot: $viewModel.slots[index],
                    index: index,
                    chooseAction: { viewModel.chooseModel(for: index) }
                )
            }

            // CoreML 与参数
            HStack(spacing: 16) {
                Toggle("ONNX 启用 CoreML EP", isOn: $viewModel.coreMLEnabled)
                    .toggleStyle(.checkbox)
                VStack(alignment: .leading, spacing: 2) {
                    Toggle("仅使用 CPU", isOn: $viewModel.coreMLCPUOnly)
                        .toggleStyle(.checkbox)
                    Text("对 ONNX 与 CoreML 有效，对 MLX 无效")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                Spacer()

                HStack(spacing: 6) {
                    Text("批大小")
                    TextField("", value: $viewModel.batchSize, format: .number)
                        .frame(width: 48)
                        .textFieldStyle(.roundedBorder)
                    Text("测试数量")
                    TextField("", value: $viewModel.testCount, format: .number)
                        .frame(width: 56)
                        .textFieldStyle(.roundedBorder)
                    Text("最大长度")
                    TextField("", value: $viewModel.maxLength, format: .number)
                        .frame(width: 48)
                        .textFieldStyle(.roundedBorder)
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                Button(action: viewModel.loadAndBenchmark) {
                    if viewModel.isLoading {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 14, height: 14)
                        Text("加载中…")
                    } else {
                        Image(systemName: "play.fill")
                        Text(viewModel.loadReport == nil ? "加载并测试" : "重新加载并测试")
                    }
                }
                .disabled(!viewModel.canStart)
                .keyboardShortcut(.return, modifiers: .command)

                if viewModel.loadReport != nil {
                    Button("卸载") { viewModel.unload() }
                        .disabled(viewModel.isBenchmarking || viewModel.isLoading)
                }
            }

            // 状态行
            if let report = viewModel.loadReport {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(report.models.enumerated()), id: \.element.id) { index, model in
                        modelStatusLine(model, slotIndex: index)
                    }
                    Label("分词器：\(report.tokenizerDescription)", systemImage: "textformat")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Label("选择 Embedding / Reranker 模型后点击「加载并测试」", systemImage: "circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func modelRow(
        slot: Binding<ModelSlot>,
        index: Int,
        chooseAction: @escaping () -> Void
    ) -> some View {
        HStack(spacing: 8) {
            Text("模型 \(index + 1)")
                .frame(width: 54, alignment: .leading)
            Picker("角色", selection: slot.role) {
                ForEach(ModelKind.allCases) { kind in
                    Text(kind.rawValue).tag(kind)
                }
            }
            .labelsHidden()
            .frame(width: 96)
            Picker("后端", selection: slot.backend) {
                ForEach(BackendKind.allCases) { kind in
                    Text(kind.displayName).tag(kind)
                }
            }
            .labelsHidden()
            .frame(width: 92)
            TextField("模型文件或目录", text: slot.path)
                .textFieldStyle(.roundedBorder)
            Button("选择…", action: chooseAction)
        }
    }

    private func modelStatusLine(_ model: LoadedModel, slotIndex: Int) -> some View {
        HStack(spacing: 10) {
            Label("模型 \(slotIndex + 1) · \(model.kind.rawValue)（\(model.backend.rawValue)）：\(model.name)", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Label(StatFormatter.formatBytes(UInt64(model.fileBytes)), systemImage: "internaldrive")
            Label("加载 \(String(format: "%.2f", model.loadDuration))s", systemImage: "clock")
            Label(model.statusDescription, systemImage: model.backend == .mlx ? "cpu" : (model.coreMLEnabled ? "bolt.fill" : "cpu"))
                .foregroundStyle(model.backend == .mlx ? .secondary : (model.coreMLEnabled ? Color.accentColor : .secondary))
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}
