import SwiftUI

/// 底部日志区
struct LogView: View {
    @ObservedObject var viewModel: ONNXTesterViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label("运行日志", systemImage: "text.line.last.and.arrowtriangle.forward")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("清空") { viewModel.logs.removeAll() }
                    .font(.caption)
                    .buttonStyle(.borderless)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 3) {
                        ForEach(viewModel.logs) { entry in
                            HStack(alignment: .top, spacing: 6) {
                                Text(entry.timeString)
                                    .font(.caption2)
                                    .monospacedDigit()
                                    .foregroundStyle(.tertiary)
                                Text(entry.message)
                                    .font(.caption)
                                    .foregroundStyle(color(for: entry.level))
                                    .textSelection(.enabled)
                            }
                            .id(entry.id)
                        }
                    }
                }
                .onChange(of: viewModel.logs.last?.id) {
                    if let last = viewModel.logs.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(height: 130)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func color(for level: LogEntry.Level) -> Color {
        switch level {
        case .info: return .primary
        case .success: return .green
        case .warning: return .orange
        case .error: return .red
        }
    }
}
