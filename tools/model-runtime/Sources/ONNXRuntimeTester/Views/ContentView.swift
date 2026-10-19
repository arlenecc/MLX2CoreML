import SwiftUI

/// 主界面布局：上方模型加载区，中间基准测试结果，下方日志
struct ContentView: View {
    @StateObject private var viewModel = ONNXTesterViewModel()

    var body: some View {
        VStack(spacing: 0) {
            ModelLoadView(viewModel: viewModel)
            Divider()
            BenchmarkResultView(viewModel: viewModel)
            Divider()
            LogView(viewModel: viewModel)
        }
        .frame(minWidth: 940, minHeight: 660)
    }
}
