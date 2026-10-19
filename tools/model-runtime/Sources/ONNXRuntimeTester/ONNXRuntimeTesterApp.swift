import AppKit
import OnnxRuntimeBindings
import SwiftUI

/// 应用入口。
///
/// 两种运行方式：
/// 1. GUI 模式（默认）：`swift run ONNXRuntimeTester`
/// 2. 命令行基准模式：
///    `swift run ONNXRuntimeTester --embedding <模型.onnx> --reranker <模型.onnx>`
@main
struct ONNXRuntimeTesterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    init() {
        let arguments = CommandLine.arguments
        if arguments.contains("--embedding") || arguments.contains("--reranker") {
            CLI.run()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .defaultSize(width: 1020, height: 760)
        .windowToolbarStyle(.unified)
    }
}

/// 让无 bundle 的可执行文件以前台应用方式显示窗口
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}
