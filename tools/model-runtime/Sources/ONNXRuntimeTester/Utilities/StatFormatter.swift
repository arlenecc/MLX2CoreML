import Foundation

/// 数值格式化工具
enum StatFormatter {
    /// 字节数的人类可读格式，如 "1.2 GB"
    static func formatBytes(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    /// 保留一位小数
    static func formatOne(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    /// 秒，保留两位小数
    static func formatSeconds(_ value: TimeInterval) -> String {
        String(format: "%.2f", value)
    }

    /// 相似度，保留四位小数
    static func formatSimilarity(_ value: Float) -> String {
        String(format: "%.4f", value)
    }
}
