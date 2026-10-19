import SwiftUI

/// 中部基准测试结果区：性能指标卡片 + 效果样例
struct BenchmarkResultView: View {
    @ObservedObject var viewModel: ONNXTesterViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if viewModel.isBenchmarking {
                    HStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.small)
                        Text(viewModel.benchmarkProgress)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                // Embedding 对比与结果
                if !viewModel.embeddingResults.isEmpty {
                    if viewModel.embeddingResults.count == 2,
                        let a = viewModel.embeddingResults[0],
                        let b = viewModel.embeddingResults[1] {
                        compareBanner(
                            title: "Embedding 对比",
                            a: (a.modelLabel, a.metrics),
                            b: (b.modelLabel, b.metrics),
                            unit: "句/s"
                        )
                    }
                    ForEach(
                        Array(viewModel.embeddingResults.sorted(by: { $0.key < $1.key })),
                        id: \.key
                    ) { _, result in
                        embeddingSection(result)
                    }
                }

                // Reranker 对比与结果
                if !viewModel.rerankerResults.isEmpty {
                    if viewModel.rerankerResults.count == 2,
                        let a = viewModel.rerankerResults[0],
                        let b = viewModel.rerankerResults[1] {
                        compareBanner(
                            title: "Reranker 对比",
                            a: (a.modelLabel, a.metrics),
                            b: (b.modelLabel, b.metrics),
                            unit: "对/s"
                        )
                    }
                    ForEach(
                        Array(viewModel.rerankerResults.sorted(by: { $0.key < $1.key })),
                        id: \.key
                    ) { _, result in
                        rerankerSection(result)
                    }
                }

                if viewModel.embeddingResults.isEmpty && viewModel.rerankerResults.isEmpty
                    && !viewModel.isBenchmarking {
                    emptyState
                }
            }
            .padding(16)
        }
    }

    // MARK: - Embedding

    private func embeddingSection(_ result: EmbeddingBenchmark) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("Embedding 测速结果", icon: "character.cursor.ibeam")
            if !result.modelLabel.isEmpty {
                Text(result.modelLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(
                columns: [.init(.flexible()), .init(.flexible()), .init(.flexible()), .init(.flexible()), .init(.flexible())],
                spacing: 8
            ) {
                StatCard(
                    title: "向量维度",
                    value: "\(result.dimension)",
                    unit: "",
                    highlight: true,
                    systemImage: "ruler"
                )
                StatCard(
                    title: "吞吐",
                    value: StatFormatter.formatOne(result.metrics.itemsPerSecond),
                    unit: "句/s",
                    highlight: true,
                    systemImage: "speedometer"
                )
                StatCard(
                    title: "Token 吞吐",
                    value: StatFormatter.formatOne(result.metrics.tokensPerSecond),
                    unit: "tokens/s",
                    systemImage: "text.append"
                )
                StatCard(
                    title: "平均推理延迟",
                    value: StatFormatter.formatOne(result.metrics.avgLatencyMs),
                    unit: "ms/批",
                    systemImage: "timer"
                )
                StatCard(
                    title: "P90 延迟",
                    value: StatFormatter.formatOne(result.metrics.p90LatencyMs),
                    unit: "ms",
                    systemImage: "chart.bars"
                )
            }

            Text("测试配置：批大小 \(result.metrics.batchSize) · \(result.metrics.iterations) 轮 · 共 \(result.metrics.totalItems) 句 · \(result.metrics.totalTokens) tokens · 总耗时 \(StatFormatter.formatSeconds(result.metrics.totalSeconds))s")
                .font(.caption)
                .foregroundStyle(.secondary)

            if !result.similaritySamples.isEmpty {
                Text("相似度效果（余弦相似度）")
                    .font(.subheadline)
                    .fontWeight(.medium)
                VStack(spacing: 0) {
                    headerRow(["类型", "文本 A", "文本 B", "相似度"], widths: [70, nil, nil, 90])
                    ForEach(Array(result.similaritySamples.enumerated()), id: \.offset) { _, sample in
                        Divider()
                        sampleRow(
                            sample.label,
                            sample.textA,
                            sample.textB,
                            StatFormatter.formatSimilarity(sample.similarity),
                            accent: sample.label == "同主题" ? .green : .orange
                        )
                    }
                }
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
    }

    // MARK: - Reranker

    private func rerankerSection(_ result: RerankerBenchmark) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("Reranker 测速结果", icon: "arrow.up.arrow.down")
            if !result.modelLabel.isEmpty {
                Text(result.modelLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(
                columns: [.init(.flexible()), .init(.flexible()), .init(.flexible()), .init(.flexible())],
                spacing: 8
            ) {
                StatCard(
                    title: "吞吐",
                    value: StatFormatter.formatOne(result.metrics.itemsPerSecond),
                    unit: "对/s",
                    highlight: true,
                    systemImage: "speedometer"
                )
                StatCard(
                    title: "Token 吞吐",
                    value: StatFormatter.formatOne(result.metrics.tokensPerSecond),
                    unit: "tokens/s",
                    systemImage: "text.append"
                )
                StatCard(
                    title: "平均推理延迟",
                    value: StatFormatter.formatOne(result.metrics.avgLatencyMs),
                    unit: "ms/批",
                    systemImage: "timer"
                )
                StatCard(
                    title: "P90 延迟",
                    value: StatFormatter.formatOne(result.metrics.p90LatencyMs),
                    unit: "ms",
                    systemImage: "chart.bars"
                )
            }

            Text("测试配置：批大小 \(result.metrics.batchSize) · \(result.metrics.iterations) 轮 · 共 \(result.metrics.totalItems) 对 · \(result.metrics.totalTokens) tokens · 总耗时 \(StatFormatter.formatSeconds(result.metrics.totalSeconds))s")
                .font(.caption)
                .foregroundStyle(.secondary)

            if !result.rankedSamples.isEmpty {
                Text("排序效果（query：\(result.rankedSamples[0].query)）")
                    .font(.subheadline)
                    .fontWeight(.medium)
                VStack(spacing: 0) {
                    headerRow(["排名", "候选文档", "得分"], widths: [50, nil, 90])
                    ForEach(Array(result.rankedSamples.enumerated()), id: \.offset) { index, sample in
                        Divider()
                        sampleRow(
                            "\(index + 1)",
                            sample.document,
                            "",
                            String(format: "%.4f", sample.score),
                            accent: index == 0 ? .accentColor : .secondary
                        )
                    }
                }
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
    }

    // MARK: - 样例行

    private func headerRow(_ titles: [String], widths: [CGFloat?]) -> some View {
        HStack(spacing: 8) {
            ForEach(Array(titles.enumerated()), id: \.offset) { index, title in
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(width: widths[index], alignment: .leading)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func sampleRow(
        _ first: String,
        _ second: String,
        _ third: String,
        _ fourth: String,
        accent: Color
    ) -> some View {
        HStack(spacing: 8) {
            Text(first)
                .font(.caption2)
                .foregroundStyle(accent)
                .frame(width: 70, alignment: .leading)
            Text(second)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
            if !third.isEmpty {
                Text(third)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text(fourth)
                .monospacedDigit()
                .frame(width: 90, alignment: .trailing)
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    // MARK: - 辅助

    private func sectionHeader(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.headline)
    }

    /// 两个同角色模型并排对比（吞吐更高者高亮标注「更快」）
    private func compareBanner(
        title: String,
        a: (label: String, metrics: BenchmarkMetrics),
        b: (label: String, metrics: BenchmarkMetrics),
        unit: String
    ) -> some View {
        let aWin = a.metrics.itemsPerSecond >= b.metrics.itemsPerSecond
        return VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: "arrow.left.arrow.right")
                .font(.subheadline)
                .fontWeight(.medium)
            HStack(spacing: 12) {
                compareItem(label: a.label, ips: a.metrics.itemsPerSecond, win: aWin, unit: unit)
                Divider()
                compareItem(label: b.label, ips: b.metrics.itemsPerSecond, win: !aWin, unit: unit)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.accentColor.opacity(0.08))
        )
    }

    private func compareItem(label: String, ips: Double, win: Bool, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(StatFormatter.formatOne(ips))
                    .font(.system(.title3, design: .rounded).monospacedDigit())
                    .fontWeight(win ? .semibold : .regular)
                    .foregroundStyle(win ? Color.accentColor : .primary)
                Text(unit)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if win {
                Text("更快")
                    .font(.caption2)
                    .foregroundStyle(.green)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "gauge.with.dots.needle.33percent")
                .font(.system(size: 40))
                .foregroundStyle(.tertiary)
            Text("加载模型后将自动生成测试数据，测量 Embedding 与 Reranker 的推理速度")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// 单个性能指标卡片
struct StatCard: View {
    let title: String
    let value: String
    let unit: String
    var highlight = false
    var systemImage: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Label(title, systemImage: systemImage)
                .font(.caption2)
                .foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(.title3, design: .rounded).monospacedDigit())
                    .fontWeight(highlight ? .semibold : .regular)
                    .foregroundStyle(highlight ? Color.accentColor : .primary)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(highlight ? Color.accentColor.opacity(0.12) : Color(nsColor: .controlBackgroundColor))
        )
    }
}
