import ChessAnalysis
import ChessBoardUI
import ChessKit
import SwiftUI

/// Streams Stockfish's evaluation of the reviewed position: eval bar, top
/// lines in SAN, and a best-move arrow on the board.
struct AnalysisView: View {
    @EnvironmentObject private var model: AppModel

    @State private var engine = AnalysisEngine()
    @State private var update: AnalysisUpdate?
    @State private var depth = 18
    @State private var analysisTask: Task<Void, Never>?

    private let depthOptions = [12, 18, 24]

    var body: some View {
        VStack(spacing: 16) {
            if !AnalysisEngine.evaluationNetworksArePresent {
                missingNetworksNotice
            }

            GeometryReader { proxy in
                let side = min(proxy.size.width - 28, proxy.size.height)
                HStack(alignment: .center, spacing: 12) {
                    EvalBarView(fraction: update?.score?.whiteWinningFraction ?? 0.5)
                        .frame(width: 16, height: side)
                    BoardView(
                        pieces: model.pieces,
                        orientation: model.orientation,
                        arrows: bestMoveArrows
                    )
                    .frame(width: side, height: side)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .padding(.horizontal, 12)
            .frame(maxHeight: 420)

            header

            linesList

            Spacer()
        }
        .navigationTitle("Analysis")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Picker("Depth", selection: $depth) {
                    ForEach(depthOptions, id: \.self) { d in
                        Text("Depth \(d)").tag(d)
                    }
                }
                .pickerStyle(.menu)
            }
        }
        .task { await runAnalysis() }
        .onChange(of: depth) {
            Task { await runAnalysis() }
        }
        .onDisappear {
            analysisTask?.cancel()
            Task { await engine.stopSearch() }
        }
    }

    // MARK: - Engine driving

    private func runAnalysis() async {
        analysisTask?.cancel()
        update = nil
        await engine.start(multiPV: 3)
        let fen = model.fen
        let targetDepth = depth
        analysisTask = Task {
            let stream = await engine.analyze(fen: fen, depth: targetDepth)
            for await value in stream {
                if Task.isCancelled { break }
                update = value
            }
        }
    }

    // MARK: - Subviews

    private var bestMoveArrows: [BoardArrow] {
        guard
            let lan = update?.bestLine?.firstMoveLAN,
            let squares = SANConverter.squares(ofLAN: lan)
        else { return [] }
        return [BoardArrow(from: squares.from, to: squares.to, color: .blue)]
    }

    private var header: some View {
        HStack {
            Text(update?.score?.displayString ?? "…")
                .font(.title2.monospacedDigit().bold())
            Spacer()
            if let update {
                Text(update.isFinal ? "depth \(update.depth) ✓" : "depth \(update.depth)…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                ProgressView()
            }
        }
        .padding(.horizontal, 24)
    }

    private var linesList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(update?.lines ?? []) { line in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(line.score.displayString)
                        .font(.subheadline.monospacedDigit().bold())
                        .foregroundStyle(line.rank == 1 ? .primary : .secondary)
                        .frame(width: 56, alignment: .trailing)
                    Text(numberedLine(line))
                        .font(.subheadline)
                        .lineLimit(2)
                        .foregroundStyle(line.rank == 1 ? .primary : .secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 24)
    }

    /// "1. e4 e5 2. Nf3 …" numbering that respects the side to move.
    private func numberedLine(_ line: PrincipalVariation) -> String {
        var parts: [String] = []
        var moveNumber = 1
        var whiteMove = model.sideToMove == .white
        if !whiteMove, let first = line.moves.first {
            parts.append("\(moveNumber)… \(first)")
            moveNumber += 1
            whiteMove = true
        }
        for san in line.moves.dropFirst(parts.isEmpty ? 0 : 1) {
            if whiteMove {
                parts.append("\(moveNumber). \(san)")
            } else {
                parts.append(san)
                moveNumber += 1
            }
            whiteMove.toggle()
        }
        return parts.joined(separator: " ")
    }

    private var missingNetworksNotice: some View {
        Label(
            "Stockfish networks missing — run Scripts/download-nnue.sh and rebuild.",
            systemImage: "exclamationmark.triangle.fill"
        )
        .font(.caption)
        .foregroundStyle(.orange)
        .padding(.horizontal, 24)
    }
}

/// Vertical eval bar: white's share fills from the bottom.
struct EvalBarView: View {
    let fraction: Double

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .bottom) {
                Rectangle().fill(Color.black.opacity(0.85))
                Rectangle()
                    .fill(Color.white)
                    .frame(height: proxy.size.height * fraction)
                    .animation(.easeOut(duration: 0.4), value: fraction)
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(Color.secondary.opacity(0.4), lineWidth: 1)
            )
        }
    }
}
