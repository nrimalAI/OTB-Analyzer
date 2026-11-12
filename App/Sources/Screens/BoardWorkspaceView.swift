import ChessAnalysis
import ChessBoardUI
import ChessKit
import SwiftUI

/// The board workspace: Position and Analysis are two tabs over one board
/// that never moves, so fixing a square after seeing the result is one tap
/// each way. The eval bar sits beside the board; Flip lives in the top bar.
struct BoardWorkspaceView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var history: HistoryStore

    @StateObject private var session = AnalysisSession()
    @State private var selectedSquare: Square?
    @State private var toast: String?
    @State private var toastTask: Task<Void, Never>?

    private var isAnalysis: Bool { model.tab == .analysis }

    var body: some View {
        VStack(spacing: 0) {
            topBar

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    heading
                    boardRow
                    if isAnalysis {
                        AnalysisPanel(session: session, showToast: showToast)
                    } else {
                        PositionPanel(selectedSquare: $selectedSquare)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 6)
                .padding(.bottom, 12)
            }
            .scrollDismissesKeyboard(.interactively)

            bottomBar
        }
        .background(Theme.paper.ignoresSafeArea())
        .foregroundStyle(Theme.ink)
        .toolbar(.hidden, for: .navigationBar)
        .overlay(alignment: .bottom) {
            if let toast {
                Toast(message: toast)
                    .padding(.bottom, 90)
                    .transition(.opacity)
                    .allowsHitTesting(false)
            }
        }
        .task {
            let history = history
            session.onFinished = { [weak session] update, fen in
                Self.save(update, fen: fen, depth: session?.targetDepth ?? update.depth, to: history)
            }
            if isAnalysis { await startAnalysis() }
        }
        .onChange(of: model.tab) { _, tab in
            selectedSquare = nil
            if tab == .analysis {
                Task { await startAnalysis() }
            } else {
                session.stop()
            }
        }
        .onChange(of: model.depth) {
            if isAnalysis { Task { await startAnalysis() } }
        }
        .onDisappear {
            session.stop()
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 6) {
            Button(action: goBack) {
                Text("\u{2039}")
                    .sans(26, weight: .medium)
                    .frame(width: 44, height: 44)
            }
            .buttonStyle(TextButtonStyle())
            .accessibilityLabel(backLabel)

            InkSegmentedControl(
                segments: [
                    .init(value: BoardTab.position, title: "Position"),
                    .init(value: BoardTab.analysis, title: "Analysis", isUnavailable: !model.canAnalyze),
                ],
                selection: model.tab,
                height: 40,
                onSelect: { model.tab = $0 }
            )

            Button {
                model.orientation.flip()
            } label: {
                Text("Flip")
                    .sans(15, weight: .medium)
                    .padding(.horizontal, 10)
                    .frame(height: 44)
            }
            .buttonStyle(TextButtonStyle())
            .accessibilityLabel("Flip board")
        }
        .padding(.horizontal, 6)
        .frame(height: 48)
    }

    /// From a scanned position being checked, back means retake; otherwise
    /// it returns to the list of positions.
    private var returnsToScan: Bool {
        !isAnalysis && model.fromScan && model.path.contains(.scan)
    }

    private var backLabel: String {
        returnsToScan ? "Retake photo" : "Back to positions"
    }

    private func goBack() {
        model.path = returnsToScan ? [.scan] : []
    }

    // MARK: - Heading

    private var heading: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .serif(30)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .frame(height: 35)
                .accessibilityAddTraits(.isHeader)
            Text(subtitle)
                .sans(13, weight: .medium)
                .foregroundStyle(subtitleIsWarning ? Theme.accent : Theme.muted)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .frame(height: 20)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.bottom, 10)
    }

    private var title: String {
        guard isAnalysis else {
            return model.fromScan ? "Check the position" : "Set up a position"
        }
        if !AnalysisEngine.evaluationNetworksArePresent { return "Engine not installed" }
        if let score = session.update?.score { return EvalFormat.verdict(score) }
        if session.update?.isFinal == true { return "No legal moves" }
        return "Analyzing\u{2026}"
    }

    private var subtitle: String {
        guard isAnalysis else { return positionStatus }
        if !AnalysisEngine.evaluationNetworksArePresent {
            return "Stockfish networks missing \u{2014} run Scripts/download-nnue.sh and rebuild."
        }
        guard let update = session.update else { return "Stockfish is thinking\u{2026}" }
        guard let best = session.lines.first else {
            return update.isFinal ? "The game is over in this position." : "Stockfish is thinking\u{2026}"
        }
        var parts: [String] = []
        if let move = best.moves.first { parts.append("Best move \(move)") }
        parts.append("depth \(session.reachedDepth)" + (update.isFinal ? "" : "\u{2026}"))
        parts.append("White win chance \(EvalFormat.whiteWinChance(best.score))%")
        return parts.joined(separator: " \u{00B7} ")
    }

    private var positionStatus: String {
        if let issue = model.validationIssues.first {
            return issue.message.hasSuffix(".") ? issue.message : issue.message + "."
        }
        let uncertain = model.uncertainSquares.count
        if uncertain > 0 {
            return "\(uncertain) \(uncertain == 1 ? "square" : "squares") to double-check"
        }
        return model.fromScan ? "Position looks legal." : "Tap a square, then pick a piece."
    }

    private var subtitleIsWarning: Bool {
        if isAnalysis { return !AnalysisEngine.evaluationNetworksArePresent }
        return !model.validationIssues.isEmpty || !model.uncertainSquares.isEmpty
    }

    // MARK: - Board

    private var boardRow: some View {
        HStack(spacing: 6) {
            EvalBarView(fraction: isAnalysis ? session.update?.score?.whiteWinningFraction ?? 0.5 : 0.5)
                .frame(width: 12)
            BoardView(
                pieces: shownPieces,
                orientation: model.orientation,
                selected: isAnalysis ? nil : selectedSquare,
                flagged: isAnalysis ? [] : model.uncertainSquares.union(model.problemSquares),
                highlighted: lastMoveSquares,
                arrows: nextMoveArrows,
                onTap: isAnalysis ? nil : { square in
                    selectedSquare = selectedSquare == square ? nil : square
                }
            )
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var shownPieces: [Square: Piece] {
        guard isAnalysis, let pieces = session.shownPieces else { return model.pieces }
        return pieces
    }

    /// From/to squares of the move just played while stepping through a line.
    private var lastMoveSquares: Set<Square> {
        guard isAnalysis, session.ply > 0,
              let squares = session.selectedLine?.squares(ofMoveAt: session.ply - 1)
        else { return [] }
        return [squares.from, squares.to]
    }

    /// The next move of the selected line (the best move, before stepping).
    private var nextMoveArrows: [BoardArrow] {
        guard isAnalysis,
              let squares = session.selectedLine?.squares(ofMoveAt: session.ply)
        else { return [] }
        return [BoardArrow(from: squares.from, to: squares.to)]
    }

    // MARK: - Bottom bar

    @ViewBuilder
    private var bottomBar: some View {
        if !isAnalysis {
            VStack(spacing: 0) {
                Rule()
                Button("Analyze position") {
                    model.tab = .analysis
                }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(!model.canAnalyze)
                .padding(.horizontal, 16)
                .padding(.top, 10)
                .padding(.bottom, 8)
            }
        } else if let line = session.selectedLine, line.playableCount > 0 {
            VStack(spacing: 0) {
                Rule()
                LineStepper(session: session, line: line, sideToMove: model.sideToMove)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
            }
        }
    }

    // MARK: - Actions

    private func startAnalysis() async {
        guard model.canAnalyze else {
            model.tab = .position
            return
        }
        await session.run(fen: model.fen, depth: model.depth)
    }

    /// Keeps finished analyses on the home screen. The untouched starting
    /// position isn't worth a row.
    private static func save(
        _ update: AnalysisUpdate, fen: String, depth: Int, to history: HistoryStore
    ) {
        guard let best = update.bestLine,
              let placement = fen.split(separator: " ").first,
              placement != Position.standard.fen.split(separator: " ").first
        else { return }
        history.record(
            fen: fen,
            score: best.score,
            bestMove: LinePlayback.line(from: best, fen: fen).moves.first,
            depth: min(update.depth, depth))
    }

    private func showToast(_ message: String) {
        toastTask?.cancel()
        withAnimation(.easeOut(duration: 0.15)) { toast = message }
        toastTask = Task {
            try? await Task.sleep(for: .seconds(1.6))
            guard !Task.isCancelled else { return }
            withAnimation(.easeIn(duration: 0.2)) { toast = nil }
        }
    }
}

/// Vertical eval bar: ink for Black, with White's share filling from the
/// bottom.
struct EvalBarView: View {
    let fraction: Double

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .bottom) {
                Theme.ink
                Rectangle()
                    .fill(Theme.field)
                    .frame(height: max(0, proxy.size.height - 2 * Theme.ruleWidth) * fraction)
                    .padding(.bottom, Theme.ruleWidth)
                    .padding(.horizontal, Theme.ruleWidth)
                    .animation(.easeOut(duration: 0.3), value: fraction)
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Evaluation bar")
        .accessibilityValue("White \(Int((fraction * 100).rounded())) percent")
    }
}

/// Steps through the selected line: back to start, back one, and a wide
/// ink key for the next move, with the move about to be played in between.
private struct LineStepper: View {
    @ObservedObject var session: AnalysisSession
    let line: AnalysisLine
    let sideToMove: Piece.Color

    private var ply: Int { session.ply }
    private var count: Int { line.playableCount }

    var body: some View {
        HStack(spacing: 8) {
            Button(action: session.stepToStart) {
                Text("\u{00AB}").sans(20, weight: .medium)
            }
            .buttonStyle(OutlineButtonStyle(cornerRadius: 10))
            .frame(width: 48, height: 48)
            .disabled(ply == 0)
            .accessibilityLabel("Back to start")

            Button(action: session.stepBack) {
                Text("\u{2039}").sans(22, weight: .medium)
            }
            .buttonStyle(OutlineButtonStyle(cornerRadius: 10))
            .frame(width: 48, height: 48)
            .disabled(ply == 0)
            .accessibilityLabel("Previous move")

            VStack(spacing: 0) {
                Text("Move \(ply) of \(count)")
                    .eyebrow()
                Text(nextLabel)
                    .serif(19)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .frame(maxWidth: .infinity)
            .accessibilityElement(children: .combine)

            Button(action: session.stepForward) {
                Text("\u{203A}")
                    .sans(22, weight: .medium)
                    .foregroundStyle(Theme.paper)
                    .frame(width: 76, height: 48)
                    .background(Theme.ink, in: RoundedRectangle(cornerRadius: 10))
            }
            .buttonStyle(TextButtonStyle())
            .disabled(ply >= count)
            .accessibilityLabel("Next move")
        }
    }

    private var nextLabel: String {
        guard ply < count, line.moves.indices.contains(ply) else { return "End of line" }
        return "Next " + EvalFormat.moveNumber(at: ply, sideToMove: sideToMove, always: true) + line.moves[ply]
    }
}
