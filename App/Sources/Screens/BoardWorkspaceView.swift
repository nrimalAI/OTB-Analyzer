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
    /// Asking who is to move before analysing a scan that nobody has said.
    @State private var askingSideToMove = false
    @State private var showingPhoto = false

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
            let model = model
            session.onFinished = { best, depth, fen in
                Self.save(best, depth: depth, fen: fen, photo: model.photoJPEG, to: history)
            }
            if isAnalysis { startAnalysis() }
        }
        .onChange(of: model.tab) { _, tab in
            selectedSquare = nil
            if tab == .analysis {
                startAnalysis()
            } else {
                session.stop()
            }
        }
        .onChange(of: model.depth) {
            if isAnalysis { startAnalysis() }
        }
        .onDisappear {
            session.stop()
        }
        .sheet(isPresented: $showingPhoto) {
            if let photo = model.photo {
                PhotoSheet(photo: photo)
            }
        }
        .confirmationDialog("Who is to move?", isPresented: $askingSideToMove, titleVisibility: .visible) {
            Button("White to move") { analyze(with: .white) }
            Button("Black to move") { analyze(with: .black) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("A photo doesn\u{2019}t show whose turn it is.")
        }
    }

    private func analyze(with side: Piece.Color) {
        model.chooseSideToMove(side)
        show(.analysis)
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 6) {
            Button(action: goBack) {
                HStack(spacing: 4) {
                    Text("\u{2039}")
                        .sans(26, weight: .medium)
                    // Retaking is the quickest fix for a poor reading, so
                    // the way back to the camera says so.
                    if returnsToScan {
                        Text("Retake")
                            .sans(15, weight: .medium)
                            .lineLimit(1)
                            .fixedSize()
                    }
                }
                .padding(.horizontal, returnsToScan ? 6 : 0)
                .frame(minWidth: 44, minHeight: 44)
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
                onSelect: show
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
        HStack(alignment: .center, spacing: 12) {
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

            if let photo = model.photo {
                photoButton(photo)
            }
        }
        .padding(.bottom, 10)
    }

    /// The photo the position was read from, small, beside the heading: to
    /// compare the board with it while checking, and to remember the game.
    private func photoButton(_ photo: UIImage) -> some View {
        Button {
            showingPhoto = true
        } label: {
            Image(uiImage: photo)
                .resizable()
                .scaledToFill()
                .frame(width: 46, height: 46)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(Theme.ink, lineWidth: Theme.ruleWidth))
        }
        .buttonStyle(TextButtonStyle())
        .accessibilityLabel("Photo")
        .accessibilityHint("Shows the photo this position was read from")
    }

    private var title: String {
        guard isAnalysis else {
            return model.fromScan ? "Check the position" : "Set up a position"
        }
        if !AnalysisEngine.evaluationNetworksArePresent { return "Engine not installed" }
        if let best = session.lines.first { return EvalFormat.verdict(best.score) }
        if session.isFinished { return "No legal moves" }
        return "Analyzing\u{2026}"
    }

    private var subtitle: String {
        guard isAnalysis else { return positionStatus }
        if !AnalysisEngine.evaluationNetworksArePresent {
            return "Stockfish networks missing \u{2014} run Scripts/download-nnue.sh and rebuild."
        }
        guard let best = session.lines.first else {
            return session.isFinished ? "The game is over in this position." : "Stockfish is thinking\u{2026}"
        }
        var parts: [String] = []
        if let move = best.moves.first { parts.append("Best move \(move)") }
        parts.append("depth \(session.reachedDepth)" + (session.isFinished ? "" : "\u{2026}"))
        parts.append("White win chance \(EvalFormat.whiteWinChance(best.score))%")
        return parts.joined(separator: " \u{00B7} ")
    }

    private var positionStatus: String {
        let problems = model.problems
        if problems.count == 1 { return problems[0].title + "." }
        if problems.count > 1 { return "\(problems.count) problems to fix before analysis" }
        let uncertain = model.uncertainSquares.count
        if uncertain >= Self.manyDoubtful, returnsToScan {
            return "\(uncertain) squares to double-check \u{2014} a retake may be quicker"
        }
        if uncertain > 0 {
            return "\(uncertain) \(uncertain == 1 ? "square" : "squares") to double-check"
                + (model.sideToMoveKnown ? "" : " \u{00B7} who is to move?")
        }
        if !model.sideToMoveKnown { return "Who is to move? Choose below." }
        return model.fromScan ? "Position looks legal." : "Tap a square, then pick a piece."
    }

    private var subtitleIsWarning: Bool {
        if isAnalysis { return !AnalysisEngine.evaluationNetworksArePresent }
        return !model.validationIssues.isEmpty || !model.uncertainSquares.isEmpty
            || !model.sideToMoveKnown
    }

    // MARK: - Board

    private var boardRow: some View {
        HStack(spacing: 6) {
            EvalBarView(fraction: isAnalysis ? session.lines.first?.score.whiteWinningFraction ?? 0.5 : 0.5)
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
        // In Analysis a swipe across the board steps through the line:
        // leftward for the next move, rightward for the one before.
        .simultaneousGesture(stepSwipe, including: isAnalysis ? .all : .subviews)
    }

    private var stepSwipe: some Gesture {
        DragGesture(minimumDistance: 24)
            .onEnded { drag in
                let dx = drag.translation.width
                guard abs(dx) > 40, abs(dx) > 2 * abs(drag.translation.height) else { return }
                if dx < 0 { session.stepForward() } else { session.stepBack() }
            }
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
                // Why Analyze is off, right above it: the first problem with
                // its fix, and how many there are. One row, so that the tray
                // stays in view on a small phone.
                let problems = model.problems
                if let first = problems.first {
                    ProblemRow(problem: first, count: problems.count, apply: apply)
                        .padding(.horizontal, 16)
                        .padding(.top, 6)
                }
                Group {
                    if model.sideToMoveKnown {
                        Button("Analyze position") {
                            show(.analysis)
                        }
                    } else {
                        // Nobody has said who is to move: the two keys ask
                        // and analyse in one tap.
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Who is to move?")
                                .eyebrow(color: Theme.accent)
                                .accessibilityAddTraits(.isHeader)
                            HStack(spacing: 8) {
                                Button("White to move") { analyze(with: .white) }
                                Button("Black to move") { analyze(with: .black) }
                            }
                        }
                    }
                }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(!model.canAnalyze)
                .padding(.horizontal, 16)
                .padding(.top, problems.isEmpty ? 10 : 6)
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

    /// From this many unsure squares on, the heading suggests a retake.
    private static let manyDoubtful = 8

    private func apply(_ fix: PositionProblem.Fix) {
        switch fix {
        case .select(let square):
            selectedSquare = square
            let piece = model.pieces[square].map(PieceGlyph.name(for:)) ?? "empty"
            AccessibilityNotification.Announcement("\(square.notation) selected, \(piece)").post()
        case .setSideToMove(let color):
            model.chooseSideToMove(color)
        }
    }

    /// Switches tab. Going to Analysis starts the search (clearing the
    /// previous position's results) in the same update as the tab change.
    private func show(_ tab: BoardTab) {
        if tab == .analysis {
            guard model.canAnalyze else { return }
            guard model.sideToMoveKnown else {
                askingSideToMove = true
                return
            }
            session.start(fen: model.fen, depth: model.depth)
        }
        model.tab = tab
    }

    private func startAnalysis() {
        guard model.canAnalyze, model.sideToMoveKnown else {
            model.tab = .position
            return
        }
        session.start(fen: model.fen, depth: model.depth)
    }

    /// Keeps finished analyses on the home screen. The untouched starting
    /// position isn't worth a row.
    private static func save(
        _ best: AnalysisLine, depth: Int, fen: String, photo: Data?, to history: HistoryStore
    ) {
        guard let placement = fen.split(separator: " ").first,
              placement != Position.standard.fen.split(separator: " ").first
        else { return }
        history.record(fen: fen, score: best.score, bestMove: best.moves.first, depth: depth, photo: photo)
    }

    private func showToast(_ message: String) {
        toastTask?.cancel()
        withAnimation(.easeOut(duration: 0.15)) { toast = message }
        AccessibilityNotification.Announcement(message).post()
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

/// One reason the position can't be analyzed: what is wrong, what usually
/// causes it, and a key for the likeliest fix.
private struct ProblemRow: View {
    let problem: PositionProblem
    /// How many problems there are in all, this one first.
    let count: Int
    let apply: (PositionProblem.Fix) -> Void

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(problem.title)
                        .sans(14, weight: .medium)
                        .foregroundStyle(Theme.accent)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                    if count > 1 {
                        Text("1 of \(count)")
                            .mono(11)
                            .foregroundStyle(Theme.muted)
                            .lineLimit(1)
                            .fixedSize()
                    }
                }
                Text(problem.detail)
                    .sans(12)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)

            if let fix = problem.fix, let label = problem.fixLabel {
                Button {
                    apply(fix)
                } label: {
                    Text(label)
                        .sans(13, weight: .medium)
                        .lineLimit(1)
                        .padding(.horizontal, 12)
                }
                .buttonStyle(OutlineButtonStyle())
                .fixedSize(horizontal: true, vertical: false)
                .frame(height: 36)
                .accessibilityHint(fix.accessibilityHint)
            }
        }
        .frame(minHeight: 44)
    }
}

private extension PositionProblem.Fix {
    var accessibilityHint: String {
        switch self {
        case .select(let square): return "Selects \(square.notation) and shows what it likely is"
        case .setSideToMove(let color): return "Makes it \(color.name) to move"
        }
    }
}

/// The photo a position was read from, as large as the screen allows.
private struct PhotoSheet: View {
    let photo: UIImage
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Photo")
                    .eyebrow()
                Spacer()
                Button("Done") { dismiss() }
                    .sans(16, weight: .medium)
                    .buttonStyle(TextButtonStyle())
                    .frame(height: 44)
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)

            Image(uiImage: photo)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.horizontal, 12)
                .padding(.bottom, 12)
                .accessibilityLabel("Photo of the board")
        }
        .background(Theme.paper.ignoresSafeArea())
        .foregroundStyle(Theme.ink)
        .presentationDragIndicator(.visible)
    }
}
