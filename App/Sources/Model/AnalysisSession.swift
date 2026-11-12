import ChessAnalysis
import ChessKit
import Foundation

/// One engine line prepared for display and for stepping through on the
/// board.
struct AnalysisLine: Identifiable, Equatable {
    /// 1-based MultiPV rank; 1 is the engine's best line.
    let rank: Int
    let score: Score
    /// Moves to show, in SAN.
    let moves: [String]
    /// The same moves in LAN, for the arrow and last-move highlight. Only
    /// covers the moves that could be replayed.
    let lan: [String]
    /// `frames[i]` is the position after `i` moves; `frames[0]` is the
    /// analyzed position itself. Empty if the position couldn't be replayed.
    let frames: [[Square: Piece]]

    var id: Int { rank }
    /// How many moves can be stepped through.
    var playableCount: Int { max(0, frames.count - 1) }

    /// From/to squares of the move at `index`, if it can be replayed.
    func squares(ofMoveAt index: Int) -> (from: Square, to: Square)? {
        guard lan.indices.contains(index) else { return nil }
        return SANConverter.squares(ofLAN: lan[index])
    }
}

enum LinePlayback {
    /// Engine lines run 20+ plies deep and the tail is speculative; show and
    /// step through the first few moves only. Five plies fit on one row, so
    /// all three lines and the actions stay on screen under the board.
    static let maxPlies = 5

    /// Lines that can't be replayed from `fen` are dropped: the engine
    /// library delivers responses out of order, so a few lines from the
    /// previous search (possibly of another position) can arrive at the
    /// start of this one. `positionIsOver` says the position has no legal
    /// move, so nothing the engine sends can be a line of it; otherwise, if
    /// no line replays at all, the lines are kept for display rather than
    /// showing an empty analysis.
    static func lines(
        from update: AnalysisUpdate, fen: String, positionIsOver: Bool
    ) -> [AnalysisLine] {
        let all = update.lines.map { line(from: $0, fen: fen) }
        let replayable = all.filter { $0.playableCount > 0 }
        if !replayable.isEmpty { return replayable }
        return positionIsOver ? [] : all
    }

    /// Whether the side to move has no legal move (checkmate or stalemate).
    /// False when the FEN can't be read.
    static func positionIsOver(fen: String) -> Bool {
        guard let position = Position(fen: fen) else { return false }
        let board = Board(position: position)
        return !position.pieces.contains {
            $0.color == position.sideToMove && !board.legalMoves(forPieceAt: $0.square).isEmpty
        }
    }

    static func line(from pv: PrincipalVariation, fen: String) -> AnalysisLine {
        let shown = Array(pv.moves.prefix(maxPlies))
        guard let position = Position(fen: fen) else {
            return AnalysisLine(rank: pv.rank, score: pv.score, moves: shown, lan: [], frames: [])
        }

        var board = Board(position: position)
        var frames = [snapshot(of: board.position)]
        var san: [String] = []
        var lan: [String] = []

        for move in pv.lanMoves.prefix(maxPlies) {
            guard
                let parsed = EngineLANParser.parse(
                    move: move, for: board.position.sideToMove, in: board.position),
                board.canMove(pieceAt: parsed.start, to: parsed.end),
                var made = board.move(pieceAt: parsed.start, to: parsed.end)
            else { break }
            if let promoted = parsed.promotedPiece {
                made = board.completePromotion(of: made, to: promoted.kind)
            }
            san.append(made.san)
            lan.append(move)
            frames.append(snapshot(of: board.position))
        }

        // Moves past the last replayable one are still shown, just not
        // steppable.
        let moves = san + shown.dropFirst(san.count)
        return AnalysisLine(rank: pv.rank, score: pv.score, moves: moves, lan: lan, frames: frames)
    }

    private static func snapshot(of position: Position) -> [Square: Piece] {
        Dictionary(position.pieces.map { ($0.square, $0) }, uniquingKeysWith: { _, last in last })
    }
}

/// Drives Stockfish for the position on screen and tracks which line and
/// move the reader is stepping through.
@MainActor
final class AnalysisSession: ObservableObject {

    @Published private(set) var lines: [AnalysisLine] = []
    @Published private(set) var lineIndex = 0
    /// Moves played so far along the selected line.
    @Published private(set) var ply = 0
    /// Depth reached so far. Capped at the depth asked for, because leftover
    /// output from a deeper previous search can inflate the engine's figure.
    @Published private(set) var reachedDepth = 0
    /// The engine has reported at least once for the current search.
    @Published private(set) var hasResult = false
    /// The current search ran to completion.
    @Published private(set) var isFinished = false

    /// Called once per completed search that produced a line, with the best
    /// line, the depth reached, and the FEN that was searched.
    var onFinished: ((AnalysisLine, Int, String) -> Void)?

    private let engine = AnalysisEngine()
    /// Depth and FEN of the search whose results are (or will be) on show.
    private var currentSearch: String?
    private var isRunning = false
    private var positionIsOver = false
    /// Bumped on every start/stop; results from older searches are ignored.
    private var generation = 0
    /// Engine work is chained on this so starts and stops reach the engine
    /// one at a time, in order.
    private var pending: Task<Void, Never>?
    /// Reads the current search's output.
    private var consumer: Task<Void, Never>?
    private var drainTimedOut = false

    // MARK: - Engine

    /// Starts searching `fen`. Clears the previous results at once, so the
    /// caller never renders one position with another's analysis. Does
    /// nothing if this exact search is already running or already finished
    /// (e.g. coming back from a look at the Position tab).
    func start(fen: String, depth: Int) {
        let key = "\(depth)|\(fen)"
        if currentSearch == key, isRunning || isFinished { return }
        currentSearch = key

        generation += 1
        let current = generation
        positionIsOver = LinePlayback.positionIsOver(fen: fen)
        lines = []
        lineIndex = 0
        ply = 0
        reachedDepth = 0
        hasResult = false
        isFinished = false
        isRunning = true

        enqueue { [weak self] in
            await self?.begin(fen: fen, depth: depth, generation: current)
        }
    }

    /// Stops the search. Finished results stay on show; an interrupted
    /// search will start over next time.
    func stop() {
        generation += 1
        isRunning = false
        if !isFinished { currentSearch = nil }
        enqueue { [engine] in
            await engine.stopSearch()
        }
    }

    private func enqueue(_ work: @escaping @MainActor () async -> Void) {
        let previous = pending
        pending = Task {
            await previous?.value
            await work()
        }
    }

    private func begin(fen: String, depth: Int, generation current: Int) async {
        guard current == generation else { return }
        await drainPreviousSearch()
        await engine.start(multiPV: 3)
        guard current == generation else { return }

        consumer = Task { [weak self] in
            guard let self else { return }
            let stream = await self.engine.analyze(fen: fen, depth: depth)
            // Keep reading after being superseded so the old search's final
            // `bestmove` is consumed here rather than by the next search.
            for await value in stream where current == self.generation {
                self.apply(value, fen: fen, depth: depth)
            }
        }
    }

    /// Stops any search still running and waits (bounded) for its stream to
    /// finish, so two searches never read the engine's output at once.
    private func drainPreviousSearch() async {
        guard let previous = consumer else { return }
        consumer = nil
        await engine.stopSearch()
        let timeout = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            self.drainTimedOut = true
            previous.cancel()
        }
        await previous.value
        timeout.cancel()

        // Cancelling the reader ends the engine's one response stream for
        // good, so restart the engine to get a fresh one.
        if drainTimedOut {
            drainTimedOut = false
            await engine.shutdown()
        }
    }

    private func apply(_ value: AnalysisUpdate, fen: String, depth: Int) {
        let newLines = LinePlayback.lines(from: value, fen: fen, positionIsOver: positionIsOver)

        // Keep the reader's place only while the line they are stepping
        // through still starts with the moves already played.
        if lineIndex >= newLines.count {
            lineIndex = 0
            ply = 0
        } else if ply > 0 {
            let played = lines.indices.contains(lineIndex) ? Array(lines[lineIndex].lan.prefix(ply)) : []
            if played != Array(newLines[lineIndex].lan.prefix(ply)) { ply = 0 }
        }

        lines = newLines
        reachedDepth = min(value.depth, depth)
        hasResult = true
        if value.isFinal {
            isRunning = false
            isFinished = true
            if let best = newLines.first { onFinished?(best, reachedDepth, fen) }
        }
    }

    // MARK: - Stepping

    var selectedLine: AnalysisLine? {
        lines.indices.contains(lineIndex) ? lines[lineIndex] : nil
    }

    /// The position after the moves played so far, or nil before any result.
    var shownPieces: [Square: Piece]? {
        guard let line = selectedLine, line.frames.indices.contains(ply) else { return nil }
        return line.frames[ply]
    }

    func select(line index: Int, ply newPly: Int = 0) {
        guard lines.indices.contains(index) else { return }
        lineIndex = index
        ply = min(max(newPly, 0), lines[index].playableCount)
    }

    func stepToStart() { ply = 0 }
    func stepBack() { ply = max(0, ply - 1) }
    func stepForward() {
        guard let line = selectedLine else { return }
        ply = min(line.playableCount, ply + 1)
    }
}
