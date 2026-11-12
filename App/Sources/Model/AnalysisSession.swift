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

    /// Lines that can't be replayed from `fen` are dropped as long as some
    /// line can: the engine library delivers responses out of order, so a
    /// few lines from the previous search (possibly of another position)
    /// can arrive at the start of this one.
    static func lines(from update: AnalysisUpdate, fen: String) -> [AnalysisLine] {
        let all = update.lines.map { line(from: $0, fen: fen) }
        let replayable = all.filter { $0.playableCount > 0 }
        return replayable.isEmpty ? all : replayable
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

    @Published private(set) var update: AnalysisUpdate?
    @Published private(set) var lines: [AnalysisLine] = []
    @Published private(set) var lineIndex = 0
    /// Moves played so far along the selected line.
    @Published private(set) var ply = 0
    @Published private(set) var isRunning = false
    /// Depth asked for in the current search.
    private(set) var targetDepth = 0

    /// Depth reached so far. Capped at the depth asked for, because leftover
    /// output from a deeper previous search can inflate the engine's figure.
    var reachedDepth: Int { min(update?.depth ?? 0, targetDepth) }

    /// Called once per completed search with the final result and its FEN.
    var onFinished: ((AnalysisUpdate, String) -> Void)?

    private let engine = AnalysisEngine()
    private var consumer: Task<Void, Never>?
    /// Bumped on every run/stop; results from older searches are ignored.
    private var generation = 0
    /// Depth and FEN of the last search that ran to completion.
    private var finishedSearch: String?

    // MARK: - Engine

    func run(fen: String, depth: Int) async {
        // Coming back to a position that was already searched to this depth
        // (e.g. after a look at the Position tab): keep the result.
        let key = "\(depth)|\(fen)"
        if finishedSearch == key, update != nil { return }
        finishedSearch = nil

        generation += 1
        let current = generation
        targetDepth = depth
        update = nil
        lines = []
        lineIndex = 0
        ply = 0
        isRunning = true

        await engine.start(multiPV: 3)
        await drainPreviousSearch()
        guard current == generation else { return }

        consumer = Task { [weak self] in
            guard let self else { return }
            let stream = await self.engine.analyze(fen: fen, depth: depth)
            // Keep reading after being superseded so the old search's final
            // `bestmove` is consumed here rather than by the next search.
            for await value in stream where current == self.generation {
                self.apply(value, fen: fen)
                if value.isFinal { self.finishedSearch = key }
            }
        }
    }

    func stop() {
        generation += 1
        isRunning = false
        Task { await engine.stopSearch() }
    }

    /// Stops any search still running and waits (bounded) for its stream to
    /// finish, so two searches never read the engine's output at once.
    private func drainPreviousSearch() async {
        guard let previous = consumer else { return }
        consumer = nil
        await engine.stopSearch()
        let timeout = Task {
            try? await Task.sleep(for: .seconds(2))
            previous.cancel()
        }
        await previous.value
        timeout.cancel()
    }

    private func apply(_ value: AnalysisUpdate, fen: String) {
        let newLines = LinePlayback.lines(from: value, fen: fen)

        // Keep the reader's place only while the line they are stepping
        // through still starts with the moves already played.
        if lineIndex >= newLines.count {
            lineIndex = 0
            ply = 0
        } else if ply > 0 {
            let played = lines.indices.contains(lineIndex) ? Array(lines[lineIndex].lan.prefix(ply)) : []
            if played != Array(newLines[lineIndex].lan.prefix(ply)) { ply = 0 }
        }

        update = value
        lines = newLines
        if value.isFinal {
            isRunning = false
            onFinished?(value, fen)
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
