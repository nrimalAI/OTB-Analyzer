import Foundation

/// An engine evaluation, always normalized to White's perspective
/// (UCI reports from the side to move; we flip when Black is to move).
public enum Score: Equatable, Hashable, Sendable {
    case centipawns(Int)
    /// Moves to mate: positive → White mates, negative → Black mates.
    case mate(Int)

    init(cp: Double?, mate: Int?, whiteToMove: Bool) {
        let sign = whiteToMove ? 1 : -1
        if let mate, mate != 0 {
            self = .mate(mate * sign)
        } else {
            self = .centipawns(Int((cp ?? 0).rounded()) * sign)
        }
    }

    /// Display string, e.g. `"+0.35"`, `"-1.20"`, `"#5"`, `"-#3"`.
    public var displayString: String {
        switch self {
        case .centipawns(let cp):
            let pawns = Double(cp) / 100.0
            return String(format: "%+.2f", pawns)
        case .mate(let n):
            return n >= 0 ? "#\(n)" : "-#\(-n)"
        }
    }

    /// White's winning share in [0, 1] for driving an eval bar.
    /// Uses the common logistic squash of centipawns.
    public var whiteWinningFraction: Double {
        switch self {
        case .mate(let n):
            return n >= 0 ? 1.0 : 0.0
        case .centipawns(let cp):
            return 1.0 / (1.0 + exp(-Double(cp) / 400.0))
        }
    }
}

/// One engine line (MultiPV entry).
public struct PrincipalVariation: Equatable, Sendable, Identifiable {
    /// 1-based MultiPV rank; 1 is the engine's best line.
    public let rank: Int
    public let score: Score
    /// Moves in SAN (falls back to LAN for anything unconvertible).
    public let moves: [String]
    /// Raw engine moves in LAN, e.g. `["e2e4", "e7e5"]`.
    public let lanMoves: [String]
    public let depth: Int

    public var id: Int { rank }

    /// The line's first move in LAN, for drawing an arrow.
    public var firstMoveLAN: String? { lanMoves.first }

    public init(rank: Int, score: Score, moves: [String], lanMoves: [String], depth: Int) {
        self.rank = rank
        self.score = score
        self.moves = moves
        self.lanMoves = lanMoves
        self.depth = depth
    }
}

/// A snapshot of the analysis as it deepens.
public struct AnalysisUpdate: Equatable, Sendable {
    public let depth: Int
    /// Sorted by rank; `lines.first` is the best line.
    public let lines: [PrincipalVariation]
    /// Set once the engine finishes (reached target depth or was stopped).
    public let isFinal: Bool

    public var bestLine: PrincipalVariation? { lines.first }
    public var score: Score? { bestLine?.score }
}
