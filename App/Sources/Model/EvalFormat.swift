import ChessAnalysis
import ChessKit
import Foundation

/// How engine output is worded on screen: the short evaluation ("+1.9",
/// "M2"), the plain-language verdict, and move numbering.
enum EvalFormat {

    /// "+1.9", "−0.4", "0.0", "M2", "−M3" — always from White's side.
    static func label(_ score: Score) -> String {
        switch score {
        case .mate(let n):
            return (n < 0 ? "\u{2212}" : "") + "M\(abs(n))"
        case .centipawns(let cp):
            let pawns = (Double(cp) / 10).rounded() / 10
            let sign = pawns > 0 ? "+" : pawns < 0 ? "\u{2212}" : ""
            return sign + String(format: "%.1f", abs(pawns))
        }
    }

    /// "White is clearly better", "Black mates in 3", "Equal position".
    static func verdict(_ score: Score) -> String {
        switch score {
        case .mate(let n):
            return "\(n > 0 ? "White" : "Black") mates in \(abs(n))"
        case .centipawns(let cp):
            let side = cp > 0 ? "White" : "Black"
            let pawns = abs(Double(cp)) / 100
            if pawns < 0.25 { return "Equal position" }
            if pawns < 0.9 { return "\(side) is slightly better" }
            if pawns < 2.5 { return "\(side) is clearly better" }
            return "\(side) is winning"
        }
    }

    /// White's winning chances as a whole percentage.
    static func whiteWinChance(_ score: Score) -> Int {
        Int((score.whiteWinningFraction * 100).rounded())
    }

    /// The number printed before the move at `index` of a line that starts
    /// with `sideToMove` to play: "1. " before White's moves, "1… " before a
    /// Black move that opens the line (or always, when `always` is set), and
    /// nothing before Black's other moves.
    static func moveNumber(at index: Int, sideToMove: Piece.Color, always: Bool = false) -> String {
        let number = 1 + (index + (sideToMove == .black ? 1 : 0)) / 2
        let isWhiteMove = (index % 2 == 0) == (sideToMove == .white)
        if isWhiteMove { return "\(number).\u{00A0}" }
        return index == 0 || always ? "\(number)\u{2026}\u{00A0}" : ""
    }
}
