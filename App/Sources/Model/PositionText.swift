import ChessKit
import ChessVision
import Foundation

/// A position as plain text, to paste into a message or a note: who is to
/// move, the FEN, a diagram from White's side, and the engine's lines.
enum PositionText {

    /// One engine line as the text shows it.
    struct Line: Equatable {
        /// "+0.4", "M2" (`EvalFormat.label`).
        let evaluation: String
        /// "1. Bd3 dxc4 2. Bxc4" (`EvalFormat.moveNumber` numbering).
        let moves: String
    }

    /// The whole export. `verdict` ("White is slightly better") and `lines`
    /// are left out when there is no analysis yet.
    /// `searching`: the engine is still at it, so the depth is not final.
    static func export(
        pieces: [Square: Piece], fen: String, sideToMove: Piece.Color,
        verdict: String? = nil, depth: Int? = nil, searching: Bool = false, lines: [Line] = []
    ) -> String {
        var out = ["\(sideToMove.name) to move", "FEN: \(fen)", "", diagram(pieces)]
        if let verdict {
            out.append("")
            var heading = "Stockfish"
            if let depth { heading += ", depth \(depth)" + (searching ? " (still searching)" : "") }
            out.append("\(heading): \(verdict)")
            out += lines.map { "\($0.evaluation)  \($0.moves)" }
        }
        return out.joined(separator: "\n")
    }

    /// An 8x8 diagram from White's side: FEN letters for the pieces
    /// (uppercase White), dots for empty squares, ranks down the left and
    /// files along the bottom.
    static func diagram(_ pieces: [Square: Piece]) -> String {
        var rows: [String] = []
        for rank in stride(from: 8, through: 1, by: -1) {
            var cells: [String] = []
            for file in 1...8 {
                guard let square = FENBuilder.square(file: file, rank: rank) else { continue }
                cells.append(pieces[square].map(letter) ?? ".")
            }
            rows.append("\(rank)  " + cells.joined(separator: " "))
        }
        rows.append("   a b c d e f g h")
        return rows.joined(separator: "\n")
    }

    /// "K", "q", "P".
    static func letter(_ piece: Piece) -> String {
        let letter = piece.kind == .pawn ? "P" : piece.kind.notation
        return piece.color == .white ? letter : letter.lowercased()
    }

    /// The moves of a line with their numbers, as on screen: "1. e4 e5 2. Nf3",
    /// "1… Nf6 2. c4".
    static func numbered(_ moves: [String], sideToMove: Piece.Color) -> String {
        moves.enumerated().map { index, san in
            EvalFormat.moveNumber(at: index, sideToMove: sideToMove)
                .replacingOccurrences(of: "\u{00A0}", with: " ") + san
        }
        .joined(separator: " ")
    }
}
