import ChessKit

/// Converts engine LAN lines (`["e2e4", "e7e5", ...]`) into SAN for display.
public enum SANConverter {

    /// Converts as many leading moves as parse cleanly; once a move fails to
    /// parse (deep PVs occasionally trip move-gen edge cases), the remaining
    /// moves are appended in raw LAN so no information is lost.
    public static func convert(fen: String, lanMoves: [String]) -> [String] {
        guard let position = Position(fen: fen) else { return lanMoves }
        var board = Board(position: position)
        var result: [String] = []

        for (index, lan) in lanMoves.enumerated() {
            guard
                let parsed = EngineLANParser.parse(
                    move: lan, for: board.position.sideToMove, in: board.position),
                board.canMove(pieceAt: parsed.start, to: parsed.end)
            else {
                result.append(contentsOf: lanMoves[index...])
                break
            }

            guard var made = board.move(pieceAt: parsed.start, to: parsed.end) else {
                result.append(contentsOf: lanMoves[index...])
                break
            }
            if let promoted = parsed.promotedPiece {
                made = board.completePromotion(of: made, to: promoted.kind)
            }
            result.append(made.san)
        }
        return result
    }

    /// Parses a single LAN move against a FEN into (from, to) squares,
    /// for drawing a best-move arrow. Pure string parsing; no legality check.
    public static func squares(ofLAN lan: String) -> (from: Square, to: Square)? {
        guard lan.count >= 4 else { return nil }
        let from = String(lan.prefix(2))
        let to = String(lan.dropFirst(2).prefix(2))
        let valid = Set("abcdefgh12345678")
        guard from.allSatisfy(valid.contains), to.allSatisfy(valid.contains) else { return nil }
        return (Square(from), Square(to))
    }
}
