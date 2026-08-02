import ChessKit

/// Standalone square-attack detection over a raw piece map.
///
/// Used by validation to reject positions where the side *not* to move is in
/// check (impossible in a real game). Deliberately independent of ChessKit's
/// move generation, which assumes an already-legal position.
enum AttackDetector {

    /// Whether `square` is attacked by any piece of `color`.
    static func isSquare(
        _ square: Square,
        attackedBy color: Piece.Color,
        in pieces: [Square: Piece]
    ) -> Bool {
        let tf = square.file.number
        let tr = square.rank.value

        func pieceAt(_ file: Int, _ rank: Int) -> Piece? {
            guard let sq = FENBuilder.square(file: file, rank: rank) else { return nil }
            return pieces[sq]
        }

        // Pawn attacks: a pawn of `color` attacks diagonally forward.
        let pawnDir = color == .white ? 1 : -1
        for df in [-1, 1] {
            if let p = pieceAt(tf + df, tr - pawnDir), p.color == color, p.kind == .pawn {
                return true
            }
        }

        // Knight attacks.
        let knightOffsets = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        for (df, dr) in knightOffsets {
            if let p = pieceAt(tf + df, tr + dr), p.color == color, p.kind == .knight {
                return true
            }
        }

        // King attacks (adjacent squares).
        for df in -1...1 {
            for dr in -1...1 where !(df == 0 && dr == 0) {
                if let p = pieceAt(tf + df, tr + dr), p.color == color, p.kind == .king {
                    return true
                }
            }
        }

        // Sliding attacks: rook/queen on ranks+files, bishop/queen on diagonals.
        let orthogonal = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        let diagonal = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for (df, dr) in orthogonal + diagonal {
            var f = tf + df
            var r = tr + dr
            while (1...8).contains(f) && (1...8).contains(r) {
                if let p = pieceAt(f, r) {
                    if p.color == color {
                        let isOrtho = df == 0 || dr == 0
                        if p.kind == .queen
                            || (isOrtho && p.kind == .rook)
                            || (!isOrtho && p.kind == .bishop) {
                            return true
                        }
                    }
                    break  // blocked by any piece either color
                }
                f += df
                r += dr
            }
        }

        return false
    }
}
