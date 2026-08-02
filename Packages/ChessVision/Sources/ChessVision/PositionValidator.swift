import ChessKit

/// A reason a recognized/edited position can't be analyzed, with the squares
/// to highlight in the editor.
public enum ValidationIssue: Equatable, Hashable {
    case missingKing(Piece.Color)
    case multipleKings(Piece.Color)
    case pawnOnBackRank(Square)
    case tooManyPawns(Piece.Color)
    case tooManyPieces(Piece.Color)
    case sideNotToMoveInCheck

    public var message: String {
        switch self {
        case .missingKing(let c): return "\(c.displayName) has no king"
        case .multipleKings(let c): return "\(c.displayName) has more than one king"
        case .pawnOnBackRank: return "Pawns can't be on the first or last rank"
        case .tooManyPawns(let c): return "\(c.displayName) has more than 8 pawns"
        case .tooManyPieces(let c): return "\(c.displayName) has more than 16 pieces"
        case .sideNotToMoveInCheck: return "The side that just moved is still in check — check side to move"
        }
    }
}

extension Piece.Color {
    var displayName: String { self == .white ? "White" : "Black" }
}

public enum PositionValidator {

    /// Validates a piece map + side to move. Empty result means analyzable.
    public static func validate(
        pieces: [Square: Piece],
        sideToMove: Piece.Color
    ) -> [ValidationIssue] {
        var issues: [ValidationIssue] = []

        for color in Piece.Color.allCases {
            let own = pieces.values.filter { $0.color == color }
            let kings = own.filter { $0.kind == .king }
            if kings.isEmpty { issues.append(.missingKing(color)) }
            if kings.count > 1 { issues.append(.multipleKings(color)) }
            if own.filter({ $0.kind == .pawn }).count > 8 { issues.append(.tooManyPawns(color)) }
            if own.count > 16 { issues.append(.tooManyPieces(color)) }
        }

        for (square, piece) in pieces where piece.kind == .pawn {
            let rank = square.rank.value
            if rank == 1 || rank == 8 {
                issues.append(.pawnOnBackRank(square))
            }
        }

        // Only meaningful once both kings exist exactly once.
        let opponent = sideToMove.opposite
        if let opponentKing = pieces.first(where: { $0.value.kind == .king && $0.value.color == opponent }),
           !issues.contains(.missingKing(opponent)),
           !issues.contains(.multipleKings(opponent)),
           AttackDetector.isSquare(opponentKing.key, attackedBy: sideToMove, in: pieces) {
            issues.append(.sideNotToMoveInCheck)
        }

        return issues
    }

    /// Squares to highlight for a set of issues.
    public static func highlightSquares(
        for issues: [ValidationIssue],
        pieces: [Square: Piece]
    ) -> Set<Square> {
        var squares: Set<Square> = []
        for issue in issues {
            switch issue {
            case .pawnOnBackRank(let sq):
                squares.insert(sq)
            case .multipleKings(let color):
                for (sq, p) in pieces where p.kind == .king && p.color == color {
                    squares.insert(sq)
                }
            case .tooManyPawns(let color):
                for (sq, p) in pieces where p.kind == .pawn && p.color == color {
                    squares.insert(sq)
                }
            case .sideNotToMoveInCheck:
                for (sq, p) in pieces where p.kind == .king {
                    squares.insert(sq)
                }
            case .missingKing, .tooManyPieces:
                break
            }
        }
        return squares
    }
}
