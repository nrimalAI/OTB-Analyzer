import ChessKit
import ChessVision

/// Why a position can't be analyzed, in words for the person fixing it, and
/// the one step most likely to help.
struct PositionProblem: Identifiable, Equatable {
    enum Fix: Equatable {
        /// Select the square most likely to be wrong, so that its likely
        /// replacements are offered.
        case select(Square)
        /// The side in check is the side to move.
        case setSideToMove(Piece.Color)
    }

    let issue: ValidationIssue
    /// "White has no king".
    let title: String
    /// What usually causes it: "A king read as a queen is the usual cause."
    let detail: String
    let fix: Fix?

    var id: ValidationIssue { issue }

    /// The button for the fix: "Show", "Black to move".
    var fixLabel: String? {
        switch fix {
        case .select: return "Show"
        case .setSideToMove(let color): return "\(color.name) to move"
        case nil: return nil
        }
    }
}

enum PositionProblems {

    /// One problem per issue, in the validator's order. `confidence` is the
    /// recognizer's, for the squares it was unsure of: where several squares
    /// could be the culprit, the least sure is the one shown.
    static func explain(
        _ issues: [ValidationIssue], pieces: [Square: Piece], sideToMove: Piece.Color,
        confidence: [Square: Float] = [:]
    ) -> [PositionProblem] {
        /// The least sure square holding a piece that matches.
        func leastSure(_ matches: (Piece) -> Bool) -> Square? {
            pieces.filter { matches($0.value) }.keys.min { a, b in
                let ca = confidence[a] ?? 1, cb = confidence[b] ?? 1
                return ca != cb ? ca < cb : a.rawValue < b.rawValue
            }
        }

        /// The least sure of `squares`, if any.
        func leastSureOf(_ squares: [Square]) -> Square? {
            squares.min { a, b in
                let ca = confidence[a] ?? 1, cb = confidence[b] ?? 1
                return ca != cb ? ca < cb : a.rawValue < b.rawValue
            }
        }

        return inStableOrder(issues).map { issue in
            switch issue {
            case .missingKing(let color):
                // Where the suggestions would put this king first: its own
                // queen or rook, or the other side's king.
                let king = PieceChoice.piece(.king, color)
                let candidates = pieces.keys.filter {
                    CorrectionSuggestions.alternatives(for: $0, in: pieces, limit: 1).first == king
                }
                return PositionProblem(
                    issue: issue,
                    title: "\(color.name) has no king",
                    detail: "A king read as a queen is the usual cause.",
                    fix: leastSureOf(candidates).map(PositionProblem.Fix.select))

            case .multipleKings(let color):
                let count = pieces.values.filter { $0.color == color && $0.kind == .king }.count
                return PositionProblem(
                    issue: issue,
                    title: "\(color.name) has \(count) kings",
                    detail: "Only one can be: the others are probably queens.",
                    fix: leastSure { $0.color == color && $0.kind == .king }.map(PositionProblem.Fix.select))

            case .pawnOnBackRank(let square):
                return PositionProblem(
                    issue: issue,
                    title: "Pawn on \(square.notation)",
                    detail: "Pawns never stand on the first or last rank.",
                    fix: .select(square))

            case .tooManyPawns(let color):
                let count = pieces.values.filter { $0.color == color && $0.kind == .pawn }.count
                return PositionProblem(
                    issue: issue,
                    title: "\(color.name) has \(count) pawns",
                    detail: "At most 8: one is probably \(color.opposite.name)\u{2019}s, or another piece.",
                    fix: leastSure { $0.color == color && $0.kind == .pawn }.map(PositionProblem.Fix.select))

            case .tooManyPieces(let color):
                let count = pieces.values.filter { $0.color == color }.count
                return PositionProblem(
                    issue: issue,
                    title: "\(color.name) has \(count) pieces",
                    detail: "At most 16: one is probably \(color.opposite.name)\u{2019}s.",
                    fix: leastSure { $0.color == color }.map(PositionProblem.Fix.select))

            case .sideNotToMoveInCheck:
                let inCheck = sideToMove.opposite
                let bothInCheck = PositionValidator.validate(pieces: pieces, sideToMove: inCheck)
                    .contains(.sideNotToMoveInCheck)
                // A check that comes from an unsure piece is as likely a
                // misreading as the wrong side to move: show that piece
                // rather than switch sides and analyse a wrong position.
                let unsureCheckers = (bothInCheck ? [sideToMove, inCheck] : [sideToMove])
                    .flatMap { checkers(givingCheckAs: $0, in: pieces) }
                    .filter { (confidence[$0] ?? 1) < AppModel.lowConfidenceThreshold }
                if bothInCheck {
                    return PositionProblem(
                        issue: issue,
                        title: "Both kings are in check",
                        detail: "Only one can be: a piece giving check is probably misread.",
                        fix: leastSureOf(unsureCheckers).map(PositionProblem.Fix.select))
                }
                if let unsure = leastSureOf(unsureCheckers) {
                    return PositionProblem(
                        issue: issue,
                        title: "\(inCheck.name) is in check",
                        detail: "From an unsure piece on \(unsure.notation): check it, or make it \(inCheck.name) to move.",
                        fix: .select(unsure))
                }
                return PositionProblem(
                    issue: issue,
                    title: "\(inCheck.name) is in check",
                    detail: "So it must be \(inCheck.name) to move.",
                    fix: .setSideToMove(inCheck))
            }
        }
    }

    /// The pieces of `side` that give check to the other king: those whose
    /// removal leaves that king out of check. (Of a double check neither
    /// alone; it is then not offered.)
    static func checkers(givingCheckAs side: Piece.Color, in pieces: [Square: Piece]) -> [Square] {
        func inCheck(_ board: [Square: Piece]) -> Bool {
            PositionValidator.validate(pieces: board, sideToMove: side).contains(.sideNotToMoveInCheck)
        }
        guard inCheck(pieces) else { return [] }
        return pieces.filter { $0.value.color == side && $0.value.kind != .king }.keys.filter { square in
            var without = pieces
            without[square] = nil
            return !inCheck(without)
        }
    }

    /// The validator's order, except that its pawns on a back rank, which
    /// come in no fixed order, are sorted by square: the rows do not move
    /// about as the board is edited.
    static func inStableOrder(_ issues: [ValidationIssue]) -> [ValidationIssue] {
        var backRank = issues.compactMap { issue -> Square? in
            if case .pawnOnBackRank(let square) = issue { return square }
            return nil
        }
        .sorted { $0.rawValue < $1.rawValue }
        .makeIterator()
        return issues.map { issue in
            if case .pawnOnBackRank = issue, let next = backRank.next() { return .pawnOnBackRank(next) }
            return issue
        }
    }
}

extension Piece.Color {
    /// "White", "Black".
    var name: String { self == .white ? "White" : "Black" }
}
