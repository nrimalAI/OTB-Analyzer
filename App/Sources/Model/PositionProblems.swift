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

        return issues.map { issue in
            switch issue {
            case .missingKing(let color):
                return PositionProblem(
                    issue: issue,
                    title: "\(color.name) has no king",
                    detail: "A king read as a queen is the usual cause.",
                    fix: leastSure { $0.color == color && $0.kind == .queen }.map(PositionProblem.Fix.select))

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
                    detail: "At most 16: one is probably \(color.opposite.name)\u{2019}s, or not there.",
                    fix: leastSure { $0.color == color }.map(PositionProblem.Fix.select))

            case .sideNotToMoveInCheck:
                let inCheck = sideToMove.opposite
                let bothInCheck = PositionValidator.validate(pieces: pieces, sideToMove: inCheck)
                    .contains(.sideNotToMoveInCheck)
                if bothInCheck {
                    return PositionProblem(
                        issue: issue,
                        title: "Both kings are in check",
                        detail: "Only one can be: a piece giving check is probably misread.",
                        fix: nil)
                }
                return PositionProblem(
                    issue: issue,
                    title: "\(inCheck.name) is in check",
                    detail: "So it must be \(inCheck.name) to move.",
                    fix: .setSideToMove(inCheck))
            }
        }
    }
}

extension Piece.Color {
    /// "White", "Black".
    var name: String { self == .white ? "White" : "Black" }
}
