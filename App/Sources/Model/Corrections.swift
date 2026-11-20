import ChessKit
import ChessVision

/// What a square can be set to: a piece of one kind and colour, or nothing.
struct PieceChoice: Hashable {
    /// Nil for an empty square.
    let kind: Piece.Kind?
    let color: Piece.Color

    static func piece(_ kind: Piece.Kind, _ color: Piece.Color) -> PieceChoice {
        PieceChoice(kind: kind, color: color)
    }

    /// The empty square. (Its colour means nothing.)
    static let empty = PieceChoice(kind: nil, color: .white)

    init(kind: Piece.Kind?, color: Piece.Color) {
        self.kind = kind
        self.color = kind == nil ? .white : color
    }

    init(_ piece: Piece?) {
        self.init(kind: piece?.kind, color: piece?.color ?? .white)
    }

    var isEmpty: Bool { kind == nil }

    /// The piece homed on `square`, or nil for the empty square.
    func piece(on square: Square) -> Piece? {
        kind.map { Piece($0, color: color, square: square) }
    }
}

/// What a doubtful square most likely holds instead, so that fixing a
/// misread piece is one tap instead of a hunt through the tray.
///
/// The recognizer hands the app one piece per square and a confidence, not
/// its second choices, so the order comes from what goes wrong on boards the
/// models have not seen (`Training/README.md`): the side of a piece (a white
/// pawn read as black), a king read as a queen, a queen read as a rook or a
/// bishop, a pawn read as a taller piece and the other way round, and a pawn
/// seen where there is none. Then the rules of chess move a choice up or
/// out: a side without a king gets its king first where a king could have
/// been misread (a queen, a rook, the other king), and nothing is offered
/// that would make the position impossible by itself (a second king, a pawn
/// on the first or last rank, a ninth pawn, a seventeenth piece).
enum CorrectionSuggestions {

    /// Up to `limit` likely replacements for what stands on `square`, the
    /// likeliest first. Never the square's current content.
    static func alternatives(
        for square: Square, in pieces: [Square: Piece], limit: Int = 3
    ) -> [PieceChoice] {
        let current = PieceChoice(pieces[square])
        var others = pieces
        others[square] = nil

        var scored: [(choice: PieceChoice, score: Double, order: Int)] = []
        for (order, candidate) in candidates.enumerated() where candidate != current {
            guard isPossible(candidate, on: square, besides: others) else { continue }
            var score = likelihood(of: candidate, readAs: current)
            if candidate.kind == .king, !hasKing(candidate.color, in: others) {
                // A side without a king: its king is the likeliest fix where
                // that king could have been misread (its own queen or rook,
                // or the other side's king), and the one thing to offer on an
                // empty square, where it may have been missed. A pawn, a
                // knight or the other side's queen is not asked to be it.
                if current.isEmpty {
                    score = 1
                } else if score >= Self.plausible,
                          current.color == candidate.color
                            || (current.kind == .king && hasKing(current.color, in: others)) {
                    // (The other side's king only where that side has another:
                    // its only king is not taken from it.)
                    score += 2
                }
            }
            if score > 0 { scored.append((candidate, score, order)) }
        }
        scored.sort { $0.score != $1.score ? $0.score > $1.score : $0.order < $1.order }
        return scored.prefix(limit).map(\.choice)
    }

    /// The least likelihood at which a misreading is taken as plausible.
    static let plausible = 0.1

    /// Every choice in a fixed order: white pieces, black pieces, empty.
    /// Breaks ties, so that the same board always gives the same list.
    static let candidates: [PieceChoice] = {
        let kinds: [Piece.Kind] = [.king, .queen, .rook, .bishop, .knight, .pawn]
        return kinds.map { PieceChoice.piece($0, .white) }
            + kinds.map { PieceChoice.piece($0, .black) }
            + [.empty]
    }()

    /// How likely a square read as `read` holds `candidate`, before the
    /// rules of the game are asked; 0 for "no reason to think so".
    static func likelihood(of candidate: PieceChoice, readAs read: PieceChoice) -> Double {
        guard let readKind = read.kind else {
            // An empty square was read: only the rules (a missing king)
            // suggest anything for it.
            return 0
        }
        guard let kind = candidate.kind else {
            return readKind == .pawn ? 0.3 : 0.15
        }
        let sameColour = candidate.color == read.color
        if kind == readKind { return sameColour ? 0 : 0.45 }
        let weight = kindConfusion[readKind]?[kind] ?? 0.04
        return weight * (sameColour ? 1 : 0.25)
    }

    /// For a piece read as the outer kind, how likely it is the inner kind
    /// (same colour). Read from the errors listed in `Training/README.md`.
    private static let kindConfusion: [Piece.Kind: [Piece.Kind: Double]] = [
        .king: [.queen: 0.6, .rook: 0.1, .bishop: 0.08],
        .queen: [.king: 0.6, .rook: 0.3, .bishop: 0.25],
        .rook: [.queen: 0.3, .pawn: 0.2, .king: 0.1],
        .bishop: [.pawn: 0.3, .queen: 0.25, .knight: 0.15],
        .knight: [.pawn: 0.3, .bishop: 0.2],
        .pawn: [.bishop: 0.2, .knight: 0.15, .rook: 0.12],
    ]

    /// Whether putting `candidate` on `square` keeps the position possible
    /// as far as that one square can: no second king, no pawn on the first
    /// or last rank, at most eight pawns and sixteen pieces a side.
    static func isPossible(
        _ candidate: PieceChoice, on square: Square, besides others: [Square: Piece]
    ) -> Bool {
        guard let kind = candidate.kind else { return true }
        let own = others.values.filter { $0.color == candidate.color }
        switch kind {
        case .king:
            if own.contains(where: { $0.kind == .king }) { return false }
        case .pawn:
            if square.rank.value == 1 || square.rank.value == 8 { return false }
            if own.filter({ $0.kind == .pawn }).count >= 8 { return false }
        default:
            break
        }
        return own.count < 16
    }

    private static func hasKing(_ color: Piece.Color, in pieces: [Square: Piece]) -> Bool {
        pieces.values.contains { $0.kind == .king && $0.color == color }
    }
}
