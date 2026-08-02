import ChessKit

/// The four castling availabilities of a FEN. A photo can't prove castling is
/// still legal, so these are inferred optimistically from piece placement
/// (king and rook on their home squares → assume available) and the user can
/// clear them in the review UI.
public struct CastlingRights: Equatable, Sendable {
    public var whiteKingside: Bool
    public var whiteQueenside: Bool
    public var blackKingside: Bool
    public var blackQueenside: Bool

    public init(
        whiteKingside: Bool = false,
        whiteQueenside: Bool = false,
        blackKingside: Bool = false,
        blackQueenside: Bool = false
    ) {
        self.whiteKingside = whiteKingside
        self.whiteQueenside = whiteQueenside
        self.blackKingside = blackKingside
        self.blackQueenside = blackQueenside
    }

    public static let none = CastlingRights()

    /// Optimistic inference from placement: a right is assumed available when
    /// the king and the relevant rook are both on their home squares.
    public static func inferred(from pieces: [Square: Piece]) -> CastlingRights {
        func has(_ kind: Piece.Kind, _ color: Piece.Color, at square: Square) -> Bool {
            guard let p = pieces[square] else { return false }
            return p.kind == kind && p.color == color
        }
        let whiteKingHome = has(.king, .white, at: .e1)
        let blackKingHome = has(.king, .black, at: .e8)
        return CastlingRights(
            whiteKingside: whiteKingHome && has(.rook, .white, at: .h1),
            whiteQueenside: whiteKingHome && has(.rook, .white, at: .a1),
            blackKingside: blackKingHome && has(.rook, .black, at: .h8),
            blackQueenside: blackKingHome && has(.rook, .black, at: .a8)
        )
    }

    /// The FEN castling field, e.g. `"KQkq"` or `"-"`.
    public var fenField: String {
        var field = ""
        if whiteKingside { field += "K" }
        if whiteQueenside { field += "Q" }
        if blackKingside { field += "k" }
        if blackQueenside { field += "q" }
        return field.isEmpty ? "-" : field
    }
}
