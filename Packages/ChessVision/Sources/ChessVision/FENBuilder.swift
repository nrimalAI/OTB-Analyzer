import ChessKit

/// Assembles and parses FEN. Only the placement field can come from a photo;
/// side to move, castling, etc. are supplied by the review UI.
public enum FENBuilder {

    // MARK: - Square helpers

    /// `Square` from 1-based file/rank numbers. ChessKit's `(File, Rank)` init
    /// is internal, but the enum's rawValue layout is `(rank-1)*8 + (file-1)`.
    public static func square(file: Int, rank: Int) -> Square? {
        guard (1...8).contains(file), (1...8).contains(rank) else { return nil }
        return Square(rawValue: (rank - 1) * 8 + (file - 1))
    }

    // MARK: - FEN assembly

    public static func fen(
        pieces: [Square: Piece],
        sideToMove: Piece.Color,
        castling: CastlingRights,
        enPassant: String = "-",
        halfmove: Int = 0,
        fullmove: Int = 1
    ) -> String {
        [
            placement(pieces),
            sideToMove.rawValue,
            castling.fenField,
            enPassant,
            String(halfmove),
            String(fullmove),
        ].joined(separator: " ")
    }

    /// The FEN placement field (ranks 8→1, files a→h) for a piece map.
    public static func placement(_ pieces: [Square: Piece]) -> String {
        var ranks: [String] = []
        for rank in stride(from: 8, through: 1, by: -1) {
            var row = ""
            var emptyRun = 0
            for file in 1...8 {
                guard let sq = square(file: file, rank: rank) else { continue }
                if let piece = pieces[sq] {
                    if emptyRun > 0 {
                        row += String(emptyRun)
                        emptyRun = 0
                    }
                    row += fenLetter(for: piece)
                } else {
                    emptyRun += 1
                }
            }
            if emptyRun > 0 { row += String(emptyRun) }
            ranks.append(row)
        }
        return ranks.joined(separator: "/")
    }

    // MARK: - FEN parsing (placement field only)

    /// Parses a FEN placement field (the part before the first space) into a
    /// piece map. Returns nil if malformed.
    public static func pieces(fromPlacement placement: String) -> [Square: Piece]? {
        let rows = placement.split(separator: "/")
        guard rows.count == 8 else { return nil }

        var result: [Square: Piece] = [:]
        for (rowIndex, row) in rows.enumerated() {
            let rank = 8 - rowIndex
            var file = 1
            for char in row {
                if let skip = char.wholeNumberValue, (1...8).contains(skip) {
                    file += skip
                } else {
                    guard let sq = square(file: file, rank: rank),
                          let piece = piece(fenLetter: char, square: sq)
                    else { return nil }
                    result[sq] = piece
                    file += 1
                }
            }
            guard file == 9 else { return nil }
        }
        return result
    }

    /// Parses a full FEN string's placement field into a piece map.
    public static func pieces(fromFEN fen: String) -> [Square: Piece]? {
        guard let placement = fen.split(separator: " ").first else { return nil }
        return pieces(fromPlacement: String(placement))
    }

    // MARK: - Letters

    static func fenLetter(for piece: Piece) -> String {
        let letter: String
        switch piece.kind {
        case .pawn: letter = "p"
        case .knight: letter = "n"
        case .bishop: letter = "b"
        case .rook: letter = "r"
        case .queen: letter = "q"
        case .king: letter = "k"
        }
        return piece.color == .white ? letter.uppercased() : letter
    }

    static func piece(fenLetter: Character, square: Square) -> Piece? {
        let color: Piece.Color = fenLetter.isUppercase ? .white : .black
        let kind: Piece.Kind
        switch Character(fenLetter.lowercased()) {
        case "p": kind = .pawn
        case "n": kind = .knight
        case "b": kind = .bishop
        case "r": kind = .rook
        case "q": kind = .queen
        case "k": kind = .king
        default: return nil
        }
        return Piece(kind, color: color, square: square)
    }
}
