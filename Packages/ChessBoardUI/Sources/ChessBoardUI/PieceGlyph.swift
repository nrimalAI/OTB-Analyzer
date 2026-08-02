import ChessKit
import SwiftUI

/// Renders pieces with the filled Unicode chess glyphs for both colors
/// (the outline "white" glyphs read poorly at small sizes), colored per side
/// with a contrasting stroke shadow.
public enum PieceGlyph {

    public static func glyph(for kind: Piece.Kind) -> String {
        // U+FE0E forces text presentation — without it U+265F (pawn) takes its
        // default emoji presentation and renders as a missing-glyph box when a
        // color style is applied.
        switch kind {
        case .king: return "♚\u{FE0E}"
        case .queen: return "♛\u{FE0E}"
        case .rook: return "♜\u{FE0E}"
        case .bishop: return "♝\u{FE0E}"
        case .knight: return "♞\u{FE0E}"
        case .pawn: return "♟\u{FE0E}"
        }
    }
}

public struct PieceView: View {
    public let piece: Piece
    public let size: CGFloat

    public init(piece: Piece, size: CGFloat) {
        self.piece = piece
        self.size = size
    }

    public var body: some View {
        Text(PieceGlyph.glyph(for: piece.kind))
            .font(.system(size: size * 0.82))
            .foregroundStyle(piece.color == .white ? Color.white : Color.black)
            .shadow(
                color: piece.color == .white ? .black.opacity(0.8) : .white.opacity(0.35),
                radius: 0.8
            )
            .frame(width: size, height: size)
            .minimumScaleFactor(0.5)
            .accessibilityLabel("\(piece.color == .white ? "white" : "black") \(kindName)")
    }

    private var kindName: String {
        switch piece.kind {
        case .king: return "king"
        case .queen: return "queen"
        case .rook: return "rook"
        case .bishop: return "bishop"
        case .knight: return "knight"
        case .pawn: return "pawn"
        }
    }
}
