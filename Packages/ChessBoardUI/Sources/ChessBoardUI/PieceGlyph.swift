import ChessKit
import SwiftUI

/// Unicode chess glyphs. Pieces are drawn like a printed diagram: black pieces
/// are the solid glyph in ink; white pieces are the solid glyph in paper
/// colour with the outline glyph in ink on top, so they stay legible on both
/// light and dark squares.
public enum PieceGlyph {

    /// The solid ("black") glyph for a piece kind.
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

    /// The outline ("white") glyph for a piece kind.
    public static func outlineGlyph(for kind: Piece.Kind) -> String {
        switch kind {
        case .king: return "♔\u{FE0E}"
        case .queen: return "♕\u{FE0E}"
        case .rook: return "♖\u{FE0E}"
        case .bishop: return "♗\u{FE0E}"
        case .knight: return "♘\u{FE0E}"
        case .pawn: return "♙\u{FE0E}"
        }
    }

    public static func name(for kind: Piece.Kind) -> String {
        switch kind {
        case .king: return "king"
        case .queen: return "queen"
        case .rook: return "rook"
        case .bishop: return "bishop"
        case .knight: return "knight"
        case .pawn: return "pawn"
        }
    }

    /// "white queen", "black pawn".
    public static func name(for piece: Piece) -> String {
        "\(piece.color == .white ? "white" : "black") \(name(for: piece.kind))"
    }
}

public struct PieceView: View {
    public let piece: Piece
    /// Side length of the square the piece sits in.
    public let size: CGFloat

    public init(piece: Piece, size: CGFloat) {
        self.piece = piece
        self.size = size
    }

    public var body: some View {
        ZStack {
            if piece.color == .white {
                // Inset so the fill never pokes out past the outline. In the
                // system font the solid pawn is drawn noticeably larger than
                // the outline pawn, so it needs more.
                Text(PieceGlyph.glyph(for: piece.kind))
                    .foregroundStyle(BoardPalette.lightSquare)
                    .scaleEffect(piece.kind == .pawn ? 0.86 : 0.93)
                Text(PieceGlyph.outlineGlyph(for: piece.kind))
                    .foregroundStyle(BoardPalette.ink)
            } else {
                Text(PieceGlyph.glyph(for: piece.kind))
                    .foregroundStyle(BoardPalette.ink)
            }
        }
        // The system's chess glyphs are small in their em box; at the full
        // square size they fill about two thirds of the square.
        .font(.system(size: size))
        .frame(width: size, height: size)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(PieceGlyph.name(for: piece))
    }
}
