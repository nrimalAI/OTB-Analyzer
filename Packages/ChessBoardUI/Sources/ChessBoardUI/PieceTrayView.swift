import ChessKit
import SwiftUI

/// The inline piece tray under the board: white pieces, black pieces, and
/// "Empty". It acts on the currently selected square and is dimmed until one
/// is selected. The chosen piece is delivered already homed on that square.
public struct PieceTrayView: View {

    public let square: Square?
    public let onSelect: (Piece?) -> Void

    public init(square: Square?, onSelect: @escaping (Piece?) -> Void) {
        self.square = square
        self.onSelect = onSelect
    }

    private let kinds: [Piece.Kind] = [.king, .queen, .rook, .bishop, .knight, .pawn]
    private let keyHeight: CGFloat = 44
    private let gap: CGFloat = 6

    public var body: some View {
        HStack(spacing: gap) {
            VStack(spacing: gap) {
                row(color: .white, background: BoardPalette.darkSquare)
                row(color: .black, background: BoardPalette.lightSquare)
            }

            Button {
                onSelect(nil)
            } label: {
                Text("Empty")
                    .font(BoardTypography.label)
                    .foregroundStyle(BoardPalette.ink)
                    .frame(width: 58, height: keyHeight * 2 + gap)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(BoardPalette.ink, lineWidth: 1.5)
                    )
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Empty the square")
        }
        .opacity(square == nil ? 0.4 : 1)
        .animation(.easeOut(duration: 0.15), value: square == nil)
        .disabled(square == nil)
    }

    private func row(color: Piece.Color, background: Color) -> some View {
        HStack(spacing: gap) {
            ForEach(kinds, id: \.self) { kind in
                // The tray shows a piece even with nothing selected; a1 is a
                // placeholder home that is never delivered (the tray is
                // disabled until a square is selected).
                let piece = Piece(kind, color: color, square: square ?? .a1)
                Button {
                    onSelect(piece)
                } label: {
                    PieceView(piece: piece, size: 39)
                        .frame(maxWidth: .infinity)
                        .frame(height: keyHeight)
                        .background(background, in: RoundedRectangle(cornerRadius: 6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(BoardPalette.ink.opacity(0.3), lineWidth: 1)
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }
}
