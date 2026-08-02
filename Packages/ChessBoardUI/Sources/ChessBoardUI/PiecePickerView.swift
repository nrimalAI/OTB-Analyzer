import ChessKit
import SwiftUI

/// A sheet-friendly picker for setting a square's contents: all twelve pieces
/// plus "empty". The chosen piece is delivered already re-homed to `square`.
public struct PiecePickerView: View {

    public let square: Square
    public let onSelect: (Piece?) -> Void

    public init(square: Square, onSelect: @escaping (Piece?) -> Void) {
        self.square = square
        self.onSelect = onSelect
    }

    private let kinds: [Piece.Kind] = [.king, .queen, .rook, .bishop, .knight, .pawn]

    public var body: some View {
        VStack(spacing: 16) {
            Text("Set \(square.notation)")
                .font(.headline)

            ForEach([Piece.Color.white, .black], id: \.self) { color in
                HStack(spacing: 8) {
                    ForEach(kinds, id: \.self) { kind in
                        let piece = Piece(kind, color: color, square: square)
                        Button {
                            onSelect(piece)
                        } label: {
                            PieceView(piece: piece, size: 44)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .fill(Color.gray.opacity(0.2))
                                )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            Button(role: .destructive) {
                onSelect(nil)
            } label: {
                Label("Clear square", systemImage: "xmark.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .padding(.horizontal)
        }
        .padding()
    }
}
