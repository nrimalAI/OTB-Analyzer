import ChessBoardUI
import ChessKit
import SwiftUI

/// Everything under the board on the Position tab: what the selected square
/// holds, the piece tray, side to move, photo side (or Start/Clear when set
/// up by hand), and the FEN.
struct PositionPanel: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selectedSquare: Square?

    /// What's being typed in the FEN field; nil shows the board's own FEN.
    @State private var fenDraft: String?
    @FocusState private var fenFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(hint)
                .mono(12)
                .foregroundStyle(Theme.muted)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .padding(.top, 10)
                .padding(.bottom, 8)

            PieceTrayView(square: selectedSquare) { piece in
                guard let square = selectedSquare else { return }
                model.set(piece, at: square)
                fenDraft = nil
            }

            HStack(alignment: .top, spacing: 12) {
                labeled("To move") {
                    InkSegmentedControl(
                        segments: [
                            .init(value: Piece.Color.white, title: "White"),
                            .init(value: Piece.Color.black, title: "Black"),
                        ],
                        selection: model.sideToMove,
                        onSelect: {
                            model.sideToMove = $0
                            fenDraft = nil
                        }
                    )
                }

                if model.fromScan {
                    labeled("Photo taken from") {
                        InkSegmentedControl(
                            segments: [
                                .init(value: Piece.Color.white, title: "White\u{2019}s side"),
                                .init(value: Piece.Color.black, title: "Black\u{2019}s side"),
                            ],
                            selection: model.photoSide,
                            onSelect: setPhotoSide
                        )
                    }
                } else {
                    labeled("Board") {
                        HStack(spacing: 8) {
                            Button("Start") { resetBoard(toStart: true) }
                            Button("Clear") { resetBoard(toStart: false) }
                        }
                        .sans(14, weight: .medium)
                        .buttonStyle(OutlineButtonStyle())
                        .frame(height: 44)
                    }
                }
            }
            .padding(.top, 14)

            labeled("FEN") {
                TextField("FEN", text: fenBinding, prompt: Text("FEN"))
                    .mono(12)
                    .foregroundStyle(Theme.ink)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.asciiCapable)
                    .submitLabel(.done)
                    .focused($fenFocused)
                    .padding(.horizontal, 10)
                    .frame(height: 44)
                    .background(Theme.field, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(Theme.fieldBorder, lineWidth: 1)
                    )
                    .onChange(of: fenFocused) { _, focused in
                        if !focused { fenDraft = nil }
                    }
            }
            .padding(.top, 14)
        }
    }

    private func labeled<Content: View>(
        _ label: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .eyebrow()
                .lineLimit(1)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Hint

    /// "Tap a square to change it", or e.g. "c1 · white queen · 54% sure".
    private var hint: String {
        guard let square = selectedSquare else { return "Tap a square to change it" }
        var parts = [square.notation, model.pieces[square].map(PieceGlyph.name(for:)) ?? "empty"]
        if let confidence = model.confidence[square], model.uncertainSquares.contains(square) {
            parts.append("\(Int((confidence * 100).rounded()))% sure")
        }
        return parts.joined(separator: " \u{00B7} ")
    }

    // MARK: - Actions

    private func setPhotoSide(_ side: Piece.Color) {
        guard side != model.photoSide else { return }
        model.setPhotoSide(side)
        selectedSquare = selectedSquare.map(BoardGeometry.rotated)
        fenDraft = nil
    }

    private func resetBoard(toStart: Bool) {
        model.resetBoard(toStartingPosition: toStart)
        selectedSquare = nil
        fenDraft = nil
    }

    /// Typing a complete placement updates the board as you go; anything
    /// else just stays in the field until it parses or focus leaves.
    private var fenBinding: Binding<String> {
        Binding(
            get: { fenDraft ?? model.fen },
            set: { text in
                guard text != (fenDraft ?? model.fen) else { return }
                fenDraft = text
                if model.applyFEN(text) { selectedSquare = nil }
            }
        )
    }
}
