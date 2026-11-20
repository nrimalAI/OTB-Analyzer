import ChessBoardUI
import ChessKit
import SwiftUI

/// Everything under the board on the Position tab: what the selected square
/// holds and its likely replacements, the piece tray, side to move, photo
/// side and a quarter turn (or Start/Clear when set up by hand), and the FEN.
struct PositionPanel: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selectedSquare: Square?

    /// What's being typed in the FEN field; nil shows the board's own FEN.
    @State private var fenDraft: String?
    @FocusState private var fenFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            squareRow
                .padding(.top, 8)
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
                // Beside the photo side and its turn key, "White" and "Black"
                // need less room than "White's side": a fixed width, so that
                // the wider column cannot squeeze it.
                .frame(width: model.fromScan ? 120 : nil)

                if model.fromScan {
                    labeled("Photo taken from") {
                        HStack(spacing: 6) {
                            InkSegmentedControl(
                                segments: [
                                    .init(value: Piece.Color.white, title: "White\u{2019}s side"),
                                    .init(value: Piece.Color.black, title: "Black\u{2019}s side"),
                                ],
                                selection: model.photoSide,
                                onSelect: setPhotoSide
                            )
                            quarterTurnButton
                        }
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

    // MARK: - The selected square

    /// What the selected square holds and, first, what it most likely holds
    /// instead: one tap on a suggestion fixes a misread piece.
    private var squareRow: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(squareLabel)
                    .mono(12)
                    .foregroundStyle(selectedSquare == nil ? Theme.muted : Theme.ink)
                if let detail = squareDetail {
                    Text(detail)
                        .mono(11)
                        .foregroundStyle(selectedIsDoubtful ? Theme.accent : Theme.muted)
                }
            }
            .lineLimit(1)
            .minimumScaleFactor(0.75)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)

            if let square = selectedSquare, showsSuggestions {
                ForEach(suggestions, id: \.self) { choice in
                    SuggestionKey(choice: choice, square: square) {
                        model.set(choice.piece(on: square), at: square)
                        fenDraft = nil
                    }
                }
            }
        }
        .frame(minHeight: 44)
    }

    private var suggestions: [PieceChoice] {
        selectedSquare.map(model.suggestions(for:)) ?? []
    }

    private var selectedIsDoubtful: Bool {
        selectedSquare.map(model.isDoubtful) ?? false
    }

    /// Suggestions are for squares worth a second look, or for any square
    /// while the position cannot be analyzed (a king read as a queen is not
    /// always marked); a square read right gets none, to keep the row quiet.
    private var showsSuggestions: Bool {
        !suggestions.isEmpty && (selectedIsDoubtful || !model.problems.isEmpty)
    }

    /// "Tap a square to change it", or e.g. "c1 · white queen".
    private var squareLabel: String {
        guard let square = selectedSquare else { return "Tap a square to change it" }
        return square.notation + " \u{00B7} " + (model.pieces[square].map(PieceGlyph.name(for:)) ?? "empty")
    }

    /// "54% sure · likely", "Likely instead", or nothing.
    private var squareDetail: String? {
        guard let square = selectedSquare, showsSuggestions else { return nil }
        if let confidence = model.confidence[square], model.uncertainSquares.contains(square) {
            return "\(Int((confidence * 100).rounded()))% sure \u{00B7} likely"
        }
        return selectedIsDoubtful ? "Likely" : "Or perhaps"
    }

    private var quarterTurnButton: some View {
        Button {
            model.turnQuarter()
            selectedSquare = selectedSquare.map(BoardGeometry.quarterTurned)
            fenDraft = nil
        } label: {
            Image(systemName: "rotate.right")
                .font(.system(size: 17, weight: .medium))
        }
        .buttonStyle(OutlineButtonStyle())
        .frame(width: 44, height: 44)
        .accessibilityLabel("Turn the board a quarter")
        .accessibilityHint("For a board read sideways, with the pieces in files instead of ranks")
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

/// A one-tap replacement for the selected square, drawn like a key of the
/// tray with an accent outline: a piece, or "Empty".
private struct SuggestionKey: View {
    let choice: PieceChoice
    let square: Square
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Group {
                if let piece = choice.piece(on: square) {
                    PieceView(piece: piece, size: 36)
                        .frame(width: 44, height: 44)
                        .background(
                            piece.color == .white ? BoardPalette.darkSquare : BoardPalette.lightSquare,
                            in: RoundedRectangle(cornerRadius: 6))
                } else {
                    Text("Empty")
                        .sans(13, weight: .medium)
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)
                        .foregroundStyle(Theme.ink)
                        .padding(.horizontal, 4)
                        .frame(width: 58, height: 44)
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(Theme.accent, lineWidth: 1.5)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(TextButtonStyle())
        .accessibilityLabel(accessibilityLabel)
    }

    private var accessibilityLabel: String {
        guard let piece = choice.piece(on: square) else { return "Make \(square.notation) empty" }
        return "Make it a \(PieceGlyph.name(for: piece))"
    }
}
