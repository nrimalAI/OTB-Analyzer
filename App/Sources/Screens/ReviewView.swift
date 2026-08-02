import ChessBoardUI
import ChessKit
import ChessVision
import SwiftUI

/// Review & correct the recognized position before analysis.
/// Low-confidence squares are tinted yellow; validation problems red.
struct ReviewView: View {
    @EnvironmentObject private var model: AppModel

    @State private var editingSquare: Square?

    private let lowConfidenceThreshold: Float = 0.85

    var body: some View {
        VStack(spacing: 16) {
            BoardView(
                pieces: model.pieces,
                orientation: model.orientation,
                selected: editingSquare,
                squareTint: tint(for:),
                onTap: { editingSquare = $0 }
            )
            .padding(.horizontal, 8)

            controls

            if !model.validationIssues.isEmpty {
                validationList
            }

            Spacer()

            Button {
                model.path.append(.analysis)
            } label: {
                Label("Analyze", systemImage: "brain.head.profile")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!model.validationIssues.isEmpty)
            .padding(.horizontal, 24)
            .padding(.bottom, 12)
        }
        .navigationTitle("Review Position")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    model.orientation.flip()
                } label: {
                    Image(systemName: "arrow.up.arrow.down")
                }
                .accessibilityLabel("Flip board")
            }
        }
        .sheet(item: $editingSquare) { square in
            PiecePickerView(square: square) { piece in
                model.set(piece, at: square)
                editingSquare = nil
            }
            .presentationDetents([.height(300)])
        }
    }

    private func tint(for square: Square) -> Color? {
        if model.problemSquares.contains(square) {
            return .red.opacity(0.45)
        }
        if let conf = model.confidence[square], conf < lowConfidenceThreshold {
            return .yellow.opacity(0.4)
        }
        return nil
    }

    private var controls: some View {
        VStack(spacing: 12) {
            Picker("Side to move", selection: $model.sideToMove) {
                Text("White to move").tag(Piece.Color.white)
                Text("Black to move").tag(Piece.Color.black)
            }
            .pickerStyle(.segmented)

            DisclosureGroup("Castling rights") {
                VStack(spacing: 4) {
                    Toggle("White O-O", isOn: $model.castling.whiteKingside)
                    Toggle("White O-O-O", isOn: $model.castling.whiteQueenside)
                    Toggle("Black O-O", isOn: $model.castling.blackKingside)
                    Toggle("Black O-O-O", isOn: $model.castling.blackQueenside)
                }
                .font(.subheadline)
            }
            .font(.subheadline)
        }
        .padding(.horizontal, 24)
    }

    private var validationList: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(model.validationIssues, id: \.self) { issue in
                Label(issue.message, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 24)
    }
}

extension Square: @retroactive Identifiable {
    public var id: Int { rawValue }
}
