import ChessAnalysis
import ChessKit
import SwiftUI

/// Everything under the board on the Analysis tab: the engine's top lines
/// (tap a move to jump there) and the row of actions for the position.
struct AnalysisPanel: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var session: AnalysisSession
    let showToast: (String) -> Void

    @Environment(\.openURL) private var openURL

    private let depthOptions = [12, 18, 24]

    var body: some View {
        VStack(spacing: 0) {
            Rule(weight: .heavy)
                .padding(.top, 10)

            ForEach(Array(session.lines.enumerated()), id: \.element.id) { index, line in
                LineRow(
                    line: line,
                    isSelected: index == session.lineIndex,
                    ply: session.ply,
                    sideToMove: model.sideToMove,
                    selectLine: {
                        if index != session.lineIndex { session.select(line: index) }
                    },
                    selectMove: { move in session.select(line: index, ply: move + 1) }
                )
                Rule()
            }

            actions
                .padding(.top, 4)
        }
    }

    private var actions: some View {
        HStack(spacing: 0) {
            Button("Copy FEN") {
                UIPasteboard.general.string = model.fen
                showToast("FEN copied")
            }
            .frame(maxWidth: .infinity)

            ShareLink(item: model.fen) {
                Text("Share")
            }
            .frame(maxWidth: .infinity)

            Button("Lichess \u{2197}") {
                if let url = Self.lichessURL(fen: model.fen) { openURL(url) }
            }
            .frame(maxWidth: .infinity)
            .accessibilityLabel("Open in Lichess")
            .accessibilityHint("Opens this position in your browser")

            Menu {
                Picker("Depth", selection: $model.depth) {
                    ForEach(depthOptions, id: \.self) { depth in
                        Text("Depth \(depth)").tag(depth)
                    }
                }
            } label: {
                Text("Depth \(model.depth)")
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .contentShape(Rectangle())
            }
            .accessibilityLabel("Search depth \(model.depth)")
        }
        .sans(14, weight: .medium)
        .foregroundStyle(Theme.ink)
        .buttonStyle(TextButtonStyle())
        .frame(height: 44)
    }

    /// Lichess's analysis board for a FEN. Opens in the browser; the app
    /// itself never goes online.
    static func lichessURL(fen: String) -> URL? {
        var components = URLComponents(string: "https://lichess.org/analysis")
        components?.queryItems = [URLQueryItem(name: "fen", value: fen)]
        return components?.url
    }
}

/// One engine line: its evaluation in a badge, then the moves as tappable
/// tokens. The selected line shows where the reader is — played moves in
/// ink, the last one on a chip, the next one in the accent colour.
private struct LineRow: View {
    let line: AnalysisLine
    let isSelected: Bool
    let ply: Int
    let sideToMove: Piece.Color
    let selectLine: () -> Void
    let selectMove: (Int) -> Void

    var body: some View {
        HStack(spacing: 12) {
            Text(EvalFormat.label(line.score))
                .mono(13, weight: .medium)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .foregroundStyle(isSelected ? Theme.paper : Theme.ink)
                .frame(width: 48)
                .padding(.vertical, 4)
                .background(isSelected ? Theme.ink : .clear, in: RoundedRectangle(cornerRadius: 4))
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .strokeBorder(Theme.ink, lineWidth: Theme.ruleWidth)
                )

            FlowLayout(spacing: 9, lineHeight: 28.5) {
                ForEach(Array(line.moves.enumerated()), id: \.offset) { index, san in
                    token(index: index, san: san)
                }
            }
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minHeight: 46)
        .contentShape(Rectangle())
        .onTapGesture(perform: selectLine)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Line \(line.rank), \(EvalFormat.label(line.score))")
    }

    private func token(index: Int, san: String) -> some View {
        let number = EvalFormat.moveNumber(at: index, sideToMove: sideToMove)
        let isNext = isSelected && index == ply
        return Button {
            selectMove(index)
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 0) {
                if !number.isEmpty {
                    Text(number)
                        .mono(12)
                        .foregroundStyle(Theme.muted)
                }
                Text(san)
                    .mono(15, weight: index == 0 || isNext ? .semibold : .regular)
                    .foregroundStyle(color(at: index))
                    .padding(.horizontal, 4)
                    .padding(.vertical, 3)
                    .background(
                        isSelected && index == ply - 1 ? Theme.ink.opacity(0.12) : .clear,
                        in: RoundedRectangle(cornerRadius: 4)
                    )
            }
            .fixedSize()
        }
        .buttonStyle(.plain)
        .allowsHitTesting(index < line.playableCount)
        .accessibilityLabel(number + san)
    }

    private func color(at index: Int) -> Color {
        if isSelected {
            if index < ply { return Theme.ink }
            return index == ply ? Theme.accent : Theme.muted
        }
        return index == 0 ? Theme.ink : Theme.muted
    }
}
