import ChessBoardUI
import SwiftUI

/// Home: the positions you've analyzed, with one primary action — scan a
/// board — and a quieter way in for setting one up by hand.
struct HomeView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var history: HistoryStore

    @State private var showAbout = false

    var body: some View {
        VStack(spacing: 0) {
            header

            ScrollView {
                LazyVStack(spacing: 0) {
                    Rule(weight: .heavy)
                    if history.items.isEmpty {
                        emptyState
                    } else {
                        ForEach(history.items) { item in
                            SavedPositionRow(item: item) { model.open(item) }
                                .contextMenu {
                                    Button(role: .destructive) {
                                        history.delete(item)
                                    } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                            Rule()
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 18)
            }

            actions
        }
        .background(Theme.paper.ignoresSafeArea())
        .foregroundStyle(Theme.ink)
        .toolbar(.hidden, for: .navigationBar)
        .sheet(isPresented: $showAbout) {
            AboutView()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Chess Scanner")
                    .eyebrow(tracking: 1.5)
                Spacer()
                Button {
                    showAbout = true
                } label: {
                    Text("About")
                        .eyebrow(tracking: 1.5)
                        .foregroundStyle(Theme.ink)
                        .padding(.vertical, 12)
                        .padding(.leading, 16)
                }
                .buttonStyle(TextButtonStyle())
                .padding(.vertical, -12)
            }
            Text("Positions")
                .serif(44)
                .tracking(-0.44)
                .accessibilityAddTraits(.isHeader)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 20)
        .padding(.top, 14)
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("No positions yet")
                .serif(21)
            Text("Scan a board or set one up by hand. Positions you analyze are kept here.")
                .sans(13)
                .foregroundStyle(Theme.muted)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 18)
    }

    private var actions: some View {
        VStack(spacing: 4) {
            Button("Scan a board") {
                model.path = [.scan]
            }
            .buttonStyle(PrimaryButtonStyle())

            Button {
                model.startFromScratch(startingPosition: true)
            } label: {
                Text("Set up a position by hand")
                    .sans(15, weight: .medium)
                    .underline()
                    .frame(maxWidth: .infinity)
                    .frame(height: 46)
            }
            .buttonStyle(TextButtonStyle())
        }
        .padding(.horizontal, 20)
        .padding(.top, 14)
        .padding(.bottom, 8)
    }
}

/// One saved position: thumbnail board, when, the verdict, and the eval.
private struct SavedPositionRow: View {
    let item: SavedPosition
    let open: () -> Void

    var body: some View {
        Button(action: open) {
            HStack(spacing: 14) {
                BoardView(
                    pieces: item.pieces,
                    showsCoordinates: false,
                    borderWidth: 1
                )
                .frame(width: 72, height: 72)
                .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 0) {
                    Text(Self.dateLabel(for: item.date))
                        .eyebrow(tracking: 0.66)
                    Text(EvalFormat.verdict(item.score))
                        .serif(21)
                        .lineLimit(2)
                        .padding(.top, 3)
                        .padding(.bottom, 2)
                    Text(subtitle)
                        .sans(13)
                        .foregroundStyle(Theme.muted)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Text(EvalFormat.label(item.score))
                    .mono(15, weight: .medium)
            }
            .padding(.vertical, 14)
            .contentShape(Rectangle())
        }
        .buttonStyle(TextButtonStyle())
        .accessibilityElement(children: .combine)
    }

    private var subtitle: String {
        let turn = item.sideToMove == .white ? "White to move" : "Black to move"
        guard let best = item.bestMove else { return turn }
        return "\(turn) · best \(best)"
    }

    /// "Today, 19:42", "Yesterday", "Sat 7 Nov".
    static func dateLabel(for date: Date, calendar: Calendar = .current) -> String {
        if calendar.isDateInToday(date) {
            return "Today, " + date.formatted(date: .omitted, time: .shortened)
        }
        if calendar.isDateInYesterday(date) {
            return "Yesterday"
        }
        return date.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated))
    }
}
