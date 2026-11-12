import SwiftUI

/// Attribution, licensing, and privacy. Shipping Stockfish (GPL-3) requires
/// distributing the license text and an offer of corresponding source — this
/// screen is that offer.
struct AboutView: View {
    @Environment(\.dismiss) private var dismiss

    private static let sourceURL = URL(string: "https://github.com/nrimalAI/OTB-Analyzer")!
    private static let stockfishURL = URL(string: "https://stockfishchess.org")!
    private static let gplURL = URL(string: "https://www.gnu.org/licenses/gpl-3.0.html")!
    private static let chesskitURL = URL(string: "https://github.com/chesskit-app")!

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Text("Done")
                        .sans(16, weight: .medium)
                        .padding(.horizontal, 12)
                        .frame(height: 44)
                }
                .buttonStyle(TextButtonStyle())
            }
            .padding(.horizontal, 8)
            .padding(.top, 8)

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    Text("Chess Scanner")
                        .eyebrow(tracking: 1.5)
                    Text("About")
                        .serif(44)
                        .tracking(-0.44)
                        .padding(.top, 6)
                        .accessibilityAddTraits(.isHeader)

                    section("Privacy") {
                        Text("Everything happens on your device. Photos, positions, and analysis stay on your iPhone \u{2014} this app makes no network connections and collects no data. The one way a position leaves is if you choose to: Share, or the Lichess button, which opens it in your browser.")
                            .sans(14)
                            .padding(.vertical, 14)
                        Rule()
                    }

                    section("Powered by") {
                        row("Stockfish 17", detail: "Chess engine \u{2014} GPL-3.0, \u{00A9} the Stockfish developers", link: Self.stockfishURL)
                        row("ChessKit", detail: "Chess rules & engine bindings \u{2014} MIT", link: Self.chesskitURL)
                        row("Board recognition", detail: "On-device CoreML models trained for this app")
                    }

                    section("Open source") {
                        row("Source code", detail: "This app is free software under GPL-3.0. Complete corresponding source, including the bundled Stockfish, is available here.", link: Self.sourceURL)
                        row("GNU GPL-3.0 license", detail: "Read the full license text", link: Self.gplURL)
                    }

                    Text("Version \(Bundle.main.shortVersion) (\(Bundle.main.buildNumber))")
                        .mono(11)
                        .foregroundStyle(Theme.muted)
                        .frame(maxWidth: .infinity)
                        .padding(.top, 24)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
        }
        .background(Theme.paper.ignoresSafeArea())
        .foregroundStyle(Theme.ink)
    }

    private func section<Content: View>(
        _ title: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title)
                .eyebrow()
                .padding(.top, 26)
                .padding(.bottom, 8)
                .accessibilityAddTraits(.isHeader)
            Rule(weight: .heavy)
            content()
        }
    }

    @ViewBuilder
    private func row(_ title: String, detail: String, link: URL? = nil) -> some View {
        let label = HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .serif(19)
                Text(detail)
                    .sans(13)
                    .foregroundStyle(Theme.muted)
                    .multilineTextAlignment(.leading)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if link != nil {
                Text("\u{2197}")
                    .sans(15, weight: .medium)
                    .accessibilityHidden(true)
            }
        }
        .padding(.vertical, 12)
        .contentShape(Rectangle())

        if let link {
            Link(destination: link) { label }
                .buttonStyle(TextButtonStyle())
        } else {
            label
        }
        Rule()
    }
}

extension Bundle {
    var shortVersion: String {
        infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
    }
    var buildNumber: String {
        infoDictionary?["CFBundleVersion"] as? String ?? "?"
    }
}
