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
        NavigationStack {
            List {
                Section("Privacy") {
                    Label {
                        Text("Everything happens on your device. Photos, positions, and analysis never leave your iPhone — this app makes no network connections and collects no data.")
                            .font(.subheadline)
                    } icon: {
                        Image(systemName: "lock.shield")
                    }
                }

                Section("Powered by") {
                    Link(destination: Self.stockfishURL) {
                        row("Stockfish 17", detail: "Chess engine — GPL-3.0, © the Stockfish developers")
                    }
                    Link(destination: Self.chesskitURL) {
                        row("ChessKit", detail: "Chess rules & engine bindings — MIT")
                    }
                    row("Board recognition", detail: "On-device CoreML models trained for this app")
                }

                Section("Open source") {
                    Link(destination: Self.sourceURL) {
                        row("Source code", detail: "This app is free software under GPL-3.0. Complete corresponding source, including the bundled Stockfish, is available here.")
                    }
                    Link(destination: Self.gplURL) {
                        row("GNU GPL-3.0 license", detail: "Read the full license text")
                    }
                }

                Section {
                    Text("Version \(Bundle.main.shortVersion) (\(Bundle.main.buildNumber))")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
                .listRowBackground(Color.clear)
            }
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func row(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).foregroundStyle(.primary)
            Text(detail).font(.caption).foregroundStyle(.secondary)
        }
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
