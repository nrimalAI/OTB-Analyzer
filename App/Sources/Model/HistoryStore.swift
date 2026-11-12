import ChessAnalysis
import ChessKit
import ChessVision
import Foundation

/// A position the engine finished analyzing, as listed on the home screen.
struct SavedPosition: Codable, Identifiable, Equatable {
    /// Full FEN; also the identity — analyzing the same position again
    /// updates its entry instead of adding another.
    let fen: String
    var date: Date
    /// Best line's evaluation from White's side: one of these is set.
    var centipawns: Int?
    var mate: Int?
    /// Best move in SAN.
    var bestMove: String?
    var depth: Int

    var id: String { fen }

    var score: Score {
        if let mate { return .mate(mate) }
        return .centipawns(centipawns ?? 0)
    }

    var pieces: [Square: Piece] {
        FENBuilder.pieces(fromFEN: fen) ?? [:]
    }

    var sideToMove: Piece.Color {
        let fields = fen.split(separator: " ")
        return fields.count > 1 && fields[1] == "b" ? .black : .white
    }
}

/// The list of analyzed positions, kept as a small JSON file on the device.
/// Nothing here leaves the phone.
@MainActor
final class HistoryStore: ObservableObject {

    @Published private(set) var items: [SavedPosition] = []

    private let fileURL: URL
    private static let limit = 100

    init(fileURL: URL = HistoryStore.defaultFileURL) {
        self.fileURL = fileURL
        load()
    }

    nonisolated static var defaultFileURL: URL {
        URL.applicationSupportDirectory.appending(path: "positions.json")
    }

    /// Adds (or refreshes and moves to the top) the entry for `fen`.
    func record(fen: String, score: Score, bestMove: String?, depth: Int, date: Date = .now) {
        var entry = SavedPosition(fen: fen, date: date, bestMove: bestMove, depth: depth)
        switch score {
        case .centipawns(let cp): entry.centipawns = cp
        case .mate(let n): entry.mate = n
        }
        items.removeAll { $0.fen == fen }
        items.insert(entry, at: 0)
        if items.count > Self.limit { items.removeLast(items.count - Self.limit) }
        save()
    }

    func delete(_ item: SavedPosition) {
        items.removeAll { $0.id == item.id }
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        items = (try? decoder.decode([SavedPosition].self, from: data)) ?? []
    }

    private func save() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted]
        guard let data = try? encoder.encode(items) else { return }
        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: fileURL, options: .atomic)
    }
}
