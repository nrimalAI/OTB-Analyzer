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
    /// The photo the position was scanned from, as a file in the store's
    /// photo folder; nil when it was set up by hand.
    var photoFile: String?

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

/// The list of analyzed positions, kept as a small JSON file on the device,
/// with the photo of each scanned one as a JPEG beside it. Nothing here
/// leaves the phone.
@MainActor
final class HistoryStore: ObservableObject {

    @Published private(set) var items: [SavedPosition] = []

    private let fileURL: URL
    /// Where the photos are: "photos" next to the list.
    let photoDirectory: URL
    private static let limit = 100

    init(fileURL: URL = HistoryStore.defaultFileURL) {
        self.fileURL = fileURL
        self.photoDirectory = fileURL.deletingLastPathComponent().appending(path: "photos")
        load()
    }

    /// The photo file of a saved position, if it has one.
    func photoURL(for item: SavedPosition) -> URL? {
        item.photoFile.map { photoDirectory.appending(path: $0) }
    }

    nonisolated static var defaultFileURL: URL {
        URL.applicationSupportDirectory.appending(path: "positions.json")
    }

    /// Adds (or refreshes and moves to the top) the entry for `fen`. A
    /// `photo` (JPEG) replaces the entry's photo; without one it keeps the
    /// photo it had.
    func record(
        fen: String, score: Score, bestMove: String?, depth: Int, photo: Data? = nil,
        date: Date = .now
    ) {
        var entry = SavedPosition(fen: fen, date: date, bestMove: bestMove, depth: depth)
        switch score {
        case .centipawns(let cp): entry.centipawns = cp
        case .mate(let n): entry.mate = n
        }
        let previous = items.first { $0.fen == fen }
        entry.photoFile = previous?.photoFile
        if let photo, let name = writePhoto(photo) {
            if let old = previous { deletePhoto(of: old) }
            entry.photoFile = name
        }
        items.removeAll { $0.fen == fen }
        items.insert(entry, at: 0)
        if items.count > Self.limit {
            items.suffix(from: Self.limit).forEach(deletePhoto)
            items.removeLast(items.count - Self.limit)
        }
        save()
    }

    func delete(_ item: SavedPosition) {
        deletePhoto(of: item)
        items.removeAll { $0.id == item.id }
        save()
    }

    /// Writes a photo under a new name; nil if it could not be written.
    private func writePhoto(_ data: Data) -> String? {
        let name = UUID().uuidString + ".jpg"
        do {
            try FileManager.default.createDirectory(at: photoDirectory, withIntermediateDirectories: true)
            try data.write(to: photoDirectory.appending(path: name), options: .atomic)
            return name
        } catch {
            return nil
        }
    }

    private func deletePhoto(of item: SavedPosition) {
        guard let url = photoURL(for: item) else { return }
        try? FileManager.default.removeItem(at: url)
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
