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
/// with the photo of each scanned one as a JPEG beside it. The app sends
/// none of it anywhere, and the photos are left out of device backups.
@MainActor
final class HistoryStore: ObservableObject {

    @Published private(set) var items: [SavedPosition] = []

    private let fileURL: URL
    /// Where the photos are: "photos" next to the list.
    let photoDirectory: URL
    static let limit = 100

    init(fileURL: URL = HistoryStore.defaultFileURL) {
        self.fileURL = fileURL
        self.photoDirectory = fileURL.deletingLastPathComponent().appending(path: "photos")
        if load() { sweepPhotos() }
    }

    /// The photo file of a saved position, if it has one.
    func photoURL(for item: SavedPosition) -> URL? {
        item.photoFile.map { photoDirectory.appending(path: $0) }
    }

    nonisolated static var defaultFileURL: URL {
        URL.applicationSupportDirectory.appending(path: "positions.json")
    }

    /// Adds (or refreshes and moves to the top) the entry for `fen`. A
    /// `photo` (JPEG) replaces the entry's photo, unless it is the very
    /// photo the entry has (a saved position reopened and analysed again);
    /// without one the entry keeps the photo it had.
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
        var replaced: SavedPosition?
        if let photo, !hasPhoto(photo, previous), let name = writePhoto(photo) {
            replaced = previous
            entry.photoFile = name
        }
        items.removeAll { $0.fen == fen }
        items.insert(entry, at: 0)
        var dropped: [SavedPosition] = []
        if items.count > Self.limit {
            dropped = Array(items.suffix(from: Self.limit))
            items.removeLast(items.count - Self.limit)
        }
        // Old files go only once the list that no longer names them is saved.
        if save() {
            if let replaced { deletePhoto(of: replaced) }
            dropped.forEach(deletePhoto)
        }
    }

    func delete(_ item: SavedPosition) {
        items.removeAll { $0.id == item.id }
        if save() { deletePhoto(of: item) }
    }

    /// Whether `item` already has exactly this photo.
    private func hasPhoto(_ photo: Data, _ item: SavedPosition?) -> Bool {
        guard let url = item.flatMap(photoURL(for:)) else { return false }
        return (try? Data(contentsOf: url)) == photo
    }

    /// Writes a photo under a new name; nil if it could not be written.
    private func writePhoto(_ data: Data) -> String? {
        let name = UUID().uuidString + ".jpg"
        do {
            try FileManager.default.createDirectory(at: photoDirectory, withIntermediateDirectories: true)
            var folder = photoDirectory
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? folder.setResourceValues(values)
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

    /// Removes photos no saved position names (left by a crash between
    /// writing a photo and saving the list). Only after the list was read
    /// without error, so that a list that failed to read keeps its photos.
    private func sweepPhotos() {
        let named = Set(items.compactMap(\.photoFile))
        let files = (try? FileManager.default.contentsOfDirectory(atPath: photoDirectory.path)) ?? []
        for file in files where !named.contains(file) {
            try? FileManager.default.removeItem(at: photoDirectory.appending(path: file))
        }
    }

    /// False when there is a list that could not be read.
    private func load() -> Bool {
        guard let data = try? Data(contentsOf: fileURL) else { return true }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let decoded = try? decoder.decode([SavedPosition].self, from: data) else { return false }
        items = decoded
        return true
    }

    @discardableResult
    private func save() -> Bool {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted]
        guard let data = try? encoder.encode(items) else { return false }
        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        return (try? data.write(to: fileURL, options: .atomic)) != nil
    }
}
