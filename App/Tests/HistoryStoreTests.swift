import ChessAnalysis
import Foundation
import Testing
import UIKit

@testable import ChessAnalyzer

@MainActor
@Suite final class HistoryStoreTests {

    private let folder = FileManager.default.temporaryDirectory
        .appending(path: "HistoryStoreTests-\(UUID().uuidString)")
    private var file: URL { folder.appending(path: "positions.json") }

    deinit {
        try? FileManager.default.removeItem(at: folder)
    }

    private let fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    private let photo = Data([0xFF, 0xD8, 0xFF, 0x01, 0x02])

    @Test func aPhotoIsKeptWithItsPositionAndSurvivesAReload() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .centipawns(120), bestMove: "Kd2", depth: 18, photo: photo)
        let item = try #require(store.items.first)
        let url = try #require(store.photoURL(for: item))
        #expect(try Data(contentsOf: url) == photo)

        let reloaded = HistoryStore(fileURL: file)
        #expect(reloaded.items.first?.photoFile == item.photoFile)
    }

    @Test func analysingAgainWithoutAPhotoKeepsThePhoto() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .centipawns(120), bestMove: "Kd2", depth: 12, photo: photo)
        let first = try #require(store.items.first?.photoFile)
        store.record(fen: fen, score: .centipawns(130), bestMove: "Kd2", depth: 18)
        #expect(store.items.count == 1)
        #expect(store.items.first?.photoFile == first)
    }

    @Test func aNewPhotoReplacesTheOldFile() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .centipawns(120), bestMove: nil, depth: 12, photo: photo)
        let old = try #require(store.photoURL(for: try #require(store.items.first)))
        store.record(fen: fen, score: .centipawns(120), bestMove: nil, depth: 18, photo: Data([1, 2, 3]))
        #expect(!FileManager.default.fileExists(atPath: old.path))
        let new = try #require(store.photoURL(for: try #require(store.items.first)))
        #expect(try Data(contentsOf: new) == Data([1, 2, 3]))
    }

    @Test func deletingAPositionDeletesItsPhoto() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .mate(2), bestMove: "Kd2", depth: 18, photo: photo)
        let item = try #require(store.items.first)
        let url = try #require(store.photoURL(for: item))
        store.delete(item)
        #expect(store.items.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func aPositionSetUpByHandHasNoPhoto() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .centipawns(0), bestMove: nil, depth: 18)
        #expect(store.photoURL(for: try #require(store.items.first)) == nil)
    }

    @Test func analysingAReopenedPositionKeepsItsPhotoFile() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .centipawns(120), bestMove: nil, depth: 18, photo: photo)
        let first = try #require(store.items.first?.photoFile)
        // Reopened, its photo comes back as the same bytes.
        store.record(fen: fen, score: .centipawns(110), bestMove: nil, depth: 24, photo: photo)
        #expect(store.items.first?.photoFile == first)
    }

    @Test func positionsDroppedFromAFullListTakeTheirPhotosWithThem() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: "8/8/8/8/8/8/8/K6k w - - 0 1", score: .centipawns(0), bestMove: nil, depth: 1, photo: photo)
        let oldest = try #require(store.photoURL(for: try #require(store.items.first)))
        for i in 0..<HistoryStore.limit {
            store.record(fen: "8/8/8/8/8/8/8/K6k w - - 0 \(i + 2)", score: .centipawns(0), bestMove: nil, depth: 1)
        }
        #expect(store.items.count == HistoryStore.limit)
        #expect(!FileManager.default.fileExists(atPath: oldest.path))
    }

    @Test func photosNoPositionNamesAreSweptOnLoad() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .centipawns(0), bestMove: nil, depth: 1, photo: photo)
        let kept = try #require(store.photoURL(for: try #require(store.items.first)))
        let stray = store.photoDirectory.appending(path: "stray.jpg")
        try photo.write(to: stray)
        _ = HistoryStore(fileURL: file)
        #expect(!FileManager.default.fileExists(atPath: stray.path))
        #expect(FileManager.default.fileExists(atPath: kept.path))
    }

    @Test func aListThatCannotBeReadKeepsItsPhotos() throws {
        try FileManager.default.createDirectory(
            at: folder.appending(path: "photos"), withIntermediateDirectories: true)
        let photoFile = folder.appending(path: "photos/kept.jpg")
        try photo.write(to: photoFile)
        try Data("not json".utf8).write(to: file)
        let store = HistoryStore(fileURL: file)
        #expect(store.items.isEmpty)
        #expect(FileManager.default.fileExists(atPath: photoFile.path))
    }

    @Test func theModelsPhotoFollowsItsBoard() throws {
        let store = HistoryStore(fileURL: file)
        store.record(fen: fen, score: .centipawns(0), bestMove: nil, depth: 1, photo: photo)
        let model = AppModel(history: store)
        model.open(try #require(store.items.first))
        #expect(model.photoJPEG == photo)
        // Clearing the board, or starting again, lets the photo go.
        model.resetBoard(toStartingPosition: false)
        #expect(model.photoJPEG == nil)
        model.open(try #require(store.items.first))
        model.startFromScratch(startingPosition: true)
        #expect(model.photoJPEG == nil)
    }

    @Test func olderListsWithoutPhotosStillLoad() throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let json = #"[{"fen":"\#(fen)","date":"2026-11-07T19:42:00Z","centipawns":35,"bestMove":"Kd2","depth":18}]"#
        try Data(json.utf8).write(to: file)
        let store = HistoryStore(fileURL: file)
        #expect(store.items.count == 1)
        #expect(store.items.first?.photoFile == nil)
    }
}

@Suite struct KeptPhotoTests {

    private func image(width: Int, height: Int) -> CGImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
            .image { context in
                UIColor.brown.setFill()
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            }
            .cgImage!
    }

    @Test func aLargePhotoIsShrunkToTheKeptSize() async throws {
        let data = try #require(await AppModel.keptCopy(of: image(width: 3024, height: 4032)))
        let kept = try #require(UIImage(data: data)?.cgImage)
        #expect(kept.height == Int(AppModel.keptPhotoSide))
        #expect(kept.width == 960)
    }

    @Test func aSmallPhotoIsNotEnlarged() async throws {
        let data = try #require(await AppModel.keptCopy(of: image(width: 600, height: 800)))
        let kept = try #require(UIImage(data: data)?.cgImage)
        #expect(kept.width == 600)
        #expect(kept.height == 800)
    }
}
