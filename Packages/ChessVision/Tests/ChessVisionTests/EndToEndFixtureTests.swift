import ChessKit
import CoreML
import ImageIO
import Foundation
import Testing

@testable import ChessVision

/// Runs the full Swift recognition pipeline (both CoreML models + geometry)
/// against real ChessReD test photos with known positions.
///
/// Skips when the exported models aren't present (`Models/*.mlpackage` at the
/// repo root — produced by `Training/export_coreml.py`), so `swift test`
/// stays green on a fresh clone.
@Suite(.serialized) struct EndToEndFixtureTests {

    /// Repo root, derived from this file's location.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // ChessVisionTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // ChessVision
            .deletingLastPathComponent()  // Packages
            .deletingLastPathComponent()  // repo root
    }

    private static var modelPackages: (corners: URL, pieces: URL) {
        let models = repoRoot.appending(path: "Models")
        return (models.appending(path: "BoardCorners.mlpackage"),
                models.appending(path: "Pieces.mlpackage"))
    }

    /// Whether the exported models are there. The tests below are skipped,
    /// not failed, when they are not.
    static var modelsPresent: Bool {
        let packages = modelPackages
        return FileManager.default.fileExists(atPath: packages.corners.path)
            && FileManager.default.fileExists(atPath: packages.pieces.path)
    }

    /// Compiled and loaded once for all tests (a static is initialised once,
    /// even when tests run in parallel).
    private static let recognizer = Result { () throws -> CoreMLBoardRecognizer in
        // .mlpackage must be compiled before loading (Xcode normally does this
        // at build time; tests do it at runtime).
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let packages = modelPackages
        let corners = try MLModel(
            contentsOf: MLModel.compileModel(at: packages.corners), configuration: config)
        let pieces = try MLModel(
            contentsOf: MLModel.compileModel(at: packages.pieces), configuration: config)
        return try CoreMLBoardRecognizer(cornerModel: corners, pieceModel: pieces)
    }

    private func makeRecognizer() throws -> CoreMLBoardRecognizer {
        try Self.recognizer.get()
    }

    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testRecognizesFixturePositions() async throws {
        let recognizer = try makeRecognizer()

        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: fixtures.appending(path: "expected.json")))
        #expect(!manifest.isEmpty)

        var totalCorrect = 0
        var totalSquares = 0

        for (file, expectedPlacement) in manifest.sorted(by: { $0.key < $1.key }) {
            let imageURL = fixtures.appending(path: file)
            guard
                let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                Issue.record("could not load \(file)")
                continue
            }

            let result = try await recognizer.recognize(image)
            let got = result.pieces.mapValues(\.piece)
            let expected = try #require(FENBuilder.pieces(fromPlacement: expectedPlacement))

            var correct = 0
            for rank in 1...8 {
                for fileNum in 1...8 {
                    let sq = FENBuilder.square(file: fileNum, rank: rank)!
                    let want = expected[sq].map { FENBuilder.fenLetter(for: $0) }
                    let have = got[sq].map { FENBuilder.fenLetter(for: $0) }
                    if want == have { correct += 1 }
                }
            }
            totalCorrect += correct
            totalSquares += 64
            print("fixture \(file): \(correct)/64 squares  (expected \(expectedPlacement), got \(FENBuilder.placement(got)))")

            // Per-board floor: recognition quality on known test images should
            // never regress below ~90% per-square.
            //
            // test_004005.jpg guards which corner is a1, and only the corner
            // model can pass it. Its position (7Q/8/1k3p2/4p3/8/p3K3/8/8)
            // points the wrong way: it scores only +2.5 the right way up
            // (BoardOrientation), less than the prior of 6 for the corner
            // model's choice, so the pieces cannot overrule a model that puts
            // a1 at the wrong end. With the grid fit the corners are exact
            // and all six pieces are read: 64/64 with a corner model that
            // knows a1 here (main's, 2026-11), 52/64, the board turned by 180
            // degrees, with one that does not (exp3). See docs/RECOGNITION.md.
            #expect(correct >= 58, "\(file) regressed: \(correct)/64")
        }

        // Aggregate floor mirrors the Python eval (~98% per-square on test).
        let accuracy = Double(totalCorrect) / Double(totalSquares)
        print(String(format: "fixture aggregate: %.1f%% per-square", accuracy * 100))
        #expect(accuracy >= 0.94)
    }

    /// Phone photos are 3:4, the fixtures are square. Each fixture is turned
    /// into a 3:4 photo (grey above and below, like extra table) and must
    /// still be read: the recognizer letterboxes it rather than stretching it.
    /// The same photos stretched to a square, which is what the app used to
    /// do, are scored alongside for comparison.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testRecognizesFixturesAsPortraitPhotos() async throws {
        let recognizer = try makeRecognizer()
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: fixtures.appending(path: "expected.json")))

        var letterboxed = 0
        var stretched = 0
        var totalSquares = 0
        for (file, expectedPlacement) in manifest.sorted(by: { $0.key < $1.key }) {
            guard
                let source = CGImageSourceCreateWithURL(
                    fixtures.appending(path: file) as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                Issue.record("could not load \(file)")
                continue
            }
            let expected = try #require(FENBuilder.pieces(fromPlacement: expectedPlacement))
            let portrait = try #require(Self.redraw(image, width: image.width, height: image.width * 4 / 3, stretch: false))
            #expect(portrait.height * 3 == portrait.width * 4)
            let squashed = try #require(Self.redraw(portrait, width: image.width, height: image.width, stretch: true))

            let good = Self.correctSquares(
                try await recognizer.recognize(portrait).pieces.mapValues(\.piece), expected)
            let bad = Self.correctSquares(
                (try? await recognizer.recognize(squashed).pieces.mapValues(\.piece)) ?? [:],
                expected)
            letterboxed += good
            stretched += bad
            totalSquares += 64
            print("portrait \(file): letterboxed \(good)/64, stretched \(bad)/64")
        }
        let accuracy = Double(letterboxed) / Double(totalSquares)
        print(String(
            format: "portrait aggregate: letterboxed %.1f%%, stretched %.1f%% per-square",
            accuracy * 100, Double(stretched) / Double(totalSquares) * 100))
        #expect(accuracy >= 0.90)
    }

    private static func correctSquares(_ got: [Square: Piece], _ expected: [Square: Piece]) -> Int {
        var correct = 0
        for rank in 1...8 {
            for file in 1...8 {
                let sq = FENBuilder.square(file: file, rank: rank)!
                if expected[sq].map({ FENBuilder.fenLetter(for: $0) })
                    == got[sq].map({ FENBuilder.fenLetter(for: $0) })
                {
                    correct += 1
                }
            }
        }
        return correct
    }

    /// Draws `image` into a grey canvas: centred at its own size, or stretched
    /// to fill it.
    private static func redraw(_ image: CGImage, width: Int, height: Int, stretch: Bool) -> CGImage? {
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.setFillColor(red: 0.45, green: 0.45, blue: 0.45, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .high
        let rect = stretch
            ? CGRect(x: 0, y: 0, width: width, height: height)
            : CGRect(
                x: (width - image.width) / 2, y: (height - image.height) / 2,
                width: image.width, height: image.height)
        context.draw(image, in: rect)
        return context.makeImage()
    }
}
