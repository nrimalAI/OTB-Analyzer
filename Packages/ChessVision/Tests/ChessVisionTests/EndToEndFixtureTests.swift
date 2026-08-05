import ChessKit
import CoreML
import ImageIO
import XCTest

@testable import ChessVision

/// Runs the full Swift recognition pipeline (both CoreML models + geometry)
/// against real ChessReD test photos with known positions.
///
/// Skips when the exported models aren't present (`Models/*.mlpackage` at the
/// repo root — produced by `Training/export_coreml.py`), so `swift test`
/// stays green on a fresh clone.
final class EndToEndFixtureTests: XCTestCase {

    /// Repo root, derived from this file's location.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // ChessVisionTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // ChessVision
            .deletingLastPathComponent()  // Packages
            .deletingLastPathComponent()  // repo root
    }

    private static var recognizer: CoreMLBoardRecognizer?
    private static var setupDone = false

    private func makeRecognizer() throws -> CoreMLBoardRecognizer {
        if Self.setupDone {
            guard let r = Self.recognizer else {
                throw XCTSkip("Models/*.mlpackage not present — run Training/export_coreml.py")
            }
            return r
        }
        Self.setupDone = true

        let models = Self.repoRoot.appending(path: "Models")
        let cornersPkg = models.appending(path: "BoardCorners.mlpackage")
        let piecesPkg = models.appending(path: "Pieces.mlpackage")
        guard FileManager.default.fileExists(atPath: cornersPkg.path),
              FileManager.default.fileExists(atPath: piecesPkg.path)
        else {
            throw XCTSkip("Models/*.mlpackage not present — run Training/export_coreml.py")
        }

        // .mlpackage must be compiled before loading (Xcode normally does this
        // at build time; tests do it at runtime).
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let corners = try MLModel(
            contentsOf: MLModel.compileModel(at: cornersPkg), configuration: config)
        let pieces = try MLModel(
            contentsOf: MLModel.compileModel(at: piecesPkg), configuration: config)
        let recognizer = try CoreMLBoardRecognizer(cornerModel: corners, pieceModel: pieces)
        Self.recognizer = recognizer
        return recognizer
    }

    func testRecognizesFixturePositions() async throws {
        let recognizer = try makeRecognizer()

        let fixtures = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: fixtures.appending(path: "expected.json")))
        XCTAssertFalse(manifest.isEmpty)

        var totalCorrect = 0
        var totalSquares = 0

        for (file, expectedPlacement) in manifest.sorted(by: { $0.key < $1.key }) {
            let imageURL = fixtures.appending(path: file)
            guard
                let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                XCTFail("could not load \(file)")
                continue
            }

            let result = try await recognizer.recognize(image)
            let got = result.pieces.mapValues(\.piece)
            let expected = try XCTUnwrap(
                FENBuilder.pieces(fromPlacement: expectedPlacement), file: #filePath)

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
            XCTAssertGreaterThanOrEqual(correct, 58, "\(file) regressed: \(correct)/64")
        }

        // Aggregate floor mirrors the Python eval (~98% per-square on test).
        let accuracy = Double(totalCorrect) / Double(totalSquares)
        print(String(format: "fixture aggregate: %.1f%% per-square", accuracy * 100))
        XCTAssertGreaterThanOrEqual(accuracy, 0.94)
    }
}
