import ChessVision
import Foundation
import ImageIO
import UIKit

/// Headless recognition verification: launch with
/// `SIMCTL_CHILD_RECOG_SELFTEST=<image path>` (simulator apps can read host
/// paths) to run the bundled CoreML models on a photo and write the resulting
/// FEN placement to Documents/recog_selftest.txt.
enum RecognitionSelfTest {

    static var requestedImagePath: String? {
        ProcessInfo.processInfo.environment["RECOG_SELFTEST"]
    }

    static func run(imagePath: String, recognizer: BoardRecognizer) async {
        let out = URL.documentsDirectory.appending(path: "recog_selftest.txt")
        func report(_ s: String) {
            print(s)
            try? s.write(to: out, atomically: true, encoding: .utf8)
        }

        let isCoreML = recognizer is CoreMLBoardRecognizer
        guard
            let source = CGImageSourceCreateWithURL(
                URL(fileURLWithPath: imagePath) as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            report("RECOG FAIL: cannot load \(imagePath)")
            return
        }
        do {
            let start = Date()
            let result = try await recognizer.recognize(image)
            let elapsed = Date().timeIntervalSince(start)
            let placement = FENBuilder.placement(result.pieces.mapValues(\.piece))
            report("RECOG OK coreml=\(isCoreML) time=\(String(format: "%.2f", elapsed))s fen=\(placement)")
        } catch {
            report("RECOG FAIL: \(error)")
        }
    }
}
