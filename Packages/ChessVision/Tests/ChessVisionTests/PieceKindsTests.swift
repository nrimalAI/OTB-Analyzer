import ChessKit
import CoreGraphics
import Foundation
import Testing

@testable import ChessVision

/// `PieceKinds` against `chess_pipeline.kinds_from_classifier`,
/// `classifier_scale` and `classifier_square`. The reference answers are in
/// `Fixtures/kinds_cases.json`, written by `Training/swift_reference.py
/// kinds-cases`; run that again after changing the Python, and a change in
/// the JSON is a change the Swift must follow.
struct PieceKindsTests {

    struct Recorded: Decodable {
        struct Kinds: Decodable {
            let cls: String
            let confidence: Float
            let probs: [String: Float]
            let out: String
            let out_confidence: Double
            /// nil: the piece was sure enough to be left as it is.
            let scores: [String: Double]?
        }
        struct Square: Decodable {
            let size: [Int]
            let box: [Double]
            let scale: Double
            let square: [Int]
        }
        let kinds: [Kinds]
        let squares: [Square]
    }

    static func recorded() throws -> Recorded {
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        return try JSONDecoder().decode(
            Recorded.self, from: Data(contentsOf: fixtures.appending(path: "kinds_cases.json")))
    }

    static func label(_ detection: PieceDetection) -> String {
        PieceSides.label(detection)
    }

    @Test func kindsAreTheReferencesOnEveryRecordedCase() throws {
        let cases = try Self.recorded().kinds
        #expect(cases.count >= 400)
        var changed = 0, leftAlone = 0
        for (index, recorded) in cases.enumerated() {
            let (kind, color) = try #require(CoreMLBoardRecognizer.pieceClass(fromLabel: recorded.cls))
            let piece = PieceDetection(
                piece: kind, color: color, confidence: recorded.confidence,
                boundingBox: CGRect(x: 10, y: 20, width: 50, height: 110),
                kindScores: [kind: Double(recorded.confidence)])
            let out = try #require(PieceKinds.apply([piece], classProbabilities: [recorded.probs]).first)
            #expect(Self.label(out) == recorded.out, "case \(index)")
            #expect(abs(Double(out.confidence) - recorded.out_confidence) < 1e-5, "case \(index)")
            #expect(out.color == color, "case \(index): the side is the piece model's")
            #expect(out.boundingBox == piece.boundingBox)
            guard let scores = recorded.scores else {
                // Left as it is: the piece model's own scores stay.
                #expect(out.kindScores == piece.kindScores, "case \(index)")
                #expect(out.confidence == piece.confidence, "case \(index)")
                leftAlone += 1
                continue
            }
            for (name, score) in scores {
                let kind = try #require(OneKingASideTests.kind(named: name))
                #expect(abs((out.kindScores[kind] ?? 0) - score) < 1e-5, "case \(index), \(name)")
            }
            #expect(out.kindScores.count == scores.count, "case \(index)")
            changed += recorded.out != recorded.cls ? 1 : 0
        }
        // The cases must exercise both outcomes.
        #expect(changed > 50 && changed < cases.count - 50)
        #expect(leftAlone > 50 && leftAlone < cases.count - 50)
    }

    @Test func squaresAreTheReferences() throws {
        let cases = try Self.recorded().squares
        #expect(cases.count >= 300)
        for (index, recorded) in cases.enumerated() {
            let scale = PieceKinds.scale(width: recorded.size[0], height: recorded.size[1])
            #expect(abs(scale - recorded.scale) < 1e-12, "case \(index)")
            let box = CGRect(
                x: recorded.box[0], y: recorded.box[1],
                width: recorded.box[2] - recorded.box[0], height: recorded.box[3] - recorded.box[1])
            let square = PieceKinds.square(of: box, scale: recorded.scale)
            #expect([square.x, square.y, square.side] == recorded.square, "case \(index)")
        }
    }

    /// A picture whose red is its column and green its row, so that where a
    /// pixel of a crop came from can be read off it.
    static func ramp(width: Int, height: Int) throws -> CGImage {
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let data = try #require(context.data).bindMemory(to: UInt8.self, capacity: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let at = (y * width + x) * 4
                data[at] = UInt8(x); data[at + 1] = UInt8(y); data[at + 2] = 0; data[at + 3] = 255
            }
        }
        return try #require(context.makeImage())
    }

    static func pixel(_ image: CGImage, _ x: Int, _ y: Int) throws -> (Int, Int) {
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try #require(CGContext(
            data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: image.width * 4, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        let data = try #require(context.data).bindMemory(
            to: UInt8.self, capacity: image.width * image.height * 4)
        let at = (y * image.width + x) * 4
        return (Int(data[at]), Int(data[at + 1]))
    }

    /// A crop of the classifier's own side is the photo's pixels as they
    /// are, origin top-left; where it runs off the photo, the edge pixels are
    /// repeated (the reference's `cv2.BORDER_REPLICATE`).
    @Test func cropsAreThePhotosPixelsWithItsEdgesRepeated() throws {
        let image = try Self.ramp(width: 200, height: 180)
        let side = PieceKinds.side
        let inside = try #require(PieceKinds.crop(image, x: 30, y: 40, side: side))
        #expect(inside.width == side && inside.height == side)
        for (x, y) in [(0, 0), (5, 9), (100, 3), (127, 127)] {
            let (r, g) = try Self.pixel(inside, x, y)
            #expect(abs(r - (30 + x)) <= 1 && abs(g - (40 + y)) <= 1, "inside at \(x), \(y): \(r), \(g)")
        }
        let off = try #require(PieceKinds.crop(image, x: -10, y: 120, side: side))
        for (x, y) in [(0, 0), (9, 0), (10, 5), (40, 59), (60, 100), (127, 127)] {
            let (r, g) = try Self.pixel(off, x, y)
            let wantX = min(max(x - 10, 0), 199), wantY = min(max(120 + y, 0), 179)
            #expect(abs(r - wantX) <= 1 && abs(g - wantY) <= 1, "off at \(x), \(y): \(r), \(g)")
        }
    }

    /// Without a classifier, or with it switched off, the pieces are the
    /// piece model's.
    @Test func withoutAClassifierThePiecesAreThePieceModels() throws {
        let piece = try OneKingASideTests.detection(
            "white_queen", confidence: 0.6, scores: ["queen": 0.6, "king": 0.3])
        #expect(PieceKinds.apply([], classProbabilities: []).isEmpty)
        let none = PieceKinds.apply([piece], classProbabilities: [[:]]).first
        #expect(none.map(Self.label) == "white_queen")
        #expect(none?.kindScores == [.queen: Double(piece.confidence)])
    }
}
