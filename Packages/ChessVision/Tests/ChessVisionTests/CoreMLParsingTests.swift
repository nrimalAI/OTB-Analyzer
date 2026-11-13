import ChessKit
import CoreGraphics
import CoreML
import Foundation
import Testing

@testable import ChessVision

struct CoreMLParsingTests {

    /// Builds a synthetic [1, 17, anchors] pose tensor with one hot anchor.
    private func poseTensor(
        anchors: Int = 16, hotAnchor: Int = 5, score: Float = 0.9,
        keypoints: [(Float, Float)]  // in 640-space, (a1, a8, h8, h1)
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 17, NSNumber(value: anchors)], dataType: .float32)
        for i in 0..<array.count { array[i] = 0 }
        array[4 * anchors + hotAnchor] = NSNumber(value: score)
        for (k, (x, y)) in keypoints.enumerated() {
            array[(5 + k * 3) * anchors + hotAnchor] = NSNumber(value: x)
            array[(6 + k * 3) * anchors + hotAnchor] = NSNumber(value: y)
            array[(7 + k * 3) * anchors + hotAnchor] = 1.0
        }
        return array
    }

    @Test func testParseCornersMapsThroughLetterbox() throws {
        // A 1280×640 photo letterboxed into the 640 model input fills the full
        // width and the rows 160..<480. Keypoints are in 640-space.
        let tensor = try poseTensor(keypoints: [(100, 400), (100, 200), (500, 200), (500, 400)])
        let corners = try CoreMLBoardRecognizer.parseCorners(
            from: tensor,
            letterbox: Letterbox(imageSize: CGSize(width: 1280, height: 640)),
            scoreThreshold: 0.25)
        #expect(corners.count == 4)
        #expect(abs(corners[0].x - 200) <= 1e-6)   // a1: x doubles
        #expect(abs(corners[0].y - 480) <= 1e-6)   // (400 - 160) × 2
        #expect(abs(corners[2].x - 1000) <= 1e-6)  // h8
        #expect(abs(corners[2].y - 80) <= 1e-6)    // (200 - 160) × 2
    }

    @Test func testParseCornersSquareImageIsPlainScaling() throws {
        let tensor = try poseTensor(keypoints: [(100, 500), (100, 100), (500, 100), (500, 500)])
        let corners = try CoreMLBoardRecognizer.parseCorners(
            from: tensor,
            letterbox: Letterbox(imageSize: CGSize(width: 1280, height: 1280)),
            scoreThreshold: 0.25)
        #expect(abs(corners[0].x - 200) <= 1e-6)
        #expect(abs(corners[0].y - 1000) <= 1e-6)
    }

    @Test func testParseCornersRespectsThreshold() throws {
        let tensor = try poseTensor(score: 0.1, keypoints: [(0, 0), (0, 0), (0, 0), (0, 0)])
        #expect(throws: RecognitionError.boardNotFound) {
            try CoreMLBoardRecognizer.parseCorners(
                from: tensor, letterbox: Letterbox(imageSize: CGSize(width: 640, height: 640)),
                scoreThreshold: 0.25)
        }
    }

    @Test func testParseCornersRejectsWrongShape() throws {
        let bad = try MLMultiArray(shape: [1, 5, 10], dataType: .float32)
        #expect(throws: RecognitionError.modelUnavailable) {
            try CoreMLBoardRecognizer.parseCorners(
                from: bad, letterbox: Letterbox(imageSize: CGSize(width: 640, height: 640)),
                scoreThreshold: 0.25)
        }
    }

    @Test func testPieceClassParsing() {
        #expect(CoreMLBoardRecognizer.pieceClass(fromLabel: "white_pawn")! == (.pawn, .white))
        #expect(CoreMLBoardRecognizer.pieceClass(fromLabel: "black_king")! == (.king, .black))
        #expect(CoreMLBoardRecognizer.pieceClass(fromLabel: "white_knight")! == (.knight, .white))
        #expect(CoreMLBoardRecognizer.pieceClass(fromLabel: "empty") == nil)
        #expect(CoreMLBoardRecognizer.pieceClass(fromLabel: "purple_wizard") == nil)
        #expect(CoreMLBoardRecognizer.pieceClass(fromLabel: "pawn") == nil)
    }

    @Test func testOrientationGuess() {
        // a1 lower than a8 → camera on White's side.
        #expect(CoreMLBoardRecognizer.orientationGuess(corners: [
                CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 100),
                CGPoint(x: 800, y: 100), CGPoint(x: 800, y: 800),
            ]) == .whiteBottom)
        // a1 higher than a8 → camera on Black's side.
        #expect(CoreMLBoardRecognizer.orientationGuess(corners: [
                CGPoint(x: 800, y: 100), CGPoint(x: 800, y: 800),
                CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 100),
            ]) == .blackBottom)
    }
}
