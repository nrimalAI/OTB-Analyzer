import ChessKit
import CoreGraphics
import CoreML
import XCTest

@testable import ChessVision

final class CoreMLParsingTests: XCTestCase {

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

    func testParseCornersScalesToImageSize() throws {
        // Keypoints at known 640-space positions; image is 1280×640 → x
        // doubles, y unchanged.
        let tensor = try poseTensor(keypoints: [(100, 500), (100, 100), (500, 100), (500, 500)])
        let corners = try CoreMLBoardRecognizer.parseCorners(
            from: tensor, imageSize: CGSize(width: 1280, height: 640), scoreThreshold: 0.25)
        XCTAssertEqual(corners.count, 4)
        XCTAssertEqual(corners[0], CGPoint(x: 200, y: 500))  // a1
        XCTAssertEqual(corners[2], CGPoint(x: 1000, y: 100))  // h8
    }

    func testParseCornersRespectsThreshold() throws {
        let tensor = try poseTensor(score: 0.1, keypoints: [(0, 0), (0, 0), (0, 0), (0, 0)])
        XCTAssertThrowsError(
            try CoreMLBoardRecognizer.parseCorners(
                from: tensor, imageSize: CGSize(width: 640, height: 640), scoreThreshold: 0.25)
        ) { error in
            XCTAssertEqual(error as? RecognitionError, .boardNotFound)
        }
    }

    func testParseCornersRejectsWrongShape() throws {
        let bad = try MLMultiArray(shape: [1, 5, 10], dataType: .float32)
        XCTAssertThrowsError(
            try CoreMLBoardRecognizer.parseCorners(
                from: bad, imageSize: CGSize(width: 640, height: 640), scoreThreshold: 0.25))
    }

    func testPieceClassParsing() {
        XCTAssertTrue(CoreMLBoardRecognizer.pieceClass(fromLabel: "white_pawn")! == (.pawn, .white))
        XCTAssertTrue(CoreMLBoardRecognizer.pieceClass(fromLabel: "black_king")! == (.king, .black))
        XCTAssertTrue(CoreMLBoardRecognizer.pieceClass(fromLabel: "white_knight")! == (.knight, .white))
        XCTAssertNil(CoreMLBoardRecognizer.pieceClass(fromLabel: "empty"))
        XCTAssertNil(CoreMLBoardRecognizer.pieceClass(fromLabel: "purple_wizard"))
        XCTAssertNil(CoreMLBoardRecognizer.pieceClass(fromLabel: "pawn"))
    }

    func testOrientationGuess() {
        // a1 lower than a8 → camera on White's side.
        XCTAssertEqual(
            CoreMLBoardRecognizer.orientationGuess(corners: [
                CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 100),
                CGPoint(x: 800, y: 100), CGPoint(x: 800, y: 800),
            ]),
            .whiteBottom)
        // a1 higher than a8 → camera on Black's side.
        XCTAssertEqual(
            CoreMLBoardRecognizer.orientationGuess(corners: [
                CGPoint(x: 800, y: 100), CGPoint(x: 800, y: 800),
                CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 100),
            ]),
            .blackBottom)
    }
}
