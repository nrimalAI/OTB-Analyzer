import ChessKit
import XCTest

@testable import ChessBoardUI

final class BoardGeometryTests: XCTestCase {

    func testSquareFromFileRank() {
        XCTAssertEqual(BoardGeometry.square(file: 1, rank: 1), .a1)
        XCTAssertEqual(BoardGeometry.square(file: 8, rank: 8), .h8)
        XCTAssertNil(BoardGeometry.square(file: 0, rank: 3))
    }

    func testWhiteBottomLayout() {
        // a1 is bottom-left: column 0, row 7.
        XCTAssertEqual(BoardGeometry.column(of: .a1, orientation: .whiteBottom), 0)
        XCTAssertEqual(BoardGeometry.row(of: .a1, orientation: .whiteBottom), 7)
        // h8 is top-right.
        XCTAssertEqual(BoardGeometry.column(of: .h8, orientation: .whiteBottom), 7)
        XCTAssertEqual(BoardGeometry.row(of: .h8, orientation: .whiteBottom), 0)
    }

    func testBlackBottomLayout() {
        // Flipped: a1 becomes top-right.
        XCTAssertEqual(BoardGeometry.column(of: .a1, orientation: .blackBottom), 7)
        XCTAssertEqual(BoardGeometry.row(of: .a1, orientation: .blackBottom), 0)
    }

    func testGridRoundTrip() {
        for orientation in [BoardDisplayOrientation.whiteBottom, .blackBottom] {
            for square in Square.allCases {
                let col = BoardGeometry.column(of: square, orientation: orientation)
                let row = BoardGeometry.row(of: square, orientation: orientation)
                XCTAssertEqual(
                    BoardGeometry.square(column: col, row: row, orientation: orientation),
                    square, "\(square) failed under \(orientation)")
            }
        }
    }

    func testSquareColors() {
        XCTAssertFalse(BoardGeometry.isLight(.a1))  // a1 is dark
        XCTAssertTrue(BoardGeometry.isLight(.h1))
        XCTAssertTrue(BoardGeometry.isLight(.a8))
        XCTAssertFalse(BoardGeometry.isLight(.h8))
        XCTAssertTrue(BoardGeometry.isLight(.e4))
        XCTAssertFalse(BoardGeometry.isLight(.d4))
    }

    func testCenterPoints() {
        let center = BoardGeometry.center(of: .a1, boardSize: 800, orientation: .whiteBottom)
        XCTAssertEqual(center.x, 50, accuracy: 0.01)
        XCTAssertEqual(center.y, 750, accuracy: 0.01)
    }

    func testOrientationFlip() {
        var o = BoardDisplayOrientation.whiteBottom
        o.flip()
        XCTAssertEqual(o, .blackBottom)
        o.flip()
        XCTAssertEqual(o, .whiteBottom)
    }

    func testGlyphsExistForAllKinds() {
        for kind in Piece.Kind.allCases {
            XCTAssertFalse(PieceGlyph.glyph(for: kind).isEmpty)
        }
    }
}
