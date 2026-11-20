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

    func testOutlineGlyphsExistForAllKinds() {
        for kind in Piece.Kind.allCases {
            XCTAssertFalse(PieceGlyph.outlineGlyph(for: kind).isEmpty)
            XCTAssertNotEqual(PieceGlyph.outlineGlyph(for: kind), PieceGlyph.glyph(for: kind))
        }
    }

    func testRotatedSquares() {
        XCTAssertEqual(BoardGeometry.rotated(.a1), .h8)
        XCTAssertEqual(BoardGeometry.rotated(.e2), .d7)
        XCTAssertEqual(BoardGeometry.rotated(.h1), .a8)
        for square in Square.allCases {
            XCTAssertEqual(BoardGeometry.rotated(BoardGeometry.rotated(square)), square)
        }
    }

    func testQuarterTurn() {
        // Clockwise seen from White's side: the corners go round a1 → a8 → h8 → h1.
        XCTAssertEqual(BoardGeometry.quarterTurned(.a1), .a8)
        XCTAssertEqual(BoardGeometry.quarterTurned(.a8), .h8)
        XCTAssertEqual(BoardGeometry.quarterTurned(.h8), .h1)
        XCTAssertEqual(BoardGeometry.quarterTurned(.h1), .a1)
        // White's pawns on the second rank end up on the b-file.
        XCTAssertEqual(BoardGeometry.quarterTurned(.e2), .b4)
        var seen = Set<Square>()
        for square in Square.allCases {
            let once = BoardGeometry.quarterTurned(square)
            seen.insert(once)
            // Two quarters are the half turn; four are none.
            XCTAssertEqual(BoardGeometry.quarterTurned(once), BoardGeometry.rotated(square))
            XCTAssertEqual(
                BoardGeometry.quarterTurned(BoardGeometry.quarterTurned(BoardGeometry.rotated(square))),
                square)
            // A quarter turn changes the colour of every square (a1 dark, a8 light).
            XCTAssertEqual(BoardGeometry.isLight(once), !BoardGeometry.isLight(square))
        }
        XCTAssertEqual(seen.count, 64)
    }

    func testArrowGeometry() throws {
        // b1 → b3 on an 800pt board (cell 100): straight up the b-file.
        let from = BoardGeometry.center(of: .b1, boardSize: 800, orientation: .whiteBottom)
        let to = BoardGeometry.center(of: .b3, boardSize: 800, orientation: .whiteBottom)
        let arrow = try XCTUnwrap(BoardGeometry.arrow(from: from, to: to, cell: 100))
        XCTAssertEqual(arrow.shaftStart.x, 150, accuracy: 0.01)
        XCTAssertEqual(arrow.shaftStart.y, 728, accuracy: 0.01)
        XCTAssertEqual(arrow.tip.y, 558, accuracy: 0.01)
        XCTAssertEqual(arrow.shaftEnd.y, 600, accuracy: 0.01)
        XCTAssertEqual(abs(arrow.headLeft.x - arrow.headRight.x), 48, accuracy: 0.01)
        XCTAssertEqual(arrow.shaftWidth, 17, accuracy: 0.01)
    }

    func testArrowBetweenSamePointIsNil() {
        let p = CGPoint(x: 50, y: 50)
        XCTAssertNil(BoardGeometry.arrow(from: p, to: p, cell: 100))
    }
}

final class BoardAccessibilityTests: XCTestCase {

    func testSquareLabels() {
        let pawn = Piece(.pawn, color: .white, square: .e4)
        XCTAssertEqual(BoardView.accessibilityLabel(square: .e4, piece: pawn, isFlagged: false), "e4, white pawn")
        XCTAssertEqual(
            BoardView.accessibilityLabel(square: .g1, piece: Piece(.queen, color: .white, square: .g1), isFlagged: true),
            "g1, white queen, double-check")
        XCTAssertEqual(
            BoardView.accessibilityLabel(square: .e2, piece: nil, isFlagged: false, isHighlighted: true),
            "e2, empty, last move")
    }

    func testArrowsAreSpoken() {
        XCTAssertEqual(BoardView.arrowsDescription([]), "")
        XCTAssertEqual(BoardView.arrowsDescription([BoardArrow(from: .g1, to: .f3)]), "Arrow from g1 to f3")
    }
}
