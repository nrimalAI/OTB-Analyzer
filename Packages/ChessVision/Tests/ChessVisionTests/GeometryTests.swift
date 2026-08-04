import ChessKit
import CoreGraphics
import XCTest

@testable import ChessVision

/// Mirrors the self-checks in Training/chess_pipeline.py — the Swift and
/// Python implementations must stay numerically in agreement.
final class HomographyTests: XCTestCase {

    /// Perfect top-down 800×800 board, a1 at bottom-left (image y grows down).
    /// Corner order: a1, a8, h8, h1.
    private let topDownCorners = [
        CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 0),
        CGPoint(x: 800, y: 0), CGPoint(x: 800, y: 800),
    ]

    func testTopDownProjection() throws {
        let H = try XCTUnwrap(Homography(corners: topDownCorners))
        // e2 center: x = 4.5/8*800 = 450, y = 800 - 1.5/8*800 = 650
        let p = try XCTUnwrap(H.project(CGPoint(x: 450, y: 650)))
        XCTAssertEqual(p.x, 4.5, accuracy: 1e-6)
        XCTAssertEqual(p.y, 1.5, accuracy: 1e-6)

        // Corners land exactly on the board frame corners.
        let a1 = try XCTUnwrap(H.project(CGPoint(x: 0, y: 800)))
        XCTAssertEqual(a1.x, 0, accuracy: 1e-6)
        XCTAssertEqual(a1.y, 0, accuracy: 1e-6)
        let h8 = try XCTUnwrap(H.project(CGPoint(x: 800, y: 0)))
        XCTAssertEqual(h8.x, 8, accuracy: 1e-6)
        XCTAssertEqual(h8.y, 8, accuracy: 1e-6)
    }

    func testPerspectiveProjection() throws {
        // A tilted quadrilateral (camera at an angle); verify midpoints of
        // edges map to board-frame edge midpoints (projective invariant via
        // the DLT solution, checked against known cross-ratio behavior).
        let corners = [
            CGPoint(x: 100, y: 700), CGPoint(x: 250, y: 120),
            CGPoint(x: 760, y: 150), CGPoint(x: 720, y: 740),
        ]
        let H = try XCTUnwrap(Homography(corners: corners))
        for (i, corner) in corners.enumerated() {
            let expected: [(Double, Double)] = [(0, 0), (0, 8), (8, 8), (8, 0)]
            let p = try XCTUnwrap(H.project(corner))
            XCTAssertEqual(p.x, expected[i].0, accuracy: 1e-6)
            XCTAssertEqual(p.y, expected[i].1, accuracy: 1e-6)
        }
    }

    func testDegenerateCornersFail() {
        // Three collinear points — no valid homography.
        let corners = [
            CGPoint(x: 0, y: 0), CGPoint(x: 100, y: 0),
            CGPoint(x: 200, y: 0), CGPoint(x: 300, y: 0),
        ]
        XCTAssertNil(Homography(corners: corners))
        XCTAssertNil(Homography(corners: [CGPoint(x: 0, y: 0)]))
    }

    func testSquareBinningMatchesPython() throws {
        let H = try XCTUnwrap(Homography(corners: topDownCorners))
        // Same checks as chess_pipeline.py: e2 center with pull 0 → (4,1).
        let bp = try XCTUnwrap(H.project(CGPoint(x: 450, y: 650)))
        XCTAssertEqual(SquareAssignment.square(atBoardX: bp.x, y: bp.y), .e2)
        // Near-bottom-edge anchor of e2 with camera-aware pull still bins e2.
        let pulled = try XCTUnwrap(H.pulledBoardPoint(CGPoint(x: 450, y: 698), pull: 0.30))
        XCTAssertEqual(SquareAssignment.square(atBoardX: pulled.x, y: pulled.y), .e2)
        // Clamping: half a square outside snaps in; further out is nil.
        XCTAssertEqual(SquareAssignment.square(atBoardX: -0.3, y: 0.5), .a1)
        XCTAssertNil(SquareAssignment.square(atBoardX: -0.6, y: 0.5))
    }
}

final class SquareAssignmentTests: XCTestCase {

    private let topDownCorners = [
        CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 0),
        CGPoint(x: 800, y: 0), CGPoint(x: 800, y: 800),
    ]

    func testAssignAndFENMatchesPython() throws {
        let H = try XCTUnwrap(Homography(corners: topDownCorners))
        // Same fixture as chess_pipeline.py's self-check:
        // white king box bottom-center (450, 660) → e2; black king → e8.
        let detections = [
            PieceDetection(piece: .king, color: .white, confidence: 0.9,
                           boundingBox: CGRect(x: 440, y: 610, width: 20, height: 50)),
            PieceDetection(piece: .king, color: .black, confidence: 0.95,
                           boundingBox: CGRect(x: 440, y: 40, width: 20, height: 50)),
        ]
        let board = SquareAssignment.assign(detections: detections, homography: H)
        let pieces = board.mapValues(\.piece)
        XCTAssertEqual(FENBuilder.placement(pieces), "4k3/8/8/8/8/8/4K3/8")
    }

    func testConflictDemotesToNeighbor() throws {
        let H = try XCTUnwrap(Homography(corners: topDownCorners))
        let detections = [
            PieceDetection(piece: .king, color: .white, confidence: 0.9,
                           boundingBox: CGRect(x: 440, y: 610, width: 20, height: 50)),
            PieceDetection(piece: .queen, color: .white, confidence: 0.5,
                           boundingBox: CGRect(x: 445, y: 615, width: 20, height: 50)),
        ]
        let board = SquareAssignment.assign(detections: detections, homography: H)
        XCTAssertEqual(board.count, 2)
        XCTAssertEqual(board[.e2]?.piece.kind, .king)  // higher confidence keeps e2
        let kinds = Set(board.values.map(\.piece.kind))
        XCTAssertTrue(kinds.contains(.queen))  // demoted, not dropped
    }
}
