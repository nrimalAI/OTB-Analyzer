import ChessKit
import CoreGraphics
import Foundation
import Testing

@testable import ChessVision

/// `BoardOrientation` against `Training/chess_pipeline.py`
/// (`white_side_score`, `orient_by_pieces`).
struct BoardOrientationTests {

    private static func board(_ placement: String) throws -> [Square: RecognizedPiece] {
        try #require(FENBuilder.pieces(fromPlacement: placement))
            .mapValues { RecognizedPiece(piece: $0, confidence: 1) }
    }

    /// The scores Python gives these positions.
    @Test func testWhiteSideScoreMatchesPython() throws {
        let start = try Self.board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        #expect(BoardOrientation.whiteSideScore(start) == 179.5)
        // The same board seen from Black's side scores the opposite.
        let turned = try Self.board("RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr")
        #expect(BoardOrientation.whiteSideScore(turned) == -179.5)
        // An endgame whose pieces point the wrong way only weakly.
        #expect(BoardOrientation.whiteSideScore(try Self.board("7Q/8/1k3p2/4p3/8/p3K3/8/8")) == 2.5)
        // Pawns about to promote count against their own side.
        #expect(BoardOrientation.whiteSideScore(try Self.board("8/P7/8/8/8/8/7p/k6K")) == -15)
        // A pawn on the first or last rank is impossible: 20 off for each.
        #expect(BoardOrientation.whiteSideScore(try Self.board("P7/8/8/8/8/8/8/7p")) == -61)
        #expect(BoardOrientation.whiteSideScore([:]) == 0)
    }

    /// A top-down 800x800 board; a detection whose box stands on the middle
    /// of a square.
    private static let topDown = [
        CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 0),
        CGPoint(x: 800, y: 0), CGPoint(x: 800, y: 800),
    ]

    private static func detections(_ placement: String) throws -> [PieceDetection] {
        try #require(FENBuilder.pieces(fromPlacement: placement)).map { square, piece in
            let x = (Double(square.file.number) - 0.5) * 100
            let y = 800 - (Double(square.rank.value) - 0.5) * 100
            return PieceDetection(
                piece: piece.kind, color: piece.color, confidence: 0.9,
                boundingBox: CGRect(x: x - 20, y: y - 60, width: 40, height: 70))
        }
    }

    private static func turned(_ corners: [CGPoint], by turn: Int) -> [CGPoint] {
        Array(corners[turn...] + corners[..<turn])
    }

    @Test func testPiecesOverruleAWrongA1() throws {
        let pieces = try Self.detections("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        for turn in 0..<4 {
            // Whichever corner the model calls a1, the start position puts it right.
            let oriented = BoardOrientation.orient(
                corners: Self.turned(Self.topDown, by: turn), detections: pieces)
            #expect(oriented == Self.topDown, "corners turned by \(turn)")
        }
    }

    @Test func testPriorKeepsTheModelsA1WhenThePiecesSayLittle() throws {
        // +2.5 the right way up, -2.5 turned by 180 degrees.
        let pieces = try Self.detections("7Q/8/1k3p2/4p3/8/p3K3/8/8")
        let upsideDown = Self.turned(Self.topDown, by: 2)
        #expect(BoardOrientation.appPrior == 6)
        #expect(BoardOrientation.orient(corners: upsideDown, detections: pieces, prior: 6) == upsideDown)
        #expect(BoardOrientation.orient(corners: upsideDown, detections: pieces, prior: 0) == Self.topDown)
        // Without pieces there is nothing to go on: the corners stay.
        #expect(BoardOrientation.orient(corners: upsideDown, detections: [], prior: 0) == upsideDown)
    }

    @Test func testCornersThatAreNotABoardComeBackUnchanged() {
        let line = [
            CGPoint(x: 0, y: 0), CGPoint(x: 100, y: 0),
            CGPoint(x: 200, y: 0), CGPoint(x: 300, y: 0),
        ]
        #expect(BoardOrientation.orient(corners: line, detections: []) == line)
        #expect(BoardOrientation.orient(corners: [.zero], detections: []) == [.zero])
    }

    /// `turns`: only the rotations named are looked at, and the head start
    /// still goes to the corners as given when they are among them.
    @Test func testOnlyTheRotationsNamedAreLookedAt() throws {
        let pieces = try Self.detections("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        let quarter = Self.turned(Self.topDown, by: 1)
        // The pieces want rotation 3 of `quarter`; it is not on offer.
        #expect(BoardOrientation.orient(corners: quarter, detections: pieces) == Self.topDown)
        let kept = BoardOrientation.orient(corners: quarter, detections: pieces, turns: [0, 2])
        #expect(kept == quarter || kept == Self.turned(quarter, by: 2))
        #expect(BoardOrientation.orient(corners: quarter, detections: pieces, turns: [1, 3]) == Self.topDown)
        // All four named is no restriction; none named leaves the corners.
        #expect(BoardOrientation.orient(
            corners: quarter, detections: pieces, turns: [0, 1, 2, 3]) == Self.topDown)
        #expect(BoardOrientation.orient(corners: quarter, detections: pieces, turns: []) == quarter)

        // The head start: +2.5 the right way up, -2.5 turned by half.
        let few = try Self.detections("7Q/8/1k3p2/4p3/8/p3K3/8/8")
        let upsideDown = Self.turned(Self.topDown, by: 2)
        #expect(BoardOrientation.orient(
            corners: upsideDown, detections: few, prior: 6, turns: [0, 2]) == upsideDown)
        #expect(BoardOrientation.orient(
            corners: upsideDown, detections: few, prior: 0, turns: [0, 2]) == Self.topDown)
        // It is the corners as given that get it, not the first rotation named.
        let quarterOff = Self.turned(Self.topDown, by: 3)   // rotation 1 of it is the right way up
        #expect(BoardOrientation.orient(
            corners: quarterOff, detections: few, prior: 6, turns: [1, 3]) == Self.topDown)
        #expect(BoardOrientation.orient(
            corners: Self.turned(Self.topDown, by: 1), detections: few, prior: 6, turns: [1, 3])
            == Self.topDown)
    }

    // MARK: - Cases written by Training/swift_reference.py

    private struct Case: Decodable {
        struct Box: Decodable {
            let cls: String
            let confidence: Float
            let box: [Double]
        }
        let corners: [[Double]]
        let pull: Double
        let prior: Double
        let detections: [Box]
        let scores: [Double]
        let turn: Int
        let placement: String
    }

    /// Random boards in perspective, corners given in a random rotation, some
    /// pieces missed, misread or found twice, some boxes beside the board
    /// (so that pieces are moved to a neighbouring square, dropped, or pulled
    /// in from the edge): the rotation, the four scores and the final
    /// placement must be the ones `chess_pipeline.py` computed.
    @Test func testMatchesPythonOnRecordedCases() throws {
        let url = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
            .appending(path: "orientation_cases.json")
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: url))
        #expect(cases.count >= 100)
        struct Allowed: Decodable { let orientation: [String: [Int]] }
        let allowed = try JSONDecoder().decode(Allowed.self, from: Data(contentsOf: url
            .deletingLastPathComponent().appending(path: "dark_cases.json"))).orientation
        #expect(allowed["0,2"]?.count == cases.count && allowed["1,3"]?.count == cases.count)

        for (index, recorded) in cases.enumerated() {
            let corners = recorded.corners.map { CGPoint(x: $0[0], y: $0[1]) }
            let detections = try recorded.detections.map { box in
                let (kind, color) = try #require(CoreMLBoardRecognizer.pieceClass(fromLabel: box.cls))
                return PieceDetection(
                    piece: kind, color: color, confidence: box.confidence,
                    boundingBox: CGRect(
                        x: box.box[0], y: box.box[1],
                        width: box.box[2] - box.box[0], height: box.box[3] - box.box[1]))
            }

            for turn in 0..<4 {
                let homography = try #require(Homography(corners: Self.turned(corners, by: turn)))
                let score = BoardOrientation.whiteSideScore(SquareAssignment.assign(
                    detections: detections, homography: homography, pull: recorded.pull))
                    + (turn == 0 ? recorded.prior : 0)
                #expect(score == recorded.scores[turn], "case \(index), turn \(turn)")
            }

            let oriented = BoardOrientation.orient(
                corners: corners, detections: detections, pull: recorded.pull, prior: recorded.prior)
            #expect(oriented == Self.turned(corners, by: recorded.turn), "case \(index)")
            // With only two of the four rotations allowed (a1 is a dark
            // square): the one `orient_by_pieces(turns=)` chose.
            for (key, turns) in [("0,2", [0, 2]), ("1,3", [1, 3])] {
                let chosen = try #require(allowed[key])[index]
                #expect(turns.contains(chosen))
                #expect(BoardOrientation.orient(
                    corners: corners, detections: detections, pull: recorded.pull,
                    prior: recorded.prior, turns: turns) == Self.turned(corners, by: chosen),
                    "case \(index), turns \(key)")
            }
            let board = SquareAssignment.assign(
                detections: detections,
                homography: try #require(Homography(corners: oriented)),
                pull: recorded.pull)
            #expect(
                FENBuilder.placement(board.mapValues(\.piece)) == recorded.placement,
                "case \(index)")
        }
    }
}
