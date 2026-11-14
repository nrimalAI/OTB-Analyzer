import ChessKit
import CoreGraphics
import Foundation
import Testing

@testable import ChessVision

/// `OneKingASide` against `chess_pipeline.one_king_a_side`. The reference
/// answers are in `Fixtures/king_cases.json`, written by
/// `Training/swift_reference.py king-cases`; run that again after changing
/// the Python, and a change in the JSON is a change the Swift must follow.
///
/// These cases are the whole contract. On photos the two cannot be compared
/// piece for piece: the Python makes up the scores of a piece from boxes that
/// overlap it (`evaluate.kind_scores_of`), the app has the model's scores for
/// the box itself (`CoreMLBoardRecognizer.kindScores`).
struct OneKingASideTests {

    // MARK: - The recorded cases

    struct Recorded: Decodable {
        struct Detection: Decodable {
            let cls: String
            let confidence: Float
            let scores: [String: Double]
        }
        let detections: [Detection]
        let out: [String]
    }

    static func kind(named name: String) -> Piece.Kind? {
        CoreMLBoardRecognizer.pieceClass(fromLabel: "white_" + name)?.0
    }

    static func detection(
        _ cls: String, confidence: Float = 0.9, index: Int = 0, scores: [String: Double] = [:]
    ) throws -> PieceDetection {
        let (kind, color) = try #require(CoreMLBoardRecognizer.pieceClass(fromLabel: cls))
        var kindScores: [Piece.Kind: Double] = [:]
        for (name, score) in scores {
            kindScores[try #require(Self.kind(named: name))] = score
        }
        return PieceDetection(
            piece: kind, color: color, confidence: confidence,
            boundingBox: CGRect(x: 60 * Double(index), y: 40, width: 50, height: 110),
            kindScores: kindScores)
    }

    static func labels(_ detections: [PieceDetection]) -> [String] {
        detections.map { PieceSides.label($0) }
    }

    @Test func testOneKingASideIsPythons() throws {
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let cases = try JSONDecoder().decode(
            [Recorded].self, from: Data(contentsOf: fixtures.appending(path: "king_cases.json")))
        #expect(cases.count >= 250)

        var given = 0, taken = 0, alone = 0
        for (number, recorded) in cases.enumerated() {
            let detections = try recorded.detections.enumerated().map { index, d in
                try Self.detection(d.cls, confidence: d.confidence, index: index, scores: d.scores)
            }
            let result = OneKingASide.apply(detections)
            #expect(Self.labels(result) == recorded.out, "case \(number)")
            // A piece keeps its side, confidence, box and scores.
            #expect(result.count == detections.count)
            for (before, after) in zip(detections, result) {
                #expect(after.color == before.color, "case \(number)")
                #expect(after.confidence == before.confidence, "case \(number)")
                #expect(after.boundingBox == before.boundingBox, "case \(number)")
                #expect(after.kindScores == before.kindScores, "case \(number)")
                // Shown with the score of the kind it became, if it changed.
                #expect(after.shownConfidence == (after.piece == before.piece
                    ? nil : Float(before.kindScores[after.piece] ?? 0)), "case \(number)")
            }
            let kings = detections.filter { $0.piece == .king }.count
            let changed = Self.labels(detections) != recorded.out
            if kings == 2 { alone += 1 } else if changed && kings < 2 { given += 1 } else if changed { taken += 1 }
        }
        // The cases cover each branch of the rule.
        #expect(given >= 50)
        #expect(taken >= 50)
        #expect(alone >= 20)
    }

    // MARK: - By hand

    private func apply(_ pieces: [(String, [String: Double])]) throws -> [String] {
        let detections = try pieces.enumerated().map { index, piece in
            try Self.detection(piece.0, index: index, scores: piece.1)
        }
        return Self.labels(OneKingASide.apply(detections))
    }

    @Test func testTwoKingsAreLeftAlone() throws {
        #expect(try apply([
            ("white_king", ["king": 0.9]), ("black_king", ["king": 0.9]), ("black_queen", ["king": 0.8]),
        ]) == ["white_king", "black_king", "black_queen"])
        // Both on one side: a wrong side, not a wrong kind.
        #expect(try apply([
            ("white_king", ["king": 0.9]), ("white_king", ["king": 0.2, "queen": 0.6]),
            ("black_queen", ["king": 0.5]),
        ]) == ["white_king", "white_king", "black_queen"])
        #expect(try apply([]) == [])
    }

    @Test func testASideWithoutAKingGetsItsBest() throws {
        // One king on the board: the other side's best becomes one.
        #expect(try apply([
            ("white_king", ["king": 0.9]), ("white_queen", ["king": 0.5]),
            ("black_queen", ["king": 0.05]), ("black_rook", ["king": 0.3]),
        ]) == ["white_king", "white_queen", "black_queen", "black_king"])
        // None: each side its own; a side whose best is under the floor none.
        #expect(try apply([
            ("white_queen", ["king": 0.1]), ("white_rook", ["king": 0.2]),
            ("black_pawn", ["king": 0.019]), ("black_queen", [:]),
        ]) == ["white_queen", "white_king", "black_pawn", "black_queen"])
        // Equal scores: the earlier piece.
        #expect(try apply([
            ("black_rook", ["king": 0.4]), ("black_queen", ["king": 0.4]), ("black_pawn", ["king": 0.39]),
        ]) == ["black_king", "black_queen", "black_pawn"])
    }

    @Test func testTheFloor() throws {
        #expect(OneKingASide.kingFloor == 0.02)
        #expect(try apply([("white_queen", ["king": 0.02])]) == ["white_king"])
        #expect(try apply([("white_queen", ["king": 0.02 - 1e-9])]) == ["white_queen"])
        let queen = try Self.detection("white_queen", scores: ["king": 0.3])
        #expect(Self.labels(OneKingASide.apply([queen], floor: 0.31)) == ["white_queen"])
    }

    @Test func testASideWithSeveralKingsKeepsItsBest() throws {
        // By the king score, not by the confidence of the box or its place
        // in the list; a side with one king is left alone.
        #expect(try apply([
            ("white_king", ["king": 0.5, "queen": 0.3]), ("white_king", ["king": 0.7]),
            ("white_king", ["king": 0.6, "rook": 0.2, "queen": 0.1]), ("black_king", ["king": 0.3]),
        ]) == ["white_queen", "white_king", "white_rook", "black_king"])
        // Equal king scores: the earlier one stays.
        #expect(try apply([
            ("white_king", ["king": 0.6, "bishop": 0.1]), ("white_king", ["king": 0.6, "knight": 0.2]),
            ("black_king", ["king": 0.9]),
        ]) == ["white_king", "white_knight", "black_king"])
        // The other kind: the best score; equal scores in the order queen,
        // rook, bishop, knight, pawn; no other score, a queen.
        #expect(try apply([
            ("black_king", ["king": 0.9]), ("black_king", ["king": 0.5, "pawn": 0.2, "knight": 0.2]),
            ("black_king", ["king": 0.4, "rook": 0.3, "queen": 0.3]), ("black_king", ["king": 0.3]),
            ("black_king", ["king": 0.2, "pawn": 0.01]),
        ]) == ["black_king", "black_knight", "black_queen", "black_queen", "black_pawn"])
        // Three kings on one side and none on the other: the other side gets
        // none, the board has more than two kings and not fewer.
        #expect(try apply([
            ("white_king", ["king": 0.9]), ("white_king", ["king": 0.8, "queen": 0.1]),
            ("white_king", ["king": 0.7]), ("black_queen", ["king": 0.6]),
        ]) == ["white_king", "white_queen", "white_queen", "black_queen"])
    }

    @Test func testAScoreThatIsNotANumberCountsAsNone() throws {
        let pieces = [
            PieceDetection(
                piece: .queen, color: .white, confidence: 0.9, boundingBox: .zero,
                kindScores: [.king: .nan]),
            PieceDetection(
                piece: .rook, color: .white, confidence: 0.9, boundingBox: .zero,
                kindScores: [.king: 0.1]),
            PieceDetection(
                piece: .rook, color: .black, confidence: 0.9, boundingBox: .zero,
                kindScores: [.king: .infinity]),
        ]
        #expect(Self.labels(OneKingASide.apply(pieces)) == ["white_queen", "white_king", "black_rook"])
    }

    // MARK: - The scores of a box

    /// Vision divides the class scores of a box by their sum and hands out
    /// the sum as the observation's confidence.
    @Test func testClassScoresAreTheLabelsTimesTheirSum() {
        let scores = CoreMLBoardRecognizer.classScores(
            labels: [("white_queen", 0.6), ("white_king", 0.375), ("black_king", 0.025)], sum: 0.8)
        #expect(scores.map(\.identifier) == ["white_queen", "white_king", "black_king"])
        for (have, want) in zip(scores.map(\.score), [0.48, 0.3, 0.02] as [Float]) {
            #expect(abs(have - want) < 1e-6)
        }
        // A sum over 1 is not cut down (two classes the model scored high).
        let over = CoreMLBoardRecognizer.classScores(
            labels: [("white_queen", 0.56), ("white_king", 0.44)], sum: 1.25)
        #expect(abs(over[0].score - 0.7) < 1e-6)
        #expect(abs(over[1].score - 0.55) < 1e-6)
    }

    /// Labels that do not add up to 1 are not what Vision is known to hand
    /// out: they are taken for the scores themselves.
    @Test func testLabelsThatDoNotAddUpToOneAreLeftAsTheyAre() {
        let scores = CoreMLBoardRecognizer.classScores(
            labels: [("white_queen", 0.48), ("white_king", 0.3)], sum: 0.8)
        #expect(scores.map(\.score) == [0.48, 0.3])
        #expect(CoreMLBoardRecognizer.classScores(labels: [], sum: 0.8).isEmpty)
    }

    // MARK: - What the user is shown

    /// A piece the rule took for another kind is shown with the model's
    /// score for that kind, so that the app marks it; which of two pieces on
    /// a square stays still goes by the model's score for its first choice.
    @Test func testAPieceTheRuleChangedIsShownWithTheScoreOfWhatItBecame() throws {
        let corners = Self.rightWayUp
        let homography = try #require(Homography(corners: corners))
        let queen = PieceDetection(
            piece: .queen, color: .white, confidence: 0.93, boundingBox: Self.box(file: 3, rank: 1),
            kindScores: [.queen: 0.93, .king: 0.05])
        // A weaker box on the same square: it must still lose to the queen
        // that became a king, and is moved to a square beside it.
        let rook = PieceDetection(
            piece: .rook, color: .white, confidence: 0.5, boundingBox: Self.box(file: 3, rank: 1))
        let blackKing = PieceDetection(
            piece: .king, color: .black, confidence: 0.9, boundingBox: Self.box(file: 4, rank: 8),
            kindScores: [.king: 0.9])
        let result = OneKingASide.apply([rook, queen, blackKing])
        #expect(Self.labels(result) == ["white_rook", "white_king", "black_king"])
        #expect(result[1].confidence == 0.93)
        #expect(result[1].shownConfidence == 0.05)
        #expect(result[0].shownConfidence == nil)
        #expect(result[2].shownConfidence == nil)

        let board = SquareAssignment.assign(detections: result, homography: homography)
        let d1 = try #require(FENBuilder.square(file: 4, rank: 1))
        let e8 = try #require(FENBuilder.square(file: 5, rank: 8))
        #expect(board[d1]?.piece.kind == .king)
        #expect(board[d1]?.confidence == 0.05)   // under the app's 0.85: marked
        #expect(board[e8]?.confidence == 0.9)
        #expect(board.count == 3)

        // A king taken away with no other score becomes a queen shown at 0.
        let kings = (0..<3).map {
            PieceDetection(
                piece: .king, color: .white, confidence: 0.9, boundingBox: Self.box(file: $0, rank: 1),
                kindScores: [.king: 0.9 - 0.1 * Double($0)])
        }
        let trimmed = OneKingASide.apply(kings)
        #expect(Self.labels(trimmed) == ["white_king", "white_queen", "white_queen"])
        #expect(trimmed.map(\.shownConfidence) == [nil, 0, 0])
        // Taken for another side only, a piece is shown as before.
        #expect(queen.taken(color: .black).shownConfidence == nil)
    }

    @Test func testKindScoresAreTheLargerOfTheTwoColours() {
        let scores = CoreMLBoardRecognizer.kindScores([
            ("white_queen", 0.48), ("white_king", 0.3), ("black_king", 0.31), ("black_queen", 0.1),
            ("white_rook", 0.0199), ("black_rook", 0.001),  // next to nothing: left out
            ("white_pawn", 0.0201),                          // just over: kept
            ("78", 0.9), ("white", 0.9), ("grey_king", 0.9), ("white_bishop", .nan),
        ])
        #expect(Set(scores.keys) == [.queen, .king, .pawn])
        #expect(abs(scores[.queen]! - 0.48) < 1e-6)
        #expect(abs(scores[.king]! - 0.31) < 1e-6)
        #expect(abs(scores[.pawn]! - 0.0201) < 1e-6)
        #expect(CoreMLBoardRecognizer.leastKindScore == OneKingASide.kingFloor)
    }

    // MARK: - In the recognizer

    private static func box(file: Int, rank: Int) -> CGRect {
        CGRect(x: Double(file) * 100 + 20, y: 800 - Double(rank) * 100 + 20, width: 60, height: 60)
    }
    private static let rightWayUp = [
        CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 0), CGPoint(x: 800, y: 0), CGPoint(x: 800, y: 800),
    ]

    /// The kings are counted before a1 is read from the pieces, where a king
    /// weighs one and a half times a queen. White: a "queen" on d1 that the
    /// model also scored as a king, and a rook on e1. Black: rooks on a1, h1,
    /// b4 and g4. As a queen and a rook White's two weigh 7 against Black's
    /// 8, and the board is turned round; with the king they weigh 8.75.
    @Test func testTheKingsAreCountedBeforeA1IsRead() throws {
        let image = try PieceSidesTests.drawnPawns().image
        let found = [
            PieceDetection(
                piece: .queen, color: .white, confidence: 0.9, boundingBox: Self.box(file: 3, rank: 1),
                kindScores: [.queen: 0.6, .king: 0.5]),
            PieceDetection(
                piece: .rook, color: .white, confidence: 0.9, boundingBox: Self.box(file: 4, rank: 1)),
        ] + [(0, 1), (7, 1), (1, 4), (6, 4)].map {
            PieceDetection(
                piece: .rook, color: .black, confidence: 0.9, boundingBox: Self.box(file: $0.0, rank: $0.1))
        }
        func place(oneKing: Bool) throws -> (corners: [CGPoint], placement: String) {
            let placed = try #require(CoreMLBoardRecognizer.place(
                found, corners: Self.rightWayUp, cornersFromModel: false, in: image,
                sidesFromColour: false, oneKing: oneKing, a1Prior: 6, pull: SquareAssignment.defaultPull))
            return (placed.corners, FENBuilder.placement(placed.pieces.mapValues(\.piece)))
        }

        let with = try place(oneKing: true)
        #expect(with.corners == Self.rightWayUp)
        #expect(with.placement == "8/8/8/8/1r4r1/8/8/r2KR2r")
        let without = try place(oneKing: false)
        #expect(without.corners == Array(Self.rightWayUp[2...] + Self.rightWayUp[..<2]))
        #expect(without.placement == "r2RQ2r/8/8/1r4r1/8/8/8/8")
    }

    /// The sides come first: the kings are counted by side. A piece model
    /// that calls every piece black, eight cream pieces on rank 2 and eight
    /// dark ones on rank 7, one of each with a score for "king". With the
    /// sides from the colours each side gets its king; counted before, all
    /// sixteen are Black's and only the best of them becomes one.
    @Test func testTheKingsAreCountedAfterTheSidesFromColour() throws {
        let (image, boxes) = try PieceSidesTests.drawnPawns()
        let found = boxes.enumerated().map { index, box in
            PieceDetection(
                piece: .pawn, color: .black, confidence: 0.9, boundingBox: box,
                kindScores: index == 3 ? [.king: 0.5] : index == 12 ? [.king: 0.9] : [:])
        }
        func placement(sidesFromColour: Bool, oneKing: Bool) throws -> String {
            let placed = try #require(CoreMLBoardRecognizer.place(
                found, corners: Self.rightWayUp, cornersFromModel: true, in: image,
                sidesFromColour: sidesFromColour, oneKing: oneKing, a1Prior: nil,  // a1 as given
                pull: SquareAssignment.defaultPull))
            return FENBuilder.placement(placed.pieces.mapValues(\.piece))
        }
        #expect(try placement(sidesFromColour: true, oneKing: true) == "8/ppppkppp/8/8/8/8/PPPKPPPP/8")
        #expect(try placement(sidesFromColour: true, oneKing: false) == "8/pppppppp/8/8/8/8/PPPPPPPP/8")
        #expect(try placement(sidesFromColour: false, oneKing: true) == "8/ppppkppp/8/8/8/8/pppppppp/8")
    }
}
