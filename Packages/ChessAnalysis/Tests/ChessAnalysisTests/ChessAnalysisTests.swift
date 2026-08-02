import XCTest

@testable import ChessAnalysis

final class ScoreTests: XCTestCase {

    func testCentipawnsNormalizedToWhitePerspective() {
        // Engine reports +50 with white to move → White is better.
        XCTAssertEqual(Score(cp: 50, mate: nil, whiteToMove: true), .centipawns(50))
        // Engine reports +50 with black to move → Black is better → -50 for White.
        XCTAssertEqual(Score(cp: 50, mate: nil, whiteToMove: false), .centipawns(-50))
    }

    func testMateNormalized() {
        XCTAssertEqual(Score(cp: nil, mate: 3, whiteToMove: true), .mate(3))
        XCTAssertEqual(Score(cp: nil, mate: 3, whiteToMove: false), .mate(-3))
        XCTAssertEqual(Score(cp: nil, mate: -2, whiteToMove: true), .mate(-2))
    }

    func testDisplayString() {
        XCTAssertEqual(Score.centipawns(35).displayString, "+0.35")
        XCTAssertEqual(Score.centipawns(-120).displayString, "-1.20")
        XCTAssertEqual(Score.centipawns(0).displayString, "+0.00")
        XCTAssertEqual(Score.mate(5).displayString, "#5")
        XCTAssertEqual(Score.mate(-3).displayString, "-#3")
    }

    func testWinningFraction() {
        XCTAssertEqual(Score.centipawns(0).whiteWinningFraction, 0.5, accuracy: 0.001)
        XCTAssertGreaterThan(Score.centipawns(300).whiteWinningFraction, 0.65)
        XCTAssertLessThan(Score.centipawns(-300).whiteWinningFraction, 0.35)
        XCTAssertEqual(Score.mate(2).whiteWinningFraction, 1.0)
        XCTAssertEqual(Score.mate(-2).whiteWinningFraction, 0.0)
    }
}

final class SANConverterTests: XCTestCase {

    static let startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    func testSimpleOpeningLine() {
        let sans = SANConverter.convert(
            fen: Self.startFEN,
            lanMoves: ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])
        XCTAssertEqual(sans, ["e4", "e5", "Nf3", "Nc6", "Bb5"])
    }

    func testCaptureNotation() {
        // 1. e4 d5 2. exd5
        let sans = SANConverter.convert(
            fen: Self.startFEN,
            lanMoves: ["e2e4", "d7d5", "e4d5"])
        XCTAssertEqual(sans, ["e4", "d5", "exd5"])
    }

    func testCastlingNotation() {
        // Italian setup where white castles kingside.
        let fen = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        let sans = SANConverter.convert(fen: fen, lanMoves: ["e1g1"])
        XCTAssertEqual(sans, ["O-O"])
    }

    func testInvalidMoveFallsBackToLAN() {
        let sans = SANConverter.convert(
            fen: Self.startFEN,
            lanMoves: ["e2e4", "zz99", "g1f3"])
        XCTAssertEqual(sans.first, "e4")
        // Remainder preserved as raw LAN.
        XCTAssertEqual(Array(sans.dropFirst()), ["zz99", "g1f3"])
    }

    func testInvalidFENPassesThroughLAN() {
        let sans = SANConverter.convert(fen: "not a fen", lanMoves: ["e2e4"])
        XCTAssertEqual(sans, ["e2e4"])
    }

    func testSquaresOfLAN() {
        let squares = SANConverter.squares(ofLAN: "e2e4")
        XCTAssertEqual(squares?.from, .e2)
        XCTAssertEqual(squares?.to, .e4)
        XCTAssertNil(SANConverter.squares(ofLAN: "e2"))
        XCTAssertNil(SANConverter.squares(ofLAN: "x9y0"))
    }

    func testPromotion() {
        let fen = "8/P6k/8/8/8/8/8/K7 w - - 0 1"
        let sans = SANConverter.convert(fen: fen, lanMoves: ["a7a8q"])
        XCTAssertEqual(sans, ["a8=Q"])
    }
}

final class FENFieldTests: XCTestCase {
    func testWhiteToMoveParsing() {
        XCTAssertTrue(AnalysisEngine.whiteToMove(fen: SANConverterTests.startFEN))
        XCTAssertFalse(
            AnalysisEngine.whiteToMove(
                fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))
        XCTAssertTrue(AnalysisEngine.whiteToMove(fen: "garbage"))
    }
}
