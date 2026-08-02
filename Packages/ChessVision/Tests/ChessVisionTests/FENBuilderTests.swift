import ChessKit
import XCTest

@testable import ChessVision

final class FENBuilderTests: XCTestCase {

    static let startPlacement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    func testSquareFromFileRank() {
        XCTAssertEqual(FENBuilder.square(file: 1, rank: 1), .a1)
        XCTAssertEqual(FENBuilder.square(file: 8, rank: 1), .h1)
        XCTAssertEqual(FENBuilder.square(file: 1, rank: 8), .a8)
        XCTAssertEqual(FENBuilder.square(file: 8, rank: 8), .h8)
        XCTAssertEqual(FENBuilder.square(file: 5, rank: 4), .e4)
        XCTAssertNil(FENBuilder.square(file: 0, rank: 4))
        XCTAssertNil(FENBuilder.square(file: 9, rank: 4))
    }

    func testPlacementRoundTrip() {
        let placements = [
            Self.startPlacement,
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",  // Italian
            "8/8/8/8/8/8/8/K6k",  // bare kings
            "4k3/8/8/8/3Q4/8/8/4K3",  // queen vs king
        ]
        for placement in placements {
            let pieces = FENBuilder.pieces(fromPlacement: placement)
            XCTAssertNotNil(pieces, placement)
            XCTAssertEqual(FENBuilder.placement(pieces!), placement)
        }
    }

    func testParseRejectsMalformed() {
        XCTAssertNil(FENBuilder.pieces(fromPlacement: "8/8/8/8/8/8/8"))        // 7 ranks
        XCTAssertNil(FENBuilder.pieces(fromPlacement: "9/8/8/8/8/8/8/8"))      // bad digit
        XCTAssertNil(FENBuilder.pieces(fromPlacement: "x7/8/8/8/8/8/8/8"))     // bad letter
        XCTAssertNil(FENBuilder.pieces(fromPlacement: "pp7/8/8/8/8/8/8/8"))    // overlong rank
    }

    func testFullFENAssembly() {
        let pieces = FENBuilder.pieces(fromPlacement: Self.startPlacement)!
        let fen = FENBuilder.fen(
            pieces: pieces,
            sideToMove: .white,
            castling: .inferred(from: pieces)
        )
        XCTAssertEqual(fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        // ChessKit accepts the assembled FEN and round-trips it.
        XCTAssertEqual(Position(fen: fen)?.fen, fen)
    }

    func testStartPositionAgainstChessKit() {
        let ours = FENBuilder.pieces(fromFEN: Position.standard.fen)!
        XCTAssertEqual(Set(ours.values), Set(Position.standard.pieces))
    }
}

final class CastlingRightsTests: XCTestCase {

    func testInferredFromStartPosition() {
        let pieces = FENBuilder.pieces(fromPlacement: FENBuilderTests.startPlacement)!
        let rights = CastlingRights.inferred(from: pieces)
        XCTAssertEqual(rights.fenField, "KQkq")
    }

    func testKingOffHomeSquareKillsBothRights() {
        var pieces = FENBuilder.pieces(fromPlacement: FENBuilderTests.startPlacement)!
        pieces[.e1] = nil
        pieces[.d1] = Piece(.king, color: .white, square: .d1)
        let rights = CastlingRights.inferred(from: pieces)
        XCTAssertEqual(rights.fenField, "kq")
    }

    func testMissingRookKillsOneSide() {
        var pieces = FENBuilder.pieces(fromPlacement: FENBuilderTests.startPlacement)!
        pieces[.h1] = nil
        let rights = CastlingRights.inferred(from: pieces)
        XCTAssertEqual(rights.fenField, "Qkq")
    }

    func testNoRightsRendersDash() {
        XCTAssertEqual(CastlingRights.none.fenField, "-")
    }
}

final class PositionValidatorTests: XCTestCase {

    private func pieces(_ placement: String) -> [Square: Piece] {
        FENBuilder.pieces(fromPlacement: placement)!
    }

    func testStartPositionIsValid() {
        let issues = PositionValidator.validate(
            pieces: pieces(FENBuilderTests.startPlacement), sideToMove: .white)
        XCTAssertTrue(issues.isEmpty)
    }

    func testMissingKing() {
        let issues = PositionValidator.validate(
            pieces: pieces("8/8/8/8/8/8/8/K7"), sideToMove: .white)
        XCTAssertTrue(issues.contains(.missingKing(.black)))
    }

    func testMultipleKings() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/8/8/8/KK6"), sideToMove: .white)
        XCTAssertTrue(issues.contains(.multipleKings(.white)))
    }

    func testPawnOnBackRank() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/8/8/8/P3K3"), sideToMove: .white)
        XCTAssertTrue(issues.contains(.pawnOnBackRank(.a1)))
    }

    func testNinePawns() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/8/1PPP4/PPPPPP2/4K3"), sideToMove: .white)
        XCTAssertTrue(issues.contains(.tooManyPawns(.white)))
    }

    func testSideNotToMoveInCheck() {
        // White queen on e-file gives check to black king on e8, but it's
        // white to move — meaning black "just moved" while in check. Invalid.
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/4Q3/8/8/4K3"), sideToMove: .white)
        XCTAssertTrue(issues.contains(.sideNotToMoveInCheck))

        // Same position with black to move is a perfectly normal check.
        let ok = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/4Q3/8/8/4K3"), sideToMove: .black)
        XCTAssertFalse(ok.contains(.sideNotToMoveInCheck))
    }

    func testBlockedSliderGivesNoCheck() {
        // Queen's path to the king is blocked by its own pawn.
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/4P3/4Q3/8/8/4K3"), sideToMove: .white)
        XCTAssertFalse(issues.contains(.sideNotToMoveInCheck))
    }

    func testKnightCheckDetected() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/3N4/8/8/8/8/4K3"), sideToMove: .white)
        XCTAssertTrue(issues.contains(.sideNotToMoveInCheck))
    }

    func testPawnCheckDetected() {
        // White pawn on d7 attacks e8.
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/3P4/8/8/8/8/8/4K3"), sideToMove: .white)
        XCTAssertTrue(issues.contains(.sideNotToMoveInCheck))
    }

    func testHighlightSquares() {
        let map = pieces("4k3/8/8/8/8/8/8/P3K3")
        let issues = PositionValidator.validate(pieces: map, sideToMove: .white)
        let highlights = PositionValidator.highlightSquares(for: issues, pieces: map)
        XCTAssertTrue(highlights.contains(.a1))
    }
}

final class StubRecognizerTests: XCTestCase {
    func testStubReturnsStartPosition() async throws {
        let context = CGContext(
            data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let image = context.makeImage()!

        let result = try await StubRecognizer().recognize(image)
        XCTAssertEqual(result.pieces.count, 32)
        XCTAssertEqual(result.pieces[.e1]?.piece.kind, .king)
        XCTAssertEqual(result.pieces[.e1]?.piece.color, .white)
        XCTAssertEqual(result.pieces[.d8]?.piece.kind, .queen)
        XCTAssertEqual(result.pieces[.d8]?.piece.color, .black)
    }
}
