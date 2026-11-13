import ChessKit
import CoreGraphics
import Foundation
import Testing

@testable import ChessVision

struct FENBuilderTests {

    static let startPlacement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    @Test func testSquareFromFileRank() {
        #expect(FENBuilder.square(file: 1, rank: 1) == .a1)
        #expect(FENBuilder.square(file: 8, rank: 1) == .h1)
        #expect(FENBuilder.square(file: 1, rank: 8) == .a8)
        #expect(FENBuilder.square(file: 8, rank: 8) == .h8)
        #expect(FENBuilder.square(file: 5, rank: 4) == .e4)
        #expect(FENBuilder.square(file: 0, rank: 4) == nil)
        #expect(FENBuilder.square(file: 9, rank: 4) == nil)
    }

    @Test func testPlacementRoundTrip() {
        let placements = [
            Self.startPlacement,
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",  // Italian
            "8/8/8/8/8/8/8/K6k",  // bare kings
            "4k3/8/8/8/3Q4/8/8/4K3",  // queen vs king
        ]
        for placement in placements {
            let pieces = FENBuilder.pieces(fromPlacement: placement)
            #expect(pieces != nil, "\(placement)")
            #expect(FENBuilder.placement(pieces!) == placement)
        }
    }

    @Test func testParseRejectsMalformed() {
        #expect(FENBuilder.pieces(fromPlacement: "8/8/8/8/8/8/8") == nil)        // 7 ranks
        #expect(FENBuilder.pieces(fromPlacement: "9/8/8/8/8/8/8/8") == nil)      // bad digit
        #expect(FENBuilder.pieces(fromPlacement: "x7/8/8/8/8/8/8/8") == nil)     // bad letter
        #expect(FENBuilder.pieces(fromPlacement: "pp7/8/8/8/8/8/8/8") == nil)    // overlong rank
    }

    @Test func testFullFENAssembly() {
        let pieces = FENBuilder.pieces(fromPlacement: Self.startPlacement)!
        let fen = FENBuilder.fen(
            pieces: pieces,
            sideToMove: .white,
            castling: .inferred(from: pieces)
        )
        #expect(fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        // ChessKit accepts the assembled FEN and round-trips it.
        #expect(Position(fen: fen)?.fen == fen)
    }

    @Test func testStartPositionAgainstChessKit() {
        let ours = FENBuilder.pieces(fromFEN: Position.standard.fen)!
        #expect(Set(ours.values) == Set(Position.standard.pieces))
    }
}

struct CastlingRightsTests {

    @Test func testInferredFromStartPosition() {
        let pieces = FENBuilder.pieces(fromPlacement: FENBuilderTests.startPlacement)!
        let rights = CastlingRights.inferred(from: pieces)
        #expect(rights.fenField == "KQkq")
    }

    @Test func testKingOffHomeSquareKillsBothRights() {
        var pieces = FENBuilder.pieces(fromPlacement: FENBuilderTests.startPlacement)!
        pieces[.e1] = nil
        pieces[.d1] = Piece(.king, color: .white, square: .d1)
        let rights = CastlingRights.inferred(from: pieces)
        #expect(rights.fenField == "kq")
    }

    @Test func testMissingRookKillsOneSide() {
        var pieces = FENBuilder.pieces(fromPlacement: FENBuilderTests.startPlacement)!
        pieces[.h1] = nil
        let rights = CastlingRights.inferred(from: pieces)
        #expect(rights.fenField == "Qkq")
    }

    @Test func testNoRightsRendersDash() {
        #expect(CastlingRights.none.fenField == "-")
    }
}

struct PositionValidatorTests {

    private func pieces(_ placement: String) -> [Square: Piece] {
        FENBuilder.pieces(fromPlacement: placement)!
    }

    @Test func testStartPositionIsValid() {
        let issues = PositionValidator.validate(
            pieces: pieces(FENBuilderTests.startPlacement), sideToMove: .white)
        #expect(issues.isEmpty)
    }

    @Test func testMissingKing() {
        let issues = PositionValidator.validate(
            pieces: pieces("8/8/8/8/8/8/8/K7"), sideToMove: .white)
        #expect(issues.contains(.missingKing(.black)))
    }

    @Test func testMultipleKings() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/8/8/8/KK6"), sideToMove: .white)
        #expect(issues.contains(.multipleKings(.white)))
    }

    @Test func testPawnOnBackRank() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/8/8/8/P3K3"), sideToMove: .white)
        #expect(issues.contains(.pawnOnBackRank(.a1)))
    }

    @Test func testNinePawns() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/8/1PPP4/PPPPPP2/4K3"), sideToMove: .white)
        #expect(issues.contains(.tooManyPawns(.white)))
    }

    @Test func testSideNotToMoveInCheck() {
        // White queen on e-file gives check to black king on e8, but it's
        // white to move — meaning black "just moved" while in check. Invalid.
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/4Q3/8/8/4K3"), sideToMove: .white)
        #expect(issues.contains(.sideNotToMoveInCheck))

        // Same position with black to move is a perfectly normal check.
        let ok = PositionValidator.validate(
            pieces: pieces("4k3/8/8/8/4Q3/8/8/4K3"), sideToMove: .black)
        #expect(!ok.contains(.sideNotToMoveInCheck))
    }

    @Test func testBlockedSliderGivesNoCheck() {
        // Queen's path to the king is blocked by its own pawn.
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/8/4P3/4Q3/8/8/4K3"), sideToMove: .white)
        #expect(!issues.contains(.sideNotToMoveInCheck))
    }

    @Test func testKnightCheckDetected() {
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/8/3N4/8/8/8/8/4K3"), sideToMove: .white)
        #expect(issues.contains(.sideNotToMoveInCheck))
    }

    @Test func testPawnCheckDetected() {
        // White pawn on d7 attacks e8.
        let issues = PositionValidator.validate(
            pieces: pieces("4k3/3P4/8/8/8/8/8/4K3"), sideToMove: .white)
        #expect(issues.contains(.sideNotToMoveInCheck))
    }

    @Test func testHighlightSquares() {
        let map = pieces("4k3/8/8/8/8/8/8/P3K3")
        let issues = PositionValidator.validate(pieces: map, sideToMove: .white)
        let highlights = PositionValidator.highlightSquares(for: issues, pieces: map)
        #expect(highlights.contains(.a1))
    }
}

struct StubRecognizerTests {
    @Test func testStubReturnsStartPosition() async throws {
        let context = CGContext(
            data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let image = context.makeImage()!

        let result = try await StubRecognizer().recognize(image)
        #expect(result.pieces.count == 32)
        #expect(result.pieces[.e1]?.piece.kind == .king)
        #expect(result.pieces[.e1]?.piece.color == .white)
        #expect(result.pieces[.d8]?.piece.kind == .queen)
        #expect(result.pieces[.d8]?.piece.color == .black)
    }
}
