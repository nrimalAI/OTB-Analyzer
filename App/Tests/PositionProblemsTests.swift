import ChessKit
import ChessVision
import Testing

@testable import ChessAnalyzer

@Suite struct PositionProblemsTests {

    private func problems(
        _ placement: String, toMove side: Piece.Color = .white, confidence: [Square: Float] = [:]
    ) throws -> [PositionProblem] {
        let pieces = try #require(FENBuilder.pieces(fromPlacement: placement))
        let issues = PositionValidator.validate(pieces: pieces, sideToMove: side)
        return PositionProblems.explain(issues, pieces: pieces, sideToMove: side, confidence: confidence)
    }

    @Test func aLegalPositionHasNone() throws {
        #expect(try problems("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").isEmpty)
    }

    @Test func aMissingKingPointsAtTheLeastSureQueen() throws {
        let found = try problems(
            "4k3/8/8/8/8/8/8/2Q1Q3", confidence: [.c1: 0.9, .e1: 0.41])
        let problem = try #require(found.first)
        #expect(problem.title == "White has no king")
        #expect(problem.fix == .select(.e1))
        #expect(problem.fixLabel == "Show")
    }

    @Test func aMissingKingWithoutAQueenHasNoFix() throws {
        let found = try problems("4k3/8/8/8/8/8/8/2R5")
        #expect(found.first?.fix == nil)
        #expect(found.first?.fixLabel == nil)
    }

    @Test func twoKingsAreCounted() throws {
        let found = try problems("4k3/8/8/8/8/8/8/2K1K3", confidence: [.c1: 0.5])
        let problem = try #require(found.first)
        #expect(problem.title == "White has 2 kings")
        #expect(problem.fix == .select(.c1))
    }

    @Test func aPawnOnTheBackRankIsShown() throws {
        let found = try problems("3Pk3/8/8/8/8/8/8/4K3")
        #expect(found.map(\.title) == ["Pawn on d8"])
        #expect(found.first?.fix == .select(.d8))
    }

    @Test func tooManyPawnsAndPieces() throws {
        let pawns = try problems("4k3/8/8/8/4P3/8/PPPPPPPP/4K3")
        #expect(pawns.first?.title == "White has 9 pawns")
        #expect(pawns.first?.detail.contains("Black") == true)
        let pieces = try problems("4k3/8/8/8/1NNN4/8/PPPPPPPP/RNBQKBNR")
        #expect(pieces.contains { $0.title == "White has 19 pieces" })
    }

    @Test func theSideInCheckIsOfferedTheMove() throws {
        // White to move, but Black's king is in check from the rook.
        let found = try problems("4k3/8/8/8/8/8/8/4RK2", toMove: .white)
        let problem = try #require(found.first)
        #expect(problem.title == "Black is in check")
        #expect(problem.fix == .setSideToMove(.black))
        #expect(problem.fixLabel == "Black to move")
    }

    @Test func bothKingsInCheckHaveNoSideToOffer() throws {
        // Each king attacked along its back rank by the other side's rook.
        let found = try problems("4k2R/8/8/8/8/8/8/4K2r", toMove: .white)
        let problem = try #require(found.first { $0.issue == .sideNotToMoveInCheck })
        #expect(problem.title == "Both kings are in check")
        #expect(problem.fix == nil)
    }
}
