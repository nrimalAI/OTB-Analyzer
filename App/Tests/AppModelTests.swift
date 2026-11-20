import ChessBoardUI
import ChessKit
import ChessVision
import Testing

@testable import ChessAnalyzer

@MainActor
@Suite struct AppModelTests {

    @Test func aQuarterTurnMovesPiecesAndTheirDoubts() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        let pieces = model.pieces
        let confidence = model.confidence

        model.turnQuarter()
        // The white "queen" on g1 is now on a2, and still doubtful there.
        #expect(model.pieces[.a2] == Piece(.queen, color: .white, square: .a2))
        #expect(model.confidence[.a2] == confidence[.g1])
        #expect(model.uncertainSquares == Set(confidence.filter { $0.value < AppModel.lowConfidenceThreshold }
            .keys.map(BoardGeometry.quarterTurned)))

        model.turnQuarter()
        model.turnQuarter()
        model.turnQuarter()
        #expect(model.pieces == pieces)
        #expect(model.confidence == confidence)
    }

    @Test func theDemoScanHasOneProblemWithAFix() throws {
        let model = AppModel()
        model.load(result: DemoScan.result)
        let problem = try #require(model.problems.first)
        #expect(model.problems.count == 1)
        #expect(problem.title == "White has no king")
        #expect(problem.fix == .select(.g1))
        #expect(model.suggestions(for: .g1).first == .piece(.king, .white))
        #expect(model.suggestions(for: .d4).first == .piece(.pawn, .white))
    }

    @Test func fixingTheDemoScanGivesTheTruePosition() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        model.set(PieceChoice.piece(.king, .white).piece(on: .g1), at: .g1)
        model.set(PieceChoice.piece(.pawn, .white).piece(on: .d4), at: .d4)
        #expect(FENBuilder.placement(model.pieces) == DemoScan.truePlacement)
        #expect(model.canAnalyze)
        // Only the knight the recognizer was unsure of, and rightly so, is left marked.
        #expect(model.uncertainSquares == [.f6])
    }
}
