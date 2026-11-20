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
        // Turning moves the pieces only.
        #expect(model.sideToMove == .white)
        #expect(model.photoSide == .white)
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

@MainActor
@Suite struct SideToMoveTests {

    private func result(_ placement: String) -> RecognitionResult {
        let pieces = FENBuilder.pieces(fromPlacement: placement) ?? [:]
        return RecognitionResult(pieces: pieces.mapValues { RecognizedPiece(piece: $0, confidence: 0.99) })
    }

    @Test func aScanLeavesTheSideToMoveOpen() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        #expect(!model.sideToMoveKnown)
        model.chooseSideToMove(.black)
        #expect(model.sideToMoveKnown)
        #expect(model.fen.split(separator: " ")[1] == "b")
    }

    @Test func choosingWhiteAnswersTooEvenThoughItIsThePlaceholder() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        model.chooseSideToMove(.white)
        #expect(model.sideToMoveKnown)
    }

    @Test func aKingInCheckSaysWhoIsToMove() {
        // Black's king on e8 in check from the rook on e1.
        let model = AppModel()
        model.load(result: result("4k3/8/8/8/8/8/8/4RK2"))
        #expect(model.sideToMoveKnown)
        #expect(model.sideToMove == .black)
        #expect(model.canAnalyze)
    }

    @Test func aSideInferredFromACheckIsAskedAgainWhenTheCheckGoes() {
        // The "rook" on e1 checks Black's king: Black must be to move...
        let model = AppModel()
        model.load(result: result("4k3/8/8/8/8/8/8/4RK2"))
        #expect(model.sideToMoveSource == .check)
        // ...until it turns out to be a pawn: then nobody has said.
        model.set(Piece(.pawn, color: .white, square: .e2), at: .e1)
        model.set(nil, at: .e1)
        #expect(model.sideToMoveSource == .unknown)
        #expect(!model.sideToMoveKnown)
        // And a check that appears while unknown answers it.
        model.set(Piece(.rook, color: .white, square: .e1), at: .e1)
        #expect(model.sideToMoveSource == .check)
        #expect(model.sideToMove == .black)
    }

    @Test func turningTheBoardReconsidersAnInferredSide() {
        // The pawn on d7 checks Black's king on e8. Turned a quarter, the
        // pawn on g5 points elsewhere: no check, and the question is open.
        let model = AppModel()
        model.load(result: result("4k3/3P4/8/8/8/8/8/6K1"))
        #expect(model.sideToMoveSource == .check)
        #expect(model.sideToMove == .black)
        model.turnQuarter()
        #expect(model.sideToMoveSource == .unknown)
    }

    @Test func aChosenSideIsNotTakenBackByEdits() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        model.chooseSideToMove(.black)
        model.set(nil, at: .f6)
        model.turnQuarter()
        model.setPhotoSide(.black)
        #expect(model.sideToMoveSource == .chosen)
        #expect(model.sideToMove == .black)
    }

    @Test func aNewScanAsksAgain() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        model.chooseSideToMove(.black)
        model.load(result: DemoScan.result)
        #expect(!model.sideToMoveKnown)
        #expect(model.sideToMove == .white)
    }

    @Test func forcedSideToMove() throws {
        func forced(_ placement: String) throws -> Piece.Color? {
            AppModel.forcedSideToMove(try #require(FENBuilder.pieces(fromPlacement: placement)))
        }
        #expect(try forced("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR") == nil)
        #expect(try forced("4k3/8/8/8/8/8/8/4RK2") == .black)
        #expect(try forced("4kr2/8/8/8/8/8/8/5K2") == .white)
        // Both in check: no answer, the validator says the position is wrong.
        #expect(try forced("4k2R/8/8/8/8/8/8/4K2r") == nil)
    }

    @Test func editingThePlaceholderFENKeepsTheQuestionOpen() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        #expect(model.applyFEN(DemoScan.truePlacement + " w - - 0 1"))
        #expect(!model.sideToMoveKnown)
        #expect(model.applyFEN(DemoScan.truePlacement + " b - - 0 1"))
        #expect(model.sideToMoveKnown)
        #expect(model.sideToMove == .black)
    }

    @Test func setUpByHandAndSavedPositionsAreKnown() {
        let model = AppModel()
        model.load(result: DemoScan.result)
        model.startFromScratch(startingPosition: true)
        #expect(model.sideToMoveKnown)
        model.load(result: DemoScan.result)
        model.open(SavedPosition(fen: DemoScan.truePlacement + " w - - 0 1", date: .now, depth: 18))
        #expect(model.sideToMoveKnown)
    }
}
