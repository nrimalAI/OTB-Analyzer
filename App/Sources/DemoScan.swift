#if DEBUG
import ChessKit
import ChessVision

/// A reading of a photo for screenshots and UI tests, in debug builds only
/// (`SIMCTL_CHILD_AUTOFLOW=scanned`, `App/UITests`): the kinds of mistake a
/// board the models have not seen gets, so that the screens for fixing them
/// can be looked at without a camera.
///
/// The position on the board is a Queen's Gambit Declined after both sides
/// castled (`r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1`).
/// It is read with White's king on g1 as a queen (so White has no king),
/// White's pawn on d4 as Black's, and Black's knight on f6 right but unsure.
enum DemoScan {
    static let truePlacement = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1"
    static let readPlacement = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2Pp4/2N1PN2/PP2BPPP/R1BQ1RQ1"

    static let unsure: [Square: Float] = [.g1: 0.54, .d4: 0.62, .f6: 0.78]

    static var result: RecognitionResult {
        let pieces = FENBuilder.pieces(fromPlacement: readPlacement) ?? [:]
        let read = Dictionary(uniqueKeysWithValues: pieces.map { square, piece in
            (square, RecognizedPiece(piece: piece, confidence: unsure[square] ?? 0.97))
        })
        return RecognitionResult(pieces: read, orientationGuess: .whiteBottom)
    }
}

extension AppModel {
    /// Opens the demo reading as if it had just come from a photo.
    func showDemoScan() {
        load(result: DemoScan.result)
        path = [.board]
    }
}
#endif
