import ChessKit
import CoreGraphics

/// Which corner of the board is a1, decided from the pieces.
///
/// A corner model can find an unfamiliar board and still not know which side
/// White sits on; the pieces usually say. Port of `white_side_score` and
/// `orient_by_pieces` in `Training/chess_pipeline.py`: keep in sync.
enum BoardOrientation {

    /// How much the corner model's own choice of a1 counts against the
    /// pieces in the app (`CoreMLBoardRecognizer`). Chosen on the ChessReD
    /// test split (`Training/README.md`, `docs/RECOGNITION.md`).
    static let appPrior = 6.0

    /// How much a board looks like one seen with White at ranks 1-2: White's
    /// pieces count for being near rank 1, Black's for being near rank 8,
    /// pawns three times and kings one and a half times as much as the rest,
    /// and a pawn on the first or last rank (impossible) costs 20. Positive
    /// when the orientation is plausible, negative when the board is upside
    /// down, near zero when it is turned by 90 degrees.
    static func whiteSideScore(_ board: [Square: RecognizedPiece]) -> Double {
        var score = 0.0
        for (square, recognized) in board {
            let rank = square.rank.value - 1
            let kind = recognized.piece.kind
            let weight = kind == .pawn ? 3.0 : kind == .king ? 1.5 : 1.0
            let towardsOwnSide = recognized.piece.color == .white
                ? 3.5 - Double(rank) : Double(rank) - 3.5
            score += weight * towardsOwnSide
            if kind == .pawn && (rank == 0 || rank == 7) {
                score -= 20
            }
        }
        return score
    }

    /// The corners (a1, a8, h8, h1) turned so that the pieces say which one
    /// is a1: of the four rotations, the one whose board has the highest
    /// `whiteSideScore`. `prior` is added to the score of the corners as
    /// given, i.e. how much to trust the corner model's own choice (0: not at
    /// all, the default, as in Python). On a tie the earlier rotation wins,
    /// the corners as given first.
    static func orient(
        corners: [CGPoint],
        detections: [PieceDetection],
        pull: Double = SquareAssignment.defaultPull,
        prior: Double = 0
    ) -> [CGPoint] {
        guard corners.count == 4 else { return corners }
        var best = corners
        var bestScore: Double?
        for turn in 0..<4 {
            let turned = Array(corners[turn...] + corners[..<turn])
            guard let homography = Homography(corners: turned) else { continue }
            var score = whiteSideScore(SquareAssignment.assign(
                detections: detections, homography: homography, pull: pull))
            if turn == 0 {
                score += prior
            }
            if bestScore == nil || score > bestScore! {
                best = turned
                bestScore = score
            }
        }
        return best
    }
}
