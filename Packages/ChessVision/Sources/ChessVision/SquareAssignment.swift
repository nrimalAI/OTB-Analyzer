import ChessKit
import CoreGraphics
import Foundation

/// A raw piece detection from the detector model, in image coordinates.
struct PieceDetection {
    let piece: Piece.Kind
    let color: Piece.Color
    let confidence: Float
    let boundingBox: CGRect  // image coordinates, origin top-left
    /// The model's score for each kind it considered for this piece, the
    /// larger of the two colours; a kind without an entry counts as 0
    /// (`kind_scores` in `chess_pipeline.one_king_a_side`). Empty where
    /// nothing is known beyond the kind the piece was taken for.
    let kindScores: [Piece.Kind: Double]
    /// What the user is shown as the confidence, where that is not
    /// `confidence`: a piece that a rule took for another kind than the model
    /// did (`OneKingASide`) is shown with the model's score for that kind,
    /// so that the app marks it for the user to check. `confidence` stays
    /// the model's score for its own first choice, as in the Python, and
    /// goes on deciding which of two pieces on a square stays.
    let shownConfidence: Float?

    init(
        piece: Piece.Kind, color: Piece.Color, confidence: Float, boundingBox: CGRect,
        kindScores: [Piece.Kind: Double] = [:], shownConfidence: Float? = nil
    ) {
        self.piece = piece
        self.color = color
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.kindScores = kindScores
        self.shownConfidence = shownConfidence
    }

    /// The piece's contact point with the board.
    var anchor: CGPoint {
        CGPoint(x: boundingBox.midX, y: boundingBox.maxY)
    }

    /// The same detection taken for another kind or side: confidence, box
    /// and scores stay. Taken for another kind, it is shown with the model's
    /// score for that kind (0 where it gave none worth the name).
    func taken(for piece: Piece.Kind? = nil, color: Piece.Color? = nil) -> PieceDetection {
        let kind = piece ?? self.piece
        return PieceDetection(
            piece: kind, color: color ?? self.color,
            confidence: confidence, boundingBox: boundingBox, kindScores: kindScores,
            shownConfidence: kind == self.piece
                ? shownConfidence : Float(OneKingASide.score(of: kind, for: self)))
    }
}

/// Maps piece detections onto board squares via the corner homography.
/// Port of `assign_squares` in `Training/chess_pipeline.py` — keep in sync.
enum SquareAssignment {

    static let defaultPull = 0.30

    /// How far outside the board, in squares, the foot of a piece may fall
    /// and still be taken back onto the rim square (`EDGE_MARGIN`). A piece on
    /// a rim square can come out just over the edge (corners a little off, a
    /// piece leaning out); further than this it is a captured or spare piece
    /// standing beside the board. Half a square, as it was at first, put
    /// those on the rim.
    static let edgeMargin = 0.25

    /// One piece per square; higher confidence wins, losers are demoted to
    /// their nearest free neighbor square or dropped.
    static func assign(
        detections: [PieceDetection],
        homography: Homography,
        pull: Double = defaultPull
    ) -> [Square: RecognizedPiece] {
        var board: [Square: RecognizedPiece] = [:]

        for det in detections.sorted(by: { $0.confidence > $1.confidence }) {
            guard let bp = homography.pulledBoardPoint(det.anchor, pull: pull),
                  let square = square(atBoardX: bp.x, y: bp.y)
            else { continue }

            if board[square] == nil {
                board[square] = recognized(det, at: square)
                continue
            }

            // Occupied by a more confident piece — nearest free 8-neighbor.
            var best: (distance: Double, square: Square)?
            let f0 = square.file.number - 1
            let r0 = square.rank.value - 1
            for df in -1...1 {
                for dr in -1...1 where !(df == 0 && dr == 0) {
                    let nf = f0 + df
                    let nr = r0 + dr
                    guard (0..<8).contains(nf), (0..<8).contains(nr),
                          let neighbor = FENBuilder.square(file: nf + 1, rank: nr + 1),
                          board[neighbor] == nil
                    else { continue }
                    let dx = bp.x - (Double(nf) + 0.5)
                    let dy = bp.y - (Double(nr) + 0.5)
                    let dist = dx * dx + dy * dy
                    if best == nil || dist < best!.distance {
                        best = (dist, neighbor)
                    }
                }
            }
            if let best {
                board[best.square] = recognized(det, at: best.square)
            }
            // else: dropped
        }
        return board
    }

    /// Board-frame point → square (`board_square`), taking a point up to
    /// `edgeMargin` outside the board back onto the rim square; further out
    /// there is no square.
    static func square(atBoardX x: Double, y: Double) -> Square? {
        // The margin first: a point that is not a number, or far away, has no
        // square (and must not reach `Int()`, which traps on such values).
        let inside = -edgeMargin..<(8 + edgeMargin)
        guard inside.contains(x), inside.contains(y) else { return nil }
        let f = min(max(Int(floor(x)), 0), 7)
        let r = min(max(Int(floor(y)), 0), 7)
        return FENBuilder.square(file: f + 1, rank: r + 1)
    }

    private static func recognized(_ det: PieceDetection, at square: Square) -> RecognizedPiece {
        RecognizedPiece(
            piece: Piece(det.piece, color: det.color, square: square),
            confidence: det.shownConfidence ?? det.confidence
        )
    }
}
