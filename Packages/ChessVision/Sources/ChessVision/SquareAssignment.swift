import ChessKit
import CoreGraphics
import Foundation

/// A raw piece detection from the detector model, in image coordinates.
struct PieceDetection {
    let piece: Piece.Kind
    let color: Piece.Color
    let confidence: Float
    let boundingBox: CGRect  // image coordinates, origin top-left

    /// The piece's contact point with the board.
    var anchor: CGPoint {
        CGPoint(x: boundingBox.midX, y: boundingBox.maxY)
    }
}

/// Maps piece detections onto board squares via the corner homography.
/// Port of `assign_squares` in `Training/chess_pipeline.py` — keep in sync.
enum SquareAssignment {

    static let defaultPull = 0.30

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

    /// Board-frame point → square, clamping near-miss anchors within half a
    /// square of the edge (pieces leaning over the rim).
    static func square(atBoardX x: Double, y: Double) -> Square? {
        var f = Int(floor(x))
        var r = Int(floor(y))
        if !(0..<8).contains(f) || !(0..<8).contains(r) {
            guard (-0.5..<8.5).contains(x), (-0.5..<8.5).contains(y) else { return nil }
            f = min(max(f, 0), 7)
            r = min(max(r, 0), 7)
        }
        return FENBuilder.square(file: f + 1, rank: r + 1)
    }

    private static func recognized(_ det: PieceDetection, at square: Square) -> RecognizedPiece {
        RecognizedPiece(
            piece: Piece(det.piece, color: det.color, square: square),
            confidence: det.confidence
        )
    }
}
