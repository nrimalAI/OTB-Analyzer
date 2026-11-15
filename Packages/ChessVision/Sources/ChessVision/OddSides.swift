import ChessKit
import CoreGraphics
import Foundation

/// The side of single pieces: a piece the model calls Black becomes White
/// when its colour is plainly the White side's in this photo.
///
/// The model's own answers say what colour each side is in the photo: the
/// median colour of the pieces it calls White and of those it calls Black.
/// A piece called Black whose colour is much nearer White's than the other
/// Black pieces' is the odd one out, and becomes White; it keeps its kind,
/// confidence and box. Unlike `PieceSides` nothing is split by colour alone
/// and no piece is changed because others are: the model is taken to be
/// right about most pieces.
///
/// One way only. A light piece in a shadow is as dark as a black one, so a
/// piece called White is never made Black by its colour.
///
/// Port of `odd_sides` and `patch_hidden` in `Training/chess_pipeline.py`:
/// keep in sync. The colours are those of `PieceSides.colours`.
enum OddSides {

    /// A piece called Black becomes White when its colour is this many times
    /// nearer White's colour than the other Black pieces' (`SIDE_NEARER`) ...
    static let nearer = 2.0
    /// ... and the two sides' colours are at least this many spreads apart
    /// (`SIDE_APART`). The spread: the median distance of a piece from its
    /// own side's colour, and not less than `spreadFloor`
    /// (`SIDE_SPREAD_FLOOR`).
    static let apart = 3.0
    static let spreadFloor = 3.0
    /// A piece is left out when another piece stands in front of it over
    /// more than this share of the patch its colour is measured on
    /// (`SIDE_HIDDEN`).
    static let hidden = 0.2

    /// The detections with the odd pieces made White, by the pieces' colours
    /// in `image`, the photo the detections' boxes are in.
    static func apply(_ detections: [PieceDetection], in image: CGImage) -> [PieceDetection] {
        apply(detections, colours: PieceSides.colours(of: detections, in: image))
    }

    /// `odd_sides`: `colours` holds, for each detection, the colour of its
    /// patch, or nil when the patch is empty. Colours are compared as
    /// (L*, chroma·a*, chroma·b*).
    ///
    /// Nothing is changed when a side has fewer than two pieces with a
    /// colour, when the pieces called White are not the lighter ones, or
    /// when the two sides' colours are less than `apart` spreads from each
    /// other. A piece with another standing in front of its colour patch
    /// (`patchHidden`) is neither counted nor changed.
    static func apply(
        _ detections: [PieceDetection],
        colours: [PieceSides.Lab?],
        chroma: Double = PieceSides.chromaWeight,
        nearer: Double = nearer,
        apart: Double = apart,
        cover: Double = hidden
    ) -> [PieceDetection] {
        guard colours.count == detections.count else { return detections }
        // The pieces that count: a colour, and nothing in front of it.
        var point: [Int: PieceSides.Point] = [:]
        var white: [Int] = [], black: [Int] = []
        for (k, colour) in colours.enumerated() {
            guard let colour, !patchHidden(detections, k, cover: cover) else { continue }
            point[k] = PieceSides.Point(colour.l, chroma * colour.a, chroma * colour.b)
            if detections[k].color == .white {
                white.append(k)
            } else {
                black.append(k)
            }
        }
        guard min(white.count, black.count) >= 2 else { return detections }
        let centreWhite = median(white.map { point[$0]! })
        let centreBlack = median(black.map { point[$0]! })
        // White must be the lighter side.
        guard !(centreWhite.x <= centreBlack.x) else { return detections }
        let distances = (white.map { distance(point[$0]!, centreWhite) }
            + black.map { distance(point[$0]!, centreBlack) })
        let spread = max(spreadFloor, GridMath.median(distances))
        guard !(distance(centreWhite, centreBlack) < apart * spread) else { return detections }

        var out = detections
        for k in black {
            // What the other Black pieces look like, without this one.
            let own = median(black.filter { $0 != k }.map { point[$0]! })
            if nearer * distance(point[k]!, centreWhite) < distance(point[k]!, own) {
                out[k] = detections[k].taken(color: .white)
            }
        }
        return out
    }

    /// `patch_hidden`: whether another piece stands in front of piece `k`
    /// where its colour is measured: a box whose foot is lower in the
    /// picture than piece k's and that covers more than `cover` of piece
    /// k's colour patch (the patch as fractions of the box, not cut to the
    /// picture). What is measured there is then the colour of the piece in
    /// front.
    static func patchHidden(_ detections: [PieceDetection], _ k: Int, cover: Double = hidden) -> Bool {
        let box = detections[k].boundingBox
        let x1 = Double(box.minX), y1 = Double(box.minY)
        let w = Double(box.width), h = Double(box.height)
        let foot = Double(box.maxY)
        let patch = PieceSides.patch
        let px1 = x1 + patch.left * w, py1 = y1 + patch.top * h
        let px2 = x1 + patch.right * w, py2 = y1 + patch.bottom * h
        let area = (px2 - px1) * (py2 - py1)
        if area <= 0 {
            return true
        }
        for (j, other) in detections.enumerated() {
            let theirs = other.boundingBox
            if j == k || Double(theirs.maxY) <= foot {
                continue
            }
            let iw = min(px2, Double(theirs.maxX)) - max(px1, Double(theirs.minX))
            let ih = min(py2, Double(theirs.maxY)) - max(py1, Double(theirs.minY))
            if iw > 0 && ih > 0 && iw * ih > cover * area {
                return true
            }
        }
        return false
    }

    /// `np.median(points, axis=0)`: each of the three numbers by itself.
    private static func median(_ points: [PieceSides.Point]) -> PieceSides.Point {
        PieceSides.Point(
            GridMath.median(points.map(\.x)),
            GridMath.median(points.map(\.y)),
            GridMath.median(points.map(\.z)))
    }

    /// `np.linalg.norm` of the difference.
    private static func distance(_ a: PieceSides.Point, _ b: PieceSides.Point) -> Double {
        a.squaredDistance(to: b).squareRoot()
    }
}
