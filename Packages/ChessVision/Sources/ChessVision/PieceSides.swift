import ChessKit
import CoreGraphics
import Foundation

/// Which side a piece belongs to, from the colours of the pieces in the photo.
///
/// A piece model that has not seen a piece set (brown "white" pieces, say)
/// can get the kind of each piece right and the side wrong. The pieces on a
/// board are still two groups of one colour each: they are split in two by
/// colour, the lighter group is White, and where that contradicts the model
/// for more than a share of the pieces, the colours decide.
///
/// Port of `colour_patch`, `two_groups` and `sides_from_colour` in
/// `Training/chess_pipeline.py` and of `piece_colours` in
/// `Training/evaluate.py`: keep in sync.
enum PieceSides {

    /// Where in a piece's box its colour is measured: the middle, as
    /// fractions of the box. The body of the piece, not the square behind
    /// its head or the board in front of its foot (`COLOUR_PATCH`).
    static let patch = (left: 0.3, right: 0.7, top: 0.2, bottom: 0.85)
    /// How much more a* and b* (the colour) count than L* (the lightness)
    /// when the pieces are split in two (`CHROMA_WEIGHT`).
    static let chromaWeight = 4.0
    /// The model's own answer stands unless the colours contradict it for
    /// more than this share of the pieces (`SIDE_DISAGREEMENT`).
    static let disagreement = 0.15

    /// A colour in CIE Lab (D65), L* from 0 to 100.
    struct Lab: Equatable {
        var l: Double
        var a: Double
        var b: Double
    }

    /// A rectangle of whole pixels, end exclusive.
    struct PixelRect: Equatable {
        var x0: Int
        var y0: Int
        var x1: Int
        var y1: Int
    }

    // MARK: - The step

    /// The detections with White and Black decided by the pieces' colours in
    /// `image` where the model's own answer does not fit them. `image` is the
    /// photo the detections' boxes are in.
    static func fromColour(_ detections: [PieceDetection], in image: CGImage) -> [PieceDetection] {
        fromColour(detections, colours: colours(of: detections, in: image))
    }

    /// `sides_from_colour`: `colours` holds, for each detection, the colour
    /// of its patch, or nil when the patch is empty. The pieces that have a
    /// colour are split in two groups (`twoGroups` on L*, chroma·a*,
    /// chroma·b*); the lighter group is White.
    ///
    /// The split can be wrong for a few pieces (a shadow, a glint), so on
    /// boards the model reads well it would cost more than it gains. It is
    /// used only when it contradicts the model for more than `disagreement`
    /// of the pieces: then every piece with a colour takes its group's side.
    /// A piece keeps its kind, confidence and box either way.
    static func fromColour(
        _ detections: [PieceDetection],
        colours: [Lab?],
        chroma: Double = chromaWeight,
        disagreement: Double = disagreement
    ) -> [PieceDetection] {
        guard colours.count == detections.count,
              let sides = sidesByColour(colours, chroma: chroma)
        else { return detections }
        var withColour = 0, differ = 0
        for (detection, side) in zip(detections, sides) {
            guard let side else { continue }
            withColour += 1
            if detection.color != side {
                differ += 1
            }
        }
        if Double(differ) <= disagreement * Double(withColour) {
            return detections
        }
        return zip(detections, sides).map { detection, side in
            guard let side else { return detection }
            return PieceDetection(
                piece: detection.piece, color: side,
                confidence: detection.confidence, boundingBox: detection.boundingBox)
        }
    }

    /// The side of each piece by its colour alone: the pieces that have a
    /// colour split in two groups, the lighter group White. A piece without a
    /// colour has no side; nil when there is nothing to split (fewer than two
    /// colours, all the same, or one group left empty).
    static func sidesByColour(_ colours: [Lab?], chroma: Double = chromaWeight) -> [Piece.Color?]? {
        let indices = colours.indices.filter { colours[$0] != nil }
        guard indices.count >= 2 else { return nil }
        let points = indices.map { index -> Point in
            let colour = colours[index]!
            return Point(colour.l, chroma * colour.a, chroma * colour.b)
        }
        guard let split = twoGroups(points), Set(split.group).count == 2 else { return nil }
        let white = split.centres.0.x >= split.centres.1.x ? 0 : 1
        var sides = [Piece.Color?](repeating: nil, count: colours.count)
        for (index, group) in zip(indices, split.group) {
            sides[index] = group == white ? .white : .black
        }
        return sides
    }

    // MARK: - Two groups

    /// A point of three numbers.
    struct Point: Equatable {
        var x: Double
        var y: Double
        var z: Double

        init(_ x: Double, _ y: Double, _ z: Double) {
            self.x = x
            self.y = y
            self.z = z
        }

        /// The squared distance, summed in NumPy's order.
        @inline(__always)
        func squaredDistance(to other: Point) -> Double {
            let dx = x - other.x, dy = y - other.y, dz = z - other.z
            return dx * dx + dy * dy + dz * dz
        }
    }

    /// `two_groups`: the points split in two by 2-means, started from the two
    /// points farthest apart (the first such pair, counting row by row).
    /// Returns the group (0 or 1) of each point and the two group means; nil
    /// when there are fewer than two points or they are all the same. A point
    /// as near to one mean as to the other goes to group 0.
    static func twoGroups(_ points: [Point]) -> (group: [Int], centres: (Point, Point))? {
        let n = points.count
        guard n >= 2 else { return nil }
        var first = 0, second = 0
        var farthest = 0.0
        for i in 0..<n {
            for j in 0..<n {
                let d = points[i].squaredDistance(to: points[j])
                if d > farthest {
                    farthest = d
                    first = i
                    second = j
                }
            }
        }
        guard farthest > 0 else { return nil }
        var centres = (points[first], points[second])
        var group: [Int]?
        for _ in 0..<50 {
            let new = points.map {
                $0.squaredDistance(to: centres.0) <= $0.squaredDistance(to: centres.1) ? 0 : 1
            }
            if new == group { break }
            group = new
            for g in 0..<2 {
                // (the mean as `np.mean(members, axis=0)` sums it: one member
                // after the other)
                var sum = Point(0, 0, 0)
                var count = 0
                for k in 0..<n where new[k] == g {
                    sum.x += points[k].x
                    sum.y += points[k].y
                    sum.z += points[k].z
                    count += 1
                }
                guard count > 0 else { continue }
                let mean = Point(sum.x / Double(count), sum.y / Double(count), sum.z / Double(count))
                if g == 0 {
                    centres.0 = mean
                } else {
                    centres.1 = mean
                }
            }
        }
        return (group!, centres)
    }

    // MARK: - Colours

    /// `colour_patch`: the part of a piece's box whose colour is measured, in
    /// whole pixels of a `width` x `height` photo; nil when it is empty.
    static func colourPatch(_ box: CGRect, width: Int, height: Int) -> PixelRect? {
        let x = Double(box.minX), y = Double(box.minY)
        let w = Double(box.width), h = Double(box.height)
        // (Python's int() cuts towards zero, as Int() does)
        let edges = [
            x + patch.left * w, x + patch.right * w,
            y + patch.top * h, y + patch.bottom * h,
        ]
        guard edges.allSatisfy({ $0.isFinite && abs($0) < 1e15 }) else { return nil }
        let rect = PixelRect(
            x0: max(Int(edges[0]), 0), y0: max(Int(edges[2]), 0),
            x1: min(Int(edges[1]) + 1, width), y1: min(Int(edges[3]) + 1, height))
        return rect.x1 > rect.x0 && rect.y1 > rect.y0 ? rect : nil
    }

    /// `piece_colours`: the median colour (each of L*, a*, b* by itself) of
    /// the patch of each detection's box in the photo, or nil for an empty
    /// patch. All nil when the photo's pixels cannot be read.
    static func colours(of detections: [PieceDetection], in image: CGImage) -> [Lab?] {
        let patches = detections.map {
            colourPatch($0.boundingBox, width: image.width, height: image.height)
        }
        let found = patches.compactMap { $0 }
        guard !found.isEmpty else { return patches.map { _ in nil } }
        // One read of the rectangle that holds every patch: a photo is
        // decoded once for it, not once a piece.
        let x0 = found.map(\.x0).min()!, y0 = found.map(\.y0).min()!
        let x1 = found.map(\.x1).max()!, y1 = found.map(\.y1).max()!
        let width = x1 - x0
        guard let pixels = GrayImage.rgba(of: image, x: x0, y: y0, width: width, height: y1 - y0)
        else { return patches.map { _ in nil } }

        return pixels.withUnsafeBufferPointer { rgba in
            patches.map { patch in
                guard let patch else { return nil }
                let count = (patch.x1 - patch.x0) * (patch.y1 - patch.y0)
                var l = [Double](repeating: 0, count: count)
                var a = [Double](repeating: 0, count: count)
                var b = [Double](repeating: 0, count: count)
                var index = 0
                var y = patch.y0
                while y < patch.y1 {
                    var offset = ((y - y0) * width + (patch.x0 - x0)) * 4
                    var x = patch.x0
                    while x < patch.x1 {
                        let colour = lab(red: rgba[offset], green: rgba[offset + 1], blue: rgba[offset + 2])
                        l[index] = colour.l
                        a[index] = colour.a
                        b[index] = colour.b
                        index += 1
                        offset += 4
                        x += 1
                    }
                    y += 1
                }
                return Lab(l: GridMath.median(l), a: GridMath.median(a), b: GridMath.median(b))
            }
        }
    }

    /// sRGB's transfer curve for the 256 byte values.
    private static let linear: [Double] = (0..<256).map { byte in
        let value = Double(byte) / 255
        return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
    }

    /// An sRGB colour in CIE Lab, with the constants of OpenCV's
    /// `COLOR_BGR2Lab` (its sRGB-to-XYZ matrix, D65 white (0.950456, 1,
    /// 1.088754), threshold 216/24389, slope 7.787).
    ///
    /// This is the formula itself. OpenCV does not compute it for float
    /// pixels: it interpolates in a 33 x 33 x 33 table, which is off by up to
    /// 0.2 in L* and 0.4 in a* and b* (L* 0.09 too low on average). The
    /// reference's numbers are therefore these to within that, which is less
    /// than the two JPEG decoders differ on a photo (`docs/RECOGNITION.md`).
    static func lab(red: UInt8, green: UInt8, blue: UInt8) -> Lab {
        let r = linear[Int(red)], g = linear[Int(green)], b = linear[Int(blue)]
        let x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456
        let y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        let z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754
        let threshold = 216.0 / 24389.0
        func f(_ t: Double) -> Double {
            t > threshold ? cbrt(t) : 7.787 * t + 16.0 / 116.0
        }
        let fy = f(y)
        return Lab(
            l: y > threshold ? 116 * fy - 16 : 903.3 * y,
            a: 500 * (f(x) - fy),
            b: 200 * (fy - f(z)))
    }
}

/// What the piece model and the colours say about one piece of a photo
/// (`CoreMLBoardRecognizer.diagnosePieceSides`), for checking the colour step
/// against the Python (`app_check --colours`).
public struct PieceSideDiagnosis: Sendable {
    /// The piece model's class, as the Python names it: `"white_pawn"`.
    public let label: String
    public let confidence: Float
    /// In pixels of the photo, origin top-left.
    public let box: CGRect
    /// The median L*, a*, b* of the middle of the box; nil when that lies
    /// outside the photo.
    public let colour: [Double]?
    /// `"white"` or `"black"` by colour alone; nil without a colour or when
    /// the pieces do not split in two.
    public let colourSide: String?
    /// The class after the step.
    public let finalLabel: String
}

extension PieceSides {

    /// `"white_pawn"`, as the Python names a detection's class.
    static func label(_ detection: PieceDetection) -> String {
        name(detection.color) + "_" + name(detection.piece)
    }

    private static func name(_ side: Piece.Color) -> String {
        side == .white ? "white" : "black"
    }

    private static func name(_ kind: Piece.Kind) -> String {
        switch kind {
        case .pawn: return "pawn"
        case .knight: return "knight"
        case .bishop: return "bishop"
        case .rook: return "rook"
        case .queen: return "queen"
        case .king: return "king"
        }
    }

    static func diagnose(_ detections: [PieceDetection], in image: CGImage) -> [PieceSideDiagnosis] {
        let colours = colours(of: detections, in: image)
        let byColour = sidesByColour(colours)
        let final = fromColour(detections, colours: colours)
        return detections.indices.map { index in
            PieceSideDiagnosis(
                label: label(detections[index]),
                confidence: detections[index].confidence,
                box: detections[index].boundingBox,
                colour: colours[index].map { [$0.l, $0.a, $0.b] },
                colourSide: byColour?[index].map(name),
                finalLabel: label(final[index]))
        }
    }
}
