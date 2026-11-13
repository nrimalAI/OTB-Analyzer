import CoreGraphics
import Foundation

/// The four outer corners of the 8×8 grid, found without a model: Swift port
/// of `find_grid` in `Training/grid_corners.py`, which is the reference this
/// must match. Keep the two in sync.
///
/// How it works:
///  1. x-corners: saddle points of the smoothed image, kept only if a ring
///     around them shows light/dark/light/dark, then refined to sub-pixel
///     (`GridXCorners`).
///  2. lattice: every small parallelogram of neighbouring x-corners proposes
///     a homography image -> integer lattice; the one that puts most
///     x-corners on integers wins and is refitted on its inliers
///     (`GridLattice`).
///  3. which 8×8: the photo is warped onto the lattice, each cell gets a grey
///     level, and the 8×8 window whose cells alternate best (and whose
///     surroundings do not) is chosen (`GridWindow`).
///  4. refinement: edge points are searched along all 18 grid lines of that
///     window and the homography is refitted to them (`GridRefine`).
///  5. validity: enough x-corners, cells alternate, the pattern stops
///     outside, the choice of window is not a near tie, the outer lines have
///     edge support, and (with a prior box) the result fills that box.
///
/// The fit returns nothing when any check fails: a learned corner model can
/// then stand in. Corners come back clockwise as seen in the image (the order
/// a1, a8, h8, h1 has when White sits at the bottom); which of them is a1 is
/// not known. On a board no model was trained on it is far more precise than
/// the corner model (`Training/README.md`).
public enum GridCorners {

    /// An accepted fit.
    public struct Fit {
        /// Outer corners of the grid in pixels of the photo, clockwise in the
        /// image.
        public let corners: [CGPoint]
        /// 0...1; above 0 for every accepted fit.
        public let confidence: Double
    }

    /// What a fit found, accepted or not: for scoring the port against the
    /// Python (`Training/app_check --grid`).
    public struct Diagnosis {
        /// The corners before the validity checks; nil when no grid was found
        /// at all.
        public let rawCorners: [CGPoint]?
        /// 0 when a validity check failed.
        public let confidence: Double
        /// How far the fit got: "lattice", "window", "refine", "final_check"
        /// or "done" (the Python's `info["stage"]`).
        public let stage: String
        /// 1 or 2: which pass the result is from.
        public let pass: Int
        /// The numbers the validity checks look at, under the Python's names.
        public let values: [String: Double]
    }

    /// The grid's outer corners, or nil when no grid passed the validity
    /// checks. `priorBox`: where the board is, in pixels of the photo, if
    /// known; without it the photo is searched twice, the second time on the
    /// board found the first time, for more pixels per square.
    public static func find(in image: CGImage, priorBox: CGRect? = nil) -> Fit? {
        let result = findGrid(image, priorBox: priorBox)
        guard result.confidence > 0, let corners = result.corners else { return nil }
        return Fit(corners: corners, confidence: result.confidence)
    }

    /// As `find`, with the diagnostics.
    public static func diagnose(_ image: CGImage, priorBox: CGRect? = nil) -> Diagnosis {
        let result = findGrid(image, priorBox: priorBox)
        return Diagnosis(
            rawCorners: result.corners, confidence: result.confidence,
            stage: result.info.stage.rawValue, pass: result.info.pass,
            values: result.info.values)
    }

    /// The fit's corners, turned to the rotation nearest to `reference`
    /// (the corner model's a1, a8, h8, h1): the grid fit does not know which
    /// corner is a1, the model at least has an opinion. As `evaluate.py
    /// --grid-corners`.
    public static func turned(_ corners: [CGPoint], nearestTo reference: [CGPoint]) -> [CGPoint] {
        guard corners.count == 4, reference.count == 4 else { return corners }
        var best = corners
        var bestDistance = Double.infinity
        for turn in 0..<4 {
            let candidate = Array(corners[turn...] + corners[..<turn])
            var distance = 0.0
            for (a, b) in zip(candidate, reference) {
                distance += Double(hypot(a.x - b.x, a.y - b.y))
            }
            if distance < bestDistance {
                best = candidate
                bestDistance = distance
            }
        }
        return best
    }

    // MARK: - The fit

    enum Stage: String {
        case start, lattice, window, refine
        case finalCheck = "final_check"
        case done
    }

    /// What one pass found (the Python's `info`).
    struct Info {
        var stage = Stage.start
        var pass = 1
        var xCorners = 0
        var inliers = 0
        var window: GridWindow.Choice?
        var borderNodes = 0
        var firstLineNodes = 0
        var innerNodes = 0
        var stats = GridRefine.Stats()
        var check: GridWindow.Check?
        var area = 0.0
        var convex = false
        var boxIoU: Double?

        var values: [String: Double] {
            var out: [String: Double] = [
                "n_xcorners": Double(xCorners), "n_inliers": Double(inliers),
                "border_nodes": Double(borderNodes), "first_line_nodes": Double(firstLineNodes),
                "inner_nodes": Double(innerNodes), "n_line_pts": Double(stats.linePoints),
                "border_support": stats.borderSupport, "area": area, "convex": convex ? 1 : 0,
            ]
            if let window {
                out["ox"] = Double(window.ox)
                out["oy"] = Double(window.oy)
                out["inside"] = Double(window.inside)
                out["ring"] = Double(window.ring)
                out["margin"] = window.margin
                out["contrast"] = window.contrast
            }
            out["shift"] = stats.shift
            out["line_rms"] = stats.lineRMS
            out["line_support"] = stats.lineSupport
            if let check {
                out["f_inside"] = check.inside
                out["f_ring_max"] = check.ringMax
                out["f_edge_rows_min"] = check.edgeRowsMin
                out["f_contrast"] = check.contrast
            }
            out["box_iou"] = boxIoU
            return out
        }
    }

    /// `find_grid`: corners (nil when nothing was found), confidence (0 when
    /// a validity check failed) and the diagnostics of the pass that counts.
    static func findGrid(
        _ image: CGImage, priorBox: CGRect?
    ) -> (corners: [CGPoint]?, confidence: Double, info: Info) {
        var (corners, info) = detect(image, priorBox: priorBox)
        var conf = corners != nil ? confidence(info, withPrior: priorBox != nil) : 0
        if priorBox == nil, let first = corners {
            // second pass on the found board alone: more pixels per square
            let xs = first.map { Double($0.x) }, ys = first.map { Double($0.y) }
            let x1 = max(xs.min()!, 0), y1 = max(ys.min()!, 0)
            let x2 = min(xs.max()!, Double(image.width)), y2 = min(ys.max()!, Double(image.height))
            if x2 - x1 > 40 && y2 - y1 > 40 {
                let box = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
                var (second, secondInfo) = detect(image, priorBox: box)
                let secondConf = second != nil ? confidence(secondInfo, withPrior: true) : 0
                secondInfo.pass = 2
                if secondPassWins(first: conf, second: secondConf) {
                    corners = second
                    info = secondInfo
                    conf = secondConf
                }
            }
        }
        return (corners, conf, info)
    }

    /// Which pass counts: the second whenever it is accepted, and also when
    /// neither is (its diagnostics are then the ones reported). A first pass
    /// that was accepted stands only if the second was not.
    static func secondPassWins(first: Double, second: Double) -> Bool {
        second > 0 || first == 0
    }

    /// `_detect`: one pass over the photo or over the prior box.
    static func detect(_ image: CGImage, priorBox: CGRect?) -> (corners: [CGPoint]?, info: Info) {
        var info = Info()
        guard let working = GrayImage.working(from: image, priorBox: priorBox) else {
            return (nil, info)
        }
        let corners = detect(
            working, imageWidth: image.width, imageHeight: image.height,
            priorBox: priorBox, info: &info)
        return (corners, info)
    }

    /// `_detect_inner`, from the working image on.
    static func detect(
        _ working: GrayImage.Working, imageWidth: Int, imageHeight: Int,
        priorBox: CGRect?, info: inout Info
    ) -> [CGPoint]? {
        let gray = working.gray
        let points = GridXCorners.find(in: gray)
        info.xCorners = points.count
        info.stage = .lattice
        guard let fit = GridLattice.fit(points) else { return nil }
        let nodes = fit.inlierNodes
        info.inliers = nodes.count
        info.stage = .window
        guard nodes.count >= 9 else { return nil }
        guard let window = GridWindow.choose(gray, fit) else { return nil }
        info.window = window
        info.stage = .refine

        let ox = window.ox, oy = window.oy
        info.borderNodes = [
            nodes.filter { $0.i == ox }.count, nodes.filter { $0.i == ox + 8 }.count,
            nodes.filter { $0.j == oy }.count, nodes.filter { $0.j == oy + 8 }.count,
        ].min()!
        let inner = nodes.filter { ox < $0.i && $0.i < ox + 8 && oy < $0.j && $0.j < oy + 8 }
        info.firstLineNodes = [
            inner.filter { $0.i == ox + 1 }.count, inner.filter { $0.i == ox + 7 }.count,
            inner.filter { $0.j == oy + 1 }.count, inner.filter { $0.j == oy + 7 }.count,
        ].min()!
        info.innerNodes = inner.count

        let start = Mat3.translation(-Double(ox), -Double(oy)) * fit.homography
        guard let refined = GridRefine.refine(gray, start, contrast: window.contrast) else {
            return nil
        }
        info.stats = refined.stats
        guard let check = GridWindow.finalCheck(gray, refined.hb) else {
            info.stage = .finalCheck
            return nil
        }
        info.check = check
        guard let toImage = refined.hb.inverse else { return nil }

        var corners = GridRefine.boardCorners.map { corner -> (x: Double, y: Double) in
            let p = toImage.apply(corner.x, corner.y)
            return (p.x / working.scale + Double(working.originX),
                    p.y / working.scale + Double(working.originY))
        }
        guard corners.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return nil }
        // clockwise in the image (y down): positive shoelace area
        var area = 0.0
        for i in 0..<4 {
            let a = corners[i], b = corners[(i + 1) % 4]
            area += a.x * b.y - b.x * a.y
        }
        area *= 0.5
        if area < 0 {
            corners.reverse()
        }
        info.area = abs(area)
        // convexity
        var crosses: [Double] = []
        for i in 0..<4 {
            let a = corners[i], b = corners[(i + 1) % 4], c = corners[(i + 2) % 4]
            crosses.append((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x))
        }
        info.convex = crosses.allSatisfy { $0 > 0 } || crosses.allSatisfy { $0 < 0 }
        if let box = priorBox {
            let bx1 = Double(box.minX), by1 = Double(box.minY)
            let bx2 = Double(box.maxX), by2 = Double(box.maxY)
            let qx1 = max(corners.map(\.x).min()!, 0), qy1 = max(corners.map(\.y).min()!, 0)
            let qx2 = min(corners.map(\.x).max()!, Double(imageWidth))
            let qy2 = min(corners.map(\.y).max()!, Double(imageHeight))
            let iw = max(0, min(bx2, qx2) - max(bx1, qx1))
            let ih = max(0, min(by2, qy2) - max(by1, qy1))
            let union = (bx2 - bx1) * (by2 - by1) + (qx2 - qx1) * (qy2 - qy1) - iw * ih
            info.boxIoU = union > 0 ? iw * ih / union : 0
        }
        info.stage = .done
        return corners.map { CGPoint(x: $0.x, y: $0.y) }
    }

    /// `confidence`: 0 when a validity check fails, otherwise a 0...1 score.
    static func confidence(_ info: Info, withPrior: Bool) -> Double {
        guard info.stage == .done, info.convex, let window = info.window, let check = info.check
        else { return 0 }
        var checks = [
            info.innerNodes >= 20,
            info.firstLineNodes >= 2,   // an inner line without x-corners: maybe shifted
            check.inside >= 0.7,
            check.ringMax <= 0.25,
            check.edgeRowsMin >= 0.25,
            window.margin >= 1,
            (info.stats.lineSupport ?? 0) >= 0.6,
            info.stats.borderSupport >= 0.16,
            (info.stats.shift ?? 1) <= 0.45,
        ]
        if withPrior {
            checks.append((info.boxIoU ?? 0) >= 0.5)
        }
        guard checks.allSatisfy({ $0 }) else { return 0 }
        return Double(min(info.innerNodes, 40)) / 40 * 0.3 + check.inside * 0.4
            + min(window.margin, 12) / 12 * 0.3
    }
}
