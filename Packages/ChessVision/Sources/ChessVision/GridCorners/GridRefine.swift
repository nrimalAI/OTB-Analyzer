import Foundation

/// Step 4 of the grid fit: edge points are searched along all 18 grid lines
/// of the chosen window (the outer ones included) and the homography is
/// refitted to them (`_line_points`, `_fit_lines` and `refine` in
/// `Training/grid_corners.py`).
enum GridRefine {

    /// The board frame's corners in (a1, a8, h8, h1) order (`BOARD_CORNERS`).
    static let boardCorners: [(x: Double, y: Double)] = [(0, 0), (0, 8), (8, 8), (8, 0)]

    /// An edge point found on a grid line.
    struct LinePoint {
        /// In the working image.
        var x: Double
        var y: Double
        /// 0: the line u = k; 1: the line v = k.
        var axis: Int
        var k: Int
        /// One of the four outer lines.
        var border: Bool
    }

    /// `_line_points`: edge points along the 18 grid lines of the board frame
    /// `hb` (working image -> [0,8]²).
    static func linePoints(
        _ gray: GrayImage, _ hb: Mat3, contrast: Double, px: Int = 40, reach: Double = 0.3
    ) -> [LinePoint]? {
        let m = 1  // margin cells
        let n = 8 + 2 * m
        guard let raw = GridWindow.warp(gray, hb, x0: -m, y0: -m, nx: n, ny: n, px: px),
              let inverse = hb.inverse
        else { return nil }
        let warped = raw.gaussianBlurred(sigma: 1.2)
        let overallMedian = GridMath.median(warped.pixels.filter { $0 >= 0 })
        let half = Int(reach * Double(px))
        let span = px / 5          // averaging along the line
        let width = 2 * half + 5
        var out: [LinePoint] = []

        for axis in 0..<2 {     // 0: lines u = k (vertical in the warp), 1: lines v = k
            let image = axis == 0 ? warped : warped.transposed
            for k in 0..<9 {
                let c = (k + m) * px
                let border = k == 0 || k == 8
                // Steps found, by kind, in the order the kinds first appear:
                // on a border line the kind is the colour of the cell inside
                // (0 dark, 1 light); on an inner line there is one kind, 2.
                var kinds: [Int] = []
                var found: [Int: [(row: Int, offset: Double)]] = [:]
                func add(_ kind: Int, _ row: Int, _ offset: Double) {
                    if found[kind] == nil {
                        kinds.append(kind)
                        found[kind] = []
                    }
                    found[kind]!.append((row, offset))
                }

                for j in 0..<8 {
                    for f in [0.3, 0.5, 0.7] {
                        let row = Int((Double(j) + f + Double(m)) * Double(px))
                        // The strip across the line, averaged along it.
                        var profile = [Float](repeating: 0, count: width)
                        var lowest = Float.infinity
                        for y in (row - span)...(row + span) {
                            for x in 0..<width {
                                let value = image[c - half - 2 + x, y]
                                profile[x] += value
                                lowest = min(lowest, value)
                            }
                        }
                        if lowest < 0 { continue }
                        let rows = Float(2 * span + 1)
                        let prof = profile.map { Double($0 / rows) }

                        if border {
                            // walk outwards from inside the cell; first clear step
                            let seq = k == 8 ? prof : Array(prof.reversed())
                            let ref = GridMath.median(Array(seq[..<max(3, half / 3)]))
                            guard let hit = seq.firstIndex(where: { abs($0 - ref) > 0.35 * contrast })
                            else { continue }
                            let steps = zip(seq.dropFirst(), seq).map { abs($0 - $1) }
                            let from = max(hit - 3, 0), to = min(hit + 1, steps.count)
                            var e = from
                            for index in from..<to where steps[index] > steps[e] {
                                e = index
                            }
                            // e counted from the inside end
                            let edge = Double(e) + 0.5
                            let position = k == 8 ? edge : Double(prof.count - 1) - edge
                            let kind = ref > Double(overallMedian) ? 1 : 0
                            add(kind, row, position - Double(half + 2))
                        } else {
                            let gradient = zip(prof.dropFirst(), prof).map { $0 - $1 }
                            let magnitude = gradient.map(abs)
                            var e = 0
                            for index in 0..<magnitude.count where magnitude[index] > magnitude[e] {
                                e = index
                            }
                            if magnitude[e] < 0.12 * contrast { continue }
                            // parabola for sub-pixel
                            guard e > 0, e < magnitude.count - 1 else { continue }
                            let denominator = magnitude[e - 1] - 2 * magnitude[e] + magnitude[e + 1]
                            let sub = denominator != 0
                                ? 0.5 * (magnitude[e - 1] - magnitude[e + 1]) / denominator : 0
                            add(2, row, Double(e) + 0.5 + sub - Double(half + 2))
                        }
                    }
                }

                var items: [(row: Int, offset: Double)] = []
                if border && !kinds.isEmpty {
                    // the true edge is the innermost step: keep the cell colour
                    // whose steps sit furthest inside
                    var best: (kind: Int, inner: Double)?
                    for kind in kinds where found[kind]!.count >= 4 {
                        let middle = GridMath.median(found[kind]!.map(\.offset))
                        let inner = k == 0 ? middle : -middle
                        // (a NaN never wins, as in Python's max)
                        if best == nil || inner > best!.inner {
                            best = (kind, inner)
                        }
                    }
                    guard let best else { continue }
                    items = found[best.kind]!
                } else {
                    items = found[2] ?? []
                }
                for item in items {
                    let u = (Double(c) + item.offset) / Double(px) - Double(m)
                    let v = Double(item.row) / Double(px) - Double(m)
                    let p = axis == 0 ? inverse.apply(u, v) : inverse.apply(v, u)
                    out.append(LinePoint(x: p.x, y: p.y, axis: axis, k: k, border: border))
                }
            }
        }
        return out
    }

    /// `_fit_lines`: the homography working image -> board from the
    /// constraint that each point lies on its grid line, robustly (six rounds
    /// of reweighting). Returns it with each point's distance from its line,
    /// in squares.
    static func fitLines(_ points: [LinePoint], rounds: Int = 6) -> (h: Mat3, residuals: [Double])? {
        let count = points.count
        guard count > 0 else { return nil }
        let n = Double(count)
        // normalise image coordinates
        let meanX = points.reduce(0) { $0 + $1.x } / n
        let meanY = points.reduce(0) { $0 + $1.y } / n
        // (the deviation is taken over x and y together, as `P.std()` does)
        let meanAll = points.reduce(0) { $0 + $1.x + $1.y } / (2 * n)
        let variance = points.reduce(0) {
            $0 + ($1.x - meanAll) * ($1.x - meanAll) + ($1.y - meanAll) * ($1.y - meanAll)
        } / (2 * n)
        let sd = variance.squareRoot() + 1e-9
        let normalise = Mat3([1 / sd, 0, -meanX / sd, 0, 1 / sd, -meanY / sd, 0, 0, 1])
        // S takes the board [0,8]² onto [-1,1]²; this is its inverse.
        let unscale = Mat3([4, 0, 4, 0, 4, 4, 0, 0, 1])

        // One row of the 9-column system per point.
        var rows = [[Double]](repeating: [Double](repeating: 0, count: 9), count: count)
        for (index, p) in points.enumerated() {
            let ph = [(p.x - meanX) / sd, (p.y - meanY) / sd, 1]
            let kn = (Double(p.k) - 4) / 4
            for t in 0..<3 {
                rows[index][3 * p.axis + t] = ph[t]
                rows[index][6 + t] = -kn * ph[t]
            }
        }

        var weights = [Double](repeating: 1, count: count)
        var h = Mat3.identity
        var residuals = [Double](repeating: 0, count: count)
        for _ in 0..<rounds {
            // The right singular vector of the smallest singular value of the
            // weighted system: the eigenvector of its 9×9 normal matrix.
            var normal = [Double](repeating: 0, count: 81)
            for index in 0..<count {
                let w2 = weights[index] * weights[index]
                let row = rows[index]
                for a in 0..<9 where row[a] != 0 {
                    for b in a..<9 where row[b] != 0 {
                        normal[a * 9 + b] += w2 * row[a] * row[b]
                    }
                }
            }
            for a in 0..<9 {
                for b in 0..<a {
                    normal[a * 9 + b] = normal[b * 9 + a]
                }
            }
            guard let eigen = GridMath.symmetricEigen(normal, n: 9) else { return nil }
            let candidate = unscale * Mat3(eigen.vectors[0]) * normalise
            guard candidate.m[8] != 0 else { return nil }
            h = candidate.normalized
            guard h.isFinite else { return nil }
            for (index, p) in points.enumerated() {
                let uv = h.apply(p.x, p.y)
                residuals[index] = (p.axis == 0 ? uv.x : uv.y) - Double(p.k)
            }
            let s = max(1.4826 * GridMath.median(residuals.map(abs)), 0.01)
            for index in 0..<count {
                let r = residuals[index] / (2 * s)
                weights[index] = 1 / (1 + r * r)
            }
        }
        return (h, residuals)
    }

    struct Stats {
        var linePoints = 0
        /// The weakest outer line's share of the 24 places where an edge was
        /// looked for.
        var borderSupport = 0.0
        /// How far the refit moved a board corner, in squares.
        var shift: Double?
        var lineRMS: Double?
        /// Share of the 432 places whose edge lies within 0.04 squares of
        /// its line.
        var lineSupport: Double?
    }

    /// `refine`: two rounds of edge search and refit; a refit that moves a
    /// corner by more than 0.45 squares is refused.
    static func refine(
        _ gray: GrayImage, _ start: Mat3, contrast: Double, rounds: Int = 2
    ) -> (hb: Mat3, stats: Stats)? {
        var hb = start
        var stats = Stats()
        for _ in 0..<rounds {
            guard let points = linePoints(gray, hb, contrast: contrast) else { return nil }
            var borderCounts: [Int] = []
            for axis in 0..<2 {
                for k in [0, 8] {
                    borderCounts.append(points.filter { $0.border && $0.axis == axis && $0.k == k }.count)
                }
            }
            stats = Stats(linePoints: points.count,
                          borderSupport: Double(borderCounts.min()!) / 24.0)
            if points.count < 60 { return (hb, stats) }
            guard let fitted = fitLines(points), let fittedInverse = fitted.h.inverse
            else { return nil }
            // refuse a refinement that runs away
            var shift = 0.0
            for corner in boardCorners {
                let inImage = fittedInverse.apply(corner.x, corner.y)
                let back = hb.apply(inImage.x, inImage.y)
                let moved = max(abs(back.x - corner.x), abs(back.y - corner.y))
                // (`np.max` gives NaN as soon as one value is NaN)
                shift = moved.isNaN || shift.isNaN ? .nan : max(shift, moved)
            }
            stats.shift = shift
            stats.lineRMS = GridMath.median(fitted.residuals.map { $0 * $0 }).squareRoot()
            stats.lineSupport = Double(fitted.residuals.filter { abs($0) < 0.04 }.count) / 432.0
            if shift > 0.45 { return (hb, stats) }
            hb = fitted.h
        }
        return (hb, stats)
    }
}
