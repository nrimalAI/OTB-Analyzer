import Foundation

/// Step 3 of the grid fit: which 8×8 block of lattice cells is the board,
/// judged by the grey of the cells (`_cell_values`, `_good_cells`,
/// `choose_window` and `final_check` in `Training/grid_corners.py`).
enum GridWindow {

    typealias Node = GridLattice.Node

    /// A lattice rectangle of cells, each with four grey patches, row `j`
    /// then column `i`.
    struct Cells {
        let x0: Int
        let y0: Int
        let nx: Int
        let ny: Int
        /// [(j * nx + i) * 4 + patch]; NaN where the cell is off the photo or
        /// behind the horizon.
        var values: [Float]

        func patches(_ i: Int, _ j: Int) -> ArraySlice<Float> {
            values[((j * nx + i) * 4)..<((j * nx + i) * 4 + 4)]
        }

        /// 0 or 1: the colour class of a cell, by its lattice coordinates.
        func parity(_ i: Int, _ j: Int) -> Int {
            GridMath.mod(x0 + i + y0 + j, 2)
        }
    }

    /// `_warp`: the lattice rectangle [x0, x0+nx] × [y0, y0+ny] at `px`
    /// pixels per cell; -1 where the photo has nothing.
    static func warp(
        _ gray: GrayImage, _ h: Mat3, x0: Int, y0: Int, nx: Int, ny: Int, px: Int
    ) -> GrayImage? {
        let p = Double(px)
        let t = Mat3([p, 0, -Double(x0) * p, 0, p, -Double(y0) * p, 0, 0, 1])
        return gray.warped(by: t * h, width: nx * px, height: ny * px)
    }

    /// `_cell_values`: the grey of each lattice cell, as the medians of four
    /// patches near its corners (a piece usually stands on the middle and
    /// hides part of the cell).
    static func cellValues(
        _ gray: GrayImage, _ h: Mat3, x0: Int, y0: Int, nx: Int, ny: Int, px: Int = 16
    ) -> Cells? {
        guard nx > 0, ny > 0, let inverse = h.inverse,
              let warped = warp(gray, h, x0: x0, y0: y0, nx: nx, ny: ny, px: px)
        else { return nil }
        let centreDepth = inverse.depth(Double(x0) + Double(nx) / 2, Double(y0) + Double(ny) / 2)
        let sign: Double = centreDepth > 0 ? 1 : centreDepth < 0 ? -1 : centreDepth  // 0 or NaN stay

        let a0 = px / 8, a1 = px * 7 / 16, b0 = px * 9 / 16, b1 = px - px / 8
        let ranges = [(a0..<a1, a0..<a1), (a0..<a1, b0..<b1), (b0..<b1, a0..<a1), (b0..<b1, b0..<b1)]
        var cells = Cells(x0: x0, y0: y0, nx: nx, ny: ny,
                          values: [Float](repeating: .nan, count: nx * ny * 4))
        var patch: [Float] = []
        for j in 0..<ny {
            for i in 0..<nx {
                let depth = inverse.depth(Double(x0 + i) + 0.5, Double(y0 + j) + 0.5)
                guard depth * sign > 1e-9 else { continue }   // behind the horizon
                for (n, (ys, xs)) in ranges.enumerated() {
                    patch.removeAll(keepingCapacity: true)
                    var negative = 0
                    for y in ys {
                        for x in xs {
                            let value = warped[i * px + x, j * px + y]
                            patch.append(value)
                            if value < 0 { negative += 1 }
                        }
                    }
                    if Double(negative) / Double(patch.count) > 0.2 { continue }
                    cells.values[(j * nx + i) * 4 + n] = GridMath.median(patch)
                }
            }
        }
        return cells
    }

    /// `_two_levels`: two cluster centres of a sample (dark, light). Nil for
    /// an empty sample.
    static func twoLevels(_ values: [Float]) -> (dark: Double, light: Double)? {
        let v = values.filter { !$0.isNaN }.map(Double.init)
        guard !v.isEmpty else { return nil }
        let sorted = v.sorted()
        var lo = GridMath.percentile(sorted: sorted, 20)
        var hi = GridMath.percentile(sorted: sorted, 80)
        for _ in 0..<10 {
            let t = (lo + hi) / 2
            let below = v.filter { $0 <= t }
            let above = v.filter { $0 > t }
            if below.isEmpty || above.isEmpty { break }
            lo = GridMath.median(below)
            hi = GridMath.median(above)
        }
        return (lo, hi)
    }

    struct Judged {
        /// At least two of a cell's four patches are near the level of its colour.
        var good: [Bool]
        /// All four are.
        var strict: [Bool]
        /// None of the four patches is missing.
        var valid: [Bool]
        var contrast: Double
        /// The cell should be a light square.
        var light: [Bool]
    }

    /// `_good_cells`: which cells show their expected colour, given that the
    /// cells in `region` are board cells. A cell is good when at least two of
    /// its four patches are near the level of its colour (a piece may hide
    /// the rest), strict when all four are: that is asked of cells outside
    /// the board, where a frame can imitate half a dark square.
    static func goodCells(_ cells: Cells, region: [Bool]) -> Judged? {
        let count = cells.nx * cells.ny
        var valid = [Bool](repeating: false, count: count)
        for c in 0..<count {
            valid[c] = !cells.values[(c * 4)..<(c * 4 + 4)].contains { $0.isNaN }
        }
        let inRegion = (0..<count).filter { region[$0] && valid[$0] }
        guard inRegion.count >= 6 else { return nil }
        guard let levels = twoLevels(inRegion.flatMap { cells.values[($0 * 4)..<($0 * 4 + 4)] })
        else { return nil }
        let contrast = levels.light - levels.dark
        guard contrast >= 20 else { return nil }

        // which parity is the light one: vote over the region
        let middle = Float((levels.dark + levels.light) / 2)
        var lightShare = [[Double]](repeating: [], count: 2)
        for c in inRegion {
            let lightish = cells.values[(c * 4)..<(c * 4 + 4)].filter { $0 > middle }.count
            lightShare[cells.parity(c % cells.nx, c / cells.nx)].append(Double(lightish) / 4)
        }
        guard !lightShare[0].isEmpty, !lightShare[1].isEmpty else { return nil }
        let mean0 = lightShare[0].reduce(0, +) / Double(lightShare[0].count)
        let mean1 = lightShare[1].reduce(0, +) / Double(lightShare[1].count)
        let lightParity = mean0 > mean1 ? 0 : 1

        var judged = Judged(
            good: [Bool](repeating: false, count: count),
            strict: [Bool](repeating: false, count: count),
            valid: valid, contrast: contrast,
            light: [Bool](repeating: false, count: count))
        let tolerance = 0.4 * contrast
        for c in 0..<count {
            let isLight = cells.parity(c % cells.nx, c / cells.nx) == lightParity
            judged.light[c] = isLight
            let expected = isLight ? levels.light : levels.dark
            let near = cells.values[(c * 4)..<(c * 4 + 4)]
                .filter { abs(Double($0) - expected) < tolerance }.count
            judged.good[c] = near >= 2 && valid[c]
            judged.strict[c] = near >= 4 && valid[c]
        }
        return judged
    }

    /// `_alternating`: how many cells of a row take part in a light/dark
    /// alternation: twice the smaller of the good light and the good dark
    /// cells. A white margin or a plain floor matches one colour only and
    /// scores 0.
    static func alternating(_ cellIndices: [Int], good: [Bool], light: [Bool]) -> Int {
        var lightCount = 0, darkCount = 0
        for c in cellIndices where good[c] {
            if light[c] { lightCount += 1 } else { darkCount += 1 }
        }
        return 2 * min(lightCount, darkCount)
    }

    /// `_ring_sides`: the cells of the four rows just outside the 8×8 block
    /// whose first cell is column `a`, row `b` (inset 1: its four outermost
    /// rows instead), as indices into an `nx`-wide grid.
    static func ringSides(a: Int, b: Int, nx: Int, inset: Int = 0) -> [[Int]] {
        let loA = a - 1 + inset, hiA = a + 8 - inset, loB = b - 1 + inset, hiB = b + 8 - inset
        return [
            (a..<(a + 8)).map { loB * nx + $0 },
            (a..<(a + 8)).map { hiB * nx + $0 },
            (b..<(b + 8)).map { $0 * nx + loA },
            (b..<(b + 8)).map { $0 * nx + hiA },
        ]
    }

    struct Choice {
        var ox: Int
        var oy: Int
        var inside: Int
        var ring: Int
        var margin: Double
        var contrast: Double
    }

    /// `choose_window`: the 8×8 window of lattice cells that is the board.
    static func choose(_ gray: GrayImage, _ fit: GridLattice.Fit) -> Choice? {
        let nodes = fit.inlierNodes
        guard !nodes.isEmpty else { return nil }
        // Extent of the lattice lines that carry at least 3 x-corners: a stray
        // inlier beside the board must not pin the window.
        var lo = [0, 0], hi = [0, 0]
        for axis in 0..<2 {
            var counts: [Int: Int] = [:]
            for node in nodes {
                counts[axis == 0 ? node.i : node.j, default: 0] += 1
            }
            let full = counts.filter { $0.value >= 3 }.map(\.key)
            guard full.count >= 3 else { return nil }
            lo[axis] = full.min()!
            hi[axis] = full.max()!
        }
        // window starts to try on each axis: every 8-wide window that holds
        // those lines or, when they span more than 8 (lines of x-corners on a
        // frame or on the floor), every window among them
        // (one more each way once the span exceeds the 7 inner lines: a line
        // of x-corners beside the board then cannot exclude the true window)
        func starts(_ axis: Int) -> ClosedRange<Int> {
            let extra = hi[axis] - lo[axis] > 6 ? 1 : 0
            return (min(lo[axis], hi[axis] - 8) - extra)...(max(lo[axis], hi[axis] - 8) + extra)
        }
        let sx = starts(0), sy = starts(1)
        let x0 = sx.lowerBound - 1, y0 = sy.lowerBound - 1
        let nx = sx.upperBound + 9 - x0, ny = sy.upperBound + 9 - y0
        guard nx <= 30, ny <= 30 else { return nil }
        guard let cells = cellValues(gray, fit.homography, x0: x0, y0: y0, nx: nx, ny: ny)
        else { return nil }

        // cells with x-corners on all four corners are board cells for sure
        var core = [Bool](repeating: false, count: nx * ny)
        for node in nodes {
            guard nodes.contains(Node(i: node.i + 1, j: node.j)),
                  nodes.contains(Node(i: node.i, j: node.j + 1)),
                  nodes.contains(Node(i: node.i + 1, j: node.j + 1))
            else { continue }
            // The Python indexes a NumPy array here, where an index below 0
            // counts from the end and one out of range is an error that
            // rejects the photo. Stray nodes far from the board get there.
            guard let row = pythonIndex(node.j - y0, ny), let column = pythonIndex(node.i - x0, nx)
            else { return nil }
            core[row * nx + column] = true
        }
        guard let judged = goodCells(cells, region: core) else { return nil }

        var candidates: [(score: Double, ox: Int, oy: Int, inside: Int, ring: Int)] = []
        for ox in sx {
            for oy in sy {
                let a = ox - x0, b = oy - y0
                var invalid = 0, inside = 0
                for j in b..<(b + 8) {
                    for i in a..<(a + 8) {
                        if !judged.valid[j * nx + i] { invalid += 1 }
                        if judged.good[j * nx + i] { inside += 1 }
                    }
                }
                if invalid > 8 { continue }
                let ring = ringSides(a: a, b: b, nx: nx)
                    .map { alternating($0, good: judged.strict, light: judged.light) }
                    .reduce(0, +)
                let within = nodes.filter {
                    $0.i >= ox && $0.i <= ox + 8 && $0.j >= oy && $0.j <= oy + 8
                }.count
                let strictlyWithin = nodes.filter {
                    $0.i > ox && $0.i < ox + 8 && $0.j > oy && $0.j < oy + 8
                }.count
                let score = Double(inside - ring) + 0.5 * Double(within) + 0.5 * Double(strictlyWithin)
                candidates.append((score, ox, oy, inside, ring))
            }
        }
        guard !candidates.isEmpty else { return nil }
        // Best first, as Python sorts the tuples in reverse.
        candidates.sort {
            if $0.score != $1.score { return $0.score > $1.score }
            if $0.ox != $1.ox { return $0.ox > $1.ox }
            if $0.oy != $1.oy { return $0.oy > $1.oy }
            if $0.inside != $1.inside { return $0.inside > $1.inside }
            return $0.ring > $1.ring
        }
        let margin = candidates.count > 1 ? candidates[0].score - candidates[1].score : 99
        let top = candidates[0]
        return Choice(ox: top.ox, oy: top.oy, inside: top.inside, ring: top.ring,
                      margin: margin, contrast: judged.contrast)
    }

    private static func pythonIndex(_ index: Int, _ count: Int) -> Int? {
        if index >= 0 { return index < count ? index : nil }
        return index >= -count ? index + count : nil
    }

    struct Check {
        /// Share of the 64 cells that show their colour.
        var inside: Double
        /// How much the best side of the ring outside still looks like more
        /// of the pattern.
        var ringMax: Double
        /// How much the weakest of the board's four outermost rows alternates.
        var edgeRowsMin: Double
        var contrast: Double
    }

    /// `final_check`: the cells of the final frame `hb` (working image ->
    /// [0,8]²).
    static func finalCheck(_ gray: GrayImage, _ hb: Mat3) -> Check? {
        guard let cells = cellValues(gray, hb, x0: -1, y0: -1, nx: 10, ny: 10, px: 20)
        else { return nil }
        var inside = [Bool](repeating: false, count: 100)
        for j in 1...8 {
            for i in 1...8 {
                inside[j * 10 + i] = true
            }
        }
        guard let judged = goodCells(cells, region: inside) else { return nil }
        let insideCells = (0..<100).filter { inside[$0] }
        guard insideCells.filter({ judged.valid[$0] }).count >= 56 else { return nil }
        // a board shifted by one shows as one side of the ring alternating well
        let sides = ringSides(a: 1, b: 1, nx: 10)
            .map { Double(alternating($0, good: judged.strict, light: judged.light)) / 8 }
        let rows = ringSides(a: 1, b: 1, nx: 10, inset: 1)
            .map { Double(alternating($0, good: judged.good, light: judged.light)) / 8 }
        return Check(
            inside: Double(insideCells.filter { judged.good[$0] }.count) / 64,
            ringMax: sides.max()!, edgeRowsMin: rows.min()!, contrast: judged.contrast)
    }
}
