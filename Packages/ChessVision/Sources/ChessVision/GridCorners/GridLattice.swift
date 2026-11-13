import Foundation

/// Step 2 of the grid fit: the homography that puts the most x-corners on
/// the nodes of an integer lattice (`fit_lattice` in
/// `Training/grid_corners.py`).
enum GridLattice {

    struct Node: Hashable {
        var i: Int
        var j: Int
    }

    struct Fit {
        /// Working image -> lattice.
        var homography: Mat3
        /// Which x-corners sit on a node.
        var inlier: [Bool]
        /// The node nearest each x-corner (meaningful for inliers).
        var node: [Node]

        /// The distinct nodes that carry an x-corner.
        var inlierNodes: Set<Node> {
            Set(zip(inlier, node).compactMap { $0 ? $1 : nil })
        }
    }

    /// `_inliers`: the x-corners within `tolerance` of a node, and each
    /// x-corner's nearest node.
    static func inliers(
        _ h: Mat3, _ points: [GridPoint], tolerance: Double
    ) -> (inlier: [Bool], node: [Node]) {
        var inlier = [Bool](repeating: false, count: points.count)
        var node = [Node](repeating: Node(i: 0, j: 0), count: points.count)
        for (index, p) in points.enumerated() {
            let uv = h.apply(Double(p.x), Double(p.y))
            let ru = uv.x.rounded(.toNearestOrEven)
            let rv = uv.y.rounded(.toNearestOrEven)
            // A point on the horizon is no inlier (in NumPy the comparisons
            // with NaN are false).
            inlier[index] = uv.x.isFinite && uv.y.isFinite
                && max(abs(uv.x - ru), abs(uv.y - rv)) < tolerance
                && max(abs(uv.x), abs(uv.y)) < 12
            node[index] = Node(i: GridMath.rint(uv.x), j: GridMath.rint(uv.y))
        }
        return (inlier, node)
    }

    /// How many distinct nodes carry an x-corner within 0.15 of them: what
    /// `inliers` would count, for the thousands of hypotheses, without
    /// allocating. Inlier nodes lie within 12 of the origin, so a 25×25
    /// table holds them.
    private static func distinctInlierNodes(
        _ h: Mat3, _ xs: [Double], _ ys: [Double], _ seen: inout [Bool]
    ) -> Int {
        let m0 = h.m[0], m1 = h.m[1], m2 = h.m[2], m3 = h.m[3], m4 = h.m[4], m5 = h.m[5]
        let m6 = h.m[6], m7 = h.m[7], m8 = h.m[8]
        let n = xs.count
        var count = 0
        seen.withUnsafeMutableBufferPointer { seen in
            var i = 0
            while i < 625 { seen[i] = false; i &+= 1 }
            xs.withUnsafeBufferPointer { xs in
                ys.withUnsafeBufferPointer { ys in
                    var k = 0
                    while k < n {
                        let x = xs[k], y = ys[k]
                        let w = m6 * x + m7 * y + m8
                        let u = (m0 * x + m1 * y + m2) / w
                        let v = (m3 * x + m4 * y + m5) / w
                        let ru = u.rounded(.toNearestOrEven), rv = v.rounded(.toNearestOrEven)
                        if u.isFinite && v.isFinite
                            && max(abs(u - ru), abs(v - rv)) < 0.15 && max(abs(u), abs(v)) < 12
                        {
                            let index = (Int(ru) + 12) * 25 + Int(rv) + 12
                            if !seen[index] {
                                seen[index] = true
                                count &+= 1
                            }
                        }
                        k &+= 1
                    }
                }
            }
        }
        return count
    }

    private static func nodes(_ h: Mat3, _ points: [GridPoint]) -> [Node] {
        points.map {
            let uv = h.apply(Double($0.x), Double($0.y))
            return Node(i: GridMath.rint(uv.x), j: GridMath.rint(uv.y))
        }
    }

    /// The 2×2 integer matrices of determinant 1 with entries in -2...2
    /// (`_UNIMODULAR`), in the Python's order.
    static let unimodular: [(a: Int, b: Int, c: Int, d: Int)] = {
        var out: [(Int, Int, Int, Int)] = []
        for a in -2...2 {
            for b in -2...2 {
                for c in -2...2 {
                    for d in -2...2 where a * d - b * c == 1 {
                        out.append((a, b, c, d))
                    }
                }
            }
        }
        return out
    }()

    static func fit(_ points: [GridPoint]) -> Fit? {
        let n = points.count
        guard n >= 9 else { return nil }

        // The seven nearest neighbours of each x-corner.
        let neighbourCount = min(7, n)
        var neighbours = [[Int]](repeating: [], count: n)
        for i in 0..<n {
            var distances = [(Float, Int)]()
            distances.reserveCapacity(n)
            for j in 0..<n {
                let dx = points[i].x - points[j].x
                let dy = points[i].y - points[j].y
                distances.append((i == j ? 1e9 : (dx * dx + dy * dy).squareRoot(), j))
            }
            distances.sort { $0.0 != $1.0 ? $0.0 < $1.0 : $0.1 < $1.1 }
            neighbours[i] = distances.prefix(neighbourCount).map(\.1)
        }

        let fx = points.map(\.x), fy = points.map(\.y)
        let xs = fx.map(Double.init), ys = fy.map(Double.init)
        var seen = [Bool](repeating: false, count: 25 * 25)

        // Every small parallelogram of neighbours proposes a lattice.
        let unit: [(x: Double, y: Double)] = [(0, 0), (1, 0), (1, 1), (0, 1)]
        var hypotheses: [(count: Int, h: Mat3)] = []
        for i in 0..<n {
            let p = points[i]
            for a in 0..<neighbourCount {
                for b in (a + 1)..<neighbourCount {
                    let qa = points[neighbours[i][a]]
                    let qb = points[neighbours[i][b]]
                    let va = (x: qa.x - p.x, y: qa.y - p.y)
                    let vb = (x: qb.x - p.x, y: qb.y - p.y)
                    let la = (va.x * va.x + va.y * va.y).squareRoot()
                    let lb = (vb.x * vb.x + vb.y * vb.y).squareRoot()
                    if abs(va.x * vb.y - va.y * vb.x) < 0.35 * la * lb { continue }
                    if max(la, lb) > 6 * min(la, lb) { continue }
                    let t = (x: p.x + va.x + vb.x, y: p.y + va.y + vb.y)
                    var nearest = 0
                    var nearestDistance = Float.infinity
                    fx.withUnsafeBufferPointer { fx in
                        fy.withUnsafeBufferPointer { fy in
                            var k = 0
                            while k < n {
                                let dx = fx[k] - t.x
                                let dy = fy[k] - t.y
                                let d = (dx * dx + dy * dy).squareRoot()
                                if d < nearestDistance {
                                    nearestDistance = d
                                    nearest = k
                                }
                                k &+= 1
                            }
                        }
                    }
                    if nearestDistance > 0.3 * min(la, lb)
                        || nearest == i || nearest == neighbours[i][a] || nearest == neighbours[i][b]
                    {
                        continue
                    }
                    let quad: [(x: Double, y: Double)] = [
                        (Double(p.x), Double(p.y)), (Double(qa.x), Double(qa.y)),
                        (Double(points[nearest].x), Double(points[nearest].y)),
                        (Double(qb.x), Double(qb.y)),
                    ]
                    guard let h = GridMath.homography(from: quad, to: unit) else { continue }
                    // distinct nodes, so a collapsed map cannot win
                    hypotheses.append((distinctInlierNodes(h, xs, ys, &seen), h))
                }
            }
        }
        guard !hypotheses.isEmpty else { return nil }
        // Most nodes first; equal ones in the order they were proposed.
        let ranked = hypotheses.enumerated()
            .sorted { $0.element.count != $1.element.count
                ? $0.element.count > $1.element.count : $0.offset < $1.offset }
            .prefix(12)
            .map(\.element)

        // Refit each of the best twelve on its inliers, tightening the tolerance.
        var best: (unique: Int, h: Mat3, inlier: [Bool], node: [Node])?
        for hypothesis in ranked {
            var h = hypothesis.h
            var found = inliers(h, points, tolerance: 0.15)
            for tolerance in [0.18, 0.18, 0.15, 0.12] {
                if found.inlier.filter({ $0 }).count < 6 { break }
                var src: [(x: Double, y: Double)] = []
                var dst: [(x: Double, y: Double)] = []
                for k in 0..<n where found.inlier[k] {
                    src.append((Double(points[k].x), Double(points[k].y)))
                    dst.append((Double(found.node[k].i), Double(found.node[k].j)))
                }
                guard let refit = GridMath.leastSquaresHomography(from: src, to: dst) else { break }
                h = refit
                found = inliers(h, points, tolerance: tolerance)
            }
            // one x-corner per node
            if found.inlier.filter({ $0 }).count >= 6 {
                let unique = Set(zip(found.inlier, found.node).compactMap { $0 ? $1 : nil }).count
                if best == nil || unique > best!.unique {
                    best = (unique, h, found.inlier, found.node)
                }
            }
        }
        guard let chosen = best else { return nil }
        var h = chosen.h
        var inlier = chosen.inlier
        var node = chosen.node

        // The winning lattice may be finer than the board's (stray points on
        // square centres or edge midpoints propose it, and every true x-corner
        // still lands on a node). Then nearly all inliers share one coset of a
        // sublattice of index 2: move to that sublattice.
        for _ in 0..<3 {
            let indices = (0..<n).filter { inlier[$0] }
            guard !indices.isEmpty else { return nil }
            let ij = indices.map { node[$0] }
            // (pieces stand on square centres and add points there, so the
            // diagonal case is accepted on a two-thirds majority)
            let cases: [(m: [Double], key: [Int], limit: Double, offsetOnI: Bool)] = [
                ([0.5, 0, 0, 1], ij.map { GridMath.mod($0.i, 2) }, 0.2, true),
                ([1, 0, 0, 0.5], ij.map { GridMath.mod($0.j, 2) }, 0.2, false),
                ([0.5, 0.5, -0.5, 0.5], ij.map { GridMath.mod($0.i + $0.j, 2) }, 0.34, true),
            ]
            var changed = false
            for item in cases {
                let share = Double(item.key.reduce(0, +)) / Double(item.key.count)
                if item.limit < share && share < 1 - item.limit { continue }
                let coset = share >= 0.5 ? 1 : 0          // the coset the nodes are on
                var keep = [Bool](repeating: false, count: n)
                for (position, index) in indices.enumerated() where item.key[position] == coset {
                    keep[index] = true
                }
                let m3 = Mat3([item.m[0], item.m[1], 0, item.m[2], item.m[3], 0, 0, 0, 1])
                let t3 = item.offsetOnI
                    ? Mat3.translation(-Double(coset), 0) : Mat3.translation(0, -Double(coset))
                h = m3 * t3 * h
                inlier = keep
                node = nodes(h, points)
                changed = true
                break
            }
            if !changed { break }
        }

        // The winning basis may be a sheared one (same lattice, slanted axes):
        // take the unimodular change of basis that makes the inliers' bounding
        // box smallest.
        let ij = (0..<n).filter { inlier[$0] }.map { node[$0] }
        guard !ij.isEmpty else { return nil }
        func spread(_ u: (a: Int, b: Int, c: Int, d: Int)) -> Double {
            // robust to a few stray inliers off the board
            let first = ij.map { Double(u.a * $0.i + u.b * $0.j) }.sorted()
            let second = ij.map { Double(u.c * $0.i + u.d * $0.j) }.sorted()
            return (GridMath.percentile(sorted: first, 92) - GridMath.percentile(sorted: first, 8))
                + (GridMath.percentile(sorted: second, 92) - GridMath.percentile(sorted: second, 8))
        }
        var bestBasis = (a: 1, b: 0, c: 0, d: 1)
        var bestSpan = spread(bestBasis)
        for u in unimodular {
            let span = spread(u)
            if span < bestSpan - 1e-9 {
                bestBasis = u
                bestSpan = span
            }
        }
        let u3 = Mat3([
            Double(bestBasis.a), Double(bestBasis.b), 0,
            Double(bestBasis.c), Double(bestBasis.d), 0,
            0, 0, 1,
        ])
        h = u3 * h
        node = node.map {
            Node(i: bestBasis.a * $0.i + bestBasis.b * $0.j, j: bestBasis.c * $0.i + bestBasis.d * $0.j)
        }
        return Fit(homography: h, inlier: inlier, node: node)
    }
}
