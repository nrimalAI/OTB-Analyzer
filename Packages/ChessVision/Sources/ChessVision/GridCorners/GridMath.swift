import Foundation

// The small amount of linear algebra and statistics the grid fit needs,
// written out so that the package depends on nothing. Each routine stands in
// for the NumPy or OpenCV call named in its comment (`Training/grid_corners.py`).

/// A 3×3 matrix, row-major: the homographies of the grid fit.
struct Mat3: Equatable {
    var m: [Double]

    init(_ m: [Double]) {
        precondition(m.count == 9)
        self.m = m
    }

    static let identity = Mat3([1, 0, 0, 0, 1, 0, 0, 0, 1])

    static func translation(_ tx: Double, _ ty: Double) -> Mat3 {
        Mat3([1, 0, tx, 0, 1, ty, 0, 0, 1])
    }

    static func * (a: Mat3, b: Mat3) -> Mat3 {
        var out = [Double](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                var sum = 0.0
                for k in 0..<3 {
                    sum += a.m[i * 3 + k] * b.m[k * 3 + j]
                }
                out[i * 3 + j] = sum
            }
        }
        return Mat3(out)
    }

    var isFinite: Bool { m.allSatisfy(\.isFinite) }

    /// `np.linalg.inv`; nil where NumPy raises (a singular matrix) and also
    /// for a matrix that is not finite, which Python carries on with only to
    /// fail a later check.
    var inverse: Mat3? {
        guard isFinite else { return nil }
        let a = m
        let c00 = a[4] * a[8] - a[5] * a[7]
        let c01 = a[5] * a[6] - a[3] * a[8]
        let c02 = a[3] * a[7] - a[4] * a[6]
        let det = a[0] * c00 + a[1] * c01 + a[2] * c02
        guard det != 0, det.isFinite else { return nil }
        let d = 1 / det
        let out = Mat3([
            c00 * d, (a[2] * a[7] - a[1] * a[8]) * d, (a[1] * a[5] - a[2] * a[4]) * d,
            c01 * d, (a[0] * a[8] - a[2] * a[6]) * d, (a[2] * a[3] - a[0] * a[5]) * d,
            c02 * d, (a[1] * a[6] - a[0] * a[7]) * d, (a[0] * a[4] - a[1] * a[3]) * d,
        ])
        return out.isFinite ? out : nil
    }

    /// The point (x, y) taken through the homography (`_apply`). Not finite
    /// when the point lies on the horizon.
    @inline(__always)
    func apply(_ x: Double, _ y: Double) -> (x: Double, y: Double) {
        let w = m[6] * x + m[7] * y + m[8]
        return ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }

    /// The third homogeneous coordinate of (x, y) taken through the matrix.
    func depth(_ x: Double, _ y: Double) -> Double {
        m[6] * x + m[7] * y + m[8]
    }

    /// Scaled so that the last element is 1.
    var normalized: Mat3 { Mat3(m.map { $0 / m[8] }) }
}

enum GridMath {

    /// `np.rint` followed by `.astype(int)`: to the nearest integer, halves
    /// to the even one. A value that is not finite gives 0; Python gives
    /// rubbish there and never reads it.
    @inline(__always)
    static func rint(_ value: Double) -> Int {
        guard value.isFinite, abs(value) < 1e15 else { return 0 }
        return Int(value.rounded(.toNearestOrEven))
    }

    /// Python's `%` for a positive modulus: never negative.
    @inline(__always)
    static func mod(_ a: Int, _ n: Int) -> Int {
        let r = a % n
        return r < 0 ? r + n : r
    }

    /// `np.percentile` (linear interpolation) of an unsorted sample; nil for
    /// an empty one, where NumPy raises.
    static func percentile(_ values: [Double], _ q: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        return percentile(sorted: values.sorted(), q)
    }

    static func percentile(sorted: [Double], _ q: Double) -> Double {
        let position = q / 100 * Double(sorted.count - 1)
        let low = Int(position.rounded(.down))
        let high = min(low + 1, sorted.count - 1)
        let t = position - Double(low)
        let a = sorted[low]
        let b = sorted[high]
        // NumPy's lerp, which is exact at both ends.
        return t >= 0.5 ? b - (b - a) * (1 - t) : a + (b - a) * t
    }

    /// `np.median`; NaN for an empty sample, as NumPy.
    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        let n = sorted.count
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }

    /// `np.median` of float32 values, in float32.
    static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        let n = sorted.count
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }

    /// Solves A·x = b by Gaussian elimination with partial pivoting; nil for
    /// a singular A.
    static func solve(_ a: [[Double]], _ b: [Double]) -> [Double]? {
        let n = b.count
        var m = a
        var x = b
        for col in 0..<n {
            var pivot = col
            for row in (col + 1)..<n where abs(m[row][col]) > abs(m[pivot][col]) {
                pivot = row
            }
            guard abs(m[pivot][col]) > 1e-300, m[pivot][col].isFinite else { return nil }
            if pivot != col {
                m.swapAt(pivot, col)
                x.swapAt(pivot, col)
            }
            for row in (col + 1)..<n {
                let factor = m[row][col] / m[col][col]
                if factor == 0 { continue }
                for k in col..<n {
                    m[row][k] -= factor * m[col][k]
                }
                x[row] -= factor * x[col]
            }
        }
        for col in stride(from: n - 1, through: 0, by: -1) {
            var sum = x[col]
            for k in (col + 1)..<n {
                sum -= m[col][k] * x[k]
            }
            x[col] = sum / m[col][col]
        }
        return x.allSatisfy(\.isFinite) ? x : nil
    }

    /// Eigenvalues (ascending) and eigenvectors of a symmetric n×n matrix
    /// given row-major, by cyclic Jacobi rotations. `vectors[k]` belongs to
    /// `values[k]`. Nil when the matrix is not finite.
    static func symmetricEigen(_ matrix: [Double], n: Int) -> (values: [Double], vectors: [[Double]])? {
        guard matrix.allSatisfy(\.isFinite) else { return nil }
        var a = matrix
        var v = [Double](repeating: 0, count: n * n)
        for i in 0..<n { v[i * n + i] = 1 }

        for _ in 0..<64 {
            var off = 0.0
            var diagonal = 0.0
            for p in 0..<n {
                diagonal += a[p * n + p] * a[p * n + p]
                for q in (p + 1)..<n {
                    off += a[p * n + q] * a[p * n + q]
                }
            }
            if off <= 1e-30 * diagonal || off == 0 { break }
            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    let apq = a[p * n + q]
                    if apq == 0 { continue }
                    let theta = (a[q * n + q] - a[p * n + p]) / (2 * apq)
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c
                    for k in 0..<n {
                        let akp = a[k * n + p]
                        let akq = a[k * n + q]
                        a[k * n + p] = c * akp - s * akq
                        a[k * n + q] = s * akp + c * akq
                    }
                    for k in 0..<n {
                        let apk = a[p * n + k]
                        let aqk = a[q * n + k]
                        a[p * n + k] = c * apk - s * aqk
                        a[q * n + k] = s * apk + c * aqk
                    }
                    for k in 0..<n {
                        let vkp = v[k * n + p]
                        let vkq = v[k * n + q]
                        v[k * n + p] = c * vkp - s * vkq
                        v[k * n + q] = s * vkp + c * vkq
                    }
                }
            }
        }
        let order = (0..<n).sorted { a[$0 * n + $0] < a[$1 * n + $1] }
        return (
            order.map { a[$0 * n + $0] },
            order.map { column in (0..<n).map { v[$0 * n + column] } }
        )
    }

    // MARK: - Homographies from points

    /// `cv2.getPerspectiveTransform`: the homography taking four points onto
    /// four others. Nil when three of them are in line.
    static func homography(
        from src: [(x: Double, y: Double)], to dst: [(x: Double, y: Double)]
    ) -> Mat3? {
        var a = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
        var b = [Double](repeating: 0, count: 8)
        for i in 0..<4 {
            let (x, y) = src[i]
            let (u, v) = dst[i]
            a[i] = [x, y, 1, 0, 0, 0, -x * u, -y * u]
            a[i + 4] = [0, 0, 0, x, y, 1, -x * v, -y * v]
            b[i] = u
            b[i + 4] = v
        }
        guard let h = solve(a, b) else { return nil }
        return Mat3(h + [1])
    }

    /// `cv2.findHomography(src, dst, 0)`: the homography that takes `src`
    /// closest to `dst` in the least-squares sense. As OpenCV: a normalised
    /// direct linear transform, then Levenberg-Marquardt on the distances in
    /// `dst`. Nil where OpenCV returns nothing (points on one line).
    static func leastSquaresHomography(
        from src: [(x: Double, y: Double)], to dst: [(x: Double, y: Double)]
    ) -> Mat3? {
        let count = src.count
        guard count >= 4, dst.count == count else { return nil }
        let n = Double(count)

        // 1. Normalised direct linear transform (HomographyEstimatorCallback::runKernel).
        var cM = (x: 0.0, y: 0.0), cm = (x: 0.0, y: 0.0)
        for i in 0..<count {
            cM.x += src[i].x; cM.y += src[i].y
            cm.x += dst[i].x; cm.y += dst[i].y
        }
        cM.x /= n; cM.y /= n; cm.x /= n; cm.y /= n
        var sM = (x: 0.0, y: 0.0), sm = (x: 0.0, y: 0.0)
        for i in 0..<count {
            sM.x += abs(src[i].x - cM.x); sM.y += abs(src[i].y - cM.y)
            sm.x += abs(dst[i].x - cm.x); sm.y += abs(dst[i].y - cm.y)
        }
        let epsilon = Double.ulpOfOne
        guard abs(sm.x) >= epsilon, abs(sm.y) >= epsilon, abs(sM.x) >= epsilon, abs(sM.y) >= epsilon
        else { return nil }
        sm.x = n / sm.x; sm.y = n / sm.y; sM.x = n / sM.x; sM.y = n / sM.y

        var ltl = [Double](repeating: 0, count: 81)
        for i in 0..<count {
            let x = (dst[i].x - cm.x) * sm.x, y = (dst[i].y - cm.y) * sm.y
            let X = (src[i].x - cM.x) * sM.x, Y = (src[i].y - cM.y) * sM.y
            let lx = [X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x]
            let ly = [0, 0, 0, X, Y, 1, -y * X, -y * Y, -y]
            for j in 0..<9 {
                for k in j..<9 {
                    ltl[j * 9 + k] += lx[j] * lx[k] + ly[j] * ly[k]
                }
            }
        }
        for j in 0..<9 {
            for k in 0..<j {
                ltl[j * 9 + k] = ltl[k * 9 + j]
            }
        }
        guard let eigen = symmetricEigen(ltl, n: 9) else { return nil }
        let h0 = Mat3(eigen.vectors[0])
        let invNorm = Mat3([1 / sm.x, 0, cm.x, 0, 1 / sm.y, cm.y, 0, 0, 1])
        let norm2 = Mat3([sM.x, 0, -cM.x * sM.x, 0, sM.y, -cM.y * sM.y, 0, 0, 1])
        let initial = invNorm * h0 * norm2
        guard initial.m[8] != 0 else { return nil }
        var h = Array(initial.normalized.m[0..<8])
        guard h.allSatisfy(\.isFinite) else { return nil }
        if count == 4 { return Mat3(h + [1]) }

        // 2. Levenberg-Marquardt on the eight free elements (LMSolver, 10 rounds).
        func residuals(_ h: [Double], jacobian: Bool) -> (r: [Double], jtj: [Double], jtr: [Double]) {
            var r = [Double](repeating: 0, count: 2 * count)
            var jtj = [Double](repeating: 0, count: 64)
            var jtr = [Double](repeating: 0, count: 8)
            for i in 0..<count {
                let Mx = src[i].x, My = src[i].y
                var ww = h[6] * Mx + h[7] * My + 1
                ww = abs(ww) > epsilon ? 1 / ww : 0
                let xi = (h[0] * Mx + h[1] * My + h[2]) * ww
                let yi = (h[3] * Mx + h[4] * My + h[5]) * ww
                r[2 * i] = xi - dst[i].x
                r[2 * i + 1] = yi - dst[i].y
                if jacobian {
                    let jx = [Mx * ww, My * ww, ww, 0, 0, 0, -Mx * ww * xi, -My * ww * xi]
                    let jy = [0, 0, 0, Mx * ww, My * ww, ww, -Mx * ww * yi, -My * ww * yi]
                    for a in 0..<8 {
                        jtr[a] += jx[a] * r[2 * i] + jy[a] * r[2 * i + 1]
                        for b in 0..<8 {
                            jtj[a * 8 + b] += jx[a] * jx[b] + jy[a] * jy[b]
                        }
                    }
                }
            }
            return (r, jtj, jtr)
        }
        func squaredNorm(_ r: [Double]) -> Double { r.reduce(0) { $0 + $1 * $1 } }

        var state = residuals(h, jacobian: true)
        var s = squaredNorm(state.r)
        var lambda = 1.0
        var lc = 0.75
        let floatEpsilon = Double(Float.ulpOfOne)
        for _ in 0..<10 {
            var damped = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
            for a in 0..<8 {
                for b in 0..<8 {
                    damped[a][b] = state.jtj[a * 8 + b]
                }
                damped[a][a] += lambda * state.jtj[a * 8 + a]
            }
            guard let d = solve(damped, state.jtr) else { break }
            let trial = zip(h, d).map { $0 - $1 }
            let trialR = residuals(trial, jacobian: false).r
            let sd = squaredNorm(trialR)
            // dS = d·(2v − A·d)
            var dS = 0.0
            for a in 0..<8 {
                var ad = 0.0
                for b in 0..<8 {
                    ad += state.jtj[a * 8 + b] * d[b]
                }
                dS += d[a] * (2 * state.jtr[a] - ad)
            }
            let gain = (s - sd) / (abs(dS) > epsilon ? dS : 1)
            if gain > 0.75 {
                lambda *= 0.5
                if lambda < lc { lambda = 0 }
            } else if gain < 0.25 {
                let t = zip(d, state.jtr).reduce(0) { $0 + $1.0 * $1.1 }
                var nu = (sd - s) / (abs(t) > epsilon ? t : 1) + 2
                nu = min(max(nu, 2), 10)
                if lambda == 0 {
                    // 1 / the largest diagonal element of the inverse of A
                    var largest = epsilon
                    let rows = (0..<8).map { a in (0..<8).map { state.jtj[a * 8 + $0] } }
                    for a in 0..<8 {
                        var unit = [Double](repeating: 0, count: 8)
                        unit[a] = 1
                        if let column = solve(rows, unit) {
                            largest = max(largest, abs(column[a]))
                        }
                    }
                    lambda = 1 / largest
                    lc = lambda
                    nu *= 0.5
                }
                lambda *= nu
            }
            if sd < s {
                s = sd
                h = trial
                state = residuals(h, jacobian: true)
            }
            let step = d.map(abs).max() ?? 0
            let worst = state.r.map(abs).max() ?? 0
            if step < floatEpsilon || worst < floatEpsilon { break }
        }
        guard h.allSatisfy(\.isFinite) else { return nil }
        return Mat3(h + [1])
    }
}
