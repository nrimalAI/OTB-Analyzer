import CoreGraphics
import Foundation

/// 3×3 projective transform from image coordinates onto the board frame.
///
/// Board frame: x = file in [0,8] (a→h), y = rank in [0,8] (rank 1→8).
/// Corners map as a1→(0,0), a8→(0,8), h8→(8,8), h1→(8,0).
///
/// Swift port of `Training/chess_pipeline.py` (`homography`, `project`,
/// `pulled_board_point`) — keep the two in sync; the Python file is the
/// executable spec, validated by `evaluate.py` against real data.
struct Homography {

    /// Row-major 3×3 matrix.
    let m: [Double]

    /// Solves the 4-point direct linear transform for corners given in
    /// (a1, a8, h8, h1) order. Returns nil for degenerate corner sets.
    init?(corners: [CGPoint]) {
        guard corners.count == 4 else { return nil }
        let dst: [(Double, Double)] = [(0, 0), (0, 8), (8, 8), (8, 0)]

        // Build the 8×9 DLT system A·h = 0.
        var A = [[Double]]()
        for (corner, (u, v)) in zip(corners, dst) {
            let x = Double(corner.x)
            let y = Double(corner.y)
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        }

        // Solve by fixing h9 = 1 (valid unless the true h9 is 0, which cannot
        // happen for a board fully in front of the camera): A'·h' = -c where
        // A' is the first 8 columns and c the 9th.
        var M = A.map { Array($0[0..<8]) }
        var b = A.map { -$0[8] }
        guard Self.gaussianSolve(&M, &b) else { return nil }
        self.m = b + [1.0]
    }

    /// Image point → board coordinates.
    func project(_ p: CGPoint) -> (x: Double, y: Double)? {
        let x = Double(p.x)
        let y = Double(p.y)
        let w = m[6] * x + m[7] * y + m[8]
        guard abs(w) > 1e-12 else { return nil }
        return ((m[0] * x + m[1] * y + m[2]) / w,
                (m[3] * x + m[4] * y + m[5]) / w)
    }

    /// Projects a piece's bbox bottom-center with camera-aware correction:
    /// the box bottom is the *front edge* of the piece's base (displaced
    /// toward the camera, i.e. straight down in image space), so step `pull`
    /// board-units along image-"up" mapped through the homography.
    func pulledBoardPoint(_ p: CGPoint, pull: Double = 0.30) -> (x: Double, y: Double)? {
        guard let base = project(p),
              let up = project(CGPoint(x: p.x, y: p.y - 1))
        else { return nil }
        let dx = up.x - base.x
        let dy = up.y - base.y
        let norm = (dx * dx + dy * dy).squareRoot()
        guard norm > 1e-9 else { return base }
        return (base.x + pull * dx / norm, base.y + pull * dy / norm)
    }

    /// In-place Gaussian elimination with partial pivoting; false if singular.
    private static func gaussianSolve(_ M: inout [[Double]], _ b: inout [Double]) -> Bool {
        let n = b.count
        for col in 0..<n {
            // Pivot
            var pivot = col
            for row in (col + 1)..<n where abs(M[row][col]) > abs(M[pivot][col]) {
                pivot = row
            }
            guard abs(M[pivot][col]) > 1e-12 else { return false }
            if pivot != col {
                M.swapAt(pivot, col)
                b.swapAt(pivot, col)
            }
            // Eliminate below
            for row in (col + 1)..<n {
                let factor = M[row][col] / M[col][col]
                if factor == 0 { continue }
                for k in col..<n {
                    M[row][k] -= factor * M[col][k]
                }
                b[row] -= factor * b[col]
            }
        }
        // Back-substitute
        for col in stride(from: n - 1, through: 0, by: -1) {
            var sum = b[col]
            for k in (col + 1)..<n {
                sum -= M[col][k] * b[k]
            }
            b[col] = sum / M[col][col]
        }
        return true
    }
}
