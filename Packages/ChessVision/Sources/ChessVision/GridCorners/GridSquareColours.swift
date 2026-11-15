import CoreGraphics
import Foundation

/// a1 is a dark square: which of the four ways the corners can be read put a
/// dark square on a1, from the grey of the 64 cells. Port of
/// `dark_corner_rotations` in `Training/grid_corners.py`: keep in sync.
///
/// On a board set up as the rules say this leaves two of the four ways, half
/// a turn apart, whatever the pieces or a model say; the pieces then choose
/// between those two only (`BoardOrientation.orient`, `turns`), so that they
/// can no longer turn a board by a quarter. A board set up with a light
/// square on a1 is read a quarter turn off.
extension GridCorners {

    /// What the cells say about a ring of four corners.
    public struct SquareColours: Equatable {
        /// The rotations of the ring that put a dark square on a1: [0, 2] or
        /// [1, 3], where rotation k starts the ring at its corner k. Nil when
        /// the cells do not say: fewer than `leastCells` of them seen, too
        /// little contrast, or no clear split between the two colours.
        public let rotations: [Int]?
        /// 0...1: how cleanly the two colours of cell split. 0 when it was
        /// not got as far as measuring it.
        public let strength: Double
    }

    /// The cells that must be seen (of 64) for the colours to count.
    static let leastCells = 40
    /// The least difference between the lighter and the darker cells, in
    /// grey levels of 255 (between the quartiles of the cells).
    static let leastContrast = 15.0
    /// The least `strength` for the colours to count.
    static let leastStrength = 0.5

    /// `dark_corner_rotations`. `corners`: four corners of the grid in
    /// pixels of the photo, in ring order (a1, a8, h8, h1 or any rotation of
    /// it; the model's, or the grid fit's).
    public static func squareColours(in photo: CGImage, corners: [CGPoint]) -> SquareColours {
        squareColours(in: photo, corners: corners, whole: nil)
    }

    /// `whole`: the whole photo at the working size (`workingImage(of:)`),
    /// when the caller has it already.
    static func squareColours(
        in photo: CGImage, corners: [CGPoint], whole: GrayImage.Working?
    ) -> SquareColours {
        guard corners.count == 4,
              corners.allSatisfy({ $0.x.isFinite && $0.y.isFinite }),
              // The same pixels as the grid fit and the models read.
              let working = whole ?? workingImage(of: photo)
        else { return SquareColours(rotations: nil, strength: 0) }
        return squareColours(working.gray, ring: corners.map {
            (Double($0.x) * working.scale, Double($0.y) * working.scale)
        })
    }

    /// From the working image on; `ring` in its pixels.
    static func squareColours(_ gray: GrayImage, ring: [(x: Double, y: Double)]) -> SquareColours {
        let none = SquareColours(rotations: nil, strength: 0)
        // The ring onto the board frame of `Homography`: its first corner is
        // a1 for rotation 0.
        guard ring.count == 4,
              let h = GridMath.homography(from: ring, to: [(0, 0), (0, 8), (8, 8), (8, 0)]),
              let cells = GridWindow.cellValues(gray, h, x0: 0, y0: 0, nx: 8, ny: 8)
        else { return none }

        // The grey of a cell: the median of its patches that are seen
        // (`np.nanmedian`).
        var seen: [(value: Float, even: Bool)] = []
        for j in 0..<8 {
            for i in 0..<8 {
                let patches = cells.patches(i, j).filter { !$0.isNaN }
                guard !patches.isEmpty else { continue }
                // (even: the colour of the cell at the ring's first corner)
                seen.append((GridMath.median(patches), (i + j) % 2 == 0))
            }
        }
        guard seen.count >= leastCells else { return none }
        let sorted = seen.map { Double($0.value) }.sorted()
        let lo = GridMath.percentile(sorted: sorted, 25)
        let hi = GridMath.percentile(sorted: sorted, 75)
        guard !(hi - lo < leastContrast) else { return none }
        let middle = (lo + hi) / 2

        var cellsOf = [0, 0], lightOf = [0, 0]   // [even, odd]
        for cell in seen {
            let k = cell.even ? 0 : 1
            cellsOf[k] += 1
            if Double(cell.value) > middle { lightOf[k] += 1 }
        }
        // (With 40 cells seen of 64 there are at least 8 of each colour.)
        guard cellsOf[0] > 0, cellsOf[1] > 0 else { return none }
        let shareEven = Double(lightOf[0]) / Double(cellsOf[0])
        let shareOdd = Double(lightOf[1]) / Double(cellsOf[1])
        let strength = abs(shareEven - shareOdd)
        guard !(strength < leastStrength) else {
            return SquareColours(rotations: nil, strength: strength)
        }
        // Rotation 0 is right when the cells of a1's colour are the dark
        // ones; a quarter turn swaps the two colours.
        return SquareColours(rotations: shareEven < shareOdd ? [0, 2] : [1, 3], strength: strength)
    }
}
