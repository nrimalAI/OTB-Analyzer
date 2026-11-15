import CoreGraphics
import Foundation

/// The corner model picks the grid fit's window. Port of `window_by_model`
/// in `Training/chess_pipeline.py`: keep in sync.
///
/// The classical fit finds the lines of the grid more exactly than a model
/// does, but it has to choose which 8×8 squares of the grid are the board,
/// and a border that looks like a rank of squares makes it choose a window
/// one rank or file off. The corner model is less exact and seldom a whole
/// square off. So the fit keeps the lines and the model picks the window.
extension GridCorners {

    /// `WINDOW_NEAR`: how far, in squares, each of the corner model's
    /// corners may lie from a whole-square shift of the fit's window for the
    /// window to be moved.
    public static let windowNear = 0.75

    /// The fit's corners after the corner model has had its say.
    public struct Window: Equatable {
        /// The four corners, in the order the fit's were given in.
        public let corners: [CGPoint]
        /// The shift made, in squares of the fit's own board frame (the one
        /// its first corner is a1 of): files, then ranks. Both 0 for none.
        public let files: Int
        public let ranks: Int

        public var moved: Bool { files != 0 || ranks != 0 }
    }

    /// `window_by_model`. `fit`: the grid fit's corners; `model`: the corner
    /// model's, in the same order (`turned(_:nearestTo:)` puts the fit's in
    /// the model's). Seen in the fit's board frame, each of the model's
    /// corners lies some way from the fit's corner; if all four lie within
    /// `near` squares of the same shift by one square (one rank, one file or
    /// both), the window is moved by it. Otherwise nothing changes: when the
    /// model's corners are near the fit's, and when they are too far off or
    /// too unlike a shift to be believed. What it gets wrong: a model that
    /// is off the same way by more than half a square on average, with
    /// every corner within `near` of the whole square (so each at least a
    /// quarter of a square off that way), moves a window that was right.
    public static func windowByModel(
        fit: [CGPoint], model: [CGPoint], near: Double = windowNear
    ) -> Window {
        let same = Window(corners: fit, files: 0, ranks: 0)
        guard fit.count == 4, model.count == 4, let h = Homography(corners: fit) else { return same }

        // How far each of the model's corners lies from the fit's, in squares.
        var offsets: [(x: Double, y: Double)] = []
        for (corner, at) in zip(model, Self.frame) {
            // (A corner that is not a number, or one on the fit's
            // horizon: left alone, as in the Python. Two on the horizon to
            // opposite sides: the Python's mean is not a number and it
            // fails; here left alone.)
            guard let p = h.project(corner), p.x.isFinite, p.y.isFinite else { return same }
            offsets.append((p.x - at.x, p.y - at.y))
        }
        // `np.round` of the mean: to the nearest whole square, halves to the
        // even one.
        let files = (offsets.reduce(0) { $0 + $1.x } / 4).rounded(.toNearestOrEven)
        let ranks = (offsets.reduce(0) { $0 + $1.y } / 4).rounded(.toNearestOrEven)
        guard files != 0 || ranks != 0, !(max(abs(files), abs(ranks)) > 1) else { return same }
        guard !offsets.contains(where: { abs($0.x - files) > near || abs($0.y - ranks) > near })
        else { return same }

        guard let moved = shifted(fit, files: Int(files), ranks: Int(ranks)) else { return same }
        return Window(corners: moved, files: Int(files), ranks: Int(ranks))
    }

    /// The window of the same grid that lies `files` and `ranks` squares on
    /// from the one with these corners, in the board frame its first corner
    /// is a1 of; the corners in the same order. Nil where the corners are
    /// not a board or the window moved lies on its horizon. (What
    /// `evaluate.py --shift-fit` does to the fit, to test `windowByModel`.)
    public static func shifted(_ corners: [CGPoint], files: Int, ranks: Int) -> [CGPoint]? {
        guard corners.count == 4, let h = Homography(corners: corners),
              let back = Mat3(h.m).inverse
        else { return nil }
        let moved = Self.frame.map { back.apply($0.x + Double(files), $0.y + Double(ranks)) }
        guard moved.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return nil }
        return moved.map { CGPoint(x: $0.x, y: $0.y) }
    }

    /// The corners of the board frame of `Homography`: a1, a8, h8, h1.
    private static let frame: [(x: Double, y: Double)] = [(0, 0), (0, 8), (8, 8), (8, 0)]
}
