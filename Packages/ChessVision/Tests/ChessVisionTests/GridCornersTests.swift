import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import ChessVision

/// The classical grid fit (`GridCorners` and its stages) against
/// `Training/grid_corners.py` and the OpenCV calls it is built on. The
/// reference values are in `Fixtures/grid_cases.json`, written by
/// `Training/swift_reference.py grid-cases`; run that again after changing
/// the Python, and a change in the JSON is a change the Swift must follow.
struct GridCornersTests {

    // MARK: - The recorded cases

    struct Recorded: Decodable {
        struct Fit: Decodable {
            let raw: [[Double]]?
            let confidence: Double
            let stage: String
            let pass: Int
            let values: [String: Double]
        }
        struct Board: Decodable {
            let file: String
            let truth: [[Double]]?
            let fit: Fit
            let box: [Double]?
            let fit_in_box: Fit?
        }
        struct Ops: Decodable {
            let width: Int
            let height: Int
            let patch: [Float]
            let gray_sum: Double
            let gray_at: [Float]
            let blur1: [Float]
            let blur2: [Float]
            let saddle: [Float]
            let dilated: [Float]
            let warp_matrix: [Double]
            let warp: [Float]
            let x_corners: [[Double]]
            let subpix_start: [[Float]]
            let subpix: [[Double]]
            let ring_points: [[Double]]
            let ring_response: [[Float]]
        }
        struct Resize: Decodable {
            let box: [Double]?
            let width: Int
            let height: Int
            let scale: Double
            let origin: [Int]?
            let step_x: Int
            let step_y: Int
            let samples: [Float]
        }
        struct Maths: Decodable {
            let src: [[Double]]
            let dst: [[Double]]
            let least_squares: [Double]
            let quad: [[Double]]
            let four_point: [Double]
            let sample: [Double]
            let median: Double
            let percentiles: [String: Double]
        }
        let boards: [String: Board]
        let photos: [String: Fit]
        let ops: Ops
        let resize: Resize
        let resize_in_box: Resize
        let resize_by_two: Resize
        let math: Maths
    }

    private static let fixtures = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!

    private static let recorded = Result {
        try JSONDecoder().decode(
            Recorded.self, from: Data(contentsOf: fixtures.appending(path: "grid_cases.json")))
    }

    private static func image(_ file: String) throws -> CGImage {
        let source = try #require(
            CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    /// The largest distance between matching corners, in squares, the two
    /// lists taken in the rotation that fits best (which corner comes first
    /// is not fixed).
    private static func distance(_ corners: [CGPoint], _ reference: [[Double]]) -> Double {
        var area = 0.0
        for i in 0..<4 {
            let a = reference[i], b = reference[(i + 1) % 4]
            area += a[0] * b[1] - b[0] * a[1]
        }
        let square = (abs(area) / 2).squareRoot() / 8
        var best = Double.infinity
        for turn in 0..<4 {
            var worst = 0.0
            for i in 0..<4 {
                let r = reference[(i + turn) % 4]
                worst = max(worst, hypot(Double(corners[i].x) - r[0], Double(corners[i].y) - r[1]))
            }
            best = min(best, worst)
        }
        return best / square
    }

    private static func largestDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? .infinity
    }

    // MARK: - The whole fit

    /// Boards drawn by the Python (the same pixels on both sides): accepted,
    /// and the corners are Python's to a hundredth of a square.
    @Test func testFindsDrawnBoardsWherePythonDoes() throws {
        let recorded = try Self.recorded.get()
        for name in ["plain", "pieces"] {
            let board = try #require(recorded.boards[name])
            let image = try Self.image(board.file)
            let python = try #require(board.fit.raw)
            let truth = try #require(board.truth)

            let fit = try #require(GridCorners.find(in: image), "\(name) rejected")
            #expect(Self.distance(fit.corners, python) < 0.01, "\(name) against Python")
            #expect(Self.distance(fit.corners, truth) < 0.05, "\(name) against the drawing")
            #expect(abs(fit.confidence - board.fit.confidence) < 0.05, "\(name)")
            #expect(GridCorners.diagnose(image).pass == board.fit.pass, "\(name)")
            print(String(
                format: "grid %@: %.4f squares from Python, %.4f from the drawing, confidence %.3f (Python %.3f)",
                name, Self.distance(fit.corners, python), Self.distance(fit.corners, truth),
                fit.confidence, board.fit.confidence))

            // clockwise as seen in the image (y down)
            var area = 0.0
            for i in 0..<4 {
                let a = fit.corners[i], b = fit.corners[(i + 1) % 4]
                area += Double(a.x * b.y - b.x * a.y)
            }
            #expect(area > 0, "\(name)")

            // The same with the board's box given: one pass, inside the box.
            let box = try #require(board.box)
            let inBox = try #require(board.fit_in_box)
            let boxed = try #require(
                GridCorners.find(in: image, priorBox: CGRect(
                    x: box[0], y: box[1], width: box[2] - box[0], height: box[3] - box[1])),
                "\(name) rejected in its box")
            let pythonInBox = try #require(inBox.raw)
            #expect(Self.distance(boxed.corners, pythonInBox) < 0.01, "\(name) in its box")
            #expect(abs(boxed.confidence - inBox.confidence) < 0.05, "\(name) in its box")
        }
    }

    /// No board: nothing comes back, as in Python.
    @Test func testRejectsWhatIsNotABoard() throws {
        let recorded = try Self.recorded.get()
        let clutter = try #require(recorded.boards["clutter"])
        #expect(clutter.fit.confidence == 0)
        let image = try Self.image(clutter.file)
        #expect(GridCorners.find(in: image) == nil)
        #expect(GridCorners.diagnose(image).confidence == 0)

        // A plain grey picture, one too small to hold a board, and a box that
        // lies outside the picture.
        let plain = try #require(LetterboxTests.solid(width: 300, height: 200, red: 120, green: 120, blue: 120))
        #expect(GridCorners.find(in: plain) == nil)
        let tiny = try #require(LetterboxTests.solid(width: 12, height: 9, red: 200, green: 10, blue: 10))
        #expect(GridCorners.find(in: tiny) == nil)
        #expect(GridCorners.find(in: plain, priorBox: CGRect(x: 500, y: 500, width: 50, height: 50)) == nil)
        #expect(GridCorners.find(in: plain, priorBox: CGRect(x: 10, y: 10, width: 0, height: 0)) == nil)
        #expect(GridCorners.find(in: plain, priorBox: CGRect(x: 1e30, y: 0, width: 50, height: 50)) == nil)
        #expect(GridCorners.find(in: plain, priorBox: CGRect(x: CGFloat.nan, y: 0, width: 50, height: 50)) == nil)
        #expect(GridCorners.find(in: plain, priorBox: .infinite) == nil)
    }

    /// The four test photos. JPEG decoders differ by a grey level here and
    /// there, so the corners are held to a twentieth of a square, not a
    /// hundredth.
    @Test func testFindsTheFixturePhotosWherePythonDoes() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.photos.count == 4)
        for (file, python) in recorded.photos.sorted(by: { $0.key < $1.key }) {
            #expect(python.confidence > 0)
            let image = try Self.image(file)
            let pythonCorners = try #require(python.raw)
            let fit = try #require(GridCorners.find(in: image), "\(file) rejected")
            let off = Self.distance(fit.corners, pythonCorners)
            print(String(format: "grid %@: %.4f squares from Python, confidence %.3f (Python %.3f)",
                         file, off, fit.confidence, python.confidence))
            #expect(off < 0.05, "\(file)")
        }
    }

    /// What the recognizer does with the two sets of corners: the fit's
    /// corners in the model's order where the fit accepts itself, the model's
    /// own where it does not.
    @Test func testRecognizerTakesTheFitOrKeepsTheModelsCorners() throws {
        let recorded = try Self.recorded.get()
        let board = try #require(recorded.boards["pieces"])
        let truth = try #require(board.truth).map { CGPoint(x: $0[0], y: $0[1]) }
        // A corner model that is half a square out and starts at another corner.
        let model = [truth[2], truth[3], truth[0], truth[1]].map { CGPoint(x: $0.x + 25, y: $0.y - 20) }

        let onBoard = CoreMLBoardRecognizer.cornersWithGridFit(
            modelCorners: model, in: try Self.image(board.file))
        for (mine, drawn) in zip(onBoard, [truth[2], truth[3], truth[0], truth[1]]) {
            // the fit's corners (within a twentieth of a 56-pixel square), in the model's order
            #expect(hypot(mine.x - drawn.x, mine.y - drawn.y) < 3)
        }

        let clutter = try #require(recorded.boards["clutter"])
        #expect(CoreMLBoardRecognizer.cornersWithGridFit(
            modelCorners: model, in: try Self.image(clutter.file)) == model)
    }

    @Test func testTurnsTheFitToTheModelsCorners() {
        let fit = [
            CGPoint(x: 100, y: 100), CGPoint(x: 900, y: 110),
            CGPoint(x: 880, y: 800), CGPoint(x: 90, y: 820),
        ]
        // The model's corners: roughly the same quadrilateral, starting two corners on.
        let model = [
            CGPoint(x: 860, y: 790), CGPoint(x: 110, y: 800),
            CGPoint(x: 120, y: 130), CGPoint(x: 880, y: 120),
        ]
        #expect(GridCorners.turned(fit, nearestTo: model) == [fit[2], fit[3], fit[0], fit[1]])
        #expect(GridCorners.turned(fit, nearestTo: fit) == fit)
        // Not four corners: left alone.
        #expect(GridCorners.turned(fit, nearestTo: []) == fit)
    }

    // MARK: - The stages

    /// `_prepare`: the grey working image of a PNG is OpenCV's, pixel for pixel.
    @Test func testWorkingImageIsOpenCVs() throws {
        let ops = try Self.recorded.get().ops
        let board = try Self.image("grid_board_pieces.png")
        let working = try #require(GrayImage.working(from: board, priorBox: nil))
        #expect(working.scale == 1)
        #expect(working.gray.width == 800 && working.gray.height == 600)
        #expect(working.gray[0, 0] == ops.gray_at[0])
        #expect(working.gray[400, 300] == ops.gray_at[1])
        #expect(working.gray[799, 599] == ops.gray_at[2])
        let sum = working.gray.pixels.reduce(0.0) { $0 + Double($1) }
        #expect(sum == ops.gray_sum)
    }

    /// A picture whose channels are formulas, so that Python can make the
    /// same one.
    private static func formulaImage(width w: Int, height h: Int) -> CGImage? {
        var bytes = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                bytes[(y * w + x) * 4] = UInt8((x * 7 + y * 3) % 256)        // R
                bytes[(y * w + x) * 4 + 1] = UInt8((x * 5 + y * 11) % 256)   // G
                bytes[(y * w + x) * 4 + 2] = UInt8((x + y * 13) % 256)       // B
            }
        }
        return bytes.withUnsafeMutableBytes { buffer in
            CGContext(
                data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)?.makeImage()
        }
    }

    /// `_prepare` on a picture larger than the working size: the shrink is
    /// `cv2.resize(..., INTER_AREA)` and the crop to a box is Python's.
    @Test func testShrinkToWorkingSizeIsOpenCVs() throws {
        let recorded = try Self.recorded.get()
        let large = try #require(Self.formulaImage(width: 1500, height: 1100))
        let twice = try #require(Self.formulaImage(width: 2000, height: 1300))
        for (reference, image, label) in [
            (recorded.resize, large, "whole"), (recorded.resize_in_box, large, "in a box"),
            (recorded.resize_by_two, twice, "by exactly two"),
        ] {
            let box = reference.box.map {
                CGRect(x: $0[0], y: $0[1], width: $0[2] - $0[0], height: $0[3] - $0[1])
            }
            let working = try #require(GrayImage.working(from: image, priorBox: box))
            #expect(working.gray.width == reference.width, "\(label)")
            #expect(working.gray.height == reference.height, "\(label)")
            #expect(abs(working.scale - reference.scale) < 1e-12, "\(label)")
            if let origin = reference.origin {
                #expect(working.originX == origin[0] && working.originY == origin[1], "\(label)")
            }
            var samples: [Float] = []
            for y in stride(from: 0, to: working.gray.height, by: reference.step_y) {
                for x in stride(from: 0, to: working.gray.width, by: reference.step_x) {
                    samples.append(working.gray[x, y])
                }
            }
            try #require(samples.count == reference.samples.count, "\(label)")
            // Pixel for pixel OpenCV's.
            #expect(samples == reference.samples, "\(label)")
        }
    }

    /// The filters on a 64×48 patch: `cv2.GaussianBlur`, the Sobel-based
    /// saddle response, `cv2.dilate` and `cv2.warpPerspective`.
    @Test func testFiltersAreOpenCVs() throws {
        let ops = try Self.recorded.get().ops
        let patch = GrayImage(width: ops.width, height: ops.height, pixels: ops.patch)

        #expect(Self.largestDifference(patch.gaussianBlurred(sigma: 1.0).pixels, ops.blur1) < 1e-3)
        let blur2 = patch.gaussianBlurred(sigma: 2.0)
        #expect(Self.largestDifference(blur2.pixels, ops.blur2) < 1e-3)

        // The saddle response is a difference of products of small
        // differences: compare it on OpenCV's own blur, relative to its size.
        let saddle = GrayImage(width: ops.width, height: ops.height, pixels: ops.blur2).saddleResponse()
        let scale = ops.saddle.max()!
        #expect(scale > 100)
        #expect(Self.largestDifference(saddle.pixels, ops.saddle) < 1e-4 * scale)
        let dilated = GrayImage(width: ops.width, height: ops.height, pixels: ops.saddle).dilated(size: 9)
        #expect(dilated.pixels == ops.dilated)

        let warped = try #require(patch.warped(by: Mat3(ops.warp_matrix), width: 40, height: 30))
        #expect(Self.largestDifference(warped.pixels, ops.warp) < 1e-3)
        // the same pixels fall outside the patch
        #expect(zip(warped.pixels, ops.warp).allSatisfy { ($0 == -1) == ($1 == -1) })
    }

    /// `x_corners` and its parts on a drawn board.
    @Test func testXCornersAreOpenCVs() throws {
        let ops = try Self.recorded.get().ops
        let board = try Self.image("grid_board_pieces.png")
        let working = try #require(GrayImage.working(from: board, priorBox: nil))
        let smooth = working.gray.gaussianBlurred(sigma: 1.0)

        // the ring test
        for (r, radius) in [4.0, 7.0].enumerated() {
            for (index, point) in ops.ring_points.enumerated() {
                let test = GridXCorners.ringTest(smooth, x: Int(point[0]), y: Int(point[1]), radius: radius)
                #expect(abs(test.response - ops.ring_response[r][index]) < 1e-2)
            }
        }

        // cornerSubPix pulls points a pixel off back onto the corner
        let start = ops.subpix_start.map { GridPoint(x: $0[0], y: $0[1]) }
        let refined = GridXCorners.cornerSubPix(smooth, start)
        for (index, (mine, theirs)) in zip(refined, ops.subpix).enumerated() {
            #expect(hypot(Double(mine.x) - theirs[0], Double(mine.y) - theirs[1]) < 1e-2,
                    "point \(index)")
        }

        // the x-corners themselves: the same ones
        let found = GridXCorners.find(in: working.gray)
        #expect(found.count == ops.x_corners.count)
        for theirs in ops.x_corners {
            let nearest = found.map { hypot(Double($0.x) - theirs[0], Double($0.y) - theirs[1]) }.min()
            #expect((nearest ?? .infinity) < 0.02, "no x-corner at \(theirs)")
        }
    }

    /// A lattice of x-corners seen in perspective, with strays: the fit puts
    /// every true corner on its node.
    @Test func testLatticeFitFindsTheNodes() throws {
        // lattice -> image; the corners of a 7×7 inner grid plus two strays
        let toImage = Mat3([52, 9, 310, -6, 47, 220, 0.004, 0.011, 1])
        var points: [GridPoint] = []
        for i in 0..<7 {
            for j in 0..<7 {
                let p = toImage.apply(Double(i), Double(j))
                points.append(GridPoint(x: Float(p.x), y: Float(p.y)))
            }
        }
        points.append(GridPoint(x: 40, y: 50))
        points.append(GridPoint(x: 900, y: 30))

        let fit = try #require(GridLattice.fit(points))
        #expect(fit.inlier[..<49].allSatisfy { $0 })
        #expect(fit.inlierNodes.count >= 49)
        // Neighbouring corners are one step apart on the lattice, along one axis.
        for i in 0..<6 {
            for j in 0..<6 {
                let a = fit.node[i * 7 + j], right = fit.node[i * 7 + j + 1], down = fit.node[(i + 1) * 7 + j]
                #expect(abs(right.i - a.i) + abs(right.j - a.j) == 1)
                #expect(abs(down.i - a.i) + abs(down.j - a.j) == 1)
            }
        }
        // Too few points: no fit.
        #expect(GridLattice.fit(Array(points[..<8])) == nil)
    }

    // MARK: - The mathematics

    @Test func testStatisticsAreNumPys() throws {
        let math = try Self.recorded.get().math
        #expect(abs(GridMath.median(math.sample) - math.median) < 1e-12)
        for (q, value) in math.percentiles {
            let rank = try #require(Double(q))
            let mine = try #require(GridMath.percentile(math.sample, rank))
            #expect(abs(mine - value) < 1e-12, "percentile \(q)")
        }
        #expect(GridMath.percentile([], 50) == nil)
        #expect(GridMath.median([Double]()).isNaN)
        #expect(GridMath.median([3, 1, 2] as [Float]) == 2)
        #expect(GridMath.median([4, 1, 3, 2] as [Float]) == 2.5)
        // to the nearest integer, halves to the even one; Python's % is never negative
        #expect(GridMath.rint(2.5) == 2 && GridMath.rint(3.5) == 4 && GridMath.rint(-2.5) == -2)
        #expect(GridMath.rint(.nan) == 0 && GridMath.rint(.infinity) == 0)
        #expect(GridMath.mod(-1, 2) == 1 && GridMath.mod(-4, 2) == 0 && GridMath.mod(5, 2) == 1)
    }

    @Test func testHomographiesAreOpenCVs() throws {
        let math = try Self.recorded.get().math
        func points(_ list: [[Double]]) -> [(x: Double, y: Double)] { list.map { ($0[0], $0[1]) } }

        // cv2.getPerspectiveTransform
        let unit: [(x: Double, y: Double)] = [(0, 0), (1, 0), (1, 1), (0, 1)]
        let four = try #require(GridMath.homography(from: points(math.quad), to: unit))
        for (mine, theirs) in zip(four.m, math.four_point) {
            #expect(abs(mine - theirs) < 1e-9 * max(1, abs(theirs)))
        }
        // three points in line: no homography
        #expect(GridMath.homography(from: [(0, 0), (1, 0), (2, 0), (3, 1)], to: unit) == nil)

        // cv2.findHomography(src, dst, 0): the same map, judged by where it
        // takes the points (in lattice units)
        let src = points(math.src), dst = points(math.dst)
        let fitted = try #require(GridMath.leastSquaresHomography(from: src, to: dst))
        let theirs = Mat3(math.least_squares)
        for p in src {
            let a = fitted.apply(p.x, p.y), b = theirs.apply(p.x, p.y)
            #expect(hypot(a.x - b.x, a.y - b.y) < 1e-5)
        }
        // all the targets on one line: nothing, as OpenCV
        #expect(GridMath.leastSquaresHomography(from: src, to: src.map { _ in (1, 2) }) == nil)
    }

    @Test func testMatricesAndEigenvectors() throws {
        let h = Mat3([52, 9, 310, -6, 47, 220, 0.004, 0.011, 1])
        let inverse = try #require(h.inverse)
        let p = h.apply(3, 4)
        let back = inverse.apply(p.x, p.y)
        #expect(abs(back.x - 3) < 1e-9 && abs(back.y - 4) < 1e-9)
        let product = h * inverse
        for (mine, identity) in zip(product.m, Mat3.identity.m) {
            #expect(abs(mine - identity) < 1e-9)
        }
        #expect(Mat3([1, 2, 3, 2, 4, 6, 0, 0, 1]).inverse == nil)       // singular
        #expect(Mat3([.nan, 0, 0, 0, 1, 0, 0, 0, 1]).inverse == nil)

        // a symmetric matrix with eigenvalues 2, 2 and 4
        let matrix = [2.0, 0, 0, 0, 3, -1, 0, -1, 3]
        let eigen = try #require(GridMath.symmetricEigen(matrix, n: 3))
        for (mine, known) in zip(eigen.values, [2.0, 2.0, 4.0]) {
            #expect(abs(mine - known) < 1e-12)
        }
        for (value, vector) in zip(eigen.values, eigen.vectors) {
            for row in 0..<3 {
                let product = (0..<3).reduce(0.0) { $0 + matrix[row * 3 + $1] * vector[$1] }
                #expect(abs(product - value * vector[row]) < 1e-12)
            }
        }
        #expect(GridMath.solve([[2, 1], [1, 3]], [3, 5]).map { abs($0[0] - 0.8) < 1e-12 && abs($0[1] - 1.4) < 1e-12 } == true)
        #expect(GridMath.solve([[1, 2], [2, 4]], [1, 2]) == nil)
    }
}
