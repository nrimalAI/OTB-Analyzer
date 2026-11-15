import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import ChessVision

/// The corner model picks the grid fit's window (`GridCorners.windowByModel`)
/// against `window_by_model` in `Training/chess_pipeline.py`. The reference
/// values are in `Fixtures/window_cases.json`, written by
/// `Training/swift_reference.py window-cases`.
struct GridFitWindowTests {

    private struct Recorded: Decodable {
        struct Case: Decodable {
            let name: String
            let fit: [[Double]]
            let model: [[Double]]
            let near: Double
            let corners: [[Double]]
            let shift: [Int]
        }
        struct Shift: Decodable {
            let ring: String
            let files: Int
            let ranks: Int
            let corners: [[Double]]
        }
        let near: Double
        let cases: [Case]
        let shifts: [Shift]
    }

    /// The drawn boards of `grid_cases.json`: the picture and its corners.
    private struct Drawn: Decodable {
        struct Board: Decodable {
            let file: String
            let truth: [[Double]]?
        }
        let boards: [String: Board]
    }

    private static let fixtures = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!

    private static let recorded = Result {
        try JSONDecoder().decode(
            Recorded.self, from: Data(contentsOf: fixtures.appending(path: "window_cases.json")))
    }

    private static func points(_ ring: [[Double]]) -> [CGPoint] {
        ring.map { CGPoint(x: $0[0], y: $0[1]) }
    }

    /// The largest distance between matching corners, in pixels.
    private static func distance(_ a: [CGPoint], _ b: [CGPoint]) -> Double {
        guard a.count == b.count else { return .infinity }
        return zip(a, b).map { Double(hypot($0.x - $1.x, $0.y - $1.y)) }.max() ?? 0
    }

    /// A drawn board with pieces on it, its true corners, and a picture with
    /// no board the fit accepts (also for `EndToEndFixtureTests`).
    static func drawnBoard() throws -> (image: CGImage, truth: [CGPoint], clutter: CGImage) {
        let drawn = try JSONDecoder().decode(
            Drawn.self, from: Data(contentsOf: fixtures.appending(path: "grid_cases.json")))
        func image(_ name: String) throws -> CGImage {
            let file = try #require(drawn.boards[name]).file
            let source = try #require(
                CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
            return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        }
        let board = try #require(drawn.boards["pieces"])
        let truth = points(try #require(board.truth))
        return (try image("pieces"), truth, try image("clutter"))
    }

    // MARK: - Against the Python

    @Test func testWindowsComeOutAsInPython() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.near == GridCorners.windowNear)
        #expect(recorded.cases.count >= 200)
        var moved = 0
        for item in recorded.cases {
            let fit = Self.points(item.fit)
            let window = GridCorners.windowByModel(
                fit: fit, model: Self.points(item.model), near: item.near)
            #expect([window.files, window.ranks] == item.shift, "\(item.name)")
            #expect(window.moved == (item.shift != [0, 0]), "\(item.name)")
            #expect(Self.distance(window.corners, Self.points(item.corners)) < 1e-6, "\(item.name)")
            if !window.moved {
                // Left alone means the very corners given.
                #expect(window.corners == fit, "\(item.name)")
            }
            if item.near == GridCorners.windowNear {
                // The tolerance not given is the reference's.
                #expect(GridCorners.windowByModel(fit: fit, model: Self.points(item.model)) == window,
                        "\(item.name)")
            }
            moved += window.moved ? 1 : 0
        }
        print("fit window: \(recorded.cases.count) cases as in Python, \(moved) of them moved")
    }

    /// What the cases are there for: each threshold is met from both sides,
    /// on every one of the fits.
    @Test func testTheCasesMeetEachThresholdFromBothSides() throws {
        let cases = try Self.recorded.get().cases
        func shifts(_ name: String) -> [[Int]] {
            cases.filter { $0.name.hasSuffix(": " + name) }.map(\.shift)
        }
        let expected: [(String, [Int])] = [
            ("the fit's own corners", [0, 0]),
            ("mean just under half a file", [0, 0]), ("mean just over half a file", [1, 0]),
            ("mean just under half a rank back", [0, 0]), ("mean just over half a rank back", [0, -1]),
            ("two ranks on", [0, 0]), ("two files back", [0, 0]), ("a file on and two ranks on", [0, 0]),
            ("just under a rank and a half", [0, 1]), ("just over a rank and a half", [0, 0]),
            ("a corner just inside, files", [0, 1]), ("a corner just outside, files", [0, 0]),
            ("a corner just inside, files back", [0, -1]), ("a corner just outside, files back", [0, 0]),
            ("a corner just inside, ranks", [1, 0]), ("a corner just outside, ranks", [0, 0]),
            ("a corner just inside, ranks up", [-1, 0]), ("a corner just outside, ranks up", [0, 0]),
            ("one corner stays", [0, 0]),
            ("0.6 off at 0.5", [0, 0]), ("0.6 off at 0.75", [0, 1]),
            ("0.9 off at 0.75", [0, 0]), ("0.9 off at 0.95", [0, 1]),
            ("a rank on, 1 corners round", [0, 0]), ("a rank on, 2 corners round", [0, 0]),
            ("a rank on, 3 corners round", [0, 0]),
        ]
        for (name, shift) in expected {
            #expect(shifts(name) == [[Int]](repeating: shift, count: 4), "\(name)")
        }
        // A square on in each of the eight directions.
        for files in -1...1 {
            for ranks in -1...1 where (files, ranks) != (0, 0) {
                #expect(shifts("a square on (\(files), \(ranks))")
                    == [[Int]](repeating: [files, ranks], count: 4))
            }
        }
    }

    /// `evaluate.py --shift-fit`: a window moved on purpose.
    @Test func testWindowsMovedOnPurposeAreWhereThePythonPutsThem() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.shifts.count >= 40)
        for item in recorded.shifts {
            let own = try #require(
                recorded.cases.first { $0.name == "\(item.ring): the fit's own corners" })
            let fit = Self.points(own.fit)
            let moved = try #require(GridCorners.shifted(fit, files: item.files, ranks: item.ranks))
            #expect(Self.distance(moved, Self.points(item.corners)) < 1e-6,
                    "\(item.ring) by \(item.files), \(item.ranks)")
        }
    }

    // MARK: - By itself

    /// A window moved on purpose is moved back when the model has the board
    /// where it was, as `evaluate.py --shift-fit` with `--fit-window`.
    @Test func testAWindowMovedOnPurposeIsMovedBack() throws {
        let board = [
            CGPoint(x: 150, y: 820), CGPoint(x: 260, y: 240),
            CGPoint(x: 790, y: 215), CGPoint(x: 930, y: 790),
        ]
        // A corner model a fifth of a square out at each corner.
        let toImage = try #require(GridMath.homography(
            from: [(0, 0), (0, 8), (8, 8), (8, 0)], to: board.map { (Double($0.x), Double($0.y)) }))
        let model = [(0.2, -0.1), (-0.15, 8.2), (8.1, 8.1), (7.8, -0.2)].map { at -> CGPoint in
            let p = toImage.apply(at.0, at.1)
            return CGPoint(x: p.x, y: p.y)
        }
        for files in -1...1 {
            for ranks in -1...1 {
                let given = try #require(GridCorners.shifted(board, files: files, ranks: ranks))
                let window = GridCorners.windowByModel(fit: given, model: model)
                #expect(window.files == -files && window.ranks == -ranks, "\(files), \(ranks)")
                #expect(Self.distance(window.corners, board) < 1e-6, "\(files), \(ranks)")
                // Moving on and back again is where it was.
                let back = try #require(GridCorners.shifted(given, files: -files, ranks: -ranks))
                #expect(Self.distance(back, board) < 1e-6, "\(files), \(ranks)")
            }
        }
        // Two squares off is more than the model is trusted with.
        let two = try #require(GridCorners.shifted(board, files: 0, ranks: 2))
        #expect(GridCorners.windowByModel(fit: two, model: model).corners == two)
        // The shift is in the frame of the corners as given: the same window
        // in another rotation is moved to the same place.
        let rank = try #require(GridCorners.shifted(board, files: 0, ranks: 1))
        for turn in 1..<4 {
            func turned(_ ring: [CGPoint]) -> [CGPoint] { Array(ring[turn...] + ring[..<turn]) }
            let window = GridCorners.windowByModel(fit: turned(rank), model: turned(model))
            #expect(window.moved, "turned by \(turn)")
            #expect(Self.distance(window.corners, turned(board)) < 1e-6, "turned by \(turn)")
        }
    }

    @Test func testCornersThatAreNotABoardAreLeftAlone() throws {
        let fit = [
            CGPoint(x: 200, y: 700), CGPoint(x: 200, y: 100),
            CGPoint(x: 800, y: 100), CGPoint(x: 800, y: 700),
        ]
        let rank = try #require(GridCorners.shifted(fit, files: 0, ranks: 1))
        #expect(GridCorners.windowByModel(fit: fit, model: rank).moved)

        // Not four corners, on either side.
        #expect(GridCorners.windowByModel(fit: Array(fit.dropLast()), model: rank).corners
            == Array(fit.dropLast()))
        #expect(GridCorners.windowByModel(fit: fit, model: Array(rank.dropLast())).corners == fit)
        #expect(GridCorners.windowByModel(fit: fit, model: []).corners == fit)
        // A fit whose corners are in line.
        let line = [
            CGPoint(x: 0, y: 0), CGPoint(x: 100, y: 100),
            CGPoint(x: 200, y: 200), CGPoint(x: 300, y: 300),
        ]
        #expect(GridCorners.windowByModel(fit: line, model: rank).corners == line)
        #expect(GridCorners.shifted(line, files: 0, ranks: 1) == nil)
        #expect(GridCorners.shifted(Array(fit.dropLast()), files: 0, ranks: 1) == nil)
        // A corner of the model's that is not a number: on a board seen
        // from above (where the projection itself gives up) and on one seen
        // at a slant (where it gives a point that is not a number).
        let slant = [
            CGPoint(x: 150, y: 820), CGPoint(x: 260, y: 240),
            CGPoint(x: 790, y: 215), CGPoint(x: 930, y: 790),
        ]
        for board in [fit, slant] {
            let on = try #require(GridCorners.shifted(board, files: 0, ranks: 1))
            #expect(GridCorners.windowByModel(fit: board, model: on).moved)
            for bad in [CGFloat.nan, .infinity, -.infinity] {
                for corner in 0..<4 {
                    // (of the fit's too, as the Python)
                    var broken = board
                    broken[corner].x = bad
                    #expect(GridCorners.windowByModel(fit: broken, model: on).files == 0)
                    #expect(GridCorners.windowByModel(fit: broken, model: on).ranks == 0)
                    #expect(GridCorners.shifted(broken, files: 0, ranks: 1) == nil)
                    var model = on
                    model[corner].x = bad
                    var window = GridCorners.windowByModel(fit: board, model: model)
                    #expect(window.corners == board && !window.moved)
                    model = on
                    model[corner].y = bad
                    window = GridCorners.windowByModel(fit: board, model: model)
                    #expect(window.corners == board && !window.moved)
                }
            }
        }
        // A tolerance of nothing moves nothing but an exact shift.
        var near = rank
        near[0].x += 5
        #expect(GridCorners.windowByModel(fit: fit, model: near).moved)
        #expect(!GridCorners.windowByModel(fit: fit, model: near, near: 0).moved)
    }

    // MARK: - In the recognizer

    /// Where the model and the fit both find the board, the model picks the
    /// fit's window; where the fit stands alone or rejects itself nothing
    /// is moved.
    @Test func testRecognizerMovesTheFitsWindowToWhereTheModelHasTheBoard() throws {
        let (image, truth, clutter) = try Self.drawnBoard()
        let found = try #require(GridCorners.find(in: image)).corners
        // A corner model that has the board one rank on from the fit's, and
        // is a quarter of a square out; and one that agrees with the fit.
        let rank = try #require(GridCorners.shifted(truth, files: 0, ranks: 1))
        let onward = rank.map { CGPoint(x: $0.x + 12, y: $0.y - 9) }
        let agreeing = truth.map { CGPoint(x: $0.x + 12, y: $0.y - 9) }

        let moved = try #require(CoreMLBoardRecognizer.corners(
            model: onward, in: image, gridFit: true, fitWindow: true))
        #expect(moved.fromModel)
        // The fit's lines, the model's window (within a twentieth of a
        // 56-pixel square of the drawing).
        #expect(Self.distance(moved.corners, rank) < 3)
        #expect(Self.distance(moved.corners, onward) > 10)

        // Switched off: the fit's window as it found it.
        let left = try #require(CoreMLBoardRecognizer.corners(
            model: onward, in: image, gridFit: true, fitWindow: false))
        #expect(Self.distance(left.corners, truth) < 3)
        #expect(left.corners == GridCorners.turned(found, nearestTo: onward))

        // The model agrees with the fit: nothing moves.
        let agreed = try #require(CoreMLBoardRecognizer.corners(
            model: agreeing, in: image, gridFit: true, fitWindow: true))
        #expect(agreed.corners == GridCorners.turned(found, nearestTo: agreeing))

        // A model two ranks on, and one with three corners a rank on and
        // the fourth where the fit has it: not to be believed, the fit's
        // window stays (the tolerance is the reference's, not a wider one).
        let twoOn = try #require(GridCorners.shifted(truth, files: 0, ranks: 2))
        var unlike = rank
        unlike[2] = truth[2]
        for model in [twoOn, unlike] {
            let kept = try #require(CoreMLBoardRecognizer.corners(
                model: model, in: image, gridFit: true, fitWindow: true))
            #expect(kept.corners == GridCorners.turned(found, nearestTo: model))
            #expect(Self.distance(kept.corners, truth) < 3)
        }

        // The fit alone, the fit rejecting itself, the fit switched off:
        // the rule has nothing to say.
        let alone = try #require(CoreMLBoardRecognizer.corners(
            model: nil, in: image, gridFit: true, fitWindow: true))
        #expect(alone.corners == found && !alone.fromModel)
        let rejected = try #require(CoreMLBoardRecognizer.corners(
            model: onward, in: clutter, gridFit: true, fitWindow: true))
        #expect(rejected.corners == onward)
        let off = try #require(CoreMLBoardRecognizer.corners(
            model: onward, in: image, gridFit: false, fitWindow: true))
        #expect(off.corners == onward)
    }
}
