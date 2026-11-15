import ChessKit
import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import ChessVision

/// a1 is a dark square (`GridCorners.squareColours`) against
/// `dark_corner_rotations` in `Training/grid_corners.py`. The reference
/// values are in `Fixtures/dark_cases.json`, written by
/// `Training/swift_reference.py dark-cases`.
struct GridSquareColoursTests {

    private struct Recorded: Decodable {
        struct Checker: Decodable {
            let name: String
            let width: Int
            let height: Int
            let cell: Int
            let origin: [Int]
            let even: Int
            let odd: Int
            let background: Int
            let swapped: [[Int]]
            /// Cells at a grey of their own: column, row, level.
            let levels: [[Int]]
            let ring: [[Double]]
            let rotations: [Int]?
            let strength: Double
        }
        struct Turned: Decodable {
            let name: String
            let width: Int
            let height: Int
            let cell: Double
            let centre: [Double]
            let cos: Double
            let sin: Double
            let even: Int
            let odd: Int
            let background: Int
            let seen: Int
            let ring: [[Double]]
            let rotations: [Int]?
            let strength: Double
        }
        struct Picture: Decodable {
            let file: String
            let label: String
            let ring: [[Double]]
            let rotations: [Int]?
            let strength: Double
        }
        let checkers: [Checker]
        let turned: [Turned]
        let boards: [Picture]
        let photos: [Picture]
    }

    private static let fixtures = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!

    private static let recorded = Result {
        try JSONDecoder().decode(
            Recorded.self, from: Data(contentsOf: fixtures.appending(path: "dark_cases.json")))
    }

    private static func image(_ file: String) throws -> CGImage {
        let source = try #require(
            CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private static func points(_ ring: [[Double]]) -> [CGPoint] {
        ring.map { CGPoint(x: $0[0], y: $0[1]) }
    }

    /// The chequered picture of a recorded case, as `swift_reference.checker`
    /// draws it: 8×8 cells from `origin`, the cells whose column and row add
    /// up to an even number at `even`, the others at `odd`, the ones listed
    /// in `swapped` the other way round, the ones in `levels` at a grey of
    /// their own.
    private static func checker(_ c: Recorded.Checker) -> GrayImage {
        var gray = GrayImage(width: c.width, height: c.height, fill: Float(c.background))
        let swapped = Set(c.swapped.map { $0[0] * 8 + $0[1] })
        let other = Dictionary(uniqueKeysWithValues: c.levels.map { ($0[0] * 8 + $0[1], $0[2]) })
        for j in 0..<8 {
            for i in 0..<8 {
                let even = ((i + j) % 2 == 0) != swapped.contains(i * 8 + j)
                let level = Float(other[i * 8 + j] ?? (even ? c.even : c.odd))
                let x0 = max(c.origin[0] + i * c.cell, 0)
                let x1 = min(max(c.origin[0] + (i + 1) * c.cell, 0), c.width)
                let y0 = max(c.origin[1] + j * c.cell, 0)
                let y1 = min(max(c.origin[1] + (j + 1) * c.cell, 0), c.height)
                guard x0 < x1, y0 < y1 else { continue }
                for y in y0..<y1 {
                    for x in x0..<x1 {
                        gray[x, y] = level
                    }
                }
            }
        }
        return gray
    }

    /// The same board turned in the picture, as `swift_reference.turned_checker`
    /// draws it: the pixel (x, y) lies on the board at (u, v), in the cell
    /// (floor u, floor v) when both are within 0..<8.
    private static func turnedChecker(_ c: Recorded.Turned) -> GrayImage {
        var gray = GrayImage(width: c.width, height: c.height, fill: Float(c.background))
        for y in 0..<c.height {
            for x in 0..<c.width {
                let dx = Double(x) - c.centre[0], dy = Double(y) - c.centre[1]
                let u = (dx * c.cos + dy * c.sin) / c.cell + 4
                let v = (-dx * c.sin + dy * c.cos) / c.cell + 4
                guard u >= 0, u < 8, v >= 0, v < 8 else { continue }
                let even = (Int(u.rounded(.down)) + Int(v.rounded(.down))) % 2 == 0
                gray[x, y] = Float(even ? c.even : c.odd)
            }
        }
        return gray
    }

    // MARK: - Against the Python

    /// Chequered pictures both sides draw from the same numbers, so that the
    /// answers must be equal to the last digit: which corner is dark, the
    /// ring in each rotation, the thresholds for the contrast (15 grey
    /// levels and 14) and for the split (0.5 and one cell less), cells seen
    /// in whole columns (40 and 32; one by one in the next test), and cells
    /// exactly at the middle grey, which are not light.
    @Test func testChequeredPicturesComeOutAsInPython() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.checkers.count >= 17)
        #expect(recorded.checkers.contains { !$0.levels.isEmpty })
        var answered = 0, refused = 0
        for c in recorded.checkers {
            let mine = GridCorners.squareColours(
                Self.checker(c), ring: c.ring.map { ($0[0], $0[1]) })
            #expect(mine.rotations == c.rotations, "\(c.name)")
            #expect(abs(mine.strength - c.strength) < 1e-12, "\(c.name): \(mine.strength)")
            if c.rotations == nil { refused += 1 } else { answered += 1 }
        }
        // Both kinds are among the cases.
        #expect(answered >= 8 && refused >= 6)
        #expect(GridCorners.leastCells == 40)
        #expect(GridCorners.leastContrast == 15)
        #expect(GridCorners.leastStrength == 0.5)
    }

    /// 40 cells must be seen, and 39 are too few: a board turned in the
    /// picture, which is cut narrower until its edge has taken cells one at
    /// a time down to 40, and one pixel further to 39.
    @Test func testFortyCellsAreEnoughAndThirtyNineAreNot() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.turned.map(\.seen) == [64, 40, 39])
        for c in recorded.turned {
            let mine = GridCorners.squareColours(
                Self.turnedChecker(c), ring: c.ring.map { ($0[0], $0[1]) })
            #expect(mine.rotations == c.rotations, "\(c.name)")
            #expect(abs(mine.strength - c.strength) < 1e-12, "\(c.name): \(mine.strength)")
            #expect((mine.rotations != nil) == (c.seen >= 40), "\(c.name)")
        }
    }

    /// The boards drawn for the grid fit (PNG: the same pixels on both
    /// sides), each with its true corners in the four rotations, a ring one
    /// square along (the colours swapped, a line of cells off the board), a
    /// ring half a square off (no cell of one colour) and a 10×10 board; and
    /// a table with no board.
    @Test func testDrawnBoardsComeOutAsInPython() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.boards.count >= 36)
        for board in recorded.boards {
            let mine = GridCorners.squareColours(
                in: try Self.image(board.file), corners: Self.points(board.ring))
            #expect(mine.rotations == board.rotations, "\(board.file), \(board.label)")
            #expect(
                abs(mine.strength - board.strength) < 1e-9,
                "\(board.file), \(board.label): \(mine.strength), Python \(board.strength)")
        }
        // A quarter turn of the ring swaps the answer, half a turn does not.
        for file in ["grid_board_plain.png", "grid_board_pieces.png", "grid_board_cut.png", "grid_board_busy.png"] {
            let turns = try (0..<4).map { k in
                try #require(recorded.boards.first { $0.file == file && $0.label == "turned by \(k)" })
                    .rotations
            }
            #expect(turns[0] != nil && turns[0] == turns[2] && turns[1] == turns[3], "\(file)")
            #expect(Set(turns[0]! + turns[1]!) == [0, 1, 2, 3], "\(file)")
        }
    }

    /// The four test photos, on the corners the Python's grid fit gives
    /// them, in each rotation. JPEG decoders differ by a grey level here and
    /// there, which can move a cell that is half hidden by a piece across
    /// the middle: the answer must be the same, the strength within two
    /// cells of 32.
    @Test func testFixturePhotosComeOutAsInPython() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.photos.count == 16)
        var worst = 0.0
        for photo in recorded.photos {
            #expect(photo.rotations != nil, "\(photo.file), \(photo.label)")
            let mine = GridCorners.squareColours(
                in: try Self.image(photo.file), corners: Self.points(photo.ring))
            #expect(mine.rotations == photo.rotations, "\(photo.file), \(photo.label)")
            worst = max(worst, abs(mine.strength - photo.strength))
        }
        print(String(format: "square colours of the fixture photos: strength within %.4f of Python", worst))
        #expect(worst <= 2.0 / 32 + 1e-9)
    }

    // MARK: - By hand

    /// Corners that are no ring of four, or not numbers: no answer, no crash.
    @Test func testCornersThatAreNotARingSayNothing() throws {
        let image = try Self.image("grid_board_plain.png")
        let none = GridCorners.SquareColours(rotations: nil, strength: 0)
        #expect(GridCorners.squareColours(in: image, corners: []) == none)
        #expect(GridCorners.squareColours(
            in: image, corners: [.zero, CGPoint(x: 100, y: 0), CGPoint(x: 100, y: 100)]) == none)
        let line = (0..<4).map { CGPoint(x: 100 + 100 * Double($0), y: 200) }
        #expect(GridCorners.squareColours(in: image, corners: line) == none)
        let notNumbers = [
            CGPoint(x: 100, y: 100), CGPoint(x: Double.nan, y: 400),
            CGPoint(x: 500, y: 400), CGPoint(x: 500, y: 100),
        ]
        #expect(GridCorners.squareColours(in: image, corners: notNumbers) == none)
        // A ring far outside the photo sees no cell.
        let outside = [
            CGPoint(x: 5000, y: 5000), CGPoint(x: 5000, y: 5400),
            CGPoint(x: 5400, y: 5400), CGPoint(x: 5400, y: 5000),
        ]
        #expect(GridCorners.squareColours(in: image, corners: outside) == none)
    }

    /// The recognizer shrinks the photo once and hands the result to the
    /// grid fit and to the square colours: both must give what they give
    /// when they shrink it themselves.
    @Test func testTheWorkingImageCanBeHandedIn() throws {
        let recorded = try Self.recorded.get()
        for file in ["grid_board_pieces.png", "test_000102.jpg"] {
            let image = try Self.image(file)
            let whole = try #require(GridCorners.workingImage(of: image))
            #expect(GrayImage.isWhole(whole, of: image))
            let fit = try #require(GridCorners.find(in: image))
            let handedFit = GridCorners.find(in: image, whole: whole)
            #expect(handedFit?.corners == fit.corners, "\(file)")
            #expect(handedFit?.confidence == fit.confidence, "\(file)")
            for turn in 0..<4 {
                let ring = Array(fit.corners[turn...] + fit.corners[..<turn])
                let alone = GridCorners.squareColours(in: image, corners: ring)
                #expect(alone.rotations != nil)
                #expect(GridCorners.squareColours(in: image, corners: ring, whole: whole) == alone)
            }
            let model = fit.corners.map { CGPoint(x: $0.x + 20, y: $0.y - 15) }
            let handed = try #require(CoreMLBoardRecognizer.corners(
                model: model, in: image, gridFit: true, fitWindow: false, whole: whole))
            let own = try #require(CoreMLBoardRecognizer.corners(
                model: model, in: image, gridFit: true, fitWindow: false))
            #expect(handed.corners == own.corners && handed.fromModel == own.fromModel)

            // A working image that is not the whole photo's (here: of its
            // top left part only) is not used.
            let part = try #require(GrayImage.working(
                from: image, priorBox: CGRect(x: 0, y: 0, width: image.width / 2, height: image.height / 2),
                pad: 0))
            #expect(part.originX == 0 && part.originY == 0)
            #expect(!GrayImage.isWhole(part, of: image))
            #expect(GridCorners.squareColours(in: image, corners: fit.corners, whole: part)
                == GridCorners.squareColours(in: image, corners: fit.corners))
        }
        #expect(recorded.photos.contains { $0.file == "test_000102.jpg" })
    }

    // MARK: - In the recognizer

    /// The bare drawn board and its true corners as recorded, which start on
    /// a light square (also for `EndToEndFixtureTests`).
    static func plainBoard() throws -> (image: CGImage, light: [CGPoint]) {
        let recorded = try Self.recorded.get()
        let light = Self.points(try #require(
            recorded.boards.first { $0.file == "grid_board_plain.png" && $0.label == "turned by 0" }).ring)
        return (try Self.image("grid_board_plain.png"), light)
    }

    /// Boxes standing on the middles of squares of the board whose corners
    /// (a1, a8, h8, h1) are `frame` (also for `EndToEndFixtureTests`).
    static func pieces(
        _ placement: String, on frame: [CGPoint]
    ) throws -> [PieceDetection] {
        let toImage = try #require(GridMath.homography(
            from: [(0, 0), (0, 8), (8, 8), (8, 0)],
            to: frame.map { (Double($0.x), Double($0.y)) }))
        return try #require(FENBuilder.pieces(fromPlacement: placement)).map { square, piece in
            let foot = toImage.apply(
                Double(square.file.number) - 0.5, Double(square.rank.value) - 0.5)
            return PieceDetection(
                piece: piece.kind, color: piece.color, confidence: 0.9,
                boundingBox: CGRect(x: foot.x - 10, y: foot.y - 30, width: 20, height: 30))
        }
    }

    private static func turned(_ corners: [CGPoint], by turn: Int) -> [CGPoint] {
        Array(corners[turn...] + corners[..<turn])
    }

    private static func place(
        _ found: [PieceDetection], corners: [CGPoint], in image: CGImage, a1Dark: Bool,
        fromModel: Bool = true, a1Prior: Double? = 6
    ) throws -> [CGPoint] {
        try #require(CoreMLBoardRecognizer.place(
            found, corners: corners, cornersFromModel: fromModel, in: image,
            sidesFromColour: false, oddSides: false, oneKing: false, queenForKing: false, a1Prior: a1Prior,
            a1Dark: a1Dark, pull: 0)).corners
    }

    /// The pieces cannot turn a board by a quarter. A drawn board whose true
    /// corners, as recorded, start on a light square: the pieces stand as if
    /// that corner were a1 (a start position, which says so loudly).
    @Test func testThePiecesChooseBetweenTheTwoWaysTheColoursAllow() throws {
        let (image, light) = try Self.plainBoard()
        #expect(GridCorners.squareColours(in: image, corners: light).rotations == [1, 3])
        let found = try Self.pieces("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQK3", on: light)

        // Without the colours the pieces have it their way, wherever the
        // corners start.
        for turn in 0..<4 {
            let corners = Self.turned(light, by: turn)
            #expect(try Self.place(found, corners: corners, in: image, a1Dark: false) == light)
        }
        // With them the answer starts on a dark square: a quarter turn away
        // from what the pieces say, and the better of the two for them.
        for turn in 0..<4 {
            let corners = Self.turned(light, by: turn)
            let allowed = turn % 2 == 0 ? [1, 3] : [0, 2]
            #expect(GridCorners.squareColours(in: image, corners: corners).rotations == allowed)
            let placed = try Self.place(found, corners: corners, in: image, a1Dark: true)
            #expect(placed == Self.turned(light, by: 1) || placed == Self.turned(light, by: 3))
            #expect(placed == BoardOrientation.orient(
                corners: corners, detections: found, pull: 0, prior: 6, turns: allowed))
            // The grid fit's corners alone: no head start, the same rule.
            let alone = try Self.place(found, corners: corners, in: image, a1Dark: true, fromModel: false)
            #expect(alone == BoardOrientation.orient(
                corners: corners, detections: found, pull: 0, prior: 0, turns: allowed))
        }
        // Where a1 is not checked against the pieces at all, the corners
        // stay as they are, light a1 or not.
        #expect(try Self.place(found, corners: light, in: image, a1Dark: true, a1Prior: nil) == light)
    }

    /// Pieces that agree with the colours are not disturbed, and half a turn
    /// is still theirs to decide.
    @Test func testHalfATurnIsStillThePieces() throws {
        let recorded = try Self.recorded.get()
        let image = try Self.image("grid_board_plain.png")
        let dark = Self.points(try #require(
            recorded.boards.first { $0.file == "grid_board_plain.png" && $0.label == "turned by 1" }).ring)
        #expect(GridCorners.squareColours(in: image, corners: dark).rotations == [0, 2])
        let found = try Self.pieces("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", on: dark)
        for a1Dark in [false, true] {
            #expect(try Self.place(found, corners: dark, in: image, a1Dark: a1Dark) == dark)
            #expect(try Self.place(
                found, corners: Self.turned(dark, by: 2), in: image, a1Dark: a1Dark) == dark)
        }
        // Corners a quarter off start on a light square: the two ways round
        // on offer are the ones a quarter on from them, and the pieces take
        // the right one of the two.
        for turn in [1, 3] {
            let quarter = Self.turned(dark, by: turn)
            #expect(GridCorners.squareColours(in: image, corners: quarter).rotations == [1, 3])
            #expect(try Self.place(found, corners: quarter, in: image, a1Dark: false) == dark)
            #expect(try Self.place(found, corners: quarter, in: image, a1Dark: true) == dark)
        }
        // An endgame that says little, upside down: the head start keeps the
        // corners as given, with the colours as without (rotation 0 is one
        // of the two allowed).
        let few = try Self.pieces("7Q/8/1k3p2/4p3/8/p3K3/8/8", on: dark)
        let upsideDown = Self.turned(dark, by: 2)
        for a1Dark in [false, true] {
            #expect(try Self.place(few, corners: upsideDown, in: image, a1Dark: a1Dark) == upsideDown)
            #expect(try Self.place(
                few, corners: upsideDown, in: image, a1Dark: a1Dark, fromModel: false) == dark)
        }
    }

    /// Where the colours say nothing, all four ways stay open, as before.
    @Test func testUnclearColoursLeaveAllFourWays() throws {
        let image = try Self.image("grid_board_clutter.png")
        let ring = [
            CGPoint(x: 150, y: 500), CGPoint(x: 150, y: 100),
            CGPoint(x: 650, y: 100), CGPoint(x: 650, y: 500),
        ]
        #expect(GridCorners.squareColours(in: image, corners: ring).rotations == nil)
        let found = try Self.pieces("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", on: ring)
        for turn in 0..<4 {
            let corners = Self.turned(ring, by: turn)
            #expect(try Self.place(found, corners: corners, in: image, a1Dark: true) == ring)
        }
    }
}
