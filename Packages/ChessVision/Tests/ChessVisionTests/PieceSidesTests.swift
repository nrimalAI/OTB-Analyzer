import ChessKit
import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import ChessVision

/// `PieceSides` against `Training/chess_pipeline.py` (`colour_patch`,
/// `two_groups`, `sides_from_colour`) and `Training/evaluate.py`
/// (`piece_colours`). The reference values are in
/// `Fixtures/colour_cases.json`, written by
/// `Training/swift_reference.py colour-cases`; run that again after changing
/// the Python, and a change in the JSON is a change the Swift must follow.
struct PieceSidesTests {

    // MARK: - The recorded cases

    struct Recorded: Decodable {
        struct Colour: Decodable {
            let rgb: [UInt8]
            let lab: [Double]
        }
        struct Patch: Decodable {
            let box: [Double]
            let width: Int
            let height: Int
            let patch: [Int]?
        }
        struct Groups: Decodable {
            let points: [[Double]]
            let group: [Int]?
            let centres: [[Double]]?
        }
        struct Detection: Decodable {
            let cls: String
            let confidence: Float
            let box: [Double]
        }
        struct Sides: Decodable {
            let detections: [Detection]
            let colours: [[Double]?]
            let sides: [String]
        }
        struct Picture: Decodable {
            let file: String
            let detections: [Detection]
            let colours: [[Double]?]
            let sides: [String]
        }
        struct Photo: Decodable {
            let file: String
            let boxes: [[Double]]
            let colours: [[Double]]
        }
        let lab: [Colour]
        let patches: [Patch]
        let groups: [Groups]
        let sides: [Sides]
        let picture: Picture
        let photo: Photo
    }

    private static let fixtures = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!

    private static let recorded = Result {
        try JSONDecoder().decode(
            Recorded.self, from: Data(contentsOf: fixtures.appending(path: "colour_cases.json")))
    }

    private static func image(_ file: String) throws -> CGImage {
        let source = try #require(
            CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private static func rect(_ box: [Double]) -> CGRect {
        CGRect(x: box[0], y: box[1], width: box[2] - box[0], height: box[3] - box[1])
    }

    private static func detection(_ recorded: Recorded.Detection) throws -> PieceDetection {
        let (kind, color) = try #require(CoreMLBoardRecognizer.pieceClass(fromLabel: recorded.cls))
        return PieceDetection(
            piece: kind, color: color, confidence: recorded.confidence,
            boundingBox: rect(recorded.box))
    }

    private static func lab(_ values: [Double]?) -> PieceSides.Lab? {
        values.map { PieceSides.Lab(l: $0[0], a: $0[1], b: $0[2]) }
    }

    /// The largest difference between two colours in any of L*, a*, b*.
    private static func distance(_ ours: PieceSides.Lab, _ theirs: [Double]) -> Double {
        max(abs(ours.l - theirs[0]), abs(ours.a - theirs[1]), abs(ours.b - theirs[2]))
    }

    // MARK: - Against Python

    /// OpenCV does not compute Lab for float pixels, it interpolates in a
    /// table: off by up to 0.2 in L* and 0.4 in a*, b* (measured on 400,000
    /// colours), 0.08 on average. `PieceSides.lab` is the formula.
    @Test func testLabIsOpenCVsToWithinItsTable() throws {
        let colours = try Self.recorded.get().lab
        #expect(colours.count >= 200)
        var worst = 0.0, sum = 0.0
        for colour in colours {
            let ours = PieceSides.lab(red: colour.rgb[0], green: colour.rgb[1], blue: colour.rgb[2])
            let off = Self.distance(ours, colour.lab)
            worst = max(worst, off)
            sum += off
            #expect(off < 0.45, "\(colour.rgb): \(ours) for \(colour.lab)")
        }
        let mean = sum / Double(colours.count)
        print("lab: difference from OpenCV, largest \(worst), mean \(mean)")
        #expect(mean < 0.2)
        let white = PieceSides.lab(red: 255, green: 255, blue: 255)
        #expect(abs(white.l - 100) < 1e-3 && abs(white.a) < 1e-3 && abs(white.b) < 1e-3)
        #expect(PieceSides.lab(red: 0, green: 0, blue: 0) == PieceSides.Lab(l: 0, a: 0, b: 0))
    }

    /// The formula, against values worked out by hand from the CIE
    /// definition with OpenCV's matrix and white point.
    @Test func testLabIsTheFormula() {
        func check(_ r: UInt8, _ g: UInt8, _ b: UInt8, _ expected: (Double, Double, Double)) {
            let ours = PieceSides.lab(red: r, green: g, blue: b)
            #expect(abs(ours.l - expected.0) < 5e-4 && abs(ours.a - expected.1) < 5e-4
                && abs(ours.b - expected.2) < 5e-4, "\((r, g, b)): \(ours)")
        }
        check(255, 0, 0, (53.2406, 80.0942, 67.2015))
        check(0, 255, 0, (87.7351, -86.1813, 83.1775))
        check(0, 0, 255, (32.2957, 79.1870, -107.8617))
        check(128, 128, 128, (53.5850, 0, 0))
        check(190, 140, 95, (62.1104, 13.6017, 31.3877))
        check(40, 36, 38, (14.6952, 2.3302, -0.6547))
        // Dark enough for the straight part of the curve.
        check(5, 9, 3, (2.1156, -1.8803, 1.9022))
        check(11, 11, 11, (3.0229, 0, 0))
    }

    @Test func testColourPatchIsPythons() throws {
        let patches = try Self.recorded.get().patches
        #expect(patches.contains { $0.patch == nil } && patches.contains { $0.patch != nil })
        for (index, patch) in patches.enumerated() {
            let ours = PieceSides.colourPatch(
                Self.rect(patch.box), width: patch.width, height: patch.height)
            #expect(ours.map { [$0.x0, $0.y0, $0.x1, $0.y1] } == patch.patch, "patch \(index)")
        }
    }

    @Test func testColourPatchOfBoxesThatAreNotBoxes() {
        // The middle of a 100 x 200 box at (10, 20): x 40...80, y 60...190.
        #expect(PieceSides.colourPatch(CGRect(x: 10, y: 20, width: 100, height: 200), width: 500, height: 500)
            == PieceSides.PixelRect(x0: 40, y0: 60, x1: 81, y1: 191))
        #expect(PieceSides.colourPatch(.null, width: 500, height: 500) == nil)
        #expect(PieceSides.colourPatch(.infinite, width: 500, height: 500) == nil)
        #expect(PieceSides.colourPatch(
            CGRect(x: Double.nan, y: 0, width: 10, height: 10), width: 500, height: 500) == nil)
        #expect(PieceSides.colourPatch(CGRect(x: 0, y: 0, width: 10, height: 10), width: 0, height: 0) == nil)
    }

    @Test func testTwoGroupsIsPythons() throws {
        let cases = try Self.recorded.get().groups
        #expect(cases.contains { $0.group == nil } && cases.contains { $0.group != nil })
        for (index, recorded) in cases.enumerated() {
            let points = recorded.points.map { PieceSides.Point($0[0], $0[1], $0[2]) }
            let ours = PieceSides.twoGroups(points)
            #expect(ours?.group == recorded.group, "points \(index)")
            guard let ours, let centres = recorded.centres else {
                #expect(ours == nil && recorded.centres == nil, "points \(index)")
                continue
            }
            for (mine, theirs) in zip([ours.centres.0, ours.centres.1], centres) {
                let off = max(abs(mine.x - theirs[0]), abs(mine.y - theirs[1]), abs(mine.z - theirs[2]))
                #expect(off < 1e-9, "points \(index): centre \(mine) for \(theirs)")
            }
        }
    }

    @Test func testSidesFromColourIsPythons() throws {
        let cases = try Self.recorded.get().sides
        var changed = 0, kept = 0
        for (index, recorded) in cases.enumerated() {
            let detections = try recorded.detections.map(Self.detection)
            let ours = PieceSides.fromColour(detections, colours: recorded.colours.map(Self.lab))
            #expect(ours.map(PieceSides.label) == recorded.sides, "detections \(index)")
            // A piece keeps its kind, confidence and box.
            #expect(ours.map(\.piece) == detections.map(\.piece), "detections \(index)")
            #expect(ours.map(\.confidence) == detections.map(\.confidence), "detections \(index)")
            #expect(ours.map(\.boundingBox) == detections.map(\.boundingBox), "detections \(index)")
            if recorded.sides == recorded.detections.map(\.cls) {
                kept += 1
            } else {
                changed += 1
            }
        }
        // Both answers are among the cases, or the test shows nothing.
        #expect(changed >= 20 && kept >= 20, "\(changed) changed, \(kept) kept")
    }

    /// The drawn picture: brown and black pieces that a model calls all
    /// black. The colours are read from the PNG, so this is the whole step.
    @Test func testReadsTheSidesOffTheDrawnPicture() throws {
        let picture = try Self.recorded.get().picture
        let image = try Self.image(picture.file)
        let detections = try picture.detections.map(Self.detection)

        let colours = PieceSides.colours(of: detections, in: image)
        #expect(colours.count == picture.colours.count)
        for (index, (ours, theirs)) in zip(colours, picture.colours).enumerated() {
            guard let ours, let theirs else {
                #expect(ours == nil && theirs == nil, "box \(index)")
                continue
            }
            // Same pixels (a PNG): what is left is OpenCV's table.
            #expect(Self.distance(ours, theirs) < 0.3, "box \(index): \(ours) for \(theirs)")
        }

        let sides = PieceSides.fromColour(detections, in: image)
        #expect(sides.map(PieceSides.label) == picture.sides)
        #expect(sides.contains { $0.color == .white } && detections.allSatisfy { $0.color == .black })
    }

    /// A photo: a JPEG, which ImageIO and OpenCV decode a grey level apart
    /// here and there; the median of a patch hardly moves.
    @Test func testColoursOfAPhotoAreNearPythons() throws {
        let photo = try Self.recorded.get().photo
        let image = try Self.image(photo.file)
        let detections = photo.boxes.map {
            PieceDetection(piece: .pawn, color: .white, confidence: 0.9, boundingBox: Self.rect($0))
        }
        let colours = PieceSides.colours(of: detections, in: image)
        var worst = 0.0
        for (index, (ours, theirs)) in zip(colours, photo.colours).enumerated() {
            let ours = try #require(ours, "box \(index)")
            let off = Self.distance(ours, theirs)
            worst = max(worst, off)
            #expect(off < 1.0, "box \(index): \(ours) for \(theirs)")
        }
        print("photo colours: largest difference from Python \(worst)")
    }

    // MARK: - By hand

    private static func pieces(_ sides: [Piece.Color]) -> [PieceDetection] {
        sides.enumerated().map { index, side in
            PieceDetection(
                piece: .pawn, color: side, confidence: 0.9,
                boundingBox: CGRect(x: Double(index) * 50, y: 0, width: 40, height: 80))
        }
    }

    private static let cream = PieceSides.Lab(l: 80, a: 2, b: 20)
    private static let ebony = PieceSides.Lab(l: 20, a: 1, b: 2)

    @Test func testTheModelStandsUnlessContradictedForMoreThanTheShare() {
        // 20 pieces, ten of each colour, the model right about all but some.
        let truth = (0..<20).map { $0 % 2 == 0 ? Piece.Color.white : .black }
        let colours: [PieceSides.Lab?] = truth.map { $0 == .white ? Self.cream : Self.ebony }
        func said(wrong: Int) -> [PieceDetection] {
            Self.pieces(truth.enumerated().map { index, side in
                index < wrong ? (side == .white ? .black : .white) : side
            })
        }
        // 3 of 20 is 15%: not more than 15%, the model's sides stand.
        #expect(PieceSides.fromColour(said(wrong: 3), colours: colours).map(\.color)
            == said(wrong: 3).map(\.color))
        // 4 of 20: the colours decide, for every piece.
        #expect(PieceSides.fromColour(said(wrong: 4), colours: colours).map(\.color) == truth)
        #expect(PieceSides.fromColour(said(wrong: 20), colours: colours).map(\.color) == truth)
        // Another share.
        #expect(PieceSides.fromColour(said(wrong: 4), colours: colours, disagreement: 0.2).map(\.color)
            == said(wrong: 4).map(\.color))
    }

    @Test func testNothingToSplit() {
        let allBlack = Self.pieces([.black, .black, .black])
        // No colours, one colour, or the same colour for all: the model's sides stand.
        #expect(PieceSides.fromColour(allBlack, colours: [nil, nil, nil]).map(\.color) == [.black, .black, .black])
        #expect(PieceSides.fromColour(allBlack, colours: [Self.cream, nil, nil]).map(\.color)
            == [.black, .black, .black])
        #expect(PieceSides.fromColour(allBlack, colours: [Self.cream, Self.cream, Self.cream]).map(\.color)
            == [.black, .black, .black])
        // A list of colours that is not the detections': nothing is done.
        #expect(PieceSides.fromColour(allBlack, colours: [Self.cream, Self.ebony]).map(\.color)
            == [.black, .black, .black])
        #expect(PieceSides.fromColour([], colours: []).isEmpty)
        // A piece without a colour keeps the model's side when the rest change.
        #expect(PieceSides.fromColour(allBlack, colours: [Self.cream, Self.ebony, nil]).map(\.color)
            == [.white, .black, .black])
    }

    @Test func testTheLighterGroupIsWhiteWhateverItsColour() {
        // Brown "white" pieces darker than cream, and still the lighter group.
        let brown = PieceSides.Lab(l: 45, a: 14, b: 28)
        let sides = PieceSides.fromColour(
            Self.pieces([.black, .black, .black, .black]),
            colours: [Self.ebony, brown, brown, Self.ebony])
        #expect(sides.map(\.color) == [.black, .white, .white, .black])
        // Equally light, told apart by colour alone: the weight on a*, b*.
        let grey = PieceSides.Lab(l: 45, a: 0, b: 0)
        let split = PieceSides.twoGroups(
            [grey, brown, grey, brown].map { PieceSides.Point($0.l, 4 * $0.a, 4 * $0.b) })
        #expect(split?.group == [0, 1, 0, 1])
    }

    @Test func testTwoGroupsTiesGoToTheFirstGroup() {
        // The middle point is as near to one end as to the other.
        let split = PieceSides.twoGroups(
            [PieceSides.Point(0, 0, 0), PieceSides.Point(1, 0, 0), PieceSides.Point(2, 0, 0)])
        #expect(split?.group == [0, 0, 1])
        #expect(split?.centres.0 == PieceSides.Point(0.5, 0, 0))
        #expect(split?.centres.1 == PieceSides.Point(2, 0, 0))
        #expect(PieceSides.twoGroups([PieceSides.Point(1, 2, 3)]) == nil)
        #expect(PieceSides.twoGroups([PieceSides.Point(1, 2, 3), PieceSides.Point(1, 2, 3)]) == nil)
    }

    /// An image the pixels of which cannot be had gives no colours, not a crash.
    @Test func testBoxesBesideThePhotoHaveNoColour() throws {
        let image = try Self.image("colour_pieces.png")
        let outside = PieceDetection(
            piece: .pawn, color: .black, confidence: 0.5,
            boundingBox: CGRect(x: 2000, y: 2000, width: 40, height: 80))
        #expect(PieceSides.colours(of: [outside], in: image) == [nil])
        #expect(PieceSides.colours(of: [], in: image).isEmpty)
        #expect(PieceSides.fromColour([outside], in: image).map(\.color) == [.black])
    }
}
