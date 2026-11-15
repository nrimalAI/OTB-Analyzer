import ChessKit
import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import ChessVision

/// The side of single pieces (`OddSides`) against `odd_sides` and
/// `patch_hidden` in `Training/chess_pipeline.py`. The reference values are
/// in `Fixtures/odd_cases.json`, written by
/// `Training/swift_reference.py odd-cases`; run that again after changing
/// the Python, and a change in the JSON is a change the Swift must follow.
struct OddSidesTests {

    private struct Recorded: Decodable {
        struct Detection: Decodable {
            let cls: String
            let confidence: Float
            let box: [Double]
        }
        struct Case: Decodable {
            let name: String
            let chroma: Double
            let nearer: Double
            let apart: Double
            let cover: Double
            let detections: [Detection]
            let colours: [[Double]?]
            let hidden: [Bool]
            let sides: [String]
            /// The drawn picture only.
            let file: String?
        }
        let nearer: Double
        let apart: Double
        let spread_floor: Double
        let hidden: Double
        let chroma: Double
        let cases: [Case]
        let picture: Case
    }

    private static let fixtures = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!

    private static let recorded = Result {
        try JSONDecoder().decode(
            Recorded.self, from: Data(contentsOf: fixtures.appending(path: "odd_cases.json")))
    }

    private static func detections(_ item: Recorded.Case) throws -> [PieceDetection] {
        try item.detections.map { recorded in
            let (kind, color) = try #require(CoreMLBoardRecognizer.pieceClass(fromLabel: recorded.cls))
            return PieceDetection(
                piece: kind, color: color, confidence: recorded.confidence,
                boundingBox: CGRect(
                    x: recorded.box[0], y: recorded.box[1],
                    width: recorded.box[2] - recorded.box[0], height: recorded.box[3] - recorded.box[1]),
                // (scores the rule has no business with)
                kindScores: kind == .queen
                    ? [.queen: Double(recorded.confidence)]
                    : [kind: Double(recorded.confidence), .queen: 0.01])
        }
    }

    private static func colours(_ item: Recorded.Case) -> [PieceSides.Lab?] {
        item.colours.map { $0.map { PieceSides.Lab(l: $0[0], a: $0[1], b: $0[2]) } }
    }

    private static func apply(_ item: Recorded.Case) throws -> [PieceDetection] {
        OddSides.apply(
            try detections(item), colours: colours(item), chroma: item.chroma, nearer: item.nearer,
            apart: item.apart, cover: item.cover)
    }

    /// The indices of the pieces a case changes.
    private static func changed(_ name: String) throws -> [Int] {
        let item = try #require(
            try recorded.get().cases.first { $0.name == name }, "no case \(name)")
        let before = try detections(item)
        return try apply(item).enumerated()
            .filter { $0.element.color != before[$0.offset].color }.map(\.offset)
    }

    // MARK: - Against the Python

    @Test func testTheNumbersAreTheReferences() throws {
        let recorded = try Self.recorded.get()
        #expect(OddSides.nearer == recorded.nearer)
        #expect(OddSides.apart == recorded.apart)
        #expect(OddSides.spreadFloor == recorded.spread_floor)
        #expect(OddSides.hidden == recorded.hidden)
        #expect(PieceSides.chromaWeight == recorded.chroma)
    }

    @Test func testComesOutAsInPython() throws {
        let recorded = try Self.recorded.get()
        #expect(recorded.cases.count >= 100)
        var fired = 0, pieces = 0, behind = 0
        for item in recorded.cases {
            let before = try Self.detections(item)
            let after = try Self.apply(item)
            #expect(after.map(PieceSides.label) == item.sides, "\(item.name)")
            for k in before.indices {
                #expect(OddSides.patchHidden(before, k, cover: item.cover) == item.hidden[k],
                        "\(item.name), piece \(k)")
            }
            // Only the side of a piece changes, and only from Black to White.
            #expect(after.count == before.count, "\(item.name)")
            for (a, b) in zip(before, after) {
                #expect(a.piece == b.piece && a.confidence == b.confidence
                    && a.boundingBox == b.boundingBox && a.kindScores == b.kindScores
                    && a.shownConfidence == b.shownConfidence, "\(item.name)")
                #expect(a.color == b.color || (a.color == .black && b.color == .white), "\(item.name)")
            }
            // With the numbers left out it is the reference's numbers.
            if item.chroma == recorded.chroma && item.nearer == recorded.nearer
                && item.apart == recorded.apart && item.cover == recorded.hidden {
                #expect(OddSides.apply(before, colours: Self.colours(item)).map(PieceSides.label)
                    == item.sides, "\(item.name)")
            }
            let changed = zip(before, after).filter { $0.color != $1.color }.count
            fired += changed > 0 ? 1 : 0
            pieces += changed
            behind += item.hidden.filter { $0 }.count
        }
        #expect(fired >= 30 && fired <= recorded.cases.count - 30)
        print("odd sides: \(recorded.cases.count) cases as in Python, a piece changed in \(fired) "
            + "(\(pieces) pieces), \(behind) pieces behind another")
    }

    /// What the cases are there for: each condition of the rule is met from
    /// both sides.
    @Test func testTheCasesMeetEachConditionFromBothSides() throws {
        let expected: [(String, [Int])] = [
            ("0 light pieces called Black", []),
            ("1 light pieces called Black", [12]),
            ("3 light pieces called Black", [12, 13, 14]),
            ("a dark piece called White", []),
            // against the other pieces called Black, as the model called them
            ("two called Black, one of them light", [3]),
            ("three light pieces of four called Black", [7, 8]),
            // the median of each number, not the mean
            ("one piece called White far off in a*", []),
            // twice as near
            ("called Black at L* 53.3", []), ("called Black at L* 53.4", [12]),
            ("called Black at L* 52, nearer 1.5", [12]), ("called Black at L* 52, nearer 3.0", []),
            // three spreads apart
            ("the sides 8.9 apart, the spread on its floor", []),
            ("the sides 9.1 apart, the spread on its floor", [12]),
            ("the sides 14.9 apart, a spread of 5", []), ("the sides 15.1 apart, a spread of 5", [13]),
            ("the sides 10 apart, apart 2.0", [12]), ("the sides 10 apart, apart 4.0", []),
            // White the lighter side
            ("the pieces called White are the darker", []),
            ("the two sides equally light", []), ("White lighter by a hair", [12]),
            ("brown pieces hardly lighter than the black", [10]),
            ("the same with the colour counting once", []),
            // two pieces a side
            ("one piece called White", []), ("two pieces called White", [5]),
            ("one piece called Black", []), ("two pieces called Black", [5]),
            ("two pieces called Black, both light", []),
            ("of three called White two have no colour", []),
            ("the piece called Black has no colour", []),
            ("no piece at all", []),
            // a piece in front
            ("0.19 of the patch behind a piece in front", [0]),
            ("0.21 of the patch behind a piece in front", []),
            ("half the patch behind a box whose foot is higher", [0]),
            ("half the patch behind a box whose foot is level", [0]),
            ("a third of the patch covered, cover 0.4", [0]),
            ("a third of the patch covered, cover 0.3", []),
            ("one of two pieces called White is hidden", []),
            ("the same with the box in front moved away", [6]),
            ("a box of no width", []),
        ]
        for (name, changed) in expected {
            #expect(try Self.changed(name) == changed, "\(name)")
        }
    }

    /// The drawn picture of `PieceSidesTests`: brown and black pieces, three
    /// of the brown ones called Black. The colours are read from the PNG, so
    /// this is the whole step.
    @Test func testMendsTheDrawnPicture() throws {
        let picture = try Self.recorded.get().picture
        let file = try #require(picture.file)
        let source = try #require(
            CGImageSourceCreateWithURL(Self.fixtures.appending(path: file) as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let before = try Self.detections(picture)

        let colours = PieceSides.colours(of: before, in: image)
        for (index, (ours, theirs)) in zip(colours, picture.colours).enumerated() {
            guard let ours, let theirs else {
                #expect(ours == nil && theirs == nil, "box \(index)")
                continue
            }
            // Same pixels (a PNG) and the same formula.
            #expect(max(abs(ours.l - theirs[0]), abs(ours.a - theirs[1]), abs(ours.b - theirs[2])) < 1e-9,
                    "box \(index)")
        }
        let after = OddSides.apply(before, in: image)
        #expect(after.map(PieceSides.label) == picture.sides)
        #expect(zip(before, after).filter { $0.color != $1.color }.count == 3)
        #expect(picture.hidden.allSatisfy { !$0 })
    }

    // MARK: - By itself

    /// Detections and colours that do not belong together, and numbers that
    /// are none, change nothing.
    @Test func testWhatIsNotAnAnswerIsLeftAlone() throws {
        let item = try #require(
            try Self.recorded.get().cases.first { $0.name == "1 light pieces called Black" })
        let before = try Self.detections(item)
        let colours = Self.colours(item)
        #expect(OddSides.apply(before, colours: colours)[12].color == .white)
        // One colour too few.
        #expect(OddSides.apply(before, colours: Array(colours.dropLast())).map(PieceSides.label)
            == before.map(PieceSides.label))
        // A colour that is not a number, of any piece that counts and in
        // any of its three numbers: nothing changes, as in the Python.
        for index in before.indices {
            for part in 0..<3 {
                var broken = colours
                broken[index] = PieceSides.Lab(
                    l: part == 0 ? .nan : 50, a: part == 1 ? .nan : 0, b: part == 2 ? .nan : 0)
                #expect(OddSides.apply(before, colours: broken).map(PieceSides.label)
                    == before.map(PieceSides.label), "colour \(index), part \(part)")
            }
        }
        var all = colours
        for index in all.indices {
            all[index] = PieceSides.Lab(l: .nan, a: .nan, b: .nan)
        }
        #expect(OddSides.apply(before, colours: all).map(PieceSides.label) == before.map(PieceSides.label))
    }

    // MARK: - In the recognizer

    private static let rightWayUp = [
        CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 0), CGPoint(x: 800, y: 0), CGPoint(x: 800, y: 800),
    ]

    /// Eight cream pawns on rank 2 and eight dark ones on rank 7, drawn; a
    /// piece model that reads them rightly but for the cream pawns on c2 and
    /// f2, which it calls Black (also for `EndToEndFixtureTests`).
    static func drawn() throws -> (image: CGImage, found: [PieceDetection]) {
        let (image, boxes) = try PieceSidesTests.drawnPawns()
        let found = boxes.enumerated().map { index, box in
            PieceDetection(
                piece: .pawn, color: index < 8 && index != 2 && index != 5 ? .white : .black,
                confidence: 0.9, boundingBox: box)
        }
        return (image, found)
    }

    private static func placement(
        _ found: [PieceDetection], in image: CGImage, oddSides: Bool, sidesFromColour: Bool = false,
        oneKing: Bool = false
    ) throws -> String {
        let placed = try #require(CoreMLBoardRecognizer.place(
            found, corners: rightWayUp, cornersFromModel: true, in: image,
            sidesFromColour: sidesFromColour, oddSides: oddSides, oneKing: oneKing,
            a1Prior: nil,  // a1 as given
            a1Dark: false, pull: SquareAssignment.defaultPull))
        return FENBuilder.placement(placed.pieces.mapValues(\.piece))
    }

    /// The step in what follows the models: off, the model's sides stand; on,
    /// the two cream pawns called Black are White's.
    @Test func testTheRecognizerMendsTheSidesWhenAsked() throws {
        let (image, found) = try Self.drawn()
        #expect(try Self.placement(found, in: image, oddSides: false) == "8/pppppppp/8/8/8/8/PPpPPpPP/8")
        #expect(try Self.placement(found, in: image, oddSides: true) == "8/pppppppp/8/8/8/8/PPPPPPPP/8")
        // With the sides from the whole board's colours as well, and that
        // step having nothing to say (2 of 16 is under its 15%).
        #expect(try Self.placement(found, in: image, oddSides: true, sidesFromColour: true)
            == "8/pppppppp/8/8/8/8/PPPPPPPP/8")
        #expect(try Self.placement(found, in: image, oddSides: false, sidesFromColour: true)
            == "8/pppppppp/8/8/8/8/PPpPPpPP/8")
    }

    /// The whole board's colours are asked before the single pieces', as in
    /// `evaluate.py`. Three cream pawns called Black and a dark one called
    /// White are four of sixteen against the colours, over that step's 15%:
    /// it puts all four right. Single pieces first, the three cream ones
    /// are mended, one of sixteen is left against the colours, and the dark
    /// pawn stays White.
    @Test func testTheWholeBoardsColoursComeBeforeTheSinglePieces() throws {
        let (image, boxes) = try PieceSidesTests.drawnPawns()
        let found = boxes.enumerated().map { index, box in
            PieceDetection(
                piece: .pawn,
                color: (index < 8 && ![2, 5, 6].contains(index)) || index == 8 ? .white : .black,
                confidence: 0.9, boundingBox: box)
        }
        #expect(try Self.placement(found, in: image, oddSides: false) == "8/Pppppppp/8/8/8/8/PPpPPppP/8")
        #expect(try Self.placement(found, in: image, oddSides: true, sidesFromColour: true)
            == "8/pppppppp/8/8/8/8/PPPPPPPP/8")
        #expect(try Self.placement(found, in: image, oddSides: false, sidesFromColour: true)
            == "8/pppppppp/8/8/8/8/PPPPPPPP/8")
        // By itself the step mends the three and leaves the dark pawn: one
        // way only.
        #expect(try Self.placement(found, in: image, oddSides: true) == "8/Pppppppp/8/8/8/8/PPPPPPPP/8")
    }

    /// What `app_check --colours` is given for this step
    /// (`PieceSides.diagnose`): which pieces stand behind another, and the
    /// class the step gives each piece from the model's own answers, not
    /// from what the other colour step made of them.
    @Test func testTheDiagnosisIsWhatTheStepDoes() throws {
        let picture = try Self.recorded.get().picture
        let file = try #require(picture.file)
        let source = try #require(
            CGImageSourceCreateWithURL(Self.fixtures.appending(path: file) as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        // The drawn pieces as recorded (three brown ones called Black), and
        // a box standing in front of the first of them.
        var recorded = try Self.detections(picture)
        let first = recorded[0].boundingBox
        recorded.append(PieceDetection(
            piece: .pawn, color: .black, confidence: 0.5,
            boundingBox: CGRect(x: first.minX, y: first.minY + 10, width: first.width, height: first.height)))
        // The same with every piece called Black: the whole board's colours
        // then change many sides and the single pieces' none.
        let allBlack = recorded.map { $0.taken(color: .black) }

        for (name, found) in [("as recorded", recorded), ("all called Black", allBlack)] {
            let diagnosis = PieceSides.diagnose(found, in: image)
            let colours = PieceSides.colours(of: found, in: image)
            #expect(diagnosis.map(\.hidden) == found.indices.map { OddSides.patchHidden(found, $0) },
                    "\(name)")
            #expect(diagnosis[0].hidden && diagnosis.filter(\.hidden).count == 1, "\(name)")
            #expect(diagnosis.map(\.oddLabel)
                == OddSides.apply(found, colours: colours).map(PieceSides.label), "\(name)")
            #expect(diagnosis.map(\.finalLabel)
                == PieceSides.fromColour(found, colours: colours).map(PieceSides.label), "\(name)")
        }
        let mended = PieceSides.diagnose(recorded, in: image).filter { $0.oddLabel != $0.label }
        #expect(mended.count >= 2 && mended.allSatisfy { $0.oddLabel.hasPrefix("white_") })
        let none = PieceSides.diagnose(allBlack, in: image)
        #expect(none.allSatisfy { $0.oddLabel == $0.label })
        #expect(none.contains { $0.finalLabel != $0.label })
    }

    /// The sides are mended before the kings are counted, since the kings
    /// are counted by side. A cream king on d2, a dark king on e7, and the
    /// cream piece on f2 called a black king with the higher score. Mended
    /// first, White has two kings and the lesser becomes a queen. Counted
    /// first, Black's two kings would lose the dark one, the real one.
    @Test func testTheSidesAreMendedBeforeTheKingsAreCounted() throws {
        var (image, found) = try Self.drawn()
        func king(_ index: Int, _ color: Piece.Color, _ scores: [Piece.Kind: Double]) {
            found[index] = PieceDetection(
                piece: .king, color: color, confidence: 0.9, boundingBox: found[index].boundingBox,
                kindScores: scores)
        }
        king(3, .white, [.king: 0.95])
        king(12, .black, [.king: 0.6])
        king(5, .black, [.king: 0.9, .queen: 0.3])
        #expect(try Self.placement(found, in: image, oddSides: true, oneKing: true)
            == "8/ppppkppp/8/8/8/8/PPPKPQPP/8")
        // The two steps one at a time, to see that the order is what counts.
        #expect(try Self.placement(found, in: image, oddSides: true) == "8/ppppkppp/8/8/8/8/PPPKPKPP/8")
        #expect(try Self.placement(found, in: image, oddSides: false, oneKing: true)
            == "8/ppppqppp/8/8/8/8/PPpKPkPP/8")
    }
}
