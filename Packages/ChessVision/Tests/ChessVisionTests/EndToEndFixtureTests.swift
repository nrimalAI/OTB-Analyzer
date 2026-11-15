import ChessKit
import CoreML
import ImageIO
import Foundation
import Testing

@testable import ChessVision

/// Runs the full Swift recognition pipeline (both CoreML models + geometry)
/// against real ChessReD test photos with known positions.
///
/// Skips when the exported models aren't present (`Models/*.mlpackage` at the
/// repo root — produced by `Training/export_coreml.py`), so `swift test`
/// stays green on a fresh clone.
@Suite(.serialized) struct EndToEndFixtureTests {

    /// Repo root, derived from this file's location.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // ChessVisionTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // ChessVision
            .deletingLastPathComponent()  // Packages
            .deletingLastPathComponent()  // repo root
    }

    /// The folder named by the environment variable
    /// `CHESSVISION_TEST_MODELS`, to run these tests on a candidate pair
    /// without putting it in `Models/`; nil when it is not set or empty.
    private static var otherModels: URL? {
        guard let path = ProcessInfo.processInfo.environment["CHESSVISION_TEST_MODELS"], !path.isEmpty
        else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// `Models/` at the repo root, or `otherModels`.
    private static var modelPackages: (corners: URL, pieces: URL) {
        let models = otherModels ?? repoRoot.appending(path: "Models")
        return (models.appending(path: "BoardCorners.mlpackage"),
                models.appending(path: "Pieces.mlpackage"))
    }

    /// Whether the exported models are there. The tests below are skipped,
    /// not failed, when `Models/` is not. A folder asked for by name is
    /// taken to be there: if it is not, or does not hold the two packages
    /// under these names, the tests fail when they load them, and a run
    /// that read no photo is not mistaken for a pass.
    static var modelsPresent: Bool {
        if otherModels != nil {
            return true
        }
        let packages = modelPackages
        return FileManager.default.fileExists(atPath: packages.corners.path)
            && FileManager.default.fileExists(atPath: packages.pieces.path)
    }

    /// Compiled and loaded once for all tests (a static is initialised once,
    /// even when tests run in parallel).
    private static let models = Result { () throws -> (corners: MLModel, pieces: MLModel) in
        // .mlpackage must be compiled before loading (Xcode normally does this
        // at build time; tests do it at runtime).
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let packages = modelPackages
        if let otherModels {
            print("models from CHESSVISION_TEST_MODELS: \(otherModels.path)")
        }
        let corners = try MLModel(
            contentsOf: MLModel.compileModel(at: packages.corners), configuration: config)
        let pieces = try MLModel(
            contentsOf: MLModel.compileModel(at: packages.pieces), configuration: config)
        return (corners, pieces)
    }

    /// The recognizer as the app builds it (every default left alone), or
    /// with the grid fit switched off, or with the colour step switched on.
    private func makeRecognizer(
        gridFit: Bool? = nil, sidesFromColour: Bool? = nil
    ) throws -> CoreMLBoardRecognizer {
        let models = try Self.models.get()
        switch (gridFit, sidesFromColour) {
        case (nil, nil):
            return try CoreMLBoardRecognizer(cornerModel: models.corners, pieceModel: models.pieces)
        case let (gridFit?, nil):
            return try CoreMLBoardRecognizer(
                cornerModel: models.corners, pieceModel: models.pieces, gridFit: gridFit)
        case let (nil, sides?):
            return try CoreMLBoardRecognizer(
                cornerModel: models.corners, pieceModel: models.pieces, sidesFromColour: sides)
        case let (gridFit?, sides?):
            return try CoreMLBoardRecognizer(
                cornerModel: models.corners, pieceModel: models.pieces, gridFit: gridFit,
                sidesFromColour: sides)
        }
    }

    /// One king a side is on unless switched off (`docs/RECOGNITION.md`, "What
    /// one king a side is worth").
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testOneKingASideIsOnByDefault() throws {
        let models = try Self.models.get()
        #expect(try makeRecognizer().oneKing == true)
        #expect(try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, oneKing: false).oneKing == false)
    }

    /// a1 is a dark square: on unless switched off (`docs/RECOGNITION.md`,
    /// "What a dark a1 is worth").
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testA1DarkIsOnByDefault() throws {
        let models = try Self.models.get()
        #expect(try makeRecognizer().a1Dark == true)
        #expect(try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, a1Dark: false).a1Dark == false)
    }

    /// The corner model picks the grid fit's window: on unless switched off
    /// (`docs/RECOGNITION.md`, "What the model's choice of window is worth"),
    /// and the recognizer hands its own setting on to where the two sets of
    /// corners meet (`corners(model:in:)`, the method `recognize` calls).
    /// The fixture photos read the same with the rule and
    /// without, so they would not notice it missing: here a drawn board and
    /// a model that has it one rank on.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testFitWindowIsOnByDefaultAndReachesTheCorners() throws {
        let models = try Self.models.get()
        let asBuilt = try makeRecognizer()
        let without = try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, fitWindow: false)
        #expect(asBuilt.fitWindow == true && CoreMLBoardRecognizer.Default.fitWindow == true)
        #expect(without.fitWindow == false)

        let (image, truth, _) = try GridFitWindowTests.drawnBoard()
        let rank = try #require(GridCorners.shifted(truth, files: 0, ranks: 1))
        func off(_ corners: [CGPoint], _ from: [CGPoint]) -> CGFloat {
            zip(corners, from).map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? .infinity
        }
        let moved = try #require(asBuilt.corners(model: rank, in: image))
        #expect(off(moved.corners, rank) < 3)
        let left = try #require(without.corners(model: rank, in: image))
        #expect(off(left.corners, truth) < 3)
        // The other defaults are the ones named.
        #expect(asBuilt.oneKing == CoreMLBoardRecognizer.Default.oneKing)
        #expect(asBuilt.a1Dark == CoreMLBoardRecognizer.Default.a1Dark)
        #expect(asBuilt.sidesFromColour == CoreMLBoardRecognizer.Default.sidesFromColour)
        #expect(CoreMLBoardRecognizer.Default.a1Prior == BoardOrientation.appPrior)
        #expect(CoreMLBoardRecognizer.Default.gridFit)
    }

    /// The side of single pieces is on unless switched off
    /// (`docs/RECOGNITION.md`, "What the side of single pieces is worth"),
    /// and the recognizer hands its own setting on to what follows the
    /// models (`place`, the method `recognize` calls). The fixture photos
    /// read the same with the step and without, so they would not notice
    /// it missing: here a drawn board with two cream pawns the model calls
    /// Black.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testOddSidesIsOnByDefaultAndReachesWhatFollowsTheModels() throws {
        let models = try Self.models.get()
        let asBuilt = try makeRecognizer()
        let without = try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, oddSides: false)
        #expect(asBuilt.oddSides == true && CoreMLBoardRecognizer.Default.oddSides == true)
        #expect(without.oddSides == false)

        let (image, found) = try OddSidesTests.drawn()
        let corners = [
            CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 0), CGPoint(x: 800, y: 0), CGPoint(x: 800, y: 800),
        ]
        func whites(_ recognizer: CoreMLBoardRecognizer) throws -> Int {
            let placed = try #require(
                recognizer.place(found, corners: corners, cornersFromModel: true, in: image))
            #expect(placed.pieces.count == 16)
            return placed.pieces.values.filter { $0.piece.color == .white }.count
        }
        #expect(found.filter { $0.color == .white }.count == 6)
        #expect(try whites(asBuilt) == 8)
        #expect(try whites(without) == 6)
    }

    /// A queen for a king is off unless asked for (through the app it gains
    /// a board of Fenit board B and loses four of Fenit A,
    /// `docs/RECOGNITION.md`, "A queen for a king"), and the recognizer
    /// hands its own setting on to what follows the models.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testQueenForKingIsOffByDefaultAndReachesWhatFollowsTheModels() throws {
        let models = try Self.models.get()
        let asBuilt = try makeRecognizer()
        let asked = try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, queenForKing: true)
        #expect(asBuilt.queenForKing == false && CoreMLBoardRecognizer.Default.queenForKing == false)
        #expect(asked.queenForKing == true && asked.oneKing == true)

        let image = try PieceSidesTests.drawnPawns().image
        let found = OneKingASideTests.kingAndQueen()
        let corners = [
            CGPoint(x: 0, y: 800), CGPoint(x: 0, y: 0), CGPoint(x: 800, y: 0), CGPoint(x: 800, y: 800),
        ]
        func kings(_ recognizer: CoreMLBoardRecognizer) throws -> Int {
            let placed = try #require(
                recognizer.place(found, corners: corners, cornersFromModel: true, in: image))
            #expect(placed.pieces.count == 3)
            return placed.pieces.values.filter { $0.piece.kind == .king }.count
        }
        #expect(try kings(asBuilt) == 1)
        #expect(try kings(asked) == 2)
        // With one king a side switched off it has nothing to act in.
        let alone = try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, oneKing: false, queenForKing: true)
        #expect(try kings(alone) == 1)
    }

    /// What `app_check --pieces` is given (`diagnosePieces`) is what the
    /// recognizer has and reads: the placement is `recognize`'s, the corners
    /// are its corners before a1 is chosen, and each piece carries the
    /// model's scores as one king a side sees them.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testThePiecesDiagnosisIsWhatTheRecognizerReads() async throws {
        let models = try Self.models.get()
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self, from: Data(contentsOf: fixtures.appending(path: "expected.json")))
        // As the app runs it; and with the colour steps off and a1 left
        // where the corners have it, so that the corners `recognize` ends
        // with are the corners before a1 is chosen.
        let asBuilt = try makeRecognizer()
        let plain = try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, a1Prior: nil, oddSides: false)
        for file in manifest.keys.sorted() {
            let source = try #require(
                CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
            let photo = DecodedImage.once(try #require(CGImageSourceCreateImageAtIndex(source, 0, nil)))
            var labels: [[String]] = []
            for recognizer in [asBuilt, plain] {
                let read = try await recognizer.recognize(photo)
                let found = try recognizer.diagnosePieces(photo)
                #expect(found.placement == FENBuilder.placement(read.pieces.mapValues(\.piece)), "\(file)")
                let corners = try #require(found.corners)
                #expect(found.squareColours == GridCorners.squareColours(in: photo, corners: corners), "\(file)")
                #expect(found.pieces.count >= read.pieces.count && !found.pieces.isEmpty, "\(file)")
                for piece in found.pieces {
                    // The highest score is the piece's confidence and names
                    // the model's answer; the scores come highest first.
                    let top = try #require(piece.scores.first)
                    #expect(top.label == piece.modelLabel && top.score == piece.confidence, "\(file)")
                    #expect(piece.scores.map(\.score) == piece.scores.map(\.score).sorted(by: >), "\(file)")
                    // What one king a side sees is made of them.
                    let kinds = CoreMLBoardRecognizer.kindScores(
                        piece.scores.map { (identifier: $0.label, score: $0.score) })
                    #expect(piece.kindScores == Dictionary(uniqueKeysWithValues: kinds.map {
                        (PieceSides.name($0.key), $0.value)
                    }), "\(file)")
                    // Only a colour step changes a class, and only the side.
                    #expect(piece.label.split(separator: "_").last
                        == piece.modelLabel.split(separator: "_").last, "\(file)")
                    #expect((piece.colour?.count ?? 3) == 3, "\(file)")
                }
                labels.append(found.pieces.map(\.label))
                if recognizer.oddSides {
                    // a1 may have been turned: the same ring, from whichever corner.
                    #expect((0..<4).contains { turn in
                        Array(corners[turn...] + corners[..<turn]) == read.corners
                    }, "\(file)")
                } else {
                    // The very corners, and the pieces as the model read them.
                    #expect(corners == read.corners, "\(file)")
                    #expect(found.pieces.allSatisfy { $0.label == $0.modelLabel }, "\(file)")
                }
            }
            // On the fixtures the side of single pieces changes nothing.
            #expect(labels[0] == labels[1], "\(file)")
            // The corners shown are those before a1 is chosen: with the
            // check of a1 on, the ones the recognizer without it ends with.
            #expect(try asBuilt.diagnosePieces(photo).corners == plain.diagnosePieces(photo).corners,
                    "\(file)")
        }
    }

    /// What `app_check --window` is given (`diagnoseFitWindow`) is what the
    /// recognizer itself makes of the two sets of corners: the fit's in the
    /// model's order where the rule is off, the window the model picks
    /// where it is on.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testTheFitWindowDiagnosisIsWhatTheRecognizerDoes() throws {
        let models = try Self.models.get()
        let asBuilt = try makeRecognizer()
        let without = try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, fitWindow: false)
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self, from: Data(contentsOf: fixtures.appending(path: "expected.json")))
        var both = 0
        for file in manifest.keys.sorted() {
            let source = try #require(
                CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
            let photo = DecodedImage.once(try #require(CGImageSourceCreateImageAtIndex(source, 0, nil)))
            guard let found = try asBuilt.diagnoseFitWindow(photo) else { continue }
            both += 1
            #expect(found.window == GridCorners.windowByModel(fit: found.fit, model: found.model), "\(file)")
            #expect(asBuilt.corners(model: found.model, in: photo)?.corners == found.window.corners,
                    "\(file)")
            #expect(without.corners(model: found.model, in: photo)?.corners == found.fit, "\(file)")
            // The setting plays no part in the diagnosis.
            let other = try #require(try without.diagnoseFitWindow(photo))
            #expect(other.model == found.model && other.fit == found.fit && other.window == found.window,
                    "\(file)")
        }
        // The model and the fit both find the board on the fixtures.
        #expect(both >= 3)
    }

    /// The recognizer hands its own `a1Dark` on to what follows the models
    /// (`place`, the method `recognize` calls). The fixture photos read the
    /// same with the rule and without, so they would not notice it missing:
    /// here a drawn board whose corners start on a light square, with
    /// pieces that say that corner is a1.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testA1DarkReachesWhatFollowsTheModels() throws {
        let models = try Self.models.get()
        let (image, light) = try GridSquareColoursTests.plainBoard()
        let found = try GridSquareColoursTests.pieces(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQK3", on: light)
        let without = try #require(try CoreMLBoardRecognizer(
            cornerModel: models.corners, pieceModel: models.pieces, a1Dark: false
        ).place(found, corners: light, cornersFromModel: true, in: image))
        #expect(without.corners == light)
        // As the app builds it: a quarter turn away from what the pieces say.
        let with = try #require(try makeRecognizer().place(
            found, corners: light, cornersFromModel: true, in: image))
        #expect(with.corners == Array(light[1...] + light[..<1])
            || with.corners == Array(light[3...] + light[..<3]))
        // The working image handed in by `recognize` changes nothing.
        let whole = try #require(GridCorners.workingImage(of: image))
        #expect(try #require(try makeRecognizer().place(
            found, corners: light, cornersFromModel: true, in: image, whole: whole)).corners
            == with.corners)
    }

    /// The colour step is off unless asked for (`docs/RECOGNITION.md`, "What
    /// the colour step is worth"). Switching it on by default is a decision
    /// to take with scores on a set nothing was tuned on, not in passing.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testColourStepIsOffByDefault() throws {
        #expect(try makeRecognizer().sidesFromColour == false)
        #expect(try makeRecognizer(sidesFromColour: true).sidesFromColour == true)
    }

    /// What the rule for the kings (`OneKingASide`) is given: the piece
    /// model's own score for every class of each box. Vision hands them out
    /// divided by their sum, with the sum as the observation's confidence
    /// (`CoreMLBoardRecognizer.classScores`); nothing in its documentation
    /// says so, so this holds it against the model run without Vision, on the
    /// same pixels: the photo drawn at the model's input size, which neither
    /// then has to resize.
    ///
    /// Twice. As the app runs the model: the pieces it finds on the four
    /// fixtures, their confidence and their scores by kind. And on boxes the
    /// model is unsure of: a box with one score says nothing about what the
    /// scores are divided by, and a piece model that reads the fixtures
    /// without a doubt has no second score worth the name on them (the first
    /// one had a single king that might be a queen; `pieces_s_render3` has
    /// none). So the second time the photos are drawn at a quarter of the
    /// size, where pieces are a few pixels high and any model hesitates, and
    /// the model's own confidence threshold is taken down to 0.01.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testPieceScoresAreTheModelsOwn() throws {
        let asTheAppRunsIt = try pieceScoresAgainstTheModel(small: false, confidenceThreshold: nil)
        print("piece scores: \(asTheAppRunsIt.boxes) pieces, "
            + "\(asTheAppRunsIt.kindScores) scores for a kind other than the piece's own")
        #expect(asTheAppRunsIt.boxes >= 40)

        let unsure = try pieceScoresAgainstTheModel(small: true, confidenceThreshold: 0.01)
        print("piece scores, small photos, threshold 0.01: \(unsure.boxes) boxes, "
            + "\(unsure.split) of them unsure between two classes, "
            + "\(unsure.kindScores) scores for a kind other than the box's own")
        // Without boxes whose second score counts, only the top score of
        // each box has been checked.
        #expect(unsure.split >= 10)
        #expect(unsure.kindScores >= 10)
        // The threshold was taken down for both runs: there are boxes under
        // the 0.25 the model is exported with.
        #expect(unsure.leastConfidence < 0.05)
        #expect(asTheAppRunsIt.leastConfidence >= 0.2)
    }

    /// Runs the piece model on the four fixtures through the recognizer
    /// (Vision) and by itself (CoreML), and expects the same boxes in the
    /// same order with the same score for each of the twelve classes, and
    /// the recognizer to make of each box what the scores say.
    ///
    /// - Parameters:
    ///   - small: draw each photo at a quarter of the model's input size, in
    ///     the middle of a grey square.
    ///   - confidenceThreshold: the model's own threshold for a box, for
    ///     both runs; nil leaves it as exported (what the app runs).
    /// - Returns: how many boxes were compared, how many of them were unsure
    ///   between two classes (the second scoring at least 0.01 and a tenth
    ///   of the first), how many scores by kind beside the box's own, and
    ///   the lowest confidence of a box.
    private func pieceScoresAgainstTheModel(
        small: Bool, confidenceThreshold: Double?
    ) throws -> (boxes: Int, split: Int, kindScores: Int, leastConfidence: Double) {
        let models = try Self.models.get()
        // A recognizer of its own: the threshold is set on its Vision model.
        let recognizer = try makeRecognizer()
        var thresholds: [String: Any] = [:]
        if let confidenceThreshold {
            // (An input under another name would be ignored by both runs.)
            #expect(models.pieces.modelDescription.inputDescriptionsByName["confidenceThreshold"] != nil)
            thresholds["confidenceThreshold"] = confidenceThreshold
            recognizer.pieceModel.featureProvider = try MLDictionaryFeatureProvider(dictionary: thresholds)
        }
        let constraint = try #require(
            models.pieces.modelDescription.inputDescriptionsByName["image"]?.imageConstraint)
        let side = constraint.pixelsWide
        #expect(side == constraint.pixelsHigh)
        // The classes in training order (`PIECE_CLASSES` in
        // `Training/chess_pipeline.py`), which the rows of the model's
        // output follow; the model says so itself.
        let classNames = ["white", "black"].flatMap { colour in
            ["king", "queen", "rook", "bishop", "knight", "pawn"].map { colour + "_" + $0 }
        }
        let labels = try #require(models.pieces.modelDescription.classLabels as? [String])
        #expect(Array(labels.prefix(12)) == classNames)
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self, from: Data(contentsOf: fixtures.appending(path: "expected.json")))
        var boxes = 0, split = 0, kindScores = 0, leastConfidence = 1.0
        let whole = Letterbox(imageSize: CGSize(width: side, height: side))
        for file in manifest.keys.sorted() {
            let source = try #require(
                CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
            let photo = DecodedImage.once(try #require(CGImageSourceCreateImageAtIndex(source, 0, nil)))
            let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
            let context = try #require(CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
            context.interpolationQuality = .high
            if small {
                context.setFillColor(red: 0.45, green: 0.45, blue: 0.45, alpha: 1)
                context.fill(CGRect(x: 0, y: 0, width: side, height: side))
                context.draw(photo, in: CGRect(
                    x: side * 3 / 8, y: side * 3 / 8, width: side / 4, height: side / 4))
            } else {
                context.draw(photo, in: CGRect(x: 0, y: 0, width: side, height: side))
            }
            let square = try #require(context.makeImage())

            let observed = try recognizer.pieceObservations(in: square)
            var inputs = thresholds
            inputs["image"] = try MLFeatureValue(cgImage: square, constraint: constraint)
            let output = try models.pieces.prediction(
                from: MLDictionaryFeatureProvider(dictionary: inputs))
            let rows = try #require(output.featureValue(for: "confidence")?.multiArrayValue)
            let places = try #require(output.featureValue(for: "coordinates")?.multiArrayValue)
            #expect(rows.shape.count == 2)
            #expect(rows.shape[0].intValue == observed.count, "\(file)")
            #expect(places.shape.map(\.intValue) == [observed.count, 4], "\(file)")
            guard rows.shape[0].intValue == observed.count, rows.shape[1].intValue >= 12,
                  places.shape.map(\.intValue) == [observed.count, 4]
            else { continue }
            if !small {
                #expect(!observed.isEmpty, "\(file)")
                // The recognizer's own entry gives a piece for every box.
                #expect(try recognizer.detectPieces(in: square, letterbox: whole).count == observed.count,
                        "\(file)")
            }

            for (index, observation) in observed.enumerated() {
                let row = (0..<12).map { rows[[index, $0] as [NSNumber]].doubleValue }
                let ranked = row.sorted(by: >)
                let top = ranked[0]
                let topClass = try #require(row.firstIndex(of: top))
                leastConfidence = min(leastConfidence, top)

                // The same box: the model gives centre and size as fractions
                // of the square, y from the top; Vision the rectangle with y
                // from the bottom. (Vision's observations come in the order
                // of the model's rows; nothing says they must.)
                let place = (0..<4).map { places[[index, $0] as [NSNumber]].doubleValue }
                let box = observation.box
                #expect(abs(box.midX - place[0]) < 1e-5, "\(file) box \(index)")
                #expect(abs((1 - box.midY) - place[1]) < 1e-5, "\(file) box \(index)")
                #expect(abs(box.width - place[2]) < 1e-5, "\(file) box \(index)")
                #expect(abs(box.height - place[3]) < 1e-5, "\(file) box \(index)")

                // Every class score, as Vision hands it out, is the model's.
                var compared = 0
                for (name, score) in observation.scores {
                    guard let k = classNames.firstIndex(of: name) else { continue }
                    #expect(abs(Double(score) - row[k]) < 1e-5, "\(file) box \(index) \(name)")
                    compared += 1
                }
                #expect(compared == 12, "\(file) box \(index)")
                if ranked[1] >= 0.01, ranked[1] >= 0.1 * top {
                    split += 1
                }

                // What the recognizer makes of the box: the class with the
                // highest score, that score as the confidence, and the
                // scores by kind, the larger of the two colours.
                let detection = try #require(
                    CoreMLBoardRecognizer.piece(from: observation, letterbox: whole))
                #expect(abs(Double(detection.confidence) - top) < 1e-5, "\(file) box \(index)")
                // (Two classes with the same highest score: either may be first.)
                if ranked[1] < top {
                    #expect(detection.piece == OneKingASide.kinds[topClass % 6], "\(file) box \(index)")
                    #expect(detection.color == (topClass < 6 ? .white : .black), "\(file) box \(index)")
                }
                for (k, kind) in OneKingASide.kinds.enumerated() {
                    let score = max(row[k], row[k + 6])
                    let have = detection.kindScores[kind]
                    if score >= CoreMLBoardRecognizer.leastKindScore + 1e-5 {
                        #expect(abs((have ?? -1) - score) < 1e-5, "\(file) box \(index) \(kind)")
                        if kind != detection.piece { kindScores += 1 }
                    } else if score < CoreMLBoardRecognizer.leastKindScore - 1e-5 {
                        #expect(have == nil, "\(file) box \(index) \(kind)")
                    }
                }
                boxes += 1
            }
        }
        return (boxes, split, kindScores, leastConfidence)
    }

    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testRecognizesFixturePositions() async throws {
        try await expectFixturesRead(by: try makeRecognizer(), label: "fixture")
    }

    /// The same photos with the grid fit switched off: the corner model on
    /// its own. With the fit on, a corner model that found no board at all
    /// would go unnoticed here, because the fit reads these photos by itself
    /// (and the pieces alone happen to put a1 right on them). This is the
    /// test that a corner model has to pass on its own corners and its own a1.
    ///
    /// What it asks: each of the model's corners within one square of the
    /// grid fit's (the corners themselves, a number that moves smoothly; the
    /// fit's corners are turned to the nearest of the model's, so this says
    /// nothing about a1), no photo that fits its position better turned by
    /// a quarter, a half or three quarters (that is the check on a1), and
    /// 94% of the squares of the four photos together. Not 58 squares a photo, as for the app's
    /// pipeline: with corners up to 0.72 squares off, as the first corner
    /// model has them on `test_000000.jpg`, the pieces of the far rank
    /// stand on the lines between squares, and each one that slips costs two
    /// squares. With the piece model before `pieces_s_render3` that photo
    /// read 58 of 64 on the Mac mini's Neural Engine and 60 on its CPU and
    /// on the CI runner, from the same pixels (with `pieces_s_render3`, 60
    /// on the Neural Engine): a count on a coin toss, not a measure of the
    /// corners.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testCornerModelAloneReadsTheFixtures() async throws {
        try await expectFixturesRead(
            by: try makeRecognizer(gridFit: false), label: "fixture, corner model alone",
            perPhotoFloor: nil, cornersWithinSquaresOfTheFit: 1.0)
    }

    /// The recognizer decodes a photo that comes straight from a file once
    /// (`DecodedImage`): handed the file's image or the decoded copy, it gives
    /// the same corners and the same board. With the grid fit off and a
    /// square photo this is the case where Vision would otherwise get the
    /// undecoded image itself.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testRecognizerDecodesThePhotoItself() async throws {
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let source = try #require(
            CGImageSourceCreateWithURL(fixtures.appending(path: "test_000000.jpg") as CFURL, nil))
        let fromFile = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(fromFile.width == fromFile.height)
        let copy = DecodedImage.once(fromFile)
        #expect(copy !== fromFile)
        for gridFit in [false, true] {
            let recognizer = try makeRecognizer(gridFit: gridFit)
            let a = try await recognizer.recognize(fromFile)
            let b = try await recognizer.recognize(copy)
            #expect(a.corners == b.corners, "grid fit \(gridFit)")
            #expect(a.pieces == b.pieces, "grid fit \(gridFit)")
            #expect(try recognizer.diagnosePieceSides(fromFile).map(\.colour)
                == recognizer.diagnosePieceSides(copy).map(\.colour), "grid fit \(gridFit)")
        }
    }

    /// The colour step is off in the app. Switched on, it must leave boards
    /// the piece model reads well as they are.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testSidesFromColourSwitchedOnLeavesTheFixturesRight() async throws {
        try await expectFixturesRead(
            by: try makeRecognizer(sidesFromColour: true), label: "fixture, sides from colour")
    }

    /// The colour step on the fixtures: every piece the model finds has a
    /// colour, the pieces split in two colours, and the split by colour alone
    /// gives as many White and as many Black pieces as the position has. (Not
    /// that the piece model's sides stand: a model that has one side wrong
    /// here, which the colours then put right, is no failure. That the final
    /// sides are right is what the two fixture tests above check.)
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testColoursSplitThePiecesOfTheFixturesAsThePositionsDo() throws {
        let recognizer = try makeRecognizer()
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: fixtures.appending(path: "expected.json")))
        for (file, placement) in manifest.sorted(by: { $0.key < $1.key }) {
            let source = try #require(
                CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
            let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            let pieces = try recognizer.diagnosePieceSides(image)
            let agree = pieces.filter { $0.colourSide.map($0.label.hasPrefix) == true }.count
            print("colour \(file): \(pieces.count) pieces, the colours agree with the model on \(agree)")
            #expect(pieces.allSatisfy { $0.colour != nil }, "\(file)")
            #expect(pieces.allSatisfy { $0.colourSide != nil }, "\(file)")
            #expect(pieces.filter { $0.colourSide == "white" }.count
                == placement.filter { $0.isLetter && $0.isUppercase }.count, "\(file)")
            #expect(pieces.filter { $0.colourSide == "black" }.count
                == placement.filter { $0.isLetter && $0.isLowercase }.count, "\(file)")
        }
    }

    /// - Parameters:
    ///   - perPhotoFloor: squares that must be right on each photo; nil for
    ///     no floor a photo.
    ///   - cornersWithinSquaresOfTheFit: if given, each corner the recognizer
    ///     returns must lie within so many squares of the grid fit's corner.
    private func expectFixturesRead(
        by recognizer: CoreMLBoardRecognizer, label: String,
        perPhotoFloor: Int? = 58, cornersWithinSquaresOfTheFit: Double? = nil
    ) async throws {

        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: fixtures.appending(path: "expected.json")))
        #expect(!manifest.isEmpty)

        var totalCorrect = 0
        var totalSquares = 0

        for (file, expectedPlacement) in manifest.sorted(by: { $0.key < $1.key }) {
            let imageURL = fixtures.appending(path: file)
            guard
                let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                Issue.record("could not load \(file)")
                continue
            }

            let result = try await recognizer.recognize(image)
            let got = result.pieces.mapValues(\.piece)
            let expected = try #require(FENBuilder.pieces(fromPlacement: expectedPlacement))

            let correct = Self.correctSquares(got, expected)
            let turned = Self.bestTurned(got, expected)
            totalCorrect += correct
            totalSquares += 64
            print("\(label) \(file): \(correct)/64 squares\(turned.correct > correct ? ", \(turned.correct)/64 TURNED BY \(turned.quarters * 90) DEGREES" : "")  (expected \(expectedPlacement), got \(FENBuilder.placement(got)))")

            // Which corner is a1: no photo may fit its position better once
            // the board read is turned by a quarter, a half or three
            // quarters. A turned endgame loses only two squares a piece,
            // which the floors below can miss (six pieces: 52 of 64, 95% of
            // the four photos together).
            #expect(correct >= turned.correct,
                    "\(file): read turned by \(turned.quarters * 90) degrees (\(correct)/64 as read, \(turned.correct)/64 turned back)")

            // Per-board floor: recognition quality on known test images should
            // never regress below ~90% per-square.
            //
            // test_004005.jpg guards which corner is a1. Its position
            // (7Q/8/1k3p2/4p3/8/p3K3/8/8) points the wrong way: it scores
            // only +2.5 the right way up (BoardOrientation), less than the
            // prior of 6 for the corner model's choice, so the pieces cannot
            // overrule a corner model that puts a1 at the wrong end. With
            // exact corners (the grid fit) all six pieces are read: 64/64
            // with a corner model that knows a1 here (the first one, in
            // `Models/` until pull request #19), 52/64, the board turned by
            // 180 degrees, with one that does not (exp3, exp4;
            // `corners_s_canon1` knows it on this photo and not on the same
            // photo as 3:4, see testRecognizesFixturesAsPortraitPhotos).
            // See docs/RECOGNITION.md.
            if let perPhotoFloor {
                #expect(correct >= perPhotoFloor, "\(file) regressed: \(correct)/64")
            }

            if let bound = cornersWithinSquaresOfTheFit {
                let fit = try #require(GridCorners.find(in: image), "\(file): the grid fit finds no board")
                let fitted = GridCorners.turned(fit.corners, nearestTo: result.corners)
                // The side of a square, from the fitted board's four edges.
                var edges = 0.0
                for k in 0..<4 {
                    let p = fitted[k], q = fitted[(k + 1) % 4]
                    edges += Double(hypot(p.x - q.x, p.y - q.y))
                }
                let square = edges / 32
                let off = zip(result.corners, fitted)
                    .map { Double(hypot($0.x - $1.x, $0.y - $1.y)) / square }.max() ?? .infinity
                print(String(format: "%@ %@: corners up to %.2f squares from the grid fit's", label, file, off))
                #expect(off < bound, "\(file): a corner is \(off) squares from the grid fit's")
            }
        }

        // Aggregate floor mirrors the Python eval (~98% per-square on test).
        let accuracy = Double(totalCorrect) / Double(totalSquares)
        print(String(format: "%@ aggregate: %.1f%% per-square", label, accuracy * 100))
        #expect(accuracy >= 0.94)
    }

    /// Phone photos are 3:4, the fixtures are square. Each fixture is turned
    /// into a 3:4 photo (grey above and below, like extra table) and must
    /// still be read: the recognizer letterboxes it rather than stretching it.
    /// The same photos stretched to a square, which is what the app used to
    /// do, are scored alongside for comparison.
    ///
    /// Asked: 90% of the squares of the four photos together, and no photo
    /// that fits its position better turned by a quarter, a half or three
    /// quarters, with one known exception. The corner model
    /// `corners_s_canon1` reads the endgame `test_004005.jpg` turned by a
    /// half when it is a 3:4 photo (and right as the square photo it is):
    /// all six pieces on the squares they would be on after half a turn.
    /// That is recorded as a known issue and does not fail the test
    /// (`docs/RECOGNITION.md`, "Known weak spot"). A turn fails in every
    /// other case: with another corner model, another photo, another turn,
    /// or with a piece wrong besides. (A misreading that is not a turn is
    /// for the 90% to catch, here as before.) The exception is entered
    /// only when the issue occurs, so nothing reports that it is no longer
    /// needed: take it out with the corner model.
    @Test(.enabled(if: modelsPresent, "Models/*.mlpackage not present: run Training/export_coreml.py"))
    func testRecognizesFixturesAsPortraitPhotos() async throws {
        let recognizer = try makeRecognizer()
        let fixtures = try #require(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: fixtures.appending(path: "expected.json")))

        var letterboxed = 0
        var stretched = 0
        var totalSquares = 0
        for (file, expectedPlacement) in manifest.sorted(by: { $0.key < $1.key }) {
            guard
                let source = CGImageSourceCreateWithURL(
                    fixtures.appending(path: file) as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                Issue.record("could not load \(file)")
                continue
            }
            let expected = try #require(FENBuilder.pieces(fromPlacement: expectedPlacement))
            let portrait = try #require(Self.redraw(image, width: image.width, height: image.width * 4 / 3, stretch: false))
            #expect(portrait.height * 3 == portrait.width * 4)
            let squashed = try #require(Self.redraw(portrait, width: image.width, height: image.width, stretch: true))

            let read = try await recognizer.recognize(portrait).pieces.mapValues(\.piece)
            let good = Self.correctSquares(read, expected)
            let bad = Self.correctSquares(
                (try? await recognizer.recognize(squashed).pieces.mapValues(\.piece)) ?? [:],
                expected)
            letterboxed += good
            stretched += bad
            totalSquares += 64
            let turned = Self.bestTurned(read, expected)
            print("portrait \(file): letterboxed \(good)/64\(turned.correct > good ? " (\(turned.correct)/64 TURNED BY \(turned.quarters * 90) DEGREES)" : ""), stretched \(bad)/64")
            if good < turned.correct {
                let what = Comment(rawValue:
                    "\(file) as a 3:4 photo: read turned by \(turned.quarters * 90) degrees (\(good)/64 as read, \(turned.correct)/64 turned back)")
                // The one case let through: this corner model, this photo,
                // every piece right once the board is turned by a half.
                if try Self.cornerModelIsCanon1(), file == "test_004005.jpg",
                   turned.quarters == 2, turned.correct == 64 {
                    withKnownIssue("corners_s_canon1 puts a1 at the wrong end of this endgame as a 3:4 photo") {
                        Issue.record(what)
                    }
                } else {
                    Issue.record(what)
                }
            }
        }
        let accuracy = Double(letterboxed) / Double(totalSquares)
        print(String(
            format: "portrait aggregate: letterboxed %.1f%%, stretched %.1f%% per-square",
            accuracy * 100, Double(stretched) / Double(totalSquares) * 100))
        #expect(accuracy >= 0.90)
    }

    /// A board read with a1 at another corner: every piece on the square it
    /// would be on after so many quarter turns (two: the opposite corner).
    private static func turned(_ got: [Square: Piece], quarters: Int) -> [Square: Piece] {
        var turned: [Square: Piece] = [:]
        for (square, piece) in got {
            var file = square.file.number
            var rank = square.rank.value
            for _ in 0..<quarters {
                (file, rank) = (rank, 9 - file)
            }
            turned[FENBuilder.square(file: file, rank: rank)!] = piece
        }
        return turned
    }

    /// How well a board read fits the position at best once it is turned
    /// by one, two or three quarters. A board read right scores less every
    /// way (the four fixtures: at most 52 of 64).
    private static func bestTurned(
        _ got: [Square: Piece], _ expected: [Square: Piece]
    ) -> (quarters: Int, correct: Int) {
        (1...3).map { ($0, correctSquares(turned(got, quarters: $0), expected)) }
            .max { $0.1 < $1.1 }!
    }

    /// Whether the corner model under test is `corners_s_canon1`, by what
    /// its package says of itself (the data it was trained on).
    private static func cornerModelIsCanon1() throws -> Bool {
        let description = try models.get().corners.modelDescription
            .metadata[MLModelMetadataKey.description] as? String
        return description?.contains("corners_s_canon1") ?? false
    }

    private static func correctSquares(_ got: [Square: Piece], _ expected: [Square: Piece]) -> Int {
        var correct = 0
        for rank in 1...8 {
            for file in 1...8 {
                let sq = FENBuilder.square(file: file, rank: rank)!
                if expected[sq].map({ FENBuilder.fenLetter(for: $0) })
                    == got[sq].map({ FENBuilder.fenLetter(for: $0) })
                {
                    correct += 1
                }
            }
        }
        return correct
    }

    /// Draws `image` into a grey canvas: centred at its own size, or stretched
    /// to fill it.
    private static func redraw(_ image: CGImage, width: Int, height: Int, stretch: Bool) -> CGImage? {
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.setFillColor(red: 0.45, green: 0.45, blue: 0.45, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .high
        let rect = stretch
            ? CGRect(x: 0, y: 0, width: width, height: height)
            : CGRect(
                x: (width - image.width) / 2, y: (height - image.height) / 2,
                width: image.width, height: image.height)
        context.draw(image, in: rect)
        return context.makeImage()
    }
}
