import ChessKit
import CoreGraphics
import CoreML
import Vision

/// The production recognizer: two CoreML models (Ultralytics YOLO11 exports)
/// plus the shared geometry in `Homography`/`SquareAssignment`.
///
/// Model interfaces (validated against the exported .mlpackages):
/// - **BoardCorners** (pose, no NMS): raw tensor `[1, 17, 8400]` — per anchor:
///   rows 0–3 box xywh, row 4 objectness/class score, rows 5–16 four
///   keypoints × (x, y, conf), all in 640×640 input-pixel space, keypoint
///   order (a1, a8, h8, h1). We take the single best-scoring anchor.
/// - **Pieces** (detect, NMS pipeline): Vision surfaces
///   `VNRecognizedObjectObservation`s with normalized bounding boxes
///   (lower-left origin) and class labels like `"white_pawn"`. Vision hands
///   out the model's score for every class of a box, but divided by their
///   sum: a label's confidence times the observation's own confidence (the
///   sum) is the score as the model gave it (`classScores`).
///
/// - **PieceKinds** (classify, optional): each piece the piece model found,
///   cut out of the photo, 128×128 in; Vision surfaces a
///   `VNClassificationObservation` for each of the twelve labels. Where the
///   bundle has it, the kind of every piece is taken from it and the piece
///   model together (`PieceKinds`); the side stays the piece model's.
///
/// After the models, two steps that need no learning
/// (`docs/RECOGNITION.md`): the corners are replaced by the classical grid
/// fit (`GridCorners`) where it accepts its own result, and which corner is
/// a1 is checked against the pieces (`BoardOrientation`). A third, White and
/// Black from the colours of the pieces where the piece model's own answer
/// does not fit them (`PieceSides`), is there but switched off: on boards
/// nobody tuned its rule on, it fails with the piece model it was made for.
/// What is on is its narrower kin, the side of single pieces: a piece
/// called Black whose colour is plainly the White side's becomes White
/// (`OddSides`). A fourth sees to it that each side has one king
/// (`OneKingASide`). And a1
/// is a dark square: the pieces choose between the two ways round that the
/// colours of the squares allow (`GridCorners.squareColours`).
///
/// The photo's pixels are decoded once for all of it (`DecodedImage`).
/// Both models see the photo letterboxed into a square (`Letterbox`), as in
/// training, never stretched. The square is passed to Vision with
/// `.scaleFill`, which for a square only resizes; `Letterbox` maps the
/// models' answers back to pixels of the original photo.
public struct CoreMLBoardRecognizer: BoardRecognizer {

    private let cornerModel: VNCoreMLModel
    /// Internal for testing (a test lowers the model's own thresholds through
    /// `VNCoreMLModel.featureProvider`, on a recognizer of its own).
    let pieceModel: VNCoreMLModel
    private let cornerScoreThreshold: Float
    private let pull: Double
    private let a1Prior: Double?
    private let gridFit: Bool
    /// Internal for testing: the app's default must stay off.
    let sidesFromColour: Bool
    /// Internal for testing.
    let oneKing: Bool
    /// Internal for testing.
    let a1Dark: Bool
    /// Internal for testing.
    let fitWindow: Bool
    /// Internal for testing.
    let oddSides: Bool
    /// Internal for testing: the app's default must stay off.
    let queenForKing: Bool
    /// The piece classifier, where there is one (`PieceKinds`).
    let kindModel: VNCoreMLModel?
    /// Internal for testing.
    let pieceKinds: Bool

    /// What the app runs with where nothing else is asked for.
    public enum Default {
        public static let pull = SquareAssignment.defaultPull
        public static let a1Prior: Double? = BoardOrientation.appPrior
        public static let gridFit = true
        public static let sidesFromColour = false
        public static let oneKing = true
        public static let a1Dark = true
        public static let fitWindow = true
        public static let oddSides = true
        public static let queenForKing = false
        public static let pieceKinds = true
    }

    /// - Parameters:
    ///   - a1Prior: which corner is a1 is checked against the pieces
    ///     (`BoardOrientation.orient`); this is how much the corner model's
    ///     own choice counts. `nil` keeps the corner model's a1 as it is.
    ///   - gridFit: take the corners from the classical grid fit
    ///     (`GridCorners`) where it accepts its own result, and from the
    ///     corner model where it does not. With it, a board the corner model
    ///     does not find can still be read.
    ///   - sidesFromColour: decide White and Black from the colours of the
    ///     pieces in the photo where they contradict the piece model for
    ///     more than a share of the pieces (`PieceSides`). Off unless asked
    ///     for: on the second held-out test set the rule as it is loses 12
    ///     points of squares on one source with `pieces_s_render1`, the
    ///     piece model it was made for, and gains nothing lasting with the
    ///     piece model before that (`docs/RECOGNITION.md`, "What the colour
    ///     step is worth"; not measured with any piece model since, neither
    ///     `pieces_s_render3` nor `pieces_s_tone1`). It is kept for the
    ///     reference's next version of the rule.
    ///   - oneKing: exactly one king a side, from the scores the piece model
    ///     gave each piece for every kind (`OneKingASide`). On: when it was
    ///     chosen no development set lost by it in total, with either piece
    ///     model it was tried with, and the second held-out test set did
    ///     not contradict that (`docs/RECOGNITION.md`, "What one king a
    ///     side is worth"). With the piece models since it is worth more:
    ///     80 boards fully right of that set's 1,619 with
    ///     `pieces_s_render3`, 109 with `pieces_s_tone1` ("What the app
    ///     reads with the third pair").
    ///   - a1Dark: a1 is a dark square, so where the colours of the squares
    ///     can be told apart the pieces choose between two ways round, half
    ///     a turn apart, and can no longer turn a board by a quarter
    ///     (`GridCorners.squareColours`). Has no effect where which corner
    ///     is a1 is not checked against the pieces (`a1Prior` nil and the
    ///     corners the model's). A board set up with a light square on a1
    ///     is read a quarter turn off.
    ///   - fitWindow: where the grid fit and the corner model both found the
    ///     board, the model picks which 8×8 window of the fit's grid is the
    ///     board: the fit's corners are moved by one square when all four of
    ///     the model's lie near the same one-square shift of them
    ///     (`GridCorners.windowByModel`). The fit takes a window one rank or
    ///     file off where a border looks like a rank of squares. Has no
    ///     effect where the fit is off, rejects itself or stands alone.
    ///   - oddSides: a piece the piece model calls Black becomes White when
    ///     its colour in the photo is plainly the White side's: twice as
    ///     near the colour of the pieces called White as that of the other
    ///     pieces called Black (`OddSides`). On: through the app it mends
    ///     15 boards of Fenit board B, the board its numbers were chosen
    ///     on, and changes no reading of the other development sets nor of
    ///     the second held-out test set as the photos are taken; drawn as
    ///     3:4 photos, one photo of ChessReD test loses a square
    ///     (`docs/RECOGNITION.md`, "What the side of single pieces is
    ///     worth").
    ///   - queenForKing: with `oneKing`, a side left without a king and
    ///     without a piece the model gave a "king" score takes its queen
    ///     for one, of several the one the model is least sure of
    ///     (`OneKingASide`). Off unless asked for: through the app it gains
    ///     one board of Fenit board B and loses four of Fenit A
    ///     (`docs/RECOGNITION.md`, "A queen for a king"). Has no effect
    ///     with `oneKing` off.
    ///   - kindModel: the piece classifier (`Models/PieceKinds.mlpackage`);
    ///     nil reads the kinds from the piece model alone.
    ///   - pieceKinds: take each piece's kind from the piece classifier and
    ///     the piece model together (`PieceKinds`), where `kindModel` is
    ///     given. On: through the app with `pieces_s_shape1` it reads more
    ///     boards of every development set fully right (Fenit board B 43 ->
    ///     53 of 100, Fenit A 230 -> 247, ChessReD test 297 -> 299) and of the
    ///     second held-out test set (640 -> 817 of 1,619), with one part of
    ///     that set worse (samryan, from straight above, 127 -> 104;
    ///     `Training/README.md`, "A piece classifier").
    public init(
        cornerModel: MLModel,
        pieceModel: MLModel,
        cornerScoreThreshold: Float = 0.25,
        pull: Double = Default.pull,
        a1Prior: Double? = Default.a1Prior,
        gridFit: Bool = Default.gridFit,
        sidesFromColour: Bool = Default.sidesFromColour,
        oneKing: Bool = Default.oneKing,
        a1Dark: Bool = Default.a1Dark,
        fitWindow: Bool = Default.fitWindow,
        oddSides: Bool = Default.oddSides,
        queenForKing: Bool = Default.queenForKing,
        kindModel: MLModel? = nil,
        pieceKinds: Bool = Default.pieceKinds
    ) throws {
        self.cornerModel = try VNCoreMLModel(for: cornerModel)
        self.pieceModel = try VNCoreMLModel(for: pieceModel)
        self.cornerScoreThreshold = cornerScoreThreshold
        self.pull = pull
        self.a1Prior = a1Prior
        self.gridFit = gridFit
        self.sidesFromColour = sidesFromColour
        self.oneKing = oneKing
        self.a1Dark = a1Dark
        self.fitWindow = fitWindow
        self.oddSides = oddSides
        self.queenForKing = queenForKing
        self.kindModel = try kindModel.map { try VNCoreMLModel(for: $0) }
        self.pieceKinds = pieceKinds
        // Keypoint coordinates come out in input-pixel space, so the parser
        // must know the model's true input side (640 for the first-generation
        // corner model, 960 for later ones).
        self.cornerInputSide = Self.imageInputSide(of: cornerModel) ?? 640
    }

    private let cornerInputSide: Double

    /// The (square) image-input side length declared by a CoreML model.
    static func imageInputSide(of model: MLModel) -> Double? {
        for input in model.modelDescription.inputDescriptionsByName.values {
            if let constraint = input.imageConstraint {
                return Double(constraint.pixelsWide)
            }
        }
        return nil
    }

    /// Loads compiled models (`.mlmodelc`) named `BoardCorners` and `Pieces`
    /// from a bundle, and `PieceKinds` where the bundle has it, on the
    /// Neural Engine + CPU (never GPU — avoids contention with camera/UI
    /// rendering).
    public init(bundle: Bundle) throws {
        guard
            let cornersURL = bundle.url(forResource: "BoardCorners", withExtension: "mlmodelc"),
            let piecesURL = bundle.url(forResource: "Pieces", withExtension: "mlmodelc")
        else { throw RecognitionError.modelUnavailable }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let kindsURL = bundle.url(forResource: "PieceKinds", withExtension: "mlmodelc")
        try self.init(
            cornerModel: MLModel(contentsOf: cornersURL, configuration: config),
            pieceModel: MLModel(contentsOf: piecesURL, configuration: config),
            // Optional: a classifier that does not load leaves the piece
            // model's kinds, not the app without a recognizer.
            kindModel: kindsURL.flatMap { try? MLModel(contentsOf: $0, configuration: config) }
        )
    }

    // MARK: - BoardRecognizer

    public func recognize(_ photo: CGImage) async throws -> RecognitionResult {
        // One decode of the photo for the models, the grid fit and the colours.
        let image = DecodedImage.once(photo)
        let letterbox = Letterbox(
            imageSize: CGSize(width: image.width, height: image.height))
        guard let square = letterbox.render(image) else {
            throw RecognitionError.boardNotFound
        }

        // The photo at the grid fit's working size, shrunk once for the fit
        // and for the square colours.
        let whole = gridFit || (a1Dark && a1Prior != nil) ? GridCorners.workingImage(of: image) : nil
        guard let located = corners(
            model: try modelCorners(in: square, letterbox: letterbox), in: image, whole: whole)
        else { throw RecognitionError.boardNotFound }
        let detections = kinds(of: try detectPieces(in: square, letterbox: letterbox), in: image)
        guard let placed = place(
            detections, corners: located.corners, cornersFromModel: located.fromModel, in: image,
            whole: whole)
        else { throw RecognitionError.boardNotFound }

        return RecognitionResult(
            pieces: placed.pieces,
            corners: placed.corners,
            orientationGuess: Self.orientationGuess(corners: placed.corners)
        )
    }

    /// The pieces with their kinds from the piece classifier and the piece
    /// model together, where there is a classifier and `pieceKinds` is on;
    /// otherwise as the piece model gave them. `image`: the photo the boxes
    /// are in pixels of. Internal for testing.
    func kinds(of found: [PieceDetection], in image: CGImage) -> [PieceDetection] {
        guard pieceKinds, let kindModel, !found.isEmpty else { return found }
        // Where the classifier cannot read the pieces (a crop or Vision
        // fails), the piece model's kinds stand: never no result for a
        // photo that read before.
        guard let probabilities = try? PieceKinds.classProbabilities(of: found, in: image, model: kindModel)
        else { return found }
        return PieceKinds.apply(found, classProbabilities: probabilities)
    }

    /// The corner model's corners where they are a board; nil for no board.
    private func modelCorners(in square: CGImage, letterbox: Letterbox) throws -> [CGPoint]? {
        do {
            let found = try detectCorners(in: square, letterbox: letterbox)
            return Homography(corners: found) != nil ? found : nil
        } catch RecognitionError.boardNotFound {
            // The grid fit may still find the board.
            return nil
        }
    }

    /// The corners: the model's, the grid fit's, or both (see
    /// `corners(model:in:gridFit:fitWindow:)`), with this recognizer's
    /// settings; nil for no board. The one place where the settings are
    /// handed on, and what `recognize` calls, so that a test can see that
    /// each of them arrives (`EndToEndFixtureTests`). Internal for testing.
    func corners(
        model: [CGPoint]?, in image: CGImage, whole: GrayImage.Working? = nil
    ) -> (corners: [CGPoint], fromModel: Bool)? {
        Self.corners(
            model: model, in: image, gridFit: gridFit, fitWindow: fitWindow, whole: whole)
    }

    /// What follows the two models, with this recognizer's settings: the one
    /// place where they are handed on, so that a test can see that each of
    /// them arrives (`EndToEndFixtureTests`). Internal for testing.
    func place(
        _ found: [PieceDetection], corners: [CGPoint], cornersFromModel: Bool, in image: CGImage,
        whole: GrayImage.Working? = nil
    ) -> (corners: [CGPoint], pieces: [Square: RecognizedPiece])? {
        Self.place(
            found, corners: corners, cornersFromModel: cornersFromModel, in: image,
            sidesFromColour: sidesFromColour, oddSides: oddSides, oneKing: oneKing,
            queenForKing: queenForKing, a1Prior: a1Prior, a1Dark: a1Dark, pull: pull, whole: whole)
    }

    /// What follows the two models, in `evaluate.py`'s order: the sides from
    /// the colours first (the whole board's, then single pieces'), since
    /// the kings are counted by side and which corner is a1 is read from
    /// sides and kinds; then one king a side; then a1, among the ways round
    /// the colours of the squares allow; then the squares. Returns the
    /// corners as finally turned and the board; nil when the corners are
    /// not a board. `whole`: the photo at the grid fit's working size, when
    /// the caller has it. Internal for testing.
    static func place(
        _ found: [PieceDetection],
        corners located: [CGPoint],
        cornersFromModel: Bool,
        in image: CGImage,
        sidesFromColour: Bool,
        oddSides: Bool,
        oneKing: Bool,
        queenForKing: Bool,
        a1Prior: Double?,
        a1Dark: Bool,
        pull: Double,
        whole: GrayImage.Working? = nil
    ) -> (corners: [CGPoint], pieces: [Square: RecognizedPiece])? {
        settle(
            sides(of: found, in: image, sidesFromColour: sidesFromColour, oddSides: oddSides),
            corners: located, cornersFromModel: cornersFromModel, in: image, oneKing: oneKing,
            queenForKing: queenForKing, a1Prior: a1Prior, a1Dark: a1Dark, pull: pull, whole: whole)
    }

    /// The second half of `place`, from the pieces with their sides as the
    /// colour steps left them: one king a side, a1, the squares. Apart, so
    /// that what is shown of those pieces (`diagnosePieces`) is the very
    /// list this is given. Internal for testing.
    static func settle(
        _ sided: [PieceDetection],
        corners located: [CGPoint],
        cornersFromModel: Bool,
        in image: CGImage,
        oneKing: Bool,
        queenForKing: Bool,
        a1Prior: Double?,
        a1Dark: Bool,
        pull: Double,
        whole: GrayImage.Working? = nil
    ) -> (corners: [CGPoint], pieces: [Square: RecognizedPiece])? {
        var corners = located
        var detections = sided
        if oneKing {
            detections = OneKingASide.apply(detections, queenForKing: queenForKing)
        }
        if let prior = orientationPrior(fromModel: cornersFromModel, configured: a1Prior) {
            // As `evaluate.py --a1-dark`: on the corners as they are by now,
            // the grid fit's where it stands, the model's where it does not.
            let turns = a1Dark
                ? GridCorners.squareColours(in: image, corners: corners, whole: whole).rotations
                : nil
            corners = BoardOrientation.orient(
                corners: corners, detections: detections, pull: pull, prior: prior, turns: turns)
        }
        guard let homography = Homography(corners: corners) else { return nil }
        return (corners, SquareAssignment.assign(
            detections: detections, homography: homography, pull: pull))
    }

    /// The pieces with their sides as the two colour steps leave them, each
    /// step where it is switched on: the whole board's colours first, then
    /// single pieces'. What one king a side is given. Internal for testing.
    static func sides(
        of found: [PieceDetection], in image: CGImage, sidesFromColour: Bool, oddSides: Bool
    ) -> [PieceDetection] {
        guard sidesFromColour || oddSides else { return found }
        var detections = found
        // One measurement for both steps: neither moves a box.
        let colours = PieceSides.colours(of: detections, in: image)
        if sidesFromColour {
            detections = PieceSides.fromColour(detections, colours: colours)
        }
        if oddSides {
            detections = OddSides.apply(detections, colours: colours)
        }
        return detections
    }

    /// Everything the recognizer has in hand just before one king a side,
    /// with its settings as they are, and what it then reads: for replaying
    /// one king a side, a1 from the pieces and the squares in the Python on
    /// exactly what the app has (`app_check --pieces`,
    /// `swift_reference.py pieces`).
    public func diagnosePieces(_ photo: CGImage) throws -> PiecesDiagnosis {
        let image = DecodedImage.once(photo)
        let letterbox = Letterbox(
            imageSize: CGSize(width: image.width, height: image.height))
        guard let square = letterbox.render(image) else {
            return PiecesDiagnosis(
                corners: nil, cornersFromModel: false, squareColours: nil, pieces: [], placement: nil)
        }
        // As `recognize`.
        let whole = gridFit || (a1Dark && a1Prior != nil) ? GridCorners.workingImage(of: image) : nil
        let located = corners(
            model: try modelCorners(in: square, letterbox: letterbox), in: image, whole: whole)
        // Each box the model takes for a piece, with all its scores.
        let found = try pieceObservations(in: square).compactMap { observation in
            Self.piece(from: observation, letterbox: letterbox).map { (observation.scores, $0) }
        }
        let model = found.map(\.1)
        // As `recognize`: the kinds, then the sides.
        let detections = Self.sides(
            of: kinds(of: model, in: image), in: image, sidesFromColour: sidesFromColour,
            oddSides: oddSides)
        let colours = PieceSides.colours(of: model, in: image)

        // What `recognize` goes on to do, on the pieces shown.
        var placement: String?
        if let located, let placed = Self.settle(
            detections, corners: located.corners, cornersFromModel: located.fromModel, in: image,
            oneKing: oneKing, queenForKing: queenForKing, a1Prior: a1Prior, a1Dark: a1Dark, pull: pull,
            whole: whole) {
            placement = FENBuilder.placement(placed.pieces.mapValues(\.piece))
        }
        return PiecesDiagnosis(
            corners: located?.corners,
            cornersFromModel: located?.fromModel ?? false,
            squareColours: located.map {
                GridCorners.squareColours(in: image, corners: $0.corners, whole: whole)
            },
            pieces: detections.indices.map { index in
                PiecesDiagnosis.Piece(
                    label: PieceSides.label(detections[index]),
                    modelLabel: PieceSides.label(model[index]),
                    confidence: detections[index].confidence,
                    box: detections[index].boundingBox,
                    scores: found[index].0.map { PiecesDiagnosis.Score(label: $0.identifier, score: $0.score) },
                    kindScores: Dictionary(uniqueKeysWithValues: detections[index].kindScores.map {
                        (PieceSides.name($0.key), $0.value)
                    }),
                    colour: colours[index].map { [$0.l, $0.a, $0.b] },
                    hidden: OddSides.patchHidden(model, index))
            },
            placement: placement)
    }

    /// What the piece model finds in a photo and what the colour step
    /// (`PieceSides`) makes of it, whether or not the step is switched on:
    /// for checking it against the Python (`app_check --colours`).
    public func diagnosePieceSides(_ photo: CGImage) throws -> [PieceSideDiagnosis] {
        let image = DecodedImage.once(photo)
        let letterbox = Letterbox(
            imageSize: CGSize(width: image.width, height: image.height))
        guard let square = letterbox.render(image) else { return [] }
        return PieceSides.diagnose(try detectPieces(in: square, letterbox: letterbox), in: image)
    }

    /// The corners the recognizer ends with before which of them is a1 is
    /// checked against the pieces, and what the colours of the squares say
    /// of them, whether or not that is switched on: for checking it against
    /// the Python (`app_check --dark`). Nil for no board.
    public func diagnoseSquareColours(
        _ photo: CGImage
    ) throws -> (corners: [CGPoint], colours: GridCorners.SquareColours)? {
        let image = DecodedImage.once(photo)
        let letterbox = Letterbox(
            imageSize: CGSize(width: image.width, height: image.height))
        guard let square = letterbox.render(image),
              let located = corners(
                  model: try modelCorners(in: square, letterbox: letterbox), in: image)
        else { return nil }
        return (located.corners, GridCorners.squareColours(in: image, corners: located.corners))
    }

    /// The corner model's corners and the grid fit's in the same order, where
    /// both found the board, and what the model makes of the fit's window,
    /// whether or not that is switched on: for checking it against the
    /// Python (`app_check --window`). Nil where either found no board.
    public func diagnoseFitWindow(
        _ photo: CGImage
    ) throws -> (model: [CGPoint], fit: [CGPoint], window: GridCorners.Window)? {
        let image = DecodedImage.once(photo)
        let letterbox = Letterbox(
            imageSize: CGSize(width: image.width, height: image.height))
        guard let square = letterbox.render(image),
              let model = try modelCorners(in: square, letterbox: letterbox),
              let found = GridCorners.find(in: image)
        else { return nil }
        let fit = GridCorners.turned(found.corners, nearestTo: model)
        return (model, fit, GridCorners.windowByModel(fit: fit, model: model))
    }

    // MARK: - Grid fit

    /// The board's corners from what the corner model and the grid fit say,
    /// as `evaluate.py --grid-corners --fit-window` combines them. Internal
    /// for testing.
    ///
    /// - The model found the board and the fit accepts its own result: the
    ///   fit's corners (it says where they are exactly), turned to the
    ///   rotation nearest the model's (it has an opinion on which is a1; the
    ///   fit has none); with `fitWindow`, moved by one square where the
    ///   model has the board in the neighbouring window of the fit's grid
    ///   (`GridCorners.windowByModel`).
    /// - The model found the board and the fit rejects itself, or is switched
    ///   off: the model's corners.
    /// - The model found no board and the fit accepts: the fit's corners, in
    ///   no particular rotation (`fromModel` false: a1 must come from the
    ///   pieces alone).
    /// - Neither: nil, no board.
    static func corners(
        model: [CGPoint]?, in image: CGImage, gridFit: Bool, fitWindow: Bool,
        whole: GrayImage.Working? = nil
    ) -> (corners: [CGPoint], fromModel: Bool)? {
        let fit = gridFit ? GridCorners.find(in: image, whole: whole) : nil
        switch (model, fit) {
        case let (model?, fit?):
            let turned = GridCorners.turned(fit.corners, nearestTo: model)
            return (
                fitWindow ? GridCorners.windowByModel(fit: turned, model: model).corners : turned,
                true)
        case let (model?, nil):
            return (model, true)
        case let (nil, fit?):
            return (fit.corners, false)
        case (nil, nil):
            return nil
        }
    }

    /// The head start the corners as given get when a1 is checked against the
    /// pieces; nil for no check. Corners that are the grid fit's alone carry
    /// no opinion on a1 to weigh against the pieces: they decide alone (0),
    /// whatever was configured, as in `evaluate.py`. Internal for testing.
    static func orientationPrior(fromModel: Bool, configured: Double?) -> Double? {
        fromModel ? configured : 0
    }

    // MARK: - Corner model

    /// `square` is the letterboxed photo; corners come back in pixels of the
    /// original photo.
    private func detectCorners(in square: CGImage, letterbox: Letterbox) throws -> [CGPoint] {
        let request = VNCoreMLRequest(model: cornerModel)
        request.imageCropAndScaleOption = .scaleFill  // square in, square out: resize only
        try VNImageRequestHandler(cgImage: square).perform([request])

        guard
            let observation = request.results?
                .compactMap({ $0 as? VNCoreMLFeatureValueObservation })
                .first(where: { $0.featureValue.multiArrayValue != nil }),
            let array = observation.featureValue.multiArrayValue
        // Not "no board": the model did not answer in the form expected. It
        // must not be mistaken for a photo without a board, which the grid
        // fit may still read.
        else { throw RecognitionError.modelUnavailable }

        return try Self.parseCorners(
            from: array,
            letterbox: letterbox,
            scoreThreshold: cornerScoreThreshold,
            inputSide: cornerInputSide
        )
    }

    /// Parses the raw `[1, 17, anchors]` pose tensor, whose keypoints are in
    /// pixels of the square model input. Internal for testing.
    static func parseCorners(
        from array: MLMultiArray,
        letterbox: Letterbox,
        scoreThreshold: Float,
        inputSide: Double = 640
    ) throws -> [CGPoint] {
        guard array.shape.count == 3,
              array.shape[1].intValue == 17
        else { throw RecognitionError.modelUnavailable }
        let anchors = array.shape[2].intValue

        // Contiguous float32 [1, 17, anchors]: index(c, a) = c * anchors + a.
        // All reads happen inside the closure — the buffer pointer must not
        // escape its scope.
        let corners: [CGPoint]? = array.withUnsafeBufferPointer(
            ofType: Float32.self
        ) { values in
            var bestAnchor = -1
            var bestScore: Float32 = 0
            for a in 0..<anchors {
                let score = values[4 * anchors + a]
                if score > bestScore {
                    bestScore = score
                    bestAnchor = a
                }
            }
            guard bestAnchor >= 0, bestScore >= scoreThreshold else { return nil }

            return (0..<4).map { k in
                letterbox.imagePoint(fromSquare: CGPoint(
                    x: Double(values[(5 + k * 3) * anchors + bestAnchor]) / inputSide,
                    y: Double(values[(6 + k * 3) * anchors + bestAnchor]) / inputSide
                ))
            }
        }
        guard let corners else { throw RecognitionError.boardNotFound }
        return corners  // (a1, a8, h8, h1) in image coordinates
    }

    // MARK: - Piece model

    /// `square` is the letterboxed photo; boxes come back in pixels of the
    /// original photo. Internal for testing.
    func detectPieces(in square: CGImage, letterbox: Letterbox) throws -> [PieceDetection] {
        try pieceObservations(in: square).compactMap { Self.piece(from: $0, letterbox: letterbox) }
    }

    /// One box of the piece model as Vision hands it out: the model's score
    /// for every class, the highest first (`classScores`), and the box as
    /// fractions of the square with a lower-left origin.
    typealias PieceObservation = (scores: [(identifier: String, score: Float)], box: CGRect)

    /// What the piece model finds in the letterboxed photo, before anything
    /// is made of it. Internal for testing.
    func pieceObservations(in square: CGImage) throws -> [PieceObservation] {
        let request = VNCoreMLRequest(model: pieceModel)
        request.imageCropAndScaleOption = .scaleFill  // square in, square out: resize only
        try VNImageRequestHandler(cgImage: square).perform([request])

        let observations = request.results?
            .compactMap { $0 as? VNRecognizedObjectObservation } ?? []
        return observations.map { obs in
            // (Vision's labels come sorted, the highest first.)
            (Self.classScores(
                labels: obs.labels.map { ($0.identifier, $0.confidence) }, sum: obs.confidence),
             obs.boundingBox)
        }
    }

    /// The piece an observation is taken for: the class with the highest
    /// score, nil when that is no piece. Internal for testing.
    static func piece(from observation: PieceObservation, letterbox: Letterbox) -> PieceDetection? {
        guard
            let top = observation.scores.first,
            let (kind, color) = pieceClass(fromLabel: top.identifier)
        else { return nil }
        // Vision boxes are fractions of the square with a lower-left origin.
        let bb = observation.box
        let rect = letterbox.imageRect(fromSquare: CGRect(
            x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height))
        return PieceDetection(
            piece: kind, color: color,
            // The model's score, as in the Python. (Vision's own number
            // for the label is its share of the box's scores: near 1
            // for a piece the model is unsure is there at all.)
            confidence: top.score,
            boundingBox: rect,
            kindScores: kindScores(observation.scores)
        )
    }

    /// The model's score for each class of one box, from what Vision makes
    /// of it: Vision divides the scores of a box by their sum, so that its
    /// labels' confidences add up to 1, and gives the sum as the confidence
    /// of the observation. Internal for testing.
    ///
    /// Vision does not document this; it is checked on macOS
    /// (`EndToEndFixtureTests.testPieceScoresAreTheModelsOwn`), not on iOS.
    /// So labels that do not add up to 1 are taken to be the scores
    /// themselves and are left as they are.
    static func classScores(
        labels: [(identifier: String, confidence: Float)], sum: Float
    ) -> [(identifier: String, score: Float)] {
        let total = labels.reduce(Float(0)) { $0 + $1.confidence }
        guard abs(total - 1) <= 1e-3 else {
            return labels.map { ($0.identifier, $0.confidence) }
        }
        return labels.map { ($0.identifier, $0.confidence * sum) }
    }

    /// The scores of one box by kind, the larger of the two colours
    /// (`PieceDetection.kindScores`). A score under `leastKindScore` is left
    /// out, and labels that are no piece. Internal for testing.
    static func kindScores(_ scores: [(identifier: String, score: Float)]) -> [Piece.Kind: Double] {
        var out: [Piece.Kind: Double] = [:]
        for (identifier, score) in scores {
            guard let (kind, _) = pieceClass(fromLabel: identifier) else { continue }
            let score = Double(score)
            guard score.isFinite, score >= leastKindScore else { continue }
            out[kind] = max(out[kind] ?? 0, score)
        }
        return out
    }

    /// The least score that counts as the model having considered a kind for
    /// a piece. `evaluate.py` sees none under it either (it asks the model
    /// for boxes down to 0.02), and the rule that uses the scores
    /// (`OneKingASide`) was measured that way: without it a king too many
    /// would become whichever kind has the largest of five scores that are
    /// all next to nothing, and not a queen.
    static let leastKindScore = 0.02

    /// `"white_pawn"` → (.pawn, .white). Internal for testing.
    static func pieceClass(fromLabel label: String) -> (Piece.Kind, Piece.Color)? {
        let parts = label.split(separator: "_")
        guard parts.count == 2 else { return nil }
        let color: Piece.Color
        switch parts[0] {
        case "white": color = .white
        case "black": color = .black
        default: return nil
        }
        let kind: Piece.Kind
        switch parts[1] {
        case "pawn": kind = .pawn
        case "knight": kind = .knight
        case "bishop": kind = .bishop
        case "rook": kind = .rook
        case "queen": kind = .queen
        case "king": kind = .king
        default: return nil
        }
        return (kind, color)
    }

    // MARK: - Orientation

    /// If the a1 corner sits lower in the image than a8, the camera is on
    /// White's side of the board.
    static func orientationGuess(corners: [CGPoint]) -> RecognizedOrientation {
        guard corners.count == 4 else { return .whiteBottom }
        let a1 = corners[0]
        let a8 = corners[1]
        return a1.y >= a8.y ? .whiteBottom : .blackBottom
    }
}

/// What the recognizer has in hand for one photo just before one king a
/// side, and what it reads (`CoreMLBoardRecognizer.diagnosePieces`).
public struct PiecesDiagnosis {

    /// The piece model's score for one label of a box.
    public struct Score {
        public let label: String
        public let score: Float
    }

    public struct Piece {
        /// The class as it stands before one king a side, as the Python
        /// names it: `"white_pawn"`.
        public let label: String
        /// The piece model's own answer, before either colour step.
        public let modelLabel: String
        /// The model's score for its own answer.
        public let confidence: Float
        /// In pixels of the photo, origin top-left.
        public let box: CGRect
        /// The model's score for every label Vision hands out for the box,
        /// the highest first, before any cut.
        public let scores: [Score]
        /// What one king a side sees: for each kind the larger score of the
        /// two colours, and nothing under
        /// `CoreMLBoardRecognizer.leastKindScore`. Keyed `"king"`, `"queen"`.
        public let kindScores: [String: Double]
        /// The median L*, a*, b* of the middle of the box; nil when that
        /// lies outside the photo.
        public let colour: [Double]?
        /// Whether another piece stands in front of the middle of the box.
        public let hidden: Bool
    }

    /// The corners before a1 is taken from the pieces (the model's, the
    /// grid fit's, or the fit's as the model moved them); nil for no board.
    public let corners: [CGPoint]?
    /// Whether the corner model had a say in which corner is a1. Where it
    /// had none (the fit alone) the pieces decide with no head start.
    public let cornersFromModel: Bool
    /// What the colours of the squares say of `corners`; nil for no board.
    public let squareColours: GridCorners.SquareColours?
    /// The pieces in the order the model gave them.
    public let pieces: [Piece]
    /// What the recognizer reads from all of this, as a placement; nil for
    /// no board.
    public let placement: String?
}
