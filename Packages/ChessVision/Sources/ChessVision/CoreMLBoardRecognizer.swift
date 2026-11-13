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
///   (lower-left origin) and class labels like `"white_pawn"`.
///
/// After the models, three steps that need no learning
/// (`docs/RECOGNITION.md`): the corners are replaced by the classical grid
/// fit (`GridCorners`) where it accepts its own result, White and Black are
/// taken from the colours of the pieces where the piece model's own answer
/// does not fit them (`PieceSides`), and which corner is a1 is checked
/// against the pieces (`BoardOrientation`).
///
/// Both models see the photo letterboxed into a square (`Letterbox`), as in
/// training, never stretched. The square is passed to Vision with
/// `.scaleFill`, which for a square only resizes; `Letterbox` maps the
/// models' answers back to pixels of the original photo.
public struct CoreMLBoardRecognizer: BoardRecognizer {

    private let cornerModel: VNCoreMLModel
    private let pieceModel: VNCoreMLModel
    private let cornerScoreThreshold: Float
    private let pull: Double
    private let a1Prior: Double?
    private let gridFit: Bool
    private let sidesFromColour: Bool

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
    ///     more than a share of the pieces (`PieceSides`).
    public init(
        cornerModel: MLModel,
        pieceModel: MLModel,
        cornerScoreThreshold: Float = 0.25,
        pull: Double = 0.30,  // = SquareAssignment.defaultPull
        a1Prior: Double? = 6,  // = BoardOrientation.appPrior
        gridFit: Bool = true,
        sidesFromColour: Bool = true
    ) throws {
        self.cornerModel = try VNCoreMLModel(for: cornerModel)
        self.pieceModel = try VNCoreMLModel(for: pieceModel)
        self.cornerScoreThreshold = cornerScoreThreshold
        self.pull = pull
        self.a1Prior = a1Prior
        self.gridFit = gridFit
        self.sidesFromColour = sidesFromColour
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
    /// from a bundle, on the Neural Engine + CPU (never GPU — avoids
    /// contention with camera/UI rendering).
    public init(bundle: Bundle) throws {
        guard
            let cornersURL = bundle.url(forResource: "BoardCorners", withExtension: "mlmodelc"),
            let piecesURL = bundle.url(forResource: "Pieces", withExtension: "mlmodelc")
        else { throw RecognitionError.modelUnavailable }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        try self.init(
            cornerModel: MLModel(contentsOf: cornersURL, configuration: config),
            pieceModel: MLModel(contentsOf: piecesURL, configuration: config)
        )
    }

    // MARK: - BoardRecognizer

    public func recognize(_ image: CGImage) async throws -> RecognitionResult {
        let letterbox = Letterbox(
            imageSize: CGSize(width: image.width, height: image.height))
        guard let square = letterbox.render(image) else {
            throw RecognitionError.boardNotFound
        }

        // Corners: the model's, the grid fit's, or both (see `corners(model:in:gridFit:)`).
        var modelCorners: [CGPoint]?
        do {
            let found = try detectCorners(in: square, letterbox: letterbox)
            if Homography(corners: found) != nil {
                modelCorners = found
            }
        } catch RecognitionError.boardNotFound {
            // The grid fit may still find the board.
        }
        guard let located = Self.corners(model: modelCorners, in: image, gridFit: gridFit) else {
            throw RecognitionError.boardNotFound
        }
        let detections = try detectPieces(in: square, letterbox: letterbox)
        guard let placed = Self.place(
            detections, corners: located.corners, cornersFromModel: located.fromModel, in: image,
            sidesFromColour: sidesFromColour, a1Prior: a1Prior, pull: pull)
        else { throw RecognitionError.boardNotFound }

        return RecognitionResult(
            pieces: placed.pieces,
            corners: placed.corners,
            orientationGuess: Self.orientationGuess(corners: placed.corners)
        )
    }

    /// What follows the two models, in `evaluate.py`'s order: the sides from
    /// the colours first, since which corner is a1 is then read from them,
    /// then the squares. Returns the corners as finally turned and the board;
    /// nil when the corners are not a board. Internal for testing.
    static func place(
        _ found: [PieceDetection],
        corners located: [CGPoint],
        cornersFromModel: Bool,
        in image: CGImage,
        sidesFromColour: Bool,
        a1Prior: Double?,
        pull: Double
    ) -> (corners: [CGPoint], pieces: [Square: RecognizedPiece])? {
        var corners = located
        var detections = found
        if sidesFromColour {
            detections = PieceSides.fromColour(detections, in: image)
        }
        if let prior = orientationPrior(fromModel: cornersFromModel, configured: a1Prior) {
            corners = BoardOrientation.orient(
                corners: corners, detections: detections, pull: pull, prior: prior)
        }
        guard let homography = Homography(corners: corners) else { return nil }
        return (corners, SquareAssignment.assign(
            detections: detections, homography: homography, pull: pull))
    }

    /// What the piece model finds in a photo and what the colour step
    /// (`PieceSides`) makes of it, whether or not the step is switched on:
    /// for checking it against the Python (`app_check --colours`).
    public func diagnosePieceSides(_ image: CGImage) throws -> [PieceSideDiagnosis] {
        let letterbox = Letterbox(
            imageSize: CGSize(width: image.width, height: image.height))
        guard let square = letterbox.render(image) else { return [] }
        return PieceSides.diagnose(try detectPieces(in: square, letterbox: letterbox), in: image)
    }

    // MARK: - Grid fit

    /// The board's corners from what the corner model and the grid fit say,
    /// as `evaluate.py --grid-corners` combines them. Internal for testing.
    ///
    /// - The model found the board and the fit accepts its own result: the
    ///   fit's corners (it says where they are exactly), turned to the
    ///   rotation nearest the model's (it has an opinion on which is a1; the
    ///   fit has none).
    /// - The model found the board and the fit rejects itself, or is switched
    ///   off: the model's corners.
    /// - The model found no board and the fit accepts: the fit's corners, in
    ///   no particular rotation (`fromModel` false: a1 must come from the
    ///   pieces alone).
    /// - Neither: nil, no board.
    static func corners(
        model: [CGPoint]?, in image: CGImage, gridFit: Bool
    ) -> (corners: [CGPoint], fromModel: Bool)? {
        let fit = gridFit ? GridCorners.find(in: image) : nil
        switch (model, fit) {
        case let (model?, fit?):
            return (GridCorners.turned(fit.corners, nearestTo: model), true)
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

    private func detectPieces(in square: CGImage, letterbox: Letterbox) throws -> [PieceDetection] {
        let request = VNCoreMLRequest(model: pieceModel)
        request.imageCropAndScaleOption = .scaleFill  // square in, square out: resize only
        try VNImageRequestHandler(cgImage: square).perform([request])

        let observations = request.results?
            .compactMap { $0 as? VNRecognizedObjectObservation } ?? []

        return observations.compactMap { obs in
            guard
                let label = obs.labels.first,
                let (kind, color) = Self.pieceClass(fromLabel: label.identifier)
            else { return nil }
            // Vision boxes are fractions of the square with a lower-left origin.
            let bb = obs.boundingBox
            let rect = letterbox.imageRect(fromSquare: CGRect(
                x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height))
            return PieceDetection(
                piece: kind, color: color,
                confidence: label.confidence,
                boundingBox: rect
            )
        }
    }

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
