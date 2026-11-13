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
/// After the models, two steps that need no learning (`docs/RECOGNITION.md`):
/// the corners are replaced by the classical grid fit (`GridCorners`) where it
/// accepts its own result, and which corner is a1 is checked against the
/// pieces (`BoardOrientation`).
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

    /// - Parameters:
    ///   - a1Prior: which corner is a1 is checked against the pieces
    ///     (`BoardOrientation.orient`); this is how much the corner model's
    ///     own choice counts. `nil` keeps the corner model's a1 as it is.
    ///   - gridFit: take the corners from the classical grid fit
    ///     (`GridCorners`) where it accepts its own result, and from the
    ///     corner model where it does not.
    public init(
        cornerModel: MLModel,
        pieceModel: MLModel,
        cornerScoreThreshold: Float = 0.25,
        pull: Double = 0.30,  // = SquareAssignment.defaultPull
        a1Prior: Double? = 6,  // = BoardOrientation.appPrior
        gridFit: Bool = true
    ) throws {
        self.cornerModel = try VNCoreMLModel(for: cornerModel)
        self.pieceModel = try VNCoreMLModel(for: pieceModel)
        self.cornerScoreThreshold = cornerScoreThreshold
        self.pull = pull
        self.a1Prior = a1Prior
        self.gridFit = gridFit
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

        var corners = try detectCorners(in: square, letterbox: letterbox)
        guard Homography(corners: corners) != nil else {
            throw RecognitionError.boardNotFound
        }
        if gridFit {
            corners = Self.cornersWithGridFit(modelCorners: corners, in: image)
        }

        let detections = try detectPieces(in: square, letterbox: letterbox)
        if let a1Prior {
            corners = BoardOrientation.orient(
                corners: corners, detections: detections, pull: pull, prior: a1Prior)
        }
        guard let homography = Homography(corners: corners) else {
            throw RecognitionError.boardNotFound
        }
        let board = SquareAssignment.assign(
            detections: detections, homography: homography, pull: pull)

        return RecognitionResult(
            pieces: board,
            corners: corners,
            orientationGuess: Self.orientationGuess(corners: corners)
        )
    }

    // MARK: - Grid fit

    /// The corner model says where the board is and which corner it takes
    /// for a1; the grid fit, where it accepts its own result, says where the
    /// corners are exactly. It does not know a1, so its corners are turned to
    /// the rotation nearest the model's. Where it rejects itself the model's
    /// corners stay. As `evaluate.py --grid-corners`. Internal for testing.
    static func cornersWithGridFit(modelCorners: [CGPoint], in image: CGImage) -> [CGPoint] {
        guard let fit = GridCorners.find(in: image) else { return modelCorners }
        return GridCorners.turned(fit.corners, nearestTo: modelCorners)
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
        else { throw RecognitionError.boardNotFound }

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
