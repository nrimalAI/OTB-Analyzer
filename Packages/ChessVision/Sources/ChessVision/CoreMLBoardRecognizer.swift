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
/// Both requests use `.scaleFill`, so model coordinates map back to the
/// original image by plain axis scaling.
public struct CoreMLBoardRecognizer: BoardRecognizer {

    private let cornerModel: VNCoreMLModel
    private let pieceModel: VNCoreMLModel
    private let cornerScoreThreshold: Float
    private let pull: Double

    public init(
        cornerModel: MLModel,
        pieceModel: MLModel,
        cornerScoreThreshold: Float = 0.25,
        pull: Double = 0.30  // = SquareAssignment.defaultPull
    ) throws {
        self.cornerModel = try VNCoreMLModel(for: cornerModel)
        self.pieceModel = try VNCoreMLModel(for: pieceModel)
        self.cornerScoreThreshold = cornerScoreThreshold
        self.pull = pull
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
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)

        let corners = try detectCorners(in: image, imageSize: CGSize(width: width, height: height))
        guard let homography = Homography(corners: corners) else {
            throw RecognitionError.boardNotFound
        }

        let detections = try detectPieces(in: image, imageSize: CGSize(width: width, height: height))
        let board = SquareAssignment.assign(
            detections: detections, homography: homography, pull: pull)

        return RecognitionResult(
            pieces: board,
            corners: corners,
            orientationGuess: Self.orientationGuess(corners: corners)
        )
    }

    // MARK: - Corner model

    private func detectCorners(in image: CGImage, imageSize: CGSize) throws -> [CGPoint] {
        let request = VNCoreMLRequest(model: cornerModel)
        request.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: image).perform([request])

        guard
            let observation = request.results?
                .compactMap({ $0 as? VNCoreMLFeatureValueObservation })
                .first(where: { $0.featureValue.multiArrayValue != nil }),
            let array = observation.featureValue.multiArrayValue
        else { throw RecognitionError.boardNotFound }

        return try Self.parseCorners(
            from: array,
            imageSize: imageSize,
            scoreThreshold: cornerScoreThreshold
        )
    }

    /// Parses the raw `[1, 17, anchors]` pose tensor. Internal for testing.
    static func parseCorners(
        from array: MLMultiArray,
        imageSize: CGSize,
        scoreThreshold: Float
    ) throws -> [CGPoint] {
        guard array.shape.count == 3,
              array.shape[1].intValue == 17
        else { throw RecognitionError.modelUnavailable }
        let anchors = array.shape[2].intValue
        let inputSide = 640.0

        // Contiguous float32 [1, 17, anchors]: index(c, a) = c * anchors + a.
        let values: UnsafeBufferPointer<Float32> = try array.withUnsafeBufferPointer(
            ofType: Float32.self) { UnsafeBufferPointer(rebasing: $0[...]) }

        var bestAnchor = -1
        var bestScore: Float32 = 0
        for a in 0..<anchors {
            let score = values[4 * anchors + a]
            if score > bestScore {
                bestScore = score
                bestAnchor = a
            }
        }
        guard bestAnchor >= 0, bestScore >= scoreThreshold else {
            throw RecognitionError.boardNotFound
        }

        let sx = Double(imageSize.width) / inputSide
        let sy = Double(imageSize.height) / inputSide
        var corners: [CGPoint] = []
        for k in 0..<4 {
            let x = Double(values[(5 + k * 3) * anchors + bestAnchor]) * sx
            let y = Double(values[(6 + k * 3) * anchors + bestAnchor]) * sy
            corners.append(CGPoint(x: x, y: y))
        }
        return corners  // (a1, a8, h8, h1) in image coordinates
    }

    // MARK: - Piece model

    private func detectPieces(in image: CGImage, imageSize: CGSize) throws -> [PieceDetection] {
        let request = VNCoreMLRequest(model: pieceModel)
        request.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: image).perform([request])

        let observations = request.results?
            .compactMap { $0 as? VNRecognizedObjectObservation } ?? []

        return observations.compactMap { obs in
            guard
                let label = obs.labels.first,
                let (kind, color) = Self.pieceClass(fromLabel: label.identifier)
            else { return nil }
            // Vision boxes are normalized with a lower-left origin.
            let bb = obs.boundingBox
            let rect = CGRect(
                x: bb.minX * imageSize.width,
                y: (1 - bb.maxY) * imageSize.height,
                width: bb.width * imageSize.width,
                height: bb.height * imageSize.height
            )
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
