import ChessKit
import CoreGraphics
import CoreImage
import CoreML
import Vision

/// The piece classifier (`Models/PieceKinds.mlpackage`, a YOLO11s-cls): a
/// third model, after the piece model, that reads each piece the piece model
/// found again, cut out of the photo at a higher resolution than the piece
/// model's 640-pixel input gives it. Port of the reference's
/// `chess_pipeline.classifier_scale`, `classifier_square` and
/// `kinds_from_classifier` (`Training/chess_pipeline.py`).
///
/// A cut-out piece shows its kind but not its side: on a set whose "white"
/// pieces are brown the classifier calls them black, and a side is told
/// from the other side's colour, which a crop does not have. So the side
/// stays the piece model's, and of the classifier only the kind is used: its
/// two sides' probabilities summed, multiplied by what the piece model said
/// (its confidence for its own kind, the rest shared evenly among the other
/// five). The kind with the largest product, normalised, is taken; its share
/// becomes the confidence, and all six shares the scores by kind that one
/// king a side is then given (`OneKingASide`). A piece the piece model gives
/// `sure` or more is left as it is: on the development sets the classifier
/// then gains as much and changes fewer pieces.
enum PieceKinds {

    /// The photo is drawn at most this many pixels on its long side before
    /// the pieces are cut out of it (the training crops were cut out of
    /// 1,280-pixel copies).
    static let photoSide = 1280
    /// The square a piece is cut out in is this many times the box's longer side.
    static let margin = 1.2
    /// The classifier's input side.
    static let side = 128
    /// A piece the piece model gives at least this is left as it is
    /// (`chess_pipeline.CLASSIFIER_SURE`).
    static let sure: Float = 0.9
    /// The kinds in the reference's order (`chess_pipeline.KINDS`): of two
    /// kinds with the same share, the first is taken.
    static let kinds: [Piece.Kind] = [.king, .queen, .rook, .bishop, .knight, .pawn]

    /// The factor the photo is drawn at for the crops (`classifier_scale`).
    static func scale(width: Int, height: Int) -> Double {
        min(1, Double(photoSide) / Double(max(width, height, 1)))
    }

    /// The square a piece is cut out in, in pixels of the photo drawn at
    /// `scale`, origin top-left (`classifier_square`). Halves round up.
    static func square(of box: CGRect, scale: Double) -> (x: Int, y: Int, side: Int) {
        let x1 = box.minX * scale, x2 = box.maxX * scale
        let y1 = box.minY * scale, y2 = box.maxY * scale
        let side = max(2, Int((max(x2 - x1, y2 - y1) * margin + 0.5).rounded(.down)))
        let x = Int(((x1 + x2) / 2 - Double(side) / 2 + 0.5).rounded(.down))
        let y = Int(((y1 + y2) / 2 - Double(side) / 2 + 0.5).rounded(.down))
        return (x, y, side)
    }

    /// The pieces with their kinds from the classifier and the piece model
    /// together (`kinds_from_classifier`). `classProbabilities[k]`: the
    /// classifier's probability for each of its twelve labels for piece k.
    static func apply(
        _ detections: [PieceDetection], classProbabilities: [[String: Float]], sure: Float = sure
    ) -> [PieceDetection] {
        zip(detections, classProbabilities).map { detection, probabilities in
            guard detection.confidence < sure else { return detection }
            // Summed in a fixed order (labels sorted), so a reading does not
            // depend on a dictionary's order.
            var fromClassifier = [Piece.Kind: Double]()
            for label in probabilities.keys.sorted() {
                guard let (kind, _) = CoreMLBoardRecognizer.pieceClass(fromLabel: label) else { continue }
                fromClassifier[kind, default: 0] += Double(probabilities[label]!)
            }
            let confidence = Double(detection.confidence)
            let rest = (1 - confidence) / Double(kinds.count - 1)
            var product = [Piece.Kind: Double]()
            for kind in kinds {
                product[kind] = (fromClassifier[kind] ?? 0) * (kind == detection.piece ? confidence : rest)
            }
            let total = kinds.reduce(0) { $0 + product[$1]! }
            guard total > 0 else {
                return PieceDetection(
                    piece: detection.piece, color: detection.color, confidence: detection.confidence,
                    boundingBox: detection.boundingBox,
                    kindScores: [detection.piece: Double(detection.confidence)])
            }
            var best = kinds[0]
            for kind in kinds where product[kind]! / total > product[best]! / total {
                best = kind
            }
            let shares = product.mapValues { $0 / total }
            // What the user is shown (the "check this piece" mark) stays the
            // piece model's own score for the kind taken: the same as before
            // where the kind did not change, and the piece model's (usually
            // low) score for the new kind where it did, so a changed piece is
            // marked. The share would hide wrong pieces (it is near 1 for most).
            let shown = best == detection.piece
                ? detection.confidence : Float(detection.kindScores[best] ?? 0)
            return PieceDetection(
                piece: best, color: detection.color, confidence: Float(shares[best]!),
                boundingBox: detection.boundingBox, kindScores: shares, shownConfidence: shown)
        }
    }

    /// Each piece cut out of `image` as the reference cuts it and read by the
    /// classifier: its probability for each label. `image` is the photo the
    /// boxes are in pixels of.
    static func classProbabilities(
        of detections: [PieceDetection], in image: CGImage, model: VNCoreMLModel
    ) throws -> [[String: Float]] {
        // A piece that will be left as it is is not read (an empty answer).
        guard detections.contains(where: { $0.confidence < sure }) else {
            return detections.map { _ in [:] }
        }
        let scale = scale(width: image.width, height: image.height)
        guard let drawn = drawn(image, scale: scale) else {
            throw RecognitionError.modelUnavailable
        }
        return try detections.map { detection in
            guard detection.confidence < sure else { return [:] }
            let (x, y, side) = square(of: detection.boundingBox, scale: scale)
            guard let crop = crop(drawn, x: x, y: y, side: side) else {
                throw RecognitionError.modelUnavailable
            }
            let request = VNCoreMLRequest(model: model)
            request.imageCropAndScaleOption = .scaleFill  // square in, square out: resize only
            try VNImageRequestHandler(cgImage: crop).perform([request])
            let observations = request.results?.compactMap { $0 as? VNClassificationObservation } ?? []
            return Dictionary(observations.map { ($0.identifier, $0.confidence) },
                              uniquingKeysWith: { first, _ in first })
        }
    }

    /// The photo drawn at `scale` (itself when that is 1), as the reference's
    /// `cv2.resize` to `round(width * scale)` by `round(height * scale)`.
    static func drawn(_ image: CGImage, scale: Double) -> CGImage? {
        guard scale < 1 else { return image }
        // Python's round(): halves to the even neighbour.
        let width = Int((Double(image.width) * scale).rounded(.toNearestOrEven))
        let height = Int((Double(image.height) * scale).rounded(.toNearestOrEven))
        guard
            let space = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }

    /// The square (x, y, side), origin top-left, cut out of `image` and
    /// resized to the classifier's side; where it runs off the image, the
    /// edge pixels are repeated (`cv2.BORDER_REPLICATE`).
    static func crop(_ image: CGImage, x: Int, y: Int, side: Int) -> CGImage? {
        // Core Image counts y from the bottom. The photo is clamped (its edge
        // pixels repeated without end), moved so that the square starts at the
        // origin, scaled, and only then cut: scaling a cut square would read
        // nothing beyond its edges and darken its border, where the reference
        // (cv2.resize of a cut with its edges repeated) does not.
        let bottom = image.height - y - side
        let placed = CIImage(cgImage: image).clampedToExtent()
            .transformed(by: CGAffineTransform(translationX: CGFloat(-x), y: CGFloat(-bottom)))
        let factor = CGFloat(Self.side) / CGFloat(side)
        let resized: CIImage
        if side > Self.side, let lanczos = CIFilter(name: "CILanczosScaleTransform") {
            // Shrinking: averaging over the pixels, as the reference's INTER_AREA.
            lanczos.setValue(placed, forKey: kCIInputImageKey)
            lanczos.setValue(factor, forKey: kCIInputScaleKey)
            lanczos.setValue(1.0, forKey: kCIInputAspectRatioKey)
            resized = lanczos.outputImage ?? placed.transformed(by: CGAffineTransform(scaleX: factor, y: factor))
        } else {
            resized = placed.transformed(by: CGAffineTransform(scaleX: factor, y: factor))
        }
        return context.createCGImage(
            resized, from: CGRect(x: 0, y: 0, width: Self.side, height: Self.side),
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    /// One Core Image context for all crops, without colour management: the
    /// pixels go through as they are, as in the reference.
    private static let context = CIContext(options: [
        .workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .cacheIntermediates: false,
    ])
}
