import ChessKit
import CoreGraphics

/// Placeholder recognizer used until the CoreML models exist (Phase 2).
/// Always "recognizes" the standard starting position with full confidence,
/// which exercises the entire capture → review → analyze flow.
public struct StubRecognizer: BoardRecognizer {
    public init() {}

    public func recognize(_ image: CGImage) async throws -> RecognitionResult {
        let placement = FENBuilder.pieces(
            fromPlacement: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        ) ?? [:]
        let recognized = placement.mapValues { RecognizedPiece(piece: $0, confidence: 1.0) }
        return RecognitionResult(pieces: recognized)
    }
}
