import CoreGraphics

/// Converts a photo of a physical chess board into a piece placement.
///
/// Phase 0 ships ``StubRecognizer``; Phase 2 replaces it with a CoreML-backed
/// implementation at the app's composition root. Nothing else changes.
public protocol BoardRecognizer {
    func recognize(_ image: CGImage) async throws -> RecognitionResult
}
