import ChessKit
import CoreGraphics

/// Which way the recognized board is oriented in the source image.
public enum RecognizedOrientation: Sendable, Equatable {
    case whiteBottom
    case blackBottom
}

/// A piece detected on a square, with the model's confidence in [0, 1].
public struct RecognizedPiece: Equatable {
    public let piece: Piece
    public let confidence: Float

    public init(piece: Piece, confidence: Float) {
        self.piece = piece
        self.confidence = confidence
    }
}

/// Output of a board recognizer: piece placement plus enough metadata for the
/// review UI (corner overlay, per-square confidence, orientation guess).
public struct RecognitionResult {
    public let pieces: [Square: RecognizedPiece]
    /// Detected board corner points in image coordinates (a1, a8, h8, h1 order).
    /// Empty when the recognizer has no geometric stage (e.g. the stub).
    public let corners: [CGPoint]
    public let orientationGuess: RecognizedOrientation

    public init(
        pieces: [Square: RecognizedPiece],
        corners: [CGPoint] = [],
        orientationGuess: RecognizedOrientation = .whiteBottom
    ) {
        self.pieces = pieces
        self.corners = corners
        self.orientationGuess = orientationGuess
    }
}

public enum RecognitionError: Error, Equatable {
    case boardNotFound
    case modelUnavailable
}
