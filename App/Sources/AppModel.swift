import ChessBoardUI
import ChessKit
import ChessVision
import SwiftUI

enum AppRoute: Hashable {
    case review
    case analysis
}

/// App-wide state for the capture → review → analyze flow.
@MainActor
final class AppModel: ObservableObject {

    @Published var path: [AppRoute] = []

    // Recognition
    /// CoreML when the models are bundled; stub keeps the flow usable without
    /// them (e.g. before running Training/export_coreml.py).
    let recognizer: BoardRecognizer =
        (try? CoreMLBoardRecognizer(bundle: .main)) ?? StubRecognizer()
    @Published var isRecognizing = false
    @Published var recognitionFailed = false

    // Position under review
    @Published var pieces: [Square: Piece] = [:]
    @Published var confidence: [Square: Float] = [:]
    @Published var sideToMove: Piece.Color = .white
    @Published var castling: CastlingRights = .none
    @Published var orientation: BoardDisplayOrientation = .whiteBottom

    // MARK: - Flow

    func startFromPhoto(_ image: UIImage) async {
        guard let cgImage = image.cgImage else {
            recognitionFailed = true
            return
        }
        isRecognizing = true
        defer { isRecognizing = false }
        do {
            let result = try await recognizer.recognize(cgImage)
            load(result: result)
            path.append(.review)
        } catch {
            recognitionFailed = true
        }
    }

    func startFromScratch(startingPosition: Bool) {
        if startingPosition {
            pieces = FENBuilder.pieces(fromFEN: Position.standard.fen) ?? [:]
        } else {
            pieces = [:]
        }
        confidence = [:]
        sideToMove = .white
        castling = .inferred(from: pieces)
        orientation = .whiteBottom
        path.append(.review)
    }

    private func load(result: RecognitionResult) {
        pieces = result.pieces.mapValues(\.piece)
        confidence = result.pieces.mapValues(\.confidence)
        sideToMove = .white
        castling = .inferred(from: pieces)
        orientation = result.orientationGuess == .whiteBottom ? .whiteBottom : .blackBottom
    }

    // MARK: - Editing

    func set(_ piece: Piece?, at square: Square) {
        if var piece {
            piece.square = square
            pieces[square] = piece
        } else {
            pieces[square] = nil
        }
        confidence[square] = nil  // user-confirmed; no longer "uncertain"
        castling = .inferred(from: pieces)
    }

    // MARK: - Validation / FEN

    var validationIssues: [ValidationIssue] {
        PositionValidator.validate(pieces: pieces, sideToMove: sideToMove)
    }

    var problemSquares: Set<Square> {
        PositionValidator.highlightSquares(for: validationIssues, pieces: pieces)
    }

    var fen: String {
        FENBuilder.fen(pieces: pieces, sideToMove: sideToMove, castling: castling)
    }
}
