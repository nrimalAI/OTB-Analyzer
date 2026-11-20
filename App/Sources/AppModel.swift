import ChessBoardUI
import ChessKit
import ChessVision
import SwiftUI

enum AppRoute: Hashable {
    /// Camera / photo picker.
    case scan
    /// The board workspace: Position and Analysis tabs over one board.
    case board
}

enum BoardTab: Hashable {
    case position
    case analysis
}

/// App-wide state for the scan → check position → analyze flow.
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

    // Position on the board
    @Published var pieces: [Square: Piece] = [:]
    @Published var confidence: [Square: Float] = [:]
    /// Setting it, from the control or a typed FEN, answers the question
    /// of who is to move.
    @Published var sideToMove: Piece.Color = .white {
        didSet { sideToMoveKnown = true }
    }
    /// False after a scan until someone says who is to move: a photo does
    /// not show it. `sideToMove` is then White only as a placeholder.
    @Published private(set) var sideToMoveKnown = true
    @Published var castling: CastlingRights = .none
    @Published var orientation: BoardDisplayOrientation = .whiteBottom

    // Workspace
    @Published var tab: BoardTab = .position
    /// Whether the position came from a photo (as opposed to being set up by
    /// hand or reopened from the saved list).
    @Published private(set) var fromScan = false
    /// Which side of the board the photo was taken from.
    @Published private(set) var photoSide: Piece.Color = .white
    @Published var depth = 18

    /// Positions the engine has analyzed, newest first.
    let history = HistoryStore()

    /// Detections below this are flagged for the user to double-check.
    static let lowConfidenceThreshold: Float = 0.85

    // MARK: - Flow

    /// Reads the board in a photo and moves on to checking the position.
    /// `minimumDuration` keeps the "reading" state on screen long enough to
    /// register when recognition itself is near-instant.
    func startFromPhoto(_ image: UIImage, minimumDuration: Duration = .zero) async {
        guard !isRecognizing else { return }
        guard let cgImage = image.normalizedUprightCGImage() else {
            recognitionFailed = true
            return
        }
        isRecognizing = true
        defer { isRecognizing = false }
        let started = ContinuousClock.now
        do {
            let result = try await recognizer.recognize(cgImage)
            let remaining = minimumDuration - (ContinuousClock.now - started)
            if remaining > .zero { try? await Task.sleep(for: remaining) }
            guard isStillScanning else { return }
            load(result: result)
            path.append(.board)
        } catch {
            if isStillScanning { recognitionFailed = true }
        }
    }

    /// False once the user has cancelled or left the scan screen; a read
    /// that finishes after that must not navigate or raise an alert.
    private var isStillScanning: Bool {
        !Task.isCancelled && path.last == .scan
    }

    func startFromScratch(startingPosition: Bool) {
        resetBoard(toStartingPosition: startingPosition)
        sideToMove = .white
        orientation = .whiteBottom
        fromScan = false
        tab = .position
        path.append(.board)
    }

    /// Reopens a saved position straight on its analysis.
    func open(_ saved: SavedPosition) {
        guard applyFEN(saved.fen) else { return }
        sideToMoveKnown = true  // A saved position was analyzed for one side.
        orientation = .whiteBottom
        fromScan = false
        tab = .analysis
        path = [.board]
    }

    /// Takes a recognizer's reading as the position to check. Internal for
    /// tests and the demo scan.
    func load(result: RecognitionResult) {
        pieces = result.pieces.mapValues(\.piece)
        confidence = result.pieces.mapValues(\.confidence)
        // A photo does not show who is to move. Only a king in check says
        // it; otherwise the person checking the board is asked.
        if let forced = Self.forcedSideToMove(pieces) {
            sideToMove = forced
        } else {
            sideToMove = .white
            sideToMoveKnown = false
        }
        castling = .inferred(from: pieces)
        orientation = result.orientationGuess == .whiteBottom ? .whiteBottom : .blackBottom
        photoSide = result.orientationGuess == .whiteBottom ? .white : .black
        fromScan = true
        tab = .position
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

    /// Starting position, or an empty board.
    func resetBoard(toStartingPosition: Bool) {
        pieces = toStartingPosition
            ? FENBuilder.pieces(fromFEN: Position.standard.fen) ?? [:]
            : [:]
        confidence = [:]
        castling = .inferred(from: pieces)
    }

    /// Corrects which side the photo was taken from. The recognizer can't
    /// always tell, and getting it wrong reads the whole board turned 180°,
    /// so switching sides turns the position back.
    func setPhotoSide(_ side: Piece.Color) {
        guard side != photoSide else { return }
        photoSide = side

        var turned: [Square: Piece] = [:]
        for (square, piece) in pieces {
            let target = BoardGeometry.rotated(square)
            var moved = piece
            moved.square = target
            turned[target] = moved
        }
        pieces = turned
        confidence = Dictionary(
            uniqueKeysWithValues: confidence.map { (BoardGeometry.rotated($0.key), $0.value) })
        castling = .inferred(from: pieces)
    }

    /// Turns the position a quarter clockwise (seen from White's side), for
    /// a photo whose board was read a quarter turn off: the pieces then
    /// stand in files where they should stand in ranks. Four turns are none.
    func turnQuarter() {
        var turned: [Square: Piece] = [:]
        for (square, piece) in pieces {
            let target = BoardGeometry.quarterTurned(square)
            var moved = piece
            moved.square = target
            turned[target] = moved
        }
        pieces = turned
        confidence = Dictionary(
            uniqueKeysWithValues: confidence.map { (BoardGeometry.quarterTurned($0.key), $0.value) })
        castling = .inferred(from: pieces)
    }

    /// Loads a typed or saved FEN. Only the placement must be well formed;
    /// side to move and castling are taken from the FEN when present.
    /// Returns false (changing nothing) if the placement doesn't parse.
    @discardableResult
    func applyFEN(_ text: String) -> Bool {
        let fields = text.split(separator: " ").map(String.init)
        guard let placement = fields.first,
              let parsed = FENBuilder.pieces(fromPlacement: placement)
        else { return false }

        pieces = parsed
        confidence = [:]
        // The FEN field shows the placeholder "w" while nobody has said who
        // is to move; editing the placement keeps that question open.
        let typedSide: Piece.Color = fields.count > 1 && fields[1] == "b" ? .black : .white
        if sideToMoveKnown || typedSide != sideToMove { sideToMove = typedSide }
        // Typed rights can only take away from what the placement allows: a
        // right without its king and rook at home is not just meaningless,
        // Stockfish reads past the board looking for the rook.
        let possible = CastlingRights.inferred(from: parsed)
        if fields.count > 2, let typed = Self.castling(fromFENField: fields[2]) {
            castling = CastlingRights(
                whiteKingside: typed.whiteKingside && possible.whiteKingside,
                whiteQueenside: typed.whiteQueenside && possible.whiteQueenside,
                blackKingside: typed.blackKingside && possible.blackKingside,
                blackQueenside: typed.blackQueenside && possible.blackQueenside
            )
        } else {
            castling = possible
        }
        return true
    }

    private static func castling(fromFENField field: String) -> CastlingRights? {
        if field == "-" { return CastlingRights.none }
        guard !field.isEmpty, field.allSatisfy({ "KQkq".contains($0) }) else { return nil }
        return CastlingRights(
            whiteKingside: field.contains("K"),
            whiteQueenside: field.contains("Q"),
            blackKingside: field.contains("k"),
            blackQueenside: field.contains("q")
        )
    }

    // MARK: - Validation / FEN

    var validationIssues: [ValidationIssue] {
        PositionValidator.validate(pieces: pieces, sideToMove: sideToMove)
    }

    var problemSquares: Set<Square> {
        PositionValidator.highlightSquares(for: validationIssues, pieces: pieces)
    }

    /// Squares the recognizer wasn't sure about and the user hasn't confirmed.
    var uncertainSquares: Set<Square> {
        Set(confidence.filter { $0.value < Self.lowConfidenceThreshold }.keys)
    }

    /// The validation issues in words, each with the step most likely to
    /// fix it.
    var problems: [PositionProblem] {
        PositionProblems.explain(
            validationIssues, pieces: pieces, sideToMove: sideToMove, confidence: confidence)
    }

    /// Squares worth a second look: unsure readings and squares a problem
    /// points at.
    func isDoubtful(_ square: Square) -> Bool {
        uncertainSquares.contains(square) || problemSquares.contains(square)
    }

    /// Likely replacements for what stands on `square`, the likeliest first.
    func suggestions(for square: Square) -> [PieceChoice] {
        CorrectionSuggestions.alternatives(for: square, in: pieces)
    }

    var canAnalyze: Bool { validationIssues.isEmpty }

    /// The side that must be to move because its king is in check, where
    /// exactly one king is; nil otherwise.
    static func forcedSideToMove(_ pieces: [Square: Piece]) -> Piece.Color? {
        let blackInCheck = PositionValidator.validate(pieces: pieces, sideToMove: .white)
            .contains(.sideNotToMoveInCheck)
        let whiteInCheck = PositionValidator.validate(pieces: pieces, sideToMove: .black)
            .contains(.sideNotToMoveInCheck)
        switch (whiteInCheck, blackInCheck) {
        case (true, false): return .white
        case (false, true): return .black
        default: return nil
        }
    }

    var fen: String {
        FENBuilder.fen(pieces: pieces, sideToMove: sideToMove, castling: castling)
    }
}
