import ChessKit

/// Exactly one king a side, where the piece model's scores allow it.
///
/// Every position has exactly two kings, and a king read as a queen (or a
/// third "king") is what a piece model gets wrong on a set it has not seen,
/// from above most of all. Port of `one_king_a_side` in
/// `Training/chess_pipeline.py`: keep in sync. The cases in
/// `Fixtures/king_cases.json` are what the Python makes of the same input.
enum OneKingASide {

    /// A piece is made the king of a side that has none only if the model
    /// gave "king" at least this score for it (`KING_FLOOR`).
    static let kingFloor = 0.02

    /// The order in which equal scores are settled (`KINDS`).
    static let kinds: [Piece.Kind] = [.king, .queen, .rook, .bishop, .knight, .pawn]

    /// The detections with one king on each side, from the score the model
    /// gave each piece for every kind (`PieceDetection.kindScores`).
    ///
    /// Nothing is done to a board that has exactly two kings, also when both
    /// are on one side: that is a wrong side (a set whose "white" pieces the
    /// model takes for black), not a wrong kind, and moving kings about then
    /// makes it worse. With fewer than two kings, a side that has pieces and
    /// no king gets one: its piece with the highest "king" score, if that is
    /// at least `floor`. With more than two, a side with several kings keeps
    /// the one with the highest "king" score and each of the others becomes
    /// its best-scoring other kind (a queen when it has no other score).
    /// Equal scores: the earlier piece, the earlier kind in `kinds`.
    /// A piece keeps its side, confidence, box and scores. (What the app
    /// shows for a piece changed here is the model's score for the kind it
    /// became: `PieceDetection.shownConfidence`.)
    ///
    /// The rule takes every position to have two kings. A position set up
    /// without one (a few in the test sets are) can only lose by it.
    ///
    /// `queenForKing` (`queen_for_king`; off unless asked for): where a side
    /// without a king has no piece with a "king" score of `floor`, its queen
    /// is made the king, and of several queens the one the model is least
    /// sure of (equally sure: the earlier). A king the model is sure is a
    /// queen has no "king" score to go by; on a board set up without a king
    /// this turns a real queen into one. The queen crowned is shown with
    /// the model's score for "king", next to nothing, so the app marks it.
    static func apply(
        _ detections: [PieceDetection], floor: Double = kingFloor, queenForKing: Bool = false
    ) -> [PieceDetection] {
        var out = detections
        let total = out.filter { $0.piece == .king }.count
        if total == 2 {
            return out
        }
        for side in [Piece.Color.white, .black] {
            let indices = out.indices.filter { out[$0].color == side }
            guard let first = indices.first else { continue }
            let kings = indices.filter { out[$0].piece == .king }
            if total < 2 && kings.isEmpty {
                var best = first
                for index in indices.dropFirst()
                where score(of: .king, for: out[index]) > score(of: .king, for: out[best]) {
                    best = index
                }
                var crowned: Int? = best
                if score(of: .king, for: out[best]) < floor {
                    crowned = nil
                    if queenForKing {
                        for index in indices where out[index].piece == .queen {
                            // the lowest confidence; equal ones, the earlier
                            if crowned.map({ out[index].confidence < out[$0].confidence }) ?? true {
                                crowned = index
                            }
                        }
                    }
                }
                if let crowned {
                    out[crowned] = out[crowned].taken(for: .king)
                }
            } else if total > 2 && kings.count > 1 {
                var keep = kings[0]
                for index in kings.dropFirst()
                where score(of: .king, for: out[index]) > score(of: .king, for: out[keep]) {
                    keep = index
                }
                for index in kings where index != keep {
                    var kind = Piece.Kind.queen
                    for other in kinds.dropFirst(2)
                    where score(of: other, for: out[index]) > score(of: kind, for: out[index]) {
                        kind = other
                    }
                    out[index] = out[index].taken(for: kind)
                }
            }
        }
        return out
    }

    /// The model's score for a kind; none, and one that is not finite (not a
    /// number, or infinite), count as 0. The recognizer hands in neither.
    static func score(of kind: Piece.Kind, for detection: PieceDetection) -> Double {
        guard let score = detection.kindScores[kind], score.isFinite else { return 0 }
        return score
    }
}
