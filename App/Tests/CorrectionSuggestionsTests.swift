import ChessKit
import ChessVision
import Testing

@testable import ChessAnalyzer

@Suite struct CorrectionSuggestionsTests {

    private func board(_ placement: String) throws -> [Square: Piece] {
        try #require(FENBuilder.pieces(fromPlacement: placement))
    }

    @Test func aSideWithoutAKingIsOfferedItsKingFirst() throws {
        // White's king on e1 read as a queen: two white queens, no white king.
        let pieces = try board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQQBNR")
        let choices = CorrectionSuggestions.alternatives(for: .e1, in: pieces)
        #expect(choices.first == .piece(.king, .white))
    }

    @Test func aMissingKingIsNotOfferedForAPawn() throws {
        // White has no king; the pawn on d4 is most likely White's pawn, not its king.
        let pieces = try board("4k3/8/8/8/3p4/8/8/3Q4")
        #expect(CorrectionSuggestions.alternatives(for: .d4, in: pieces).first == .piece(.pawn, .white))
        #expect(!CorrectionSuggestions.alternatives(for: .d4, in: pieces).contains(.piece(.king, .white)))
        #expect(CorrectionSuggestions.alternatives(for: .d1, in: pieces).first == .piece(.king, .white))
    }

    @Test func aQueenIsOfferedTheKingAndOtherTallPieces() throws {
        let pieces = try board("4k3/8/8/8/3Q4/8/8/4K3")
        let choices = CorrectionSuggestions.alternatives(for: .d4, in: pieces)
        // White has a king, so no second one; the queen's usual confusions.
        #expect(!choices.contains(.piece(.king, .white)))
        #expect(choices.first == .piece(.queen, .black))
        #expect(choices.contains(.piece(.rook, .white)))
    }

    @Test func aPawnIsOfferedTheOtherSideFirst() throws {
        // On unseen boards the side of a pawn is what goes wrong most.
        let pieces = try board("4k3/8/8/8/4p3/8/8/4K3")
        let choices = CorrectionSuggestions.alternatives(for: .e4, in: pieces)
        #expect(choices.first == .piece(.pawn, .white))
        #expect(choices.contains(.empty))
    }

    @Test func neverOffersWhatIsAlreadyThere() throws {
        let pieces = try board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
        for square in pieces.keys {
            let current = PieceChoice(pieces[square])
            #expect(!CorrectionSuggestions.alternatives(for: square, in: pieces).contains(current))
        }
    }

    @Test func neverOffersAPawnOnTheFirstOrLastRank() throws {
        let pieces = try board("4k1b1/8/8/8/8/8/8/2B1K3")
        for square in [Square.c1, .g8] {
            let choices = CorrectionSuggestions.alternatives(for: square, in: pieces)
            #expect(!choices.contains { $0.kind == .pawn }, "\(square)")
            #expect(!choices.isEmpty)
        }
    }

    @Test func neverOffersANinthPawnOrASeventeenthPiece() throws {
        // Black's pawn on e5 must not become White's ninth pawn.
        let pieces = try board("4k3/8/8/4p3/8/8/PPPPPPPP/4K3")
        let choices = CorrectionSuggestions.alternatives(for: .e5, in: pieces)
        #expect(!choices.contains(.piece(.pawn, .white)))
        // Sixteen white pieces: a black knight cannot turn white.
        let full = try board("4k3/8/8/4n3/8/8/PPPPPPPP/RNBQKBNR")
        #expect(!CorrectionSuggestions.alternatives(for: .e5, in: full).contains { $0.color == .white && !$0.isEmpty })
    }

    @Test func anEmptySquareIsOfferedOnlyAMissingKing() throws {
        let complete = try board("4k3/8/8/8/8/8/8/4K3")
        #expect(CorrectionSuggestions.alternatives(for: .d4, in: complete).isEmpty)
        let noBlackKing = try board("8/8/8/8/8/8/8/4K3")
        #expect(CorrectionSuggestions.alternatives(for: .d4, in: noBlackKing) == [.piece(.king, .black)])
    }

    @Test func theSameBoardAlwaysGivesTheSameList() throws {
        let pieces = try board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R")
        for square in pieces.keys {
            let first = CorrectionSuggestions.alternatives(for: square, in: pieces)
            #expect(first.count == 3)
            #expect(CorrectionSuggestions.alternatives(for: square, in: pieces) == first)
        }
    }

    @Test func aChoiceIsHomedOnItsSquare() {
        #expect(PieceChoice.piece(.knight, .black).piece(on: .f6) == Piece(.knight, color: .black, square: .f6))
        #expect(PieceChoice.empty.piece(on: .f6) == nil)
        #expect(PieceChoice(nil) == .empty)
    }
}
