import ChessKit
import ChessVision
import Testing

@testable import ChessAnalyzer

@Suite struct PositionTextTests {

    @Test func theDiagramIsFromWhitesSide() throws {
        let pieces = try #require(FENBuilder.pieces(fromPlacement: "4k3/8/8/8/8/8/4P3/4K3"))
        #expect(PositionText.diagram(pieces) == """
            8  . . . . k . . .
            7  . . . . . . . .
            6  . . . . . . . .
            5  . . . . . . . .
            4  . . . . . . . .
            3  . . . . . . . .
            2  . . . . P . . .
            1  . . . . K . . .
               a b c d e f g h
            """)
    }

    @Test func lettersAreFENLetters() {
        #expect(PositionText.letter(Piece(.knight, color: .white, square: .a1)) == "N")
        #expect(PositionText.letter(Piece(.pawn, color: .black, square: .a7)) == "p")
        #expect(PositionText.letter(Piece(.king, color: .black, square: .e8)) == "k")
    }

    @Test func movesAreNumberedAsOnScreen() {
        #expect(PositionText.numbered(["e4", "e5", "Nf3"], sideToMove: .white) == "1. e4 e5 2. Nf3")
        #expect(PositionText.numbered(["Nf6", "c4", "e6"], sideToMove: .black) == "1… Nf6 2. c4 e6")
    }

    @Test func theExportCarriesEverything() throws {
        let fen = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"
        let pieces = try #require(FENBuilder.pieces(fromFEN: fen))
        let text = PositionText.export(
            pieces: pieces, fen: fen, sideToMove: .black,
            verdict: "White is winning", depth: 18,
            lines: [.init(evaluation: "+5.2", moves: "1… Kd7 2. Kd2")])
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        #expect(lines[0] == "Black to move")
        #expect(lines[1] == "FEN: " + fen)
        #expect(lines.contains("Stockfish, depth 18: White is winning"))
        #expect(lines.last == "+5.2  1… Kd7 2. Kd2")
        // Without analysis: the position only.
        let bare = PositionText.export(pieces: pieces, fen: fen, sideToMove: .black)
        #expect(!bare.contains("Stockfish"))
        #expect(bare.hasSuffix("   a b c d e f g h"))
    }
}
