import ChessKit
import SwiftUI

/// An arrow drawn on the board (e.g. the engine's best move).
public struct BoardArrow: Equatable {
    public let from: Square
    public let to: Square
    public let color: Color

    public init(from: Square, to: Square, color: Color = BoardPalette.accent) {
        self.from = from
        self.to = to
        self.color = color
    }
}

/// Renders a chess position as a printed diagram: paper squares, ink border,
/// outline/solid pieces. Squares can be selected, flagged (to double-check),
/// highlighted (last move), tapped, and overlaid with arrows. Pure display —
/// all state lives with the caller.
public struct BoardView: View {

    let pieces: [Square: Piece]
    let orientation: BoardDisplayOrientation
    let selected: Square?
    let flagged: Set<Square>
    let highlighted: Set<Square>
    let arrows: [BoardArrow]
    let showsCoordinates: Bool
    let borderWidth: CGFloat
    let onTap: ((Square) -> Void)?

    public init(
        pieces: [Square: Piece],
        orientation: BoardDisplayOrientation = .whiteBottom,
        selected: Square? = nil,
        flagged: Set<Square> = [],
        highlighted: Set<Square> = [],
        arrows: [BoardArrow] = [],
        showsCoordinates: Bool = true,
        borderWidth: CGFloat = 1.5,
        onTap: ((Square) -> Void)? = nil
    ) {
        self.pieces = pieces
        self.orientation = orientation
        self.selected = selected
        self.flagged = flagged
        self.highlighted = highlighted
        self.arrows = arrows
        self.showsCoordinates = showsCoordinates
        self.borderWidth = borderWidth
        self.onTap = onTap
    }

    public var body: some View {
        GeometryReader { proxy in
            let boardSize = min(proxy.size.width, proxy.size.height)
            let cell = boardSize / 8

            ZStack(alignment: .topLeading) {
                Checkerboard()
                    .fill(BoardPalette.darkSquare)
                    .background(BoardPalette.lightSquare)

                VStack(spacing: 0) {
                    ForEach(0..<8, id: \.self) { row in
                        HStack(spacing: 0) {
                            ForEach(0..<8, id: \.self) { column in
                                squareView(column: column, row: row, cell: cell)
                            }
                        }
                    }
                }

                ForEach(Array(arrows.enumerated()), id: \.offset) { _, arrow in
                    arrowView(arrow, boardSize: boardSize, cell: cell)
                }

                Rectangle()
                    .strokeBorder(BoardPalette.ink, lineWidth: borderWidth)
                    .allowsHitTesting(false)
            }
            .frame(width: boardSize, height: boardSize)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    // MARK: - Squares

    @ViewBuilder
    private func squareView(column: Int, row: Int, cell: CGFloat) -> some View {
        if let square = BoardGeometry.square(column: column, row: row, orientation: orientation) {
            let isFlagged = flagged.contains(square)
            let piece = pieces[square]
            ZStack {
                if isFlagged {
                    Rectangle().fill(BoardPalette.accent.opacity(0.16))
                } else if highlighted.contains(square) {
                    Rectangle().fill(BoardPalette.accent.opacity(0.2))
                }
                if let piece {
                    PieceView(piece: piece, size: cell)
                }
                if showsCoordinates {
                    coordinateLabels(square: square, column: column, row: row)
                }
                if selected == square {
                    Rectangle().strokeBorder(BoardPalette.ink, lineWidth: 3)
                } else if isFlagged {
                    Rectangle().strokeBorder(BoardPalette.accent, lineWidth: 2.5)
                }
            }
            .frame(width: cell, height: cell)
            .contentShape(Rectangle())
            .onTapGesture { onTap?(square) }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(accessibilityLabel(square: square, piece: piece, isFlagged: isFlagged))
            .accessibilityAddTraits(accessibilityTraits(for: square))
        }
    }

    private func accessibilityLabel(square: Square, piece: Piece?, isFlagged: Bool) -> String {
        var label = "\(square.notation), \(piece.map(PieceGlyph.name(for:)) ?? "empty")"
        if isFlagged { label += ", double-check" }
        return label
    }

    private func accessibilityTraits(for square: Square) -> AccessibilityTraits {
        var traits: AccessibilityTraits = []
        if onTap != nil { traits.formUnion(.isButton) }
        if selected == square { traits.formUnion(.isSelected) }
        return traits
    }

    /// Rank numbers down the left edge (top-left of the square), file letters
    /// along the bottom (bottom-right of the square).
    private func coordinateLabels(square: Square, column: Int, row: Int) -> some View {
        ZStack {
            if column == 0 {
                Text("\(square.rank.value)")
                    .padding(.top, 2)
                    .padding(.leading, 3)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
            if row == 7 {
                Text(square.file.rawValue)
                    .padding(.bottom, 2)
                    .padding(.trailing, 3)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
            }
        }
        .font(.system(size: 9, weight: .medium, design: .monospaced))
        .foregroundStyle(BoardPalette.coordinate)
    }

    // MARK: - Arrows

    @ViewBuilder
    private func arrowView(_ arrow: BoardArrow, boardSize: CGFloat, cell: CGFloat) -> some View {
        let from = BoardGeometry.center(of: arrow.from, boardSize: boardSize, orientation: orientation)
        let to = BoardGeometry.center(of: arrow.to, boardSize: boardSize, orientation: orientation)
        if let geometry = BoardGeometry.arrow(from: from, to: to, cell: cell) {
            ZStack {
                Path { path in
                    path.move(to: geometry.shaftStart)
                    path.addLine(to: geometry.shaftEnd)
                }
                .stroke(arrow.color, style: StrokeStyle(lineWidth: geometry.shaftWidth, lineCap: .round))
                Path { path in
                    path.move(to: geometry.tip)
                    path.addLine(to: geometry.headLeft)
                    path.addLine(to: geometry.headRight)
                    path.closeSubpath()
                }
                .fill(arrow.color)
            }
            // Flatten first so the shaft/head overlap doesn't show through.
            .compositingGroup()
            .opacity(0.9)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
        }
    }
}

/// The 32 dark squares of an 8×8 board. The top-left square is light in
/// either orientation (a8 from White's side, h1 from Black's).
private struct Checkerboard: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let cell = min(rect.width, rect.height) / 8
        for row in 0..<8 {
            for column in 0..<8 where (row + column) % 2 == 1 {
                path.addRect(CGRect(
                    x: rect.minX + CGFloat(column) * cell,
                    y: rect.minY + CGFloat(row) * cell,
                    width: cell, height: cell))
            }
        }
        return path
    }
}
