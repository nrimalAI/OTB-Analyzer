import ChessKit
import SwiftUI

/// An arrow drawn on the board (e.g. the engine's best move).
public struct BoardArrow: Equatable {
    public let from: Square
    public let to: Square
    public let color: Color

    public init(from: Square, to: Square, color: Color = .blue) {
        self.from = from
        self.to = to
        self.color = color
    }
}

/// Renders a chess position. Squares can be tinted (confidence / validation),
/// selected, tapped, and overlaid with arrows. Pure display — all state lives
/// with the caller.
public struct BoardView: View {

    let pieces: [Square: Piece]
    let orientation: BoardDisplayOrientation
    let selected: Square?
    let squareTint: (Square) -> Color?
    let arrows: [BoardArrow]
    let onTap: ((Square) -> Void)?

    public init(
        pieces: [Square: Piece],
        orientation: BoardDisplayOrientation = .whiteBottom,
        selected: Square? = nil,
        squareTint: @escaping (Square) -> Color? = { _ in nil },
        arrows: [BoardArrow] = [],
        onTap: ((Square) -> Void)? = nil
    ) {
        self.pieces = pieces
        self.orientation = orientation
        self.selected = selected
        self.squareTint = squareTint
        self.arrows = arrows
        self.onTap = onTap
    }

    private let lightColor = Color(red: 0.93, green: 0.89, blue: 0.79)
    private let darkColor = Color(red: 0.55, green: 0.44, blue: 0.30)

    public var body: some View {
        GeometryReader { proxy in
            let boardSize = min(proxy.size.width, proxy.size.height)
            let cell = boardSize / 8

            ZStack(alignment: .topLeading) {
                // Squares + pieces
                VStack(spacing: 0) {
                    ForEach(0..<8, id: \.self) { row in
                        HStack(spacing: 0) {
                            ForEach(0..<8, id: \.self) { column in
                                squareView(column: column, row: row, cell: cell)
                            }
                        }
                    }
                }

                // Arrows on top
                ForEach(Array(arrows.enumerated()), id: \.offset) { _, arrow in
                    ArrowShape(
                        from: BoardGeometry.center(
                            of: arrow.from, boardSize: boardSize, orientation: orientation),
                        to: BoardGeometry.center(
                            of: arrow.to, boardSize: boardSize, orientation: orientation),
                        thickness: cell * 0.18
                    )
                    .fill(arrow.color.opacity(0.7))
                    .allowsHitTesting(false)
                }
            }
            .frame(width: boardSize, height: boardSize)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    @ViewBuilder
    private func squareView(column: Int, row: Int, cell: CGFloat) -> some View {
        if let square = BoardGeometry.square(column: column, row: row, orientation: orientation) {
            ZStack {
                Rectangle()
                    .fill(BoardGeometry.isLight(square) ? lightColor : darkColor)
                if let tint = squareTint(square) {
                    Rectangle().fill(tint)
                }
                if selected == square {
                    Rectangle()
                        .strokeBorder(Color.accentColor, lineWidth: max(2, cell * 0.07))
                }
                if let piece = pieces[square] {
                    PieceView(piece: piece, size: cell)
                }
                coordinateLabels(square: square, column: column, row: row, cell: cell)
            }
            .frame(width: cell, height: cell)
            .contentShape(Rectangle())
            .onTapGesture { onTap?(square) }
        }
    }

    @ViewBuilder
    private func coordinateLabels(square: Square, column: Int, row: Int, cell: CGFloat) -> some View {
        let labelColor = BoardGeometry.isLight(square) ? darkColor : lightColor
        ZStack(alignment: .topLeading) {
            Color.clear
            if column == 0 {
                Text("\(square.rank.value)")
                    .font(.system(size: cell * 0.2, weight: .semibold))
                    .foregroundStyle(labelColor)
                    .padding(1.5)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if row == 7 {
                Text(square.file.rawValue)
                    .font(.system(size: cell * 0.2, weight: .semibold))
                    .foregroundStyle(labelColor)
                    .padding(1.5)
            }
        }
    }
}

/// A tapered arrow between two points.
struct ArrowShape: Shape {
    let from: CGPoint
    let to: CGPoint
    let thickness: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let dx = to.x - from.x
        let dy = to.y - from.y
        let length = sqrt(dx * dx + dy * dy)
        guard length > 1 else { return path }

        let dirX = dx / length
        let dirY = dy / length
        let perpX = -dirY
        let perpY = dirX

        let headLength = min(thickness * 2.2, length * 0.4)
        let headWidth = thickness * 2.0
        let shaftEndX = to.x - dirX * headLength
        let shaftEndY = to.y - dirY * headLength
        let half = thickness / 2

        path.move(to: CGPoint(x: from.x + perpX * half, y: from.y + perpY * half))
        path.addLine(to: CGPoint(x: shaftEndX + perpX * half, y: shaftEndY + perpY * half))
        path.addLine(to: CGPoint(x: shaftEndX + perpX * headWidth / 2, y: shaftEndY + perpY * headWidth / 2))
        path.addLine(to: to)
        path.addLine(to: CGPoint(x: shaftEndX - perpX * headWidth / 2, y: shaftEndY - perpY * headWidth / 2))
        path.addLine(to: CGPoint(x: shaftEndX - perpX * half, y: shaftEndY - perpY * half))
        path.addLine(to: CGPoint(x: from.x - perpX * half, y: from.y - perpY * half))
        path.closeSubpath()
        return path
    }
}
