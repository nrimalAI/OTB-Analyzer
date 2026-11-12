import ChessKit
import SwiftUI

/// Which color sits at the bottom of the rendered board.
public enum BoardDisplayOrientation: Sendable, Equatable {
    case whiteBottom
    case blackBottom

    public mutating func flip() {
        self = self == .whiteBottom ? .blackBottom : .whiteBottom
    }
}

/// Pure layout math shared by the board, arrows, and hit testing — kept free
/// of views so it's unit-testable.
public enum BoardGeometry {

    /// `Square` from 1-based file/rank (ChessKit's `(File, Rank)` init is
    /// internal; rawValue layout is `(rank-1)*8 + (file-1)`).
    public static func square(file: Int, rank: Int) -> Square? {
        guard (1...8).contains(file), (1...8).contains(rank) else { return nil }
        return Square(rawValue: (rank - 1) * 8 + (file - 1))
    }

    /// Grid column (0–7, left to right) for a square under an orientation.
    public static func column(of square: Square, orientation: BoardDisplayOrientation) -> Int {
        let f = square.file.number - 1
        return orientation == .whiteBottom ? f : 7 - f
    }

    /// Grid row (0–7, top to bottom) for a square under an orientation.
    public static func row(of square: Square, orientation: BoardDisplayOrientation) -> Int {
        let r = square.rank.value - 1
        return orientation == .whiteBottom ? 7 - r : r
    }

    /// The square at a grid (column, row) under an orientation.
    public static func square(
        column: Int, row: Int, orientation: BoardDisplayOrientation
    ) -> Square? {
        let file = orientation == .whiteBottom ? column + 1 : 8 - column
        let rank = orientation == .whiteBottom ? 8 - row : row + 1
        return square(file: file, rank: rank)
    }

    /// Center point of a square inside a board of the given side length.
    public static func center(
        of square: Square, boardSize: CGFloat, orientation: BoardDisplayOrientation
    ) -> CGPoint {
        let cell = boardSize / 8
        let col = CGFloat(column(of: square, orientation: orientation))
        let row = CGFloat(row(of: square, orientation: orientation))
        return CGPoint(x: (col + 0.5) * cell, y: (row + 0.5) * cell)
    }

    /// Whether a square is rendered as a light square.
    public static func isLight(_ square: Square) -> Bool {
        (square.file.number + square.rank.value) % 2 == 1
    }
}

// MARK: - Rotation

extension BoardGeometry {
    /// The diametrically opposite square (a1 ↔ h8): where a piece lands when
    /// the board is turned 180°, e.g. after a photo taken from Black's side
    /// was read as if taken from White's.
    public static func rotated(_ square: Square) -> Square {
        Square(rawValue: 63 - square.rawValue) ?? square
    }
}

// MARK: - Arrows

/// The points of a move arrow: a round-capped shaft plus a triangular head.
public struct ArrowGeometry: Equatable {
    public let shaftStart: CGPoint
    public let shaftEnd: CGPoint
    public let tip: CGPoint
    public let headLeft: CGPoint
    public let headRight: CGPoint
    public let shaftWidth: CGFloat
}

extension BoardGeometry {
    /// Arrow between two square centers, in the proportions of the design:
    /// the shaft starts 0.22 of a cell out from the origin square's center and
    /// the head stops 0.08 of a cell short of the target's. Nil when the two
    /// points coincide.
    public static func arrow(from: CGPoint, to: CGPoint, cell: CGFloat) -> ArrowGeometry? {
        let dx = to.x - from.x
        let dy = to.y - from.y
        let length = (dx * dx + dy * dy).squareRoot()
        guard length > cell * 0.5 else { return nil }
        let ux = dx / length
        let uy = dy / length

        let tip = CGPoint(x: to.x - ux * cell * 0.08, y: to.y - uy * cell * 0.08)
        let base = CGPoint(x: tip.x - ux * cell * 0.42, y: tip.y - uy * cell * 0.42)
        let halfHead = cell * 0.24
        return ArrowGeometry(
            shaftStart: CGPoint(x: from.x + ux * cell * 0.22, y: from.y + uy * cell * 0.22),
            shaftEnd: base,
            tip: tip,
            headLeft: CGPoint(x: base.x - uy * halfHead, y: base.y + ux * halfHead),
            headRight: CGPoint(x: base.x + uy * halfHead, y: base.y - ux * halfHead),
            shaftWidth: cell * 0.17
        )
    }
}
