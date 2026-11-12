import SwiftUI

/// Fonts for the few pieces of text the board package draws. The defaults
/// are system fonts; a host app can swap in its own once at launch.
public enum BoardTypography {
    /// Rank/file labels in the board's corners.
    public static var coordinate = Font.system(size: 9, weight: .medium, design: .monospaced)
    /// Text keys such as the tray's "Empty".
    public static var label = Font.system(size: 13, weight: .medium)
}

/// The "scoresheet" board colours: a printed diagram on warm paper. Shared by
/// the board, the piece tray, and the app's own theme so they never drift.
public enum BoardPalette {
    /// Near-black ink used for pieces, borders, and text.
    public static let ink = Color(red: 0x26 / 255, green: 0x22 / 255, blue: 0x1D / 255)
    /// Light squares, and the fill of white pieces.
    public static let lightSquare = Color(red: 0xFB / 255, green: 0xF8 / 255, blue: 0xF1 / 255)
    public static let darkSquare = Color(red: 0xD6 / 255, green: 0xCB / 255, blue: 0xB6 / 255)
    /// Rank/file labels in the board's corners.
    public static let coordinate = Color(red: 0x4A / 255, green: 0x44 / 255, blue: 0x3C / 255)
    /// Brick red (oklch 0.5 0.14 28): best-move arrow, squares to double-check.
    public static let accent = Color(red: 164 / 255, green: 60 / 255, blue: 51 / 255)
}
