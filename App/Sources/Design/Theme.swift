import ChessBoardUI
import SwiftUI

/// The "scoresheet" look: warm paper, near-black ink, one brick-red accent.
/// Serif for titles and verdicts, sans for controls, mono for notation and
/// small caps labels.
enum Theme {
    static let paper = Color(red: 0xF4 / 255, green: 0xF0 / 255, blue: 0xE6 / 255)
    static let ink = BoardPalette.ink
    static let muted = Color(red: 0x6E / 255, green: 0x66 / 255, blue: 0x5B / 255)
    /// Lighter surface for fields (same as the board's light squares).
    static let field = BoardPalette.lightSquare
    static let accent = BoardPalette.accent
    static let hairline = BoardPalette.ink.opacity(0.16)
    static let fieldBorder = BoardPalette.ink.opacity(0.3)
    /// Background of the scan screen.
    static let darkroom = Color(red: 0x0D / 255, green: 0x0D / 255, blue: 0x0C / 255)

    /// Weight of the rules that open a list and of control outlines.
    static let ruleWidth: CGFloat = 1.5
}

// MARK: - Type

/// A fixed-size design font that still follows Dynamic Type.
private struct ThemeFont: ViewModifier {
    @ScaledMetric private var size: CGFloat
    private let weight: Font.Weight
    private let design: Font.Design

    init(size: CGFloat, weight: Font.Weight, design: Font.Design) {
        _size = ScaledMetric(wrappedValue: size, relativeTo: .body)
        self.weight = weight
        self.design = design
    }

    func body(content: Content) -> some View {
        content.font(.system(size: size, weight: weight, design: design))
    }
}

extension View {
    /// Titles and verdicts.
    func serif(_ size: CGFloat, weight: Font.Weight = .medium) -> some View {
        modifier(ThemeFont(size: size, weight: weight, design: .serif))
    }

    /// Controls and body copy.
    func sans(_ size: CGFloat, weight: Font.Weight = .regular) -> some View {
        modifier(ThemeFont(size: size, weight: weight, design: .default))
    }

    /// Notation, evaluations, and labels.
    func mono(_ size: CGFloat, weight: Font.Weight = .regular) -> some View {
        modifier(ThemeFont(size: size, weight: weight, design: .monospaced))
    }

    /// Small uppercase mono label above a control or section.
    func eyebrow(tracking: CGFloat = 0.9) -> some View {
        self
            .mono(11, weight: .medium)
            .textCase(.uppercase)
            .tracking(tracking)
            .foregroundStyle(Theme.muted)
    }
}
