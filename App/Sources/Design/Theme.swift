import ChessBoardUI
import CoreText
import SwiftUI
import UIKit

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

/// The three bundled typefaces (all SIL Open Font License; the license texts
/// ship in App/Resources/Fonts and are credited on the About screen).
enum Typeface: Hashable {
    /// Newsreader — titles and verdicts.
    case serif
    /// IBM Plex Sans — controls and body copy.
    case sans
    /// IBM Plex Mono — notation, evaluations, and labels.
    case mono
}

extension Typeface {
    /// OpenType variation axis tags.
    private static let weightAxis = 0x7767_6874  // 'wght'
    private static let opticalSizeAxis = 0x6F70_737A  // 'opsz'

    private struct Key: Hashable {
        let face: Typeface
        let size: CGFloat
        let weight: Int
    }
    private static var cache: [Key: UIFont] = [:]

    /// The bundled font at a point size, or the closest system font if it
    /// isn't registered (e.g. the file is missing from the bundle).
    func uiFont(size: CGFloat, weight: Font.Weight) -> UIFont {
        let key = Key(face: self, size: size, weight: Self.value(of: weight))
        if let cached = Self.cache[key] { return cached }
        let font = resolve(size: size, weight: weight)
        Self.cache[key] = font
        return font
    }

    private func resolve(size: CGFloat, weight: Font.Weight) -> UIFont {
        let value = Self.value(of: weight)
        switch self {
        case .serif:
            // Variable font: weight plus an optical size that follows the
            // point size, as browsers do, so large titles get finer detail.
            let opticalSize = min(max(size, 6), 72)
            if let font = Self.variableFont(
                family: "Newsreader", size: size,
                axes: [Self.weightAxis: CGFloat(value), Self.opticalSizeAxis: opticalSize]) {
                return font
            }
            return Self.systemFont(size: size, weight: weight, design: .serif)
        case .sans:
            if let font = Self.variableFont(
                family: "IBM Plex Sans", size: size, axes: [Self.weightAxis: CGFloat(value)]) {
                return font
            }
            return Self.systemFont(size: size, weight: weight, design: .default)
        case .mono:
            let name = value >= 600 ? "IBMPlexMono-SemiBold" : value >= 500 ? "IBMPlexMono-Medium" : "IBMPlexMono-Regular"
            return UIFont(name: name, size: size)
                ?? Self.systemFont(size: size, weight: weight, design: .monospaced)
        }
    }

    private static func variableFont(family: String, size: CGFloat, axes: [Int: CGFloat]) -> UIFont? {
        let descriptor = UIFontDescriptor(fontAttributes: [
            .family: family,
            UIFontDescriptor.AttributeName(rawValue: kCTFontVariationAttribute as String): axes,
        ])
        let font = UIFont(descriptor: descriptor, size: size)
        // An unknown family silently resolves to a system font.
        return font.familyName == family ? font : nil
    }

    private static func systemFont(size: CGFloat, weight: Font.Weight, design: UIFontDescriptor.SystemDesign) -> UIFont {
        let base = UIFont.systemFont(ofSize: size, weight: uiWeight(of: weight))
        guard let descriptor = base.fontDescriptor.withDesign(design) else { return base }
        return UIFont(descriptor: descriptor, size: size)
    }

    private static func value(of weight: Font.Weight) -> Int {
        switch weight {
        case .medium: return 500
        case .semibold: return 600
        case .bold: return 700
        default: return 400
        }
    }

    private static func uiWeight(of weight: Font.Weight) -> UIFont.Weight {
        switch weight {
        case .medium: return .medium
        case .semibold: return .semibold
        case .bold: return .bold
        default: return .regular
        }
    }
}

/// A design-size font that still follows Dynamic Type.
private struct ThemeFont: ViewModifier {
    @ScaledMetric private var size: CGFloat
    private let weight: Font.Weight
    private let face: Typeface

    init(size: CGFloat, weight: Font.Weight, face: Typeface) {
        _size = ScaledMetric(wrappedValue: size, relativeTo: .body)
        self.weight = weight
        self.face = face
    }

    func body(content: Content) -> some View {
        content.font(Font(face.uiFont(size: size, weight: weight)))
    }
}

extension View {
    /// Titles and verdicts.
    func serif(_ size: CGFloat, weight: Font.Weight = .medium) -> some View {
        modifier(ThemeFont(size: size, weight: weight, face: .serif))
    }

    /// Controls and body copy.
    func sans(_ size: CGFloat, weight: Font.Weight = .regular) -> some View {
        modifier(ThemeFont(size: size, weight: weight, face: .sans))
    }

    /// Notation, evaluations, and labels.
    func mono(_ size: CGFloat, weight: Font.Weight = .regular) -> some View {
        modifier(ThemeFont(size: size, weight: weight, face: .mono))
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

extension Theme {
    /// Hands the app's typefaces to the board package, which otherwise
    /// draws its few labels in system fonts.
    static func applyBoardTypography() {
        BoardTypography.coordinate = Font(Typeface.mono.uiFont(size: 9, weight: .medium))
        BoardTypography.label = Font(Typeface.sans.uiFont(size: 13, weight: .medium))
    }
}
