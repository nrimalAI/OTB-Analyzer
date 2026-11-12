import SwiftUI

// MARK: - Buttons

/// Solid ink button: the one primary action of a screen.
struct PrimaryButtonStyle: ButtonStyle {
    var height: CGFloat = 54
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .sans(17, weight: .semibold)
            .foregroundStyle(Theme.paper)
            .frame(maxWidth: .infinity)
            .frame(height: height)
            .background(Theme.ink, in: RoundedRectangle(cornerRadius: 10))
            .opacity(isEnabled ? (configuration.isPressed ? 0.8 : 1) : 0.35)
            .contentShape(Rectangle())
    }
}

/// Ink outline on paper: secondary actions and stepper keys.
struct OutlineButtonStyle: ButtonStyle {
    var cornerRadius: CGFloat = 8
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Theme.ink)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(
                configuration.isPressed ? Theme.ink.opacity(0.08) : .clear,
                in: RoundedRectangle(cornerRadius: cornerRadius)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .strokeBorder(Theme.ink, lineWidth: Theme.ruleWidth)
            )
            .opacity(isEnabled ? 1 : 0.3)
            .contentShape(Rectangle())
    }
}

/// Plain text button (top bars, action rows). Dims while pressed.
struct TextButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(isEnabled ? (configuration.isPressed ? 0.5 : 1) : 0.35)
            .contentShape(Rectangle())
    }
}

// MARK: - Segmented control

/// Two-or-more-way switch: ink outline, the chosen segment filled with ink.
struct InkSegmentedControl<Value: Hashable>: View {
    struct Segment: Identifiable {
        let value: Value
        let title: String
        /// Drawn faint and disabled (e.g. Analysis while the position is
        /// illegal).
        var isUnavailable = false
        var id: Value { value }
    }

    let segments: [Segment]
    let selection: Value
    var height: CGFloat = 44
    let onSelect: (Value) -> Void

    var body: some View {
        HStack(spacing: 0) {
            ForEach(segments) { segment in
                let isOn = segment.value == selection
                Button {
                    onSelect(segment.value)
                } label: {
                    Text(segment.title)
                        .sans(14, weight: .medium)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                        .padding(.horizontal, 4)
                        .foregroundStyle(foreground(isOn: isOn, isUnavailable: segment.isUnavailable))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(isOn ? Theme.ink : .clear)
                        .contentShape(Rectangle())
                }
                .buttonStyle(SegmentButtonStyle())
                .disabled(segment.isUnavailable)
                .accessibilityAddTraits(isOn ? .isSelected : [])
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Theme.ink, lineWidth: Theme.ruleWidth)
        )
    }

    private func foreground(isOn: Bool, isUnavailable: Bool) -> Color {
        if isOn { return Theme.paper }
        return isUnavailable ? Theme.ink.opacity(0.35) : Theme.ink
    }
}

/// Leaves the segment's label exactly as drawn; the control colours its own
/// selected and unavailable states.
private struct SegmentButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
    }
}

// MARK: - Rules

/// Horizontal rule. `.heavy` opens a list; `.hairline` separates rows.
struct Rule: View {
    enum Weight { case heavy, hairline }
    var weight: Weight = .hairline

    var body: some View {
        Rectangle()
            .fill(weight == .heavy ? Theme.ink : Theme.hairline)
            .frame(height: weight == .heavy ? Theme.ruleWidth : 1)
    }
}

// MARK: - Toast

/// Short confirmation pill near the bottom of the screen ("FEN copied").
struct Toast: View {
    let message: String

    var body: some View {
        Text(message)
            .sans(14, weight: .medium)
            .foregroundStyle(Theme.paper)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Theme.ink, in: Capsule())
            .shadow(color: .black.opacity(0.25), radius: 10, y: 6)
            .accessibilityAddTraits(.isStaticText)
    }
}

// MARK: - Flow layout

/// Lays children out left to right, wrapping onto new lines — used for the
/// move tokens of an engine line.
struct FlowLayout: Layout {
    var spacing: CGFloat = 0
    var lineHeight: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        let rows = arrange(subviews: subviews, width: width)
        let usedWidth = rows.map(\.width).max() ?? 0
        return CGSize(
            width: proposal.width ?? usedWidth,
            height: CGFloat(max(rows.count, 1)) * lineHeight
        )
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = arrange(subviews: subviews, width: bounds.width)
        for (rowIndex, row) in rows.enumerated() {
            let midY = bounds.minY + (CGFloat(rowIndex) + 0.5) * lineHeight
            for item in row.items {
                subviews[item.index].place(
                    at: CGPoint(x: bounds.minX + item.x, y: midY),
                    anchor: .leading,
                    proposal: .unspecified
                )
            }
        }
    }

    private struct Row {
        var items: [(index: Int, x: CGFloat)] = []
        var width: CGFloat = 0
    }

    private func arrange(subviews: Subviews, width: CGFloat) -> [Row] {
        var rows: [Row] = [Row()]
        var x: CGFloat = 0
        for (index, subview) in subviews.enumerated() {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width {
                rows.append(Row())
                x = 0
            }
            rows[rows.count - 1].items.append((index, x))
            rows[rows.count - 1].width = x + size.width
            x += size.width + spacing
        }
        return rows
    }
}
