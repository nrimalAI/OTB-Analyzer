import ChessVision
import CoreGraphics
import Foundation

/// Why a photo gave no position, in words for the person holding the phone,
/// with what to change for the next one.
enum ScanFailure: Equatable {
    /// The recognizer found no board in the photo.
    case noBoard
    /// The photo itself could not be opened.
    case unreadablePhoto
    /// The recognizer failed for another reason (its models did not load).
    case readerUnavailable

    init(_ error: Error) {
        switch error as? RecognitionError {
        case .boardNotFound: self = .noBoard
        case .modelUnavailable, nil: self = .readerUnavailable
        }
    }

    var title: String {
        switch self {
        case .noBoard: return "No board found"
        case .unreadablePhoto: return "Couldn\u{2019}t open the photo"
        case .readerUnavailable: return "Couldn\u{2019}t read this photo"
        }
    }

    /// What to do differently, most useful first.
    var advice: [String] {
        switch self {
        case .noBoard:
            return [
                "Get all four corners of the board inside the frame.",
                "From your seat, raise the phone and angle it down at the board.",
                "Keep glare from lamps and windows off the squares.",
            ]
        case .unreadablePhoto:
            return ["Choose another photo, or take one with the camera."]
        case .readerUnavailable:
            return ["Try another photo, or set the position up by hand."]
        }
    }

    /// Whether another photo could help (as opposed to setting up by hand).
    var retakeHelps: Bool { self != .readerUnavailable }
}

/// The framing hint on the camera screen, from which way the phone points.
enum FramingAdvice: Equatable {
    /// Nothing to say beyond "fill the frame".
    case frame
    /// The camera looks too far toward the horizon: a board on a table is
    /// then seen from low down, where tall pieces hide the ones behind them
    /// (`Training/README.md`, views 12 to 32 degrees above the board).
    case tiltDown

    /// The camera's angle from straight down, in degrees, for the direction
    /// of gravity in the phone's own axes (Core Motion's `gravity`: x to the
    /// right of the screen, y up the screen, z out of the screen). The back
    /// camera looks along −z. Nil when the reading says nothing (zero).
    static func cameraAngle(gravityX x: Double, y: Double, z: Double) -> Double? {
        let length = (x * x + y * y + z * z).squareRoot()
        guard length > 0.1 else { return nil }
        let cosine = max(-1, min(1, -z / length))
        return acos(cosine) * 180 / .pi
    }

    /// Past this angle from straight down (about 32 degrees above the table)
    /// the hint asks to tilt the phone down; it goes away again below
    /// `clearAngle`, so that it does not flicker at the edge. A photo from a
    /// player's seat, 40 to 60 degrees above the table, is left alone; so is
    /// one from straight above, though the models read that one less well.
    static let tiltAngle = 58.0
    static let clearAngle = 50.0

    /// The hint for a camera angle, given the hint on show now.
    static func next(after current: FramingAdvice, cameraAngle angle: Double?) -> FramingAdvice {
        guard let angle else { return .frame }
        switch current {
        case .frame: return angle > tiltAngle ? .tiltDown : .frame
        case .tiltDown: return angle < clearAngle ? .frame : .tiltDown
        }
    }

    var message: String {
        switch self {
        case .frame: return "From your seat, the whole board inside the corners"
        case .tiltDown: return "Raise the phone and angle it down at the board"
        }
    }
}

/// The lines of a chessboard seen in perspective from a player's seat, to
/// draw as the framing guide: the near edge across the whole rectangle, the
/// far edge `farWidth` of it, the ranks closer together toward the far edge
/// as a camera sees them (a projective map of the board, not even spacing).
struct PerspectiveGrid {
    struct Line: Equatable {
        let from: CGPoint
        let to: CGPoint
    }

    /// Nine lines across (the ranks' edges, far to near) then nine along
    /// (the files' edges, left to right).
    let lines: [Line]

    init(in rect: CGRect, farWidth: CGFloat = 0.62, height: CGFloat = 0.78) {
        let near = rect.width
        let far = rect.width * farWidth
        let h = rect.height * height
        let top = rect.midY - h / 2
        let midX = rect.midX
        /// Screen height of a point `v` of the way from the far edge to the
        /// near one: v = 0 at the top, 1 at the bottom.
        func y(_ v: CGFloat) -> CGFloat { top + h * far * v / (near - (near - far) * v) }
        func width(atY y: CGFloat) -> CGFloat { far + (near - far) * (y - top) / h }

        var lines: [Line] = []
        for i in 0...8 {
            let yi = y(CGFloat(i) / 8)
            let half = width(atY: yi) / 2
            lines.append(Line(from: CGPoint(x: midX - half, y: yi), to: CGPoint(x: midX + half, y: yi)))
        }
        for i in 0...8 {
            let t = CGFloat(i) / 8 - 0.5
            lines.append(Line(from: CGPoint(x: midX + t * far, y: top), to: CGPoint(x: midX + t * near, y: top + h)))
        }
        self.lines = lines
    }
}
