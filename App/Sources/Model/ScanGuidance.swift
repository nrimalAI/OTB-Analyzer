import ChessVision
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
                "Hold the phone over the board, not off to the side.",
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
    /// then seen nearly edge-on, and its squares cannot be told apart.
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

    /// Past this angle from straight down the hint asks to tilt the phone
    /// down; it goes away again below `clearAngle`, so that it does not
    /// flicker at the edge.
    static let tiltAngle = 70.0
    static let clearAngle = 62.0

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
        case .frame: return "Fit the whole board inside the corners"
        case .tiltDown: return "Tilt the phone down over the board"
        }
    }
}
