import ChessVision
import Testing

@testable import ChessAnalyzer

@Suite struct ScanGuidanceTests {

    @Test func failuresFromTheRecognizer() {
        #expect(ScanFailure(RecognitionError.boardNotFound) == .noBoard)
        #expect(ScanFailure(RecognitionError.modelUnavailable) == .readerUnavailable)
        struct Other: Error {}
        #expect(ScanFailure(Other()) == .readerUnavailable)
    }

    @Test func eachFailureSaysWhatToChange() {
        for failure in [ScanFailure.noBoard, .unreadablePhoto, .readerUnavailable] {
            #expect(!failure.title.isEmpty)
            #expect(!failure.advice.isEmpty)
        }
        #expect(ScanFailure.noBoard.advice.first?.contains("four corners") == true)
        #expect(ScanFailure.noBoard.retakeHelps)
        #expect(!ScanFailure.readerUnavailable.retakeHelps)
    }

    @Test func cameraAngleFromGravity() throws {
        // Flat over the table, screen up: the back camera looks straight down.
        #expect(try #require(FramingAdvice.cameraAngle(gravityX: 0, y: 0, z: -1)) < 0.001)
        // Upright in portrait: it looks at the horizon.
        #expect(abs(try #require(FramingAdvice.cameraAngle(gravityX: 0, y: -1, z: 0)) - 90) < 0.001)
        // Halfway, and the length of the vector does not matter.
        #expect(abs(try #require(FramingAdvice.cameraAngle(gravityX: 0, y: -0.5, z: -0.5)) - 45) < 0.001)
        #expect(FramingAdvice.cameraAngle(gravityX: 0, y: 0, z: 0) == nil)
    }

    @Test func theTiltHintDoesNotFlicker() {
        #expect(FramingAdvice.next(after: .frame, cameraAngle: 45) == .frame)
        #expect(FramingAdvice.next(after: .frame, cameraAngle: 66) == .frame)
        #expect(FramingAdvice.next(after: .frame, cameraAngle: 75) == .tiltDown)
        // Once shown, it stays until the phone is clearly tilted down.
        #expect(FramingAdvice.next(after: .tiltDown, cameraAngle: 66) == .tiltDown)
        #expect(FramingAdvice.next(after: .tiltDown, cameraAngle: 55) == .frame)
        #expect(FramingAdvice.next(after: .tiltDown, cameraAngle: nil) == .frame)
    }
}
