import ChessVision
import CoreGraphics
import Testing
import UIKit

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
        // From a seat (40-55 degrees from straight down) and from above: no hint.
        #expect(FramingAdvice.next(after: .frame, cameraAngle: 5) == .frame)
        #expect(FramingAdvice.next(after: .frame, cameraAngle: 45) == .frame)
        #expect(FramingAdvice.next(after: .frame, cameraAngle: 55) == .frame)
        // Low over the table: tilt down.
        #expect(FramingAdvice.next(after: .frame, cameraAngle: 62) == .tiltDown)
        // Once shown, it stays until the phone is clearly tilted down.
        #expect(FramingAdvice.next(after: .tiltDown, cameraAngle: 55) == .tiltDown)
        #expect(FramingAdvice.next(after: .tiltDown, cameraAngle: 45) == .frame)
        #expect(FramingAdvice.next(after: .tiltDown, cameraAngle: nil) == .frame)
    }

    @Test func theGuideIsABoardInPerspective() throws {
        let rect = CGRect(x: 0, y: 0, width: 300, height: 300)
        let grid = PerspectiveGrid(in: rect, farWidth: 0.6, height: 0.8)
        #expect(grid.lines.count == 18)
        let ranks = Array(grid.lines.prefix(9))
        // The far edge is narrower than the near one, which spans the frame.
        #expect(abs((ranks[0].to.x - ranks[0].from.x) - 180) < 0.001)
        #expect(abs((ranks[8].to.x - ranks[8].from.x) - 300) < 0.001)
        #expect(abs(ranks[0].from.y - 30) < 0.001)
        #expect(abs(ranks[8].from.y - 270) < 0.001)
        // Ranks get deeper toward the near edge, as a camera sees them.
        let depths = zip(ranks.dropFirst(), ranks).map { $0.from.y - $1.from.y }
        #expect(zip(depths.dropFirst(), depths).allSatisfy { $0 > $1 })
        // Every rank line ends on the outer files.
        let files = Array(grid.lines.suffix(9))
        for rank in ranks {
            for edge in [files[0], files[8]] {
                let t = (rank.from.y - edge.from.y) / (edge.to.y - edge.from.y)
                let x = edge.from.x + t * (edge.to.x - edge.from.x)
                #expect(abs(x - (edge == files[0] ? rank.from.x : rank.to.x)) < 0.001)
            }
        }
    }
}

@MainActor
@Suite struct ScanFailureFlowTests {

    @Test func aPhotoWithNoBoardSaysSo() async {
        let model = AppModel()
        model.path = [.scan]
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let blank = UIGraphicsImageRenderer(size: CGSize(width: 600, height: 800), format: format).image { context in
            UIColor.brown.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 600, height: 800))
        }
        #expect(await model.startFromPhoto(blank))
        #expect(model.scanFailure == .noBoard)
        #expect(model.path == [.scan])
    }

    @Test func aPhotoThatCannotBeOpenedSaysSo() async {
        let model = AppModel()
        model.path = [.scan]
        #expect(await model.startFromPhoto(UIImage()))
        #expect(model.scanFailure == .unreadablePhoto)
    }
}
