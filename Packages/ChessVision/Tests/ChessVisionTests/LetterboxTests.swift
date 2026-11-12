import CoreGraphics
import XCTest

@testable import ChessVision

final class LetterboxTests: XCTestCase {

    func testPortraitPhotoIsCentredWithSideBars() {
        let box = Letterbox(imageSize: CGSize(width: 3000, height: 4000))
        XCTAssertEqual(box.content.minX, 0.125, accuracy: 1e-9)
        XCTAssertEqual(box.content.width, 0.75, accuracy: 1e-9)
        XCTAssertEqual(box.content.minY, 0, accuracy: 1e-9)
        XCTAssertEqual(box.content.height, 1, accuracy: 1e-9)
        XCTAssertFalse(box.isIdentity)
    }

    func testPointsMapBackToImagePixels() {
        let box = Letterbox(imageSize: CGSize(width: 3000, height: 4000))
        // Corners of the content area are the corners of the photo.
        let topLeft = box.imagePoint(fromSquare: CGPoint(x: 0.125, y: 0))
        XCTAssertEqual(topLeft.x, 0, accuracy: 1e-6)
        XCTAssertEqual(topLeft.y, 0, accuracy: 1e-6)
        let bottomRight = box.imagePoint(fromSquare: CGPoint(x: 0.875, y: 1))
        XCTAssertEqual(bottomRight.x, 3000, accuracy: 1e-6)
        XCTAssertEqual(bottomRight.y, 4000, accuracy: 1e-6)
        // The centre stays the centre.
        let centre = box.imagePoint(fromSquare: CGPoint(x: 0.5, y: 0.5))
        XCTAssertEqual(centre.x, 1500, accuracy: 1e-6)
        XCTAssertEqual(centre.y, 2000, accuracy: 1e-6)
    }

    func testRectsKeepTheirShape() {
        // A box that is square in the letterboxed input is square in the photo:
        // that is the whole point of not stretching.
        let box = Letterbox(imageSize: CGSize(width: 3000, height: 4000))
        let rect = box.imageRect(fromSquare: CGRect(x: 0.4, y: 0.4, width: 0.1, height: 0.1))
        XCTAssertEqual(rect.width, rect.height, accuracy: 1e-6)
        XCTAssertEqual(rect.width, 400, accuracy: 1e-6)
        XCTAssertEqual(rect.minX, 1100, accuracy: 1e-6)
        XCTAssertEqual(rect.minY, 1600, accuracy: 1e-6)
    }

    func testSquareImageIsLeftAlone() throws {
        let box = Letterbox(imageSize: CGSize(width: 64, height: 64))
        XCTAssertTrue(box.isIdentity)
        let image = try XCTUnwrap(Self.solid(width: 64, height: 64, red: 255, green: 0, blue: 0))
        XCTAssertTrue(box.render(image) === image)
    }

    func testRenderPadsWithGreyAndKeepsThePhotoInTheMiddle() throws {
        // A red 300×400 photo → 400×400 square: grey bars 50 px wide left and
        // right, red in between.
        let image = try XCTUnwrap(Self.solid(width: 300, height: 400, red: 255, green: 0, blue: 0))
        let box = Letterbox(imageSize: CGSize(width: 300, height: 400))
        let square = try XCTUnwrap(box.render(image))
        XCTAssertEqual(square.width, 400)
        XCTAssertEqual(square.height, 400)

        let bar = try XCTUnwrap(Self.pixel(of: square, x: 20, y: 200))
        XCTAssertEqual(bar.red, 114, accuracy: 2)
        XCTAssertEqual(bar.green, 114, accuracy: 2)
        XCTAssertEqual(bar.blue, 114, accuracy: 2)
        let middle = try XCTUnwrap(Self.pixel(of: square, x: 200, y: 200))
        XCTAssertEqual(middle.red, 255, accuracy: 2)
        XCTAssertEqual(middle.green, 0, accuracy: 2)
        let insideEdge = try XCTUnwrap(Self.pixel(of: square, x: 60, y: 10))
        XCTAssertEqual(insideEdge.red, 255, accuracy: 2)
    }

    func testRenderCapsTheSide() throws {
        let image = try XCTUnwrap(Self.solid(width: 1500, height: 2000, red: 0, green: 0, blue: 255))
        let square = try XCTUnwrap(
            Letterbox(imageSize: CGSize(width: 1500, height: 2000)).render(image, maxSide: 640))
        XCTAssertEqual(square.width, 640)
        XCTAssertEqual(square.height, 640)
    }

    // MARK: - Helpers

    private static func context(width: Int, height: Int) -> CGContext? {
        CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    }

    static func solid(width: Int, height: Int, red: Int, green: Int, blue: Int) -> CGImage? {
        guard let ctx = context(width: width, height: height) else { return nil }
        ctx.setFillColor(
            red: CGFloat(red) / 255, green: CGFloat(green) / 255, blue: CGFloat(blue) / 255,
            alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    /// Pixel at (x, y) counted from the top-left.
    static func pixel(of image: CGImage, x: Int, y: Int) -> (red: Int, green: Int, blue: Int)? {
        guard let ctx = context(width: image.width, height: image.height) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let data = ctx.data else { return nil }
        let p = data.advanced(by: y * ctx.bytesPerRow + x * 4)
            .assumingMemoryBound(to: UInt8.self)
        return (Int(p[0]), Int(p[1]), Int(p[2]))
    }
}
