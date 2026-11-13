import CoreGraphics
import Foundation
import Testing

@testable import ChessVision

struct LetterboxTests {

    @Test func testPortraitPhotoIsCentredWithSideBars() {
        let box = Letterbox(imageSize: CGSize(width: 3000, height: 4000))
        #expect(abs(box.content.minX - 0.125) <= 1e-9)
        #expect(abs(box.content.width - 0.75) <= 1e-9)
        #expect(abs(box.content.minY - 0) <= 1e-9)
        #expect(abs(box.content.height - 1) <= 1e-9)
        #expect(!box.isIdentity)
    }

    @Test func testPointsMapBackToImagePixels() {
        let box = Letterbox(imageSize: CGSize(width: 3000, height: 4000))
        // Corners of the content area are the corners of the photo.
        let topLeft = box.imagePoint(fromSquare: CGPoint(x: 0.125, y: 0))
        #expect(abs(topLeft.x - 0) <= 1e-6)
        #expect(abs(topLeft.y - 0) <= 1e-6)
        let bottomRight = box.imagePoint(fromSquare: CGPoint(x: 0.875, y: 1))
        #expect(abs(bottomRight.x - 3000) <= 1e-6)
        #expect(abs(bottomRight.y - 4000) <= 1e-6)
        // The centre stays the centre.
        let centre = box.imagePoint(fromSquare: CGPoint(x: 0.5, y: 0.5))
        #expect(abs(centre.x - 1500) <= 1e-6)
        #expect(abs(centre.y - 2000) <= 1e-6)
    }

    @Test func testRectsKeepTheirShape() {
        // A box that is square in the letterboxed input is square in the photo:
        // that is the whole point of not stretching.
        let box = Letterbox(imageSize: CGSize(width: 3000, height: 4000))
        let rect = box.imageRect(fromSquare: CGRect(x: 0.4, y: 0.4, width: 0.1, height: 0.1))
        #expect(abs(rect.width - rect.height) <= 1e-6)
        #expect(abs(rect.width - 400) <= 1e-6)
        #expect(abs(rect.minX - 1100) <= 1e-6)
        #expect(abs(rect.minY - 1600) <= 1e-6)
    }

    @Test func testSquareImageIsLeftAlone() throws {
        let box = Letterbox(imageSize: CGSize(width: 64, height: 64))
        #expect(box.isIdentity)
        let image = try #require(Self.solid(width: 64, height: 64, red: 255, green: 0, blue: 0))
        #expect(box.render(image) === image)
    }

    @Test func testRenderPadsWithGreyAndKeepsThePhotoInTheMiddle() throws {
        // A red 300×400 photo → 400×400 square: grey bars 50 px wide left and
        // right, red in between.
        let image = try #require(Self.solid(width: 300, height: 400, red: 255, green: 0, blue: 0))
        let box = Letterbox(imageSize: CGSize(width: 300, height: 400))
        let square = try #require(box.render(image))
        #expect(square.width == 400)
        #expect(square.height == 400)

        let bar = try #require(Self.pixel(of: square, x: 20, y: 200))
        #expect(abs(bar.red - 114) <= 2)
        #expect(abs(bar.green - 114) <= 2)
        #expect(abs(bar.blue - 114) <= 2)
        let middle = try #require(Self.pixel(of: square, x: 200, y: 200))
        #expect(abs(middle.red - 255) <= 2)
        #expect(abs(middle.green - 0) <= 2)
        let insideEdge = try #require(Self.pixel(of: square, x: 60, y: 10))
        #expect(abs(insideEdge.red - 255) <= 2)
    }

    @Test func testRenderCapsTheSide() throws {
        let image = try #require(Self.solid(width: 1500, height: 2000, red: 0, green: 0, blue: 255))
        let square = try #require(Letterbox(imageSize: CGSize(width: 1500, height: 2000)).render(image, maxSide: 640))
        #expect(square.width == 640)
        #expect(square.height == 640)
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
