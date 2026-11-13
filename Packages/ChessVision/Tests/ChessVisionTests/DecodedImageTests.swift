import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import ChessVision

/// `DecodedImage`: every stage of the recognition reads one decode of the
/// photo. See `docs/RECOGNITION.md` for what an undecoded JPEG does instead.
struct DecodedImageTests {

    private static let fixtures = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!

    /// As ImageIO hands it over: not decoded yet.
    private static func fromFile(_ file: String) throws -> CGImage {
        let source = try #require(
            CGImageSourceCreateWithURL(fixtures.appending(path: file) as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private static func bytes(_ image: CGImage, _ rect: (x: Int, y: Int, width: Int, height: Int)? = nil) throws -> [UInt8] {
        let r = rect ?? (0, 0, image.width, image.height)
        return try #require(GrayImage.rgba(of: image, x: r.x, y: r.y, width: r.width, height: r.height))
    }

    @Test func testAnImageFromAFileIsCopiedOnceAndABitmapIsLeftAlone() throws {
        for file in ["test_000102.jpg", "colour_pieces.png"] {
            let image = try Self.fromFile(file)
            #expect(image.utType != nil, "\(file): ImageIO names the file type")
            let copy = DecodedImage.once(image)
            #expect(copy !== image, "\(file)")
            #expect(copy.utType == nil, "\(file)")
            #expect(copy.width == image.width && copy.height == image.height, "\(file)")
            // The copy is a bitmap: it is not copied again.
            #expect(DecodedImage.once(copy) === copy, "\(file)")
        }
    }

    /// The copy holds the pixels of a first whole read of the file, in the
    /// file's own colour space.
    @Test func testTheCopyIsAWholeReadOfTheFile() throws {
        for file in ["test_000102.jpg", "test_004005.jpg", "colour_pieces.png"] {
            let whole = try Self.bytes(try Self.fromFile(file))
            let copy = DecodedImage.once(try Self.fromFile(file))
            #expect(try Self.bytes(copy) == whole, "\(file)")
            #expect(copy.colorSpace?.name == (try Self.fromFile(file)).colorSpace?.name, "\(file)")
        }
    }

    /// What the copy is for: a part of it read by itself is that part of the
    /// whole, and it stays what it was after the photo has been drawn scaled.
    /// (Neither holds for a JPEG that is not decoded yet.)
    @Test func testEveryReadOfTheCopyIsTheSamePixels() throws {
        let copy = DecodedImage.once(try Self.fromFile("test_000102.jpg"))
        let whole = try Self.bytes(copy)
        let rect = (x: 701, y: 433, width: 1203, height: 977)

        func part(of image: CGImage) throws -> [UInt8] { try Self.bytes(image, rect) }
        var expected = [UInt8]()
        expected.reserveCapacity(rect.width * rect.height * 4)
        for y in rect.y..<(rect.y + rect.height) {
            let start = (y * copy.width + rect.x) * 4
            expected.append(contentsOf: whole[start..<(start + rect.width * 4)])
        }
        #expect(try part(of: copy) == expected)

        // Drawn scaled, as the letterbox does for a photo that is not square.
        let letterbox = Letterbox(imageSize: CGSize(width: copy.width, height: copy.height + 600))
        _ = try #require(letterbox.render(copy))
        #expect(try Self.bytes(copy) == whole)
        #expect(try part(of: copy) == expected)
    }

    /// The grid fit, the colours and the recognizer take the copy themselves:
    /// handed the file's image or the copy, they give the same answer.
    @Test func testTheStagesDecodeForThemselves() throws {
        let image = try Self.fromFile("test_000102.jpg")
        let copy = DecodedImage.once(image)
        let fromFile = GridCorners.diagnose(image)
        let fromCopy = GridCorners.diagnose(copy)
        #expect(fromFile.confidence == fromCopy.confidence)
        #expect(fromFile.rawCorners == fromCopy.rawCorners)
        #expect(fromFile.values == fromCopy.values)

        let pieces = [
            PieceDetection(piece: .pawn, color: .white, confidence: 0.9,
                           boundingBox: CGRect(x: 900, y: 800, width: 160, height: 300)),
            PieceDetection(piece: .king, color: .black, confidence: 0.8,
                           boundingBox: CGRect(x: 1700.5, y: 1500.25, width: 190, height: 420)),
        ]
        #expect(PieceSides.colours(of: pieces, in: image) == PieceSides.colours(of: pieces, in: copy))
    }
}
