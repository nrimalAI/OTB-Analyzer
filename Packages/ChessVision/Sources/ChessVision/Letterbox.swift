import CoreGraphics

/// Fits an image into a square without changing its shape: scaled to fit,
/// centred, the rest filled with grey.
///
/// The models take a square input. They were trained on letterboxed photos
/// (Ultralytics pads with grey 114), so the app must feed them the same thing.
/// Stretching a 3:4 phone photo to a square instead squashes the board by a
/// quarter, which costs accuracy; it went unnoticed because the ChessReD test
/// photos are square.
///
/// All geometry is in fractions of the square, top-left origin, so it does not
/// depend on the size the square is rendered at or on a model's input size.
struct Letterbox: Equatable {

    /// Size of the original image, in pixels.
    let imageSize: CGSize

    /// Where the image sits inside the unit square.
    let content: CGRect

    init(imageSize: CGSize) {
        self.imageSize = imageSize
        let longSide = max(imageSize.width, imageSize.height, 1)
        let width = imageSize.width / longSide
        let height = imageSize.height / longSide
        content = CGRect(x: (1 - width) / 2, y: (1 - height) / 2, width: width, height: height)
    }

    /// Whether the square is just the image itself.
    var isIdentity: Bool { imageSize.width == imageSize.height }

    /// A point given as fractions of the square → pixels of the original image.
    func imagePoint(fromSquare point: CGPoint) -> CGPoint {
        CGPoint(
            x: (point.x - content.minX) / content.width * imageSize.width,
            y: (point.y - content.minY) / content.height * imageSize.height
        )
    }

    /// A rectangle given as fractions of the square → pixels of the original image.
    func imageRect(fromSquare rect: CGRect) -> CGRect {
        let origin = imagePoint(fromSquare: rect.origin)
        return CGRect(
            x: origin.x,
            y: origin.y,
            width: rect.width / content.width * imageSize.width,
            height: rect.height / content.height * imageSize.height
        )
    }

    /// The image drawn into a grey square of at most `maxSide` pixels per
    /// side. A square image is returned as it is.
    func render(_ image: CGImage, maxSide: Int = 1280) -> CGImage? {
        if isIdentity { return image }
        let side = min(max(image.width, image.height), maxSide)
        guard
            let space = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: side, height: side,
                bitsPerComponent: 8, bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }

        let grey = CGFloat(114) / 255  // Ultralytics' letterbox padding value
        context.setFillColor(red: grey, green: grey, blue: grey, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        context.interpolationQuality = .high
        // CoreGraphics counts y from the bottom; the image is centred, so the
        // gap below it equals the gap above it.
        let s = CGFloat(side)
        context.draw(image, in: CGRect(
            x: content.minX * s,
            y: (1 - content.maxY) * s,
            width: content.width * s,
            height: content.height * s))
        return context.makeImage()
    }
}
