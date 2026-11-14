import CoreGraphics

/// A photo whose pixels are decoded once, into memory.
///
/// An image straight from a compressed file (a JPEG from ImageIO) is decoded
/// anew whenever it is drawn, and how it is drawn decides which pixels come
/// out: scaled (the letterbox), cropped (the grid fit's second pass, the
/// colours of the pieces) or whole, CoreGraphics returns bytes that differ
/// from one full decode in 10-30% of places. A whole read made after a
/// scaled draw was not even the same from run to run when it was found
/// (2026-11-13, through `app_check` with the models loaded; a later probe
/// without them did not reproduce that part; `docs/RECOGNITION.md`). The
/// grid fit follows its pixels, so the same photo could be a board in one run
/// and none in the next.
///
/// Every entry point of the recognition therefore takes its pixels from one
/// full decode: the models, the grid fit and the colours then see the same
/// photo, the same in every run, and the nearest to what `cv2.imread` gives
/// the Python reference.
public enum DecodedImage {

    /// The image as a bitmap in memory. An image that already is one (the
    /// app's own upright redraw, or a copy made here) is returned as it is;
    /// anything that came from an image file is drawn once, 1:1, in its own
    /// colour space where that is an RGB one (sRGB otherwise, as
    /// `GrayImage.rgba` reads it; transparency on black). If no bitmap can be
    /// made the image comes back as it was.
    ///
    /// Public for a caller that draws the photo itself before recognising it
    /// (`app_check --redraw`): it should draw the copy.
    public static func once(_ image: CGImage) -> CGImage {
        // ImageIO names the file type of an image it made; a bitmap drawn in
        // memory has none. (Checked on macOS for JPEG, HEIC and PNG from
        // ImageIO and AppKit against bitmaps from CoreGraphics, CoreImage and
        // VideoToolbox; `DecodedImageTests` pins it. Two exceptions, neither
        // on the app's path: a PNG is copied though it needs no copy, and a
        // crop of an undecoded JPEG has no type and is not copied; its reads
        // agree with each other.)
        guard image.utType != nil else { return image }
        let spaces = [
            image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil },
            CGColorSpace(name: CGColorSpace.sRGB),
        ]
        for case let space? in spaces {
            guard let context = CGContext(
                data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: 0, space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { continue }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            if let copy = context.makeImage() {
                return copy
            }
        }
        return image
    }
}
