import UIKit

extension UIImage {
    /// Redraws the image so its pixels are upright, discarding the EXIF
    /// orientation flag.
    ///
    /// Camera and library photos usually store rotated pixels plus an
    /// orientation tag; `cgImage` alone ignores the tag, which fed the
    /// recognizer sideways boards and produced garbage positions. Also caps
    /// the long edge (48 MP originals are wasted work — the models see 640px).
    func normalizedUprightCGImage(maxDimension: CGFloat = 3072) -> CGImage? {
        let longEdge = max(size.width, size.height)
        guard longEdge > 0 else { return nil }
        let scale = min(1, maxDimension / longEdge)
        let target = CGSize(
            width: (size.width * scale).rounded(),
            height: (size.height * scale).rounded()
        )

        if imageOrientation == .up, scale == 1 {
            return cgImage
        }

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: target, format: format)
        // UIImage.draw honors imageOrientation, so the output is upright.
        let upright = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }
        return upright.cgImage
    }
}
