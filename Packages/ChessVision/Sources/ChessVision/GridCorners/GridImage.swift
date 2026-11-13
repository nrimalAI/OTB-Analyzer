import CoreGraphics
import Foundation

/// A one-channel float image, row by row from the top: what OpenCV calls
/// CV_32F. The image operations of the grid fit, each written to give what
/// the OpenCV call named in its comment gives (`Training/grid_corners.py`).
struct GrayImage {
    let width: Int
    let height: Int
    var pixels: [Float]

    init(width: Int, height: Int, pixels: [Float]) {
        precondition(pixels.count == width * height)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    init(width: Int, height: Int, fill: Float = 0) {
        self.init(width: width, height: height,
                  pixels: [Float](repeating: fill, count: width * height))
    }

    @inline(__always)
    subscript(x: Int, y: Int) -> Float {
        get { pixels[y * width + x] }
        set { pixels[y * width + x] = newValue }
    }

    /// Rows and columns exchanged (`W.T`).
    var transposed: GrayImage {
        var out = GrayImage(width: height, height: width)
        for y in 0..<height {
            for x in 0..<width {
                out.pixels[x * height + y] = pixels[y * width + x]
            }
        }
        return out
    }
}

// MARK: - From a photo (`_prepare`)

extension GrayImage {

    /// The long side of the working image, pixels (`WORK`).
    static let workingSide = 1000

    struct Working {
        let gray: GrayImage
        /// Working pixels per photo pixel.
        let scale: Double
        /// Where the working image starts in the photo.
        let originX: Int
        let originY: Int
    }

    /// The part of the photo to search, as grey levels 0...255: the prior box
    /// with a tenth added on each side (or the whole photo), shrunk so that
    /// its long side is at most 1000 pixels.
    static func working(from image: CGImage, priorBox: CGRect?, pad: Double = 0.10) -> Working? {
        let w = image.width
        let h = image.height
        var x1 = 0, y1 = 0, x2 = w, y2 = h
        if let box = priorBox {
            guard box.minX.isFinite, box.minY.isFinite, box.maxX.isFinite, box.maxY.isFinite
            else { return nil }
            let px = pad * Double(box.width)
            let py = pad * Double(box.height)
            let left = max(0, Double(box.minX) - px), top = max(0, Double(box.minY) - py)
            let right = min(Double(w), Double(box.maxX) + px)
            let bottom = min(Double(h), Double(box.maxY) + py)
            // A box that lies outside the photo leaves nothing.
            guard left < right, top < bottom else { return nil }
            x1 = Int(left)
            y1 = Int(top)
            x2 = Int(right)
            y2 = Int(bottom)
        }
        let cw = x2 - x1
        let ch = y2 - y1
        guard cw > 0, ch > 0 else { return nil }
        guard let rgba = Self.rgba(of: image, x: x1, y: y1, width: cw, height: ch) else { return nil }

        let s = min(1.0, Double(workingSide) / Double(max(cw, ch)))
        if s < 1 {
            let dw = Int((Double(cw) * s).rounded(.toNearestOrEven))
            let dh = Int((Double(ch) * s).rounded(.toNearestOrEven))
            guard dw > 0, dh > 0 else { return nil }
            let gray = Self.areaResizedGray(
                rgba: rgba, width: cw, height: ch, toWidth: dw, toHeight: dh, scale: 1 / s)
            return Working(gray: gray, scale: s, originX: x1, originY: y1)
        }
        var gray = GrayImage(width: cw, height: ch)
        for i in 0..<(cw * ch) {
            gray.pixels[i] = Float(Self.grey(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2]))
        }
        return Working(gray: gray, scale: s, originX: x1, originY: y1)
    }

    /// OpenCV's 8-bit BGR-to-grey: 0.299 R + 0.587 G + 0.114 B in 15-bit
    /// fixed point (the constants of OpenCV 4; checked equal to
    /// `cv2.cvtColor` on all 2^24 colours).
    @inline(__always)
    static func grey(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> UInt8 {
        UInt8((Int(r) * 9798 + Int(g) * 19235 + Int(b) * 3735 + 16384) >> 15)
    }

    /// The pixels of a rectangle of the image as RGBX bytes, in the image's
    /// own colour space where that is an RGB one, so that the numbers are the
    /// ones stored in the file (what `cv2.imread` returns).
    private static func rgba(of image: CGImage, x: Int, y: Int, width: Int, height: Int) -> [UInt8]? {
        let whole = x == 0 && y == 0 && width == image.width && height == image.height
        guard let part = whole
            ? image
            : image.cropping(to: CGRect(x: x, y: y, width: width, height: height))
        else { return nil }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let spaces = [
            image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil },
            CGColorSpace(name: CGColorSpace.sRGB),
        ]
        for case let space? in spaces {
            let drawn = bytes.withUnsafeMutableBytes { buffer -> Bool in
                guard let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                else { return false }
                context.interpolationQuality = .none
                context.draw(part, in: CGRect(x: 0, y: 0, width: width, height: height))
                return true
            }
            if drawn { return bytes }
        }
        return nil
    }

    /// One entry of OpenCV's area-resize table: source index, destination
    /// index, weight.
    private struct AreaTap {
        let source: Int
        let destination: Int
        let weight: Float
    }

    /// `computeResizeAreaTab`: how much of each source pixel falls into each
    /// destination pixel along one axis.
    private static func areaTable(source: Int, destination: Int, scale: Double) -> [AreaTap] {
        var taps: [AreaTap] = []
        taps.reserveCapacity(destination * (Int(scale) + 2))
        for d in 0..<destination {
            let f1 = Double(d) * scale
            let f2 = f1 + scale
            let cell = min(scale, Double(source) - f1)
            var s1 = Int(f1.rounded(.up))
            var s2 = Int(f2.rounded(.down))
            s2 = min(s2, source - 1)
            s1 = min(s1, s2)
            if Double(s1) - f1 > 1e-3 {
                taps.append(AreaTap(source: s1 - 1, destination: d,
                                    weight: Float((Double(s1) - f1) / cell)))
            }
            for s in s1..<s2 {
                taps.append(AreaTap(source: s, destination: d, weight: Float(1.0 / cell)))
            }
            if f2 - Double(s2) > 1e-3 {
                taps.append(AreaTap(source: s2, destination: d,
                                    weight: Float(min(min(f2 - Double(s2), 1.0), cell) / cell)))
            }
        }
        return taps
    }

    /// The same for a shrink by exactly two, where OpenCV takes a shortcut
    /// that rounds halves up: (a + b + c + d + 2) >> 2.
    private static func halvedGray(
        rgba: [UInt8], width: Int, height: Int, toWidth dw: Int, toHeight dh: Int
    ) -> GrayImage {
        var out = GrayImage(width: dw, height: dh)
        rgba.withUnsafeBufferPointer { src in
            out.pixels.withUnsafeMutableBufferPointer { out in
                var dy = 0
                while dy < dh {
                    // (an odd side leaves a last row or column of single pixels)
                    let y0 = dy * 2, y1 = min(dy * 2 + 1, height - 1)
                    var dx = 0
                    while dx < dw {
                        let x0 = dx * 2, x1 = min(dx * 2 + 1, width - 1)
                        let a = src.baseAddress! + (y0 * width + x0) * 4
                        let b = src.baseAddress! + (y0 * width + x1) * 4
                        let c = src.baseAddress! + (y1 * width + x0) * 4
                        let d = src.baseAddress! + (y1 * width + x1) * 4
                        let red = (Int(a[0]) + Int(b[0]) + Int(c[0]) + Int(d[0]) + 2) >> 2
                        let green = (Int(a[1]) + Int(b[1]) + Int(c[1]) + Int(d[1]) + 2) >> 2
                        let blue = (Int(a[2]) + Int(b[2]) + Int(c[2]) + Int(d[2]) + 2) >> 2
                        out[dy * dw + dx] = Float(grey(UInt8(red), UInt8(green), UInt8(blue)))
                        dx &+= 1
                    }
                    dy &+= 1
                }
            }
        }
        return out
    }

    /// `cv2.resize(..., interpolation=cv2.INTER_AREA)` on the three colour
    /// channels, each rounded to a byte, then OpenCV's grey conversion.
    private static func areaResizedGray(
        rgba: [UInt8], width: Int, height: Int, toWidth dw: Int, toHeight dh: Int, scale: Double
    ) -> GrayImage {
        if scale == 2 {
            return halvedGray(rgba: rgba, width: width, height: height, toWidth: dw, toHeight: dh)
        }
        let xTaps = areaTable(source: width, destination: dw, scale: scale)
        let yTaps = areaTable(source: height, destination: dh, scale: scale)
        var out = GrayImage(width: dw, height: dh)
        var row = [Float](repeating: 0, count: dw * 3)
        var sum = [Float](repeating: 0, count: dw * 3)

        @inline(__always) func byte(_ value: Float) -> UInt8 {
            UInt8(min(max(value.rounded(.toNearestOrEven), 0), 255))
        }
        func flush(_ dy: Int, _ sum: UnsafeMutablePointer<Float>, _ out: UnsafeMutablePointer<Float>) {
            for dx in 0..<dw {
                out[dy * dw + dx] = Float(grey(
                    byte(sum[dx * 3]), byte(sum[dx * 3 + 1]), byte(sum[dx * 3 + 2])))
            }
        }

        // Plain `while` loops over pointers throughout this file: they stay
        // fast in a build without optimisation, where `for` over a range and
        // array subscripts cost many times more.
        let channels = dw * 3
        rgba.withUnsafeBufferPointer { src in
            row.withUnsafeMutableBufferPointer { rowBuffer in
                sum.withUnsafeMutableBufferPointer { sumBuffer in
                    out.pixels.withUnsafeMutableBufferPointer { outBuffer in
                        xTaps.withUnsafeBufferPointer { xTaps in
                            let row = rowBuffer.baseAddress!, sum = sumBuffer.baseAddress!
                            let taps = xTaps.baseAddress!, tapCount = xTaps.count
                            var previous = -1
                            for tap in yTaps {
                                let line = src.baseAddress! + tap.source * width * 4
                                var i = 0
                                while i < channels { row[i] = 0; i &+= 1 }
                                var t = 0
                                while t < tapCount {
                                    let x = taps[t]
                                    let s = line + x.source * 4
                                    let d = row + x.destination * 3
                                    d[0] += Float(s[0]) * x.weight
                                    d[1] += Float(s[1]) * x.weight
                                    d[2] += Float(s[2]) * x.weight
                                    t &+= 1
                                }
                                let weight = tap.weight
                                i = 0
                                if tap.destination != previous {
                                    if previous >= 0 {
                                        flush(previous, sum, outBuffer.baseAddress!)
                                    }
                                    while i < channels { sum[i] = weight * row[i]; i &+= 1 }
                                    previous = tap.destination
                                } else {
                                    while i < channels { sum[i] += weight * row[i]; i &+= 1 }
                                }
                            }
                            if previous >= 0 {
                                flush(previous, sum, outBuffer.baseAddress!)
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}

// MARK: - Filters

extension GrayImage {

    /// OpenCV's default border: the image mirrored about its edge pixel
    /// (BORDER_REFLECT_101).
    @inline(__always)
    static func reflect(_ i: Int, _ n: Int) -> Int {
        if n == 1 { return 0 }
        var i = i
        while i < 0 || i >= n {
            i = i < 0 ? -i : 2 * n - 2 - i
        }
        return i
    }

    /// The kernel `cv2.GaussianBlur(image, (0, 0), sigma)` uses on a float
    /// image: 8 sigma + 1 taps, made odd.
    static func gaussianKernel(sigma: Double) -> [Float] {
        let size = Int((sigma * 8 + 1).rounded(.toNearestOrEven)) | 1
        let centre = Double(size - 1) / 2
        var kernel = (0..<size).map { i -> Double in
            let x = Double(i) - centre
            return exp(-x * x / (2 * sigma * sigma))
        }
        let total = kernel.reduce(0, +)
        kernel = kernel.map { $0 / total }
        return kernel.map { Float($0) }
    }

    /// `cv2.GaussianBlur(image, (0, 0), sigma)`.
    func gaussianBlurred(sigma: Double) -> GrayImage {
        let kernel = Self.gaussianKernel(sigma: sigma)
        let radius = kernel.count / 2
        let w = width, h = height
        var temp = [Float](repeating: 0, count: w * h)
        var out = [Float](repeating: 0, count: w * h)
        // Each output is the centre tap, then the symmetric pairs added in
        // order of distance, as OpenCV adds them.
        var padded = [Float](repeating: 0, count: w + 2 * radius)
        var columns = [Int](repeating: 0, count: w + 2 * radius)
        for x in 0..<(w + 2 * radius) {
            columns[x] = Self.reflect(x - radius, w)
        }
        var rows = [Int](repeating: 0, count: h + 2 * radius)
        for y in 0..<(h + 2 * radius) {
            rows[y] = Self.reflect(y - radius, h) * w
        }

        pixels.withUnsafeBufferPointer { src in
            temp.withUnsafeMutableBufferPointer { temp in
                padded.withUnsafeMutableBufferPointer { paddedBuffer in
                    let padded = paddedBuffer.baseAddress!
                    let count = w + 2 * radius
                    var y = 0
                    while y < h {
                        // Along x.
                        let line = src.baseAddress! + y * w
                        let o = temp.baseAddress! + y * w
                        var x = 0
                        while x < count { padded[x] = line[columns[x]]; x &+= 1 }
                        let centre = padded + radius
                        Self.scale(o, centre, kernel[radius], w)
                        var i = 1
                        while i <= radius {
                            Self.addScaledSum(o, centre + i, centre - i, kernel[radius + i], w)
                            i &+= 1
                        }
                        y &+= 1
                    }
                }
            }
        }
        temp.withUnsafeBufferPointer { temp in
            out.withUnsafeMutableBufferPointer { out in
                var y = 0
                while y < h {
                    // Along y.
                    let c = y + radius
                    let o = out.baseAddress! + y * w
                    Self.scale(o, temp.baseAddress! + rows[c], kernel[radius], w)
                    var i = 1
                    while i <= radius {
                        Self.addScaledSum(
                            o, temp.baseAddress! + rows[c + i], temp.baseAddress! + rows[c - i],
                            kernel[radius + i], w)
                        i &+= 1
                    }
                    y &+= 1
                }
            }
        }
        return GrayImage(width: w, height: h, pixels: out)
    }

    /// out[x] = weight * a[x]
    @inline(__always)
    private static func scale(
        _ out: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ weight: Float, _ count: Int
    ) {
        var x = 0
        while x < count {
            out[x] = weight * a[x]
            x &+= 1
        }
    }

    /// out[x] += weight * (a[x] + b[x])
    @inline(__always)
    private static func addScaledSum(
        _ out: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
        _ weight: Float, _ count: Int
    ) {
        var x = 0
        while x < count {
            out[x] += weight * (a[x] + b[x])
            x &+= 1
        }
    }

    /// The determinant of the Hessian with its sign turned, so that saddle
    /// points are positive: `ixy*ixy - ixx*iyy` from three 3×3 Sobel filters
    /// (`cv2.Sobel(g, CV_32F, 2, 0)`, `(0, 2)` and `(1, 1)`), negatives set
    /// to 0.
    func saddleResponse() -> GrayImage {
        let w = width, h = height
        var out = GrayImage(width: w, height: h)
        pixels.withUnsafeBufferPointer { p in
            out.pixels.withUnsafeMutableBufferPointer { o in
                for y in 0..<h {
                    let ym = Self.reflect(y - 1, h) * w
                    let y0 = y * w
                    let yp = Self.reflect(y + 1, h) * w
                    for x in 0..<w {
                        let xm = Self.reflect(x - 1, w)
                        let xp = Self.reflect(x + 1, w)
                        let a = p[ym + xm], b = p[ym + x], c = p[ym + xp]
                        let d = p[y0 + xm], e = p[y0 + x], f = p[y0 + xp]
                        let g = p[yp + xm], i = p[yp + x], j = p[yp + xp]
                        // second derivative along x, smoothed 1 2 1 along y
                        let ixx = (a - 2 * b + c) + 2 * (d - 2 * e + f) + (g - 2 * i + j)
                        // second derivative along y, smoothed 1 2 1 along x
                        let iyy = (a - 2 * d + g) + 2 * (b - 2 * e + i) + (c - 2 * f + j)
                        let ixy = (j - g) - (c - a)
                        let value = ixy * ixy - ixx * iyy
                        o[y0 + x] = value < 0 ? 0 : value
                    }
                }
            }
        }
        return out
    }

    /// `cv2.dilate(image, np.ones((size, size)))`: the largest value in the
    /// size×size window around each pixel, the outside of the image ignored.
    func dilated(size: Int) -> GrayImage {
        let radius = size / 2
        let w = width, h = height
        var temp = [Float](repeating: 0, count: w * h)
        var out = [Float](repeating: 0, count: w * h)
        pixels.withUnsafeBufferPointer { p in
            temp.withUnsafeMutableBufferPointer { t in
                var y = 0
                while y < h {
                    let line = p.baseAddress! + y * w
                    let o = t.baseAddress! + y * w
                    var x = 0
                    while x < w {
                        var i = x - radius < 0 ? 0 : x - radius
                        let last = x + radius > w - 1 ? w - 1 : x + radius
                        var best = line[i]
                        i &+= 1
                        while i <= last {
                            if line[i] > best { best = line[i] }
                            i &+= 1
                        }
                        o[x] = best
                        x &+= 1
                    }
                    y &+= 1
                }
            }
        }
        temp.withUnsafeBufferPointer { t in
            out.withUnsafeMutableBufferPointer { o in
                var y = 0
                while y < h {
                    let first = y - radius < 0 ? 0 : y - radius
                    let last = y + radius > h - 1 ? h - 1 : y + radius
                    let line = o.baseAddress! + y * w
                    var x = 0
                    while x < w { line[x] = t[first * w + x]; x &+= 1 }
                    var i = first + 1
                    while i <= last {
                        let other = t.baseAddress! + i * w
                        x = 0
                        while x < w {
                            if other[x] > line[x] { line[x] = other[x] }
                            x &+= 1
                        }
                        i &+= 1
                    }
                    y &+= 1
                }
            }
        }
        return GrayImage(width: w, height: h, pixels: out)
    }
}

// MARK: - Sampling

extension GrayImage {

    /// `cv2.warpPerspective(image, matrix, (width, height), flags=INTER_LINEAR,
    /// borderMode=BORDER_CONSTANT, borderValue=-1)`: `matrix` takes this image
    /// onto the output; pixels with no source are -1. As in OpenCV, source
    /// positions are rounded to 1/32 pixel before interpolating.
    func warped(by matrix: Mat3, width outWidth: Int, height outHeight: Int) -> GrayImage? {
        guard let inverse = matrix.inverse else { return nil }
        let m0 = inverse.m[0], m1 = inverse.m[1], m2 = inverse.m[2]
        let m3 = inverse.m[3], m4 = inverse.m[4], m5 = inverse.m[5]
        let m6 = inverse.m[6], m7 = inverse.m[7], m8 = inverse.m[8]
        let border: Float = -1
        let w = width, h = height
        var out = GrayImage(width: outWidth, height: outHeight)
        let limit = Double(Int32.max)
        let block = 64

        pixels.withUnsafeBufferPointer { p in
            out.pixels.withUnsafeMutableBufferPointer { o in
                var y = 0
                while y < outHeight {
                    let line = o.baseAddress! + y * outWidth
                    let fy0 = Double(y)
                    var x = 0
                    while x < outWidth {
                        // OpenCV works in blocks of 64 columns and adds the
                        // offset inside the block; so does this, to round alike.
                        let start = x - x % block
                        let x1 = Double(x - start)
                        let x0 = m0 * Double(start) + m1 * fy0 + m2
                        let y0 = m3 * Double(start) + m4 * fy0 + m5
                        let w0 = m6 * Double(start) + m7 * fy0 + m8
                        var depth = w0 + m6 * x1
                        depth = depth != 0 ? 32 / depth : 0
                        let fx = max(-limit - 1, min(limit, (x0 + m0 * x1) * depth))
                        let fy = max(-limit - 1, min(limit, (y0 + m3 * x1) * depth))
                        if !(fx.isFinite && fy.isFinite) {
                            line[x] = border
                            x &+= 1
                            continue
                        }
                        let ix = Int(fx.rounded(.toNearestOrEven))
                        let iy = Int(fy.rounded(.toNearestOrEven))
                        let sx = min(max(ix >> 5, -32768), 32767)
                        let sy = min(max(iy >> 5, -32768), 32767)
                        let ax = Float(ix & 31) / 32
                        let ay = Float(iy & 31) / 32
                        let w00 = (1 - ay) * (1 - ax)
                        let w01 = (1 - ay) * ax
                        let w10 = ay * (1 - ax)
                        let w11 = ay * ax
                        if sx >= 0, sx < w - 1, sy >= 0, sy < h - 1 {
                            let i = sy * w + sx
                            line[x] = p[i] * w00 + p[i + 1] * w01 + p[i + w] * w10 + p[i + w + 1] * w11
                        } else if sx >= w || sx + 1 < 0 || sy >= h || sy + 1 < 0 {
                            line[x] = border
                        } else {
                            let v00 = sx >= 0 && sx < w && sy >= 0 && sy < h ? p[sy * w + sx] : border
                            let v01 = sx + 1 >= 0 && sx + 1 < w && sy >= 0 && sy < h
                                ? p[sy * w + sx + 1] : border
                            let v10 = sx >= 0 && sx < w && sy + 1 >= 0 && sy + 1 < h
                                ? p[(sy + 1) * w + sx] : border
                            let v11 = sx + 1 >= 0 && sx + 1 < w && sy + 1 >= 0 && sy + 1 < h
                                ? p[(sy + 1) * w + sx + 1] : border
                            line[x] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
                        }
                        x &+= 1
                    }
                    y &+= 1
                }
            }
        }
        return out
    }

    /// `cv2.getRectSubPix` for one pixel: the image at a point between
    /// pixels, the nearest edge pixel standing in for the outside.
    @inline(__always)
    func bilinear(_ x: Double, _ y: Double) -> Float {
        let fx = x.rounded(.down)
        let fy = y.rounded(.down)
        let a = Float(x - fx)
        let b = Float(y - fy)
        let x0 = min(max(Int(fx), 0), width - 1)
        let x1 = min(max(Int(fx) + 1, 0), width - 1)
        let y0 = min(max(Int(fy), 0), height - 1)
        let y1 = min(max(Int(fy) + 1, 0), height - 1)
        return (1 - a) * (1 - b) * pixels[y0 * width + x0] + a * (1 - b) * pixels[y0 * width + x1]
            + (1 - a) * b * pixels[y1 * width + x0] + a * b * pixels[y1 * width + x1]
    }
}
