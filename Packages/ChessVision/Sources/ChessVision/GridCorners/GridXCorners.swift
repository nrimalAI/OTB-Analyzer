import Foundation

/// A point of the working image in float32, as the Python keeps its x-corners.
struct GridPoint: Equatable {
    var x: Float
    var y: Float
}

/// Step 1 of the grid fit: the x-corners, the points where four squares meet
/// (`x_corners` and `_ring_test` in `Training/grid_corners.py`).
enum GridXCorners {

    /// Sixteen directions around a point (`RING`).
    static let ring: [(cos: Double, sin: Double)] = (0..<16).map {
        let angle = Double($0) * Double.pi / 8
        return (cos(angle), sin(angle))
    }

    /// Sub-pixel saddle points that look like chessboard x-corners.
    static func find(in gray: GrayImage, maxPoints: Int = 260) -> [GridPoint] {
        let peaks = strongestPeaks(in: gray, count: maxPoints * 4)
        guard !peaks.isEmpty else { return [] }
        let smooth = gray.gaussianBlurred(sigma: 1.0)
        let points = Array(ringFiltered(peaks, smooth).prefix(maxPoints))
        guard !points.isEmpty else { return [] }

        let refined = cornerSubPix(smooth, points)
        var out: [GridPoint] = []
        for (before, after) in zip(points, refined) {
            let dx = after.x - before.x
            let dy = after.y - before.y
            if (dx * dx + dy * dy).squareRoot() < 4 {
                out.append(after)
            }
        }
        return out
    }

    /// The local maxima of the saddle response, strongest first, at least 8
    /// pixels from the edge.
    static func strongestPeaks(in gray: GrayImage, count: Int) -> [(value: Float, x: Int, y: Int)] {
        let w = gray.width, h = gray.height
        let border = 8
        guard w > 2 * border, h > 2 * border else { return [] }
        let saddle = gray.gaussianBlurred(sigma: 2.0).saddleResponse()
        let highest = saddle.dilated(size: 9)
        var candidates: [(value: Float, x: Int, y: Int)] = []
        saddle.pixels.withUnsafeBufferPointer { saddle in
            highest.pixels.withUnsafeBufferPointer { highest in
                var y = border
                while y < h - border {
                    var i = y * w + border
                    let end = y * w + w - border
                    while i < end {
                        let value = saddle[i]
                        if value > 0 && value == highest[i] {
                            candidates.append((value, i - y * w, y))
                        }
                        i &+= 1
                    }
                    y &+= 1
                }
            }
        }
        // Strongest first; equal ones stay in reading order.
        return candidates.enumerated()
            .sorted { $0.element.value != $1.element.value
                ? $0.element.value > $1.element.value : $0.offset < $1.offset }
            .prefix(count)
            .map(\.element)
    }

    /// The peaks around which the image goes light, dark, light, dark, at a
    /// radius of 4 or of 7 pixels.
    static func ringFiltered(
        _ peaks: [(value: Float, x: Int, y: Int)], _ smooth: GrayImage
    ) -> [GridPoint] {
        var points: [GridPoint] = []
        for peak in peaks {
            var keep = false
            for radius in [4.0, 7.0] {
                let test = ringTest(smooth, x: peak.x, y: peak.y, radius: radius)
                if test.transitions == 4 && test.response > 0.4 * test.contrast && test.contrast > 18 {
                    keep = true
                }
            }
            if keep {
                points.append(GridPoint(x: Float(peak.x), y: Float(peak.y)))
            }
        }
        return points
    }

    /// The ChESS-like test: on a ring around an x-corner the image goes
    /// light, dark, light, dark. Returns the response (opposite points alike,
    /// points a quarter turn apart different), the number of light/dark
    /// changes around the ring, and the ring's contrast.
    static func ringTest(
        _ image: GrayImage, x: Int, y: Int, radius: Double
    ) -> (response: Float, transitions: Int, contrast: Float) {
        var values = [Float](repeating: 0, count: 16)
        for (n, direction) in ring.enumerated() {
            // float32 arithmetic, as NumPy does it for float32 points
            let sx = Int((Float(x) + Float(radius * direction.cos)).rounded(.toNearestOrEven))
            let sy = Int((Float(y) + Float(radius * direction.sin)).rounded(.toNearestOrEven))
            values[n] = image[min(max(sx, 0), image.width - 1), min(max(sy, 0), image.height - 1)]
        }
        var sumResponse: Float = 0
        for n in 0..<4 {
            sumResponse += abs(values[n] + values[n + 8] - values[n + 4] - values[n + 12])
        }
        var difference: Float = 0
        for n in 0..<8 {
            difference += abs(values[n] - values[n + 8])
        }
        let highest = values.max()!
        let lowest = values.min()!
        let middle = (highest + lowest) / 2
        var transitions = 0
        for n in 0..<16 where (values[n] > middle) != (values[(n + 15) % 16] > middle) {
            transitions += 1
        }
        return (sumResponse - difference, transitions, highest - lowest)
    }

    /// `cv2.cornerSubPix(image, points, (4, 4), (-1, -1), (EPS + MAX_ITER, 30,
    /// 0.01))`: each point moved to where the image gradients around it say
    /// the corner is.
    static func cornerSubPix(_ image: GrayImage, _ points: [GridPoint]) -> [GridPoint] {
        let half = 4
        let side = 2 * half + 1
        let maxIterations = 30
        let epsilon = 0.01 * 0.01
        guard image.width >= half * 2 + 5, image.height >= half * 2 + 5 else { return points }

        var mask = [Float](repeating: 0, count: side * side)
        for i in 0..<side {
            let y = Float(i - half) / Float(half)
            let vy = exp(-y * y)
            for j in 0..<side {
                let x = Float(j - half) / Float(half)
                mask[i * side + j] = vy * exp(-x * x)
            }
        }

        let patchSide = side + 2
        var patch = [Float](repeating: 0, count: patchSide * patchSide)
        var out = points
        for (index, start) in points.enumerated() {
            var current = start
            var iteration = 0
            var error = 0.0
            repeat {
                // An 11×11 patch around the point, interpolated.
                let left = Double(current.x) - Double(patchSide - 1) * 0.5
                let top = Double(current.y) - Double(patchSide - 1) * 0.5
                for i in 0..<patchSide {
                    for j in 0..<patchSide {
                        patch[i * patchSide + j] = image.bilinear(left + Double(j), top + Double(i))
                    }
                }
                var a = 0.0, b = 0.0, c = 0.0, bb1 = 0.0, bb2 = 0.0
                for i in 0..<side {
                    let py = Double(i - half)
                    for j in 0..<side {
                        let m = Double(mask[i * side + j])
                        let centre = (i + 1) * patchSide + (j + 1)
                        let tgx = Double(patch[centre + 1] - patch[centre - 1])
                        let tgy = Double(patch[centre + patchSide] - patch[centre - patchSide])
                        let gxx = tgx * tgx * m
                        let gxy = tgx * tgy * m
                        let gyy = tgy * tgy * m
                        let px = Double(j - half)
                        a += gxx
                        b += gxy
                        c += gyy
                        bb1 += gxx * px + gxy * py
                        bb2 += gxy * px + gyy * py
                    }
                }
                let det = a * c - b * b
                if abs(det) <= Double.ulpOfOne * Double.ulpOfOne { break }
                let scale = 1 / det
                let next = GridPoint(
                    x: Float(Double(current.x) + c * scale * bb1 - b * scale * bb2),
                    y: Float(Double(current.y) - b * scale * bb1 + a * scale * bb2))
                let dx = Double(next.x - current.x)
                let dy = Double(next.y - current.y)
                error = dx * dx + dy * dy
                current = next
                if current.x < 0 || current.x >= Float(image.width)
                    || current.y < 0 || current.y >= Float(image.height)
                {
                    break
                }
                iteration += 1
            } while iteration < maxIterations && error > epsilon
            // Too far from where it started means it did not converge: keep the start.
            if abs(current.x - start.x) > Float(half) || abs(current.y - start.y) > Float(half) {
                current = start
            }
            out[index] = current
        }
        return out
    }
}
