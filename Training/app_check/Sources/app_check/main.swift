// Runs the app's recognizer, exactly as the app runs it, over photos with
// known positions, and prints what `evaluate.py` prints: squares right and
// boards fully right. It exists because the Python scoring resizes photos
// with PIL while the app lets Vision do it, and a model can read the same
// photo differently in the two (pull request #4: one test fixture came out
// turned 180 degrees in the app and right in Python).
//
//   swift run -c release app_check --models ../../Models --list photos.json
//
//   --models DIR    folder with BoardCorners.mlpackage and Pieces.mlpackage
//   --list FILE     JSON object: photo path -> piece placement (first FEN field).
//                   Relative paths are taken from the folder of FILE.
//                   `evaluate.py --dump-list` writes one for a test split.
//   --redraw N      first draw each photo into an N x N square with
//                   CoreGraphics' high-quality resampling (what the app does
//                   for non-square photos, at 1280), then recognise that.
//   --portrait      first put each photo on a 3:4 grey canvas, as
//                   EndToEndFixtureTests.testRecognizesFixturesAsPortraitPhotos does.
//   --verbose       one line per photo, not only the wrong ones.
//
// Needs only the Command Line Tools (no Xcode, no XCTest).
import ChessKit
import ChessVision
import CoreGraphics
import CoreML
import Foundation
import ImageIO

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(2)
}

var modelsPath: String?
var listPath: String?
var redraw: Int?
var portrait = false
var verbose = false
var arguments = Array(CommandLine.arguments.dropFirst())
while !arguments.isEmpty {
    let flag = arguments.removeFirst()
    switch flag {
    case "--models": modelsPath = arguments.isEmpty ? nil : arguments.removeFirst()
    case "--list": listPath = arguments.isEmpty ? nil : arguments.removeFirst()
    case "--redraw": redraw = arguments.isEmpty ? nil : Int(arguments.removeFirst())
    case "--portrait": portrait = true
    case "--verbose": verbose = true
    default: fail("unknown argument \(flag)")
    }
}
guard let modelsPath, let listPath else {
    fail("usage: app_check --models DIR --list FILE [--redraw N] [--portrait] [--verbose]")
}

/// The image drawn into a canvas of the given size, grey 114 around it,
/// centred, shape kept.
func draw(_ image: CGImage, width: Int, height: Int) -> CGImage? {
    guard
        let space = CGColorSpace(name: CGColorSpace.sRGB),
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return nil }
    let grey = CGFloat(114) / 255
    context.setFillColor(red: grey, green: grey, blue: grey, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.interpolationQuality = .high
    let scale = min(CGFloat(width) / CGFloat(image.width), CGFloat(height) / CGFloat(image.height))
    let w = CGFloat(image.width) * scale
    let h = CGFloat(image.height) * scale
    context.draw(image, in: CGRect(
        x: (CGFloat(width) - w) / 2, y: (CGFloat(height) - h) / 2, width: w, height: h))
    return context.makeImage()
}

/// A placement written out as 64 squares, a8 first, "." for an empty one.
func squares(_ placement: String) -> [Character] {
    var out: [Character] = []
    for character in placement where character != "/" {
        if let empty = character.wholeNumberValue {
            out.append(contentsOf: [Character](repeating: ".", count: empty))
        } else {
            out.append(character)
        }
    }
    return out
}

let models = URL(fileURLWithPath: modelsPath)
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine  // as in the app
let recognizer: CoreMLBoardRecognizer
do {
    let corners = try MLModel(
        contentsOf: MLModel.compileModel(at: models.appending(path: "BoardCorners.mlpackage")),
        configuration: config)
    let pieces = try MLModel(
        contentsOf: MLModel.compileModel(at: models.appending(path: "Pieces.mlpackage")),
        configuration: config)
    recognizer = try CoreMLBoardRecognizer(cornerModel: corners, pieceModel: pieces)
} catch {
    fail("could not load the models in \(modelsPath): \(error)")
}

let listURL = URL(fileURLWithPath: listPath)
guard
    let data = try? Data(contentsOf: listURL),
    let list = try? JSONDecoder().decode([String: String].self, from: data),
    !list.isEmpty
else { fail("could not read \(listPath) as a JSON object of photo -> placement") }

var photos = 0
var squaresRight = 0
var exact = 0
var exactTurned = 0  // right once the board is turned 180 degrees
var notFound = 0
for (path, placement) in list.sorted(by: { $0.key < $1.key }) {
    let url = path.hasPrefix("/")
        ? URL(fileURLWithPath: path)
        : listURL.deletingLastPathComponent().appending(path: path)
    guard
        let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        var image = CGImageSourceCreateImageAtIndex(source, 0, nil),
        let expected = FENBuilder.pieces(fromPlacement: placement)
    else { fail("could not load \(path) or its placement") }
    if portrait {
        guard let canvas = draw(image, width: image.width, height: image.width * 4 / 3)
        else { fail("could not draw \(path)") }
        image = canvas
    }
    if let redraw {
        guard let canvas = draw(image, width: redraw, height: redraw)
        else { fail("could not draw \(path)") }
        image = canvas
    }

    photos += 1
    // Through FENBuilder, so that both are written the same way.
    let want = squares(FENBuilder.placement(expected))
    var have = [Character](repeating: "?", count: 64)
    var got = "board not found"
    do {
        let result = try await recognizer.recognize(image)
        got = FENBuilder.placement(result.pieces.mapValues(\.piece))
        have = squares(got)
    } catch RecognitionError.boardNotFound {
        // Counts as 64 wrong squares, as in evaluate.py.
        notFound += 1
    } catch {
        fail("\(path): \(error)")
    }
    let right = zip(want, have).filter { $0 == $1 }.count
    let turned = zip(want, have.reversed()).filter { $0 == $1 }.count
    squaresRight += right
    if right == 64 { exact += 1 }
    if right == 64 || turned == 64 { exactTurned += 1 }
    if verbose || right < 64 {
        let note = right < 64 && turned == 64 ? "  TURNED 180" : ""
        print("\(url.lastPathComponent): \(right)/64\(note)  expected \(placement)  got \(got)")
    }
}

func percent(_ part: Int, _ whole: Int) -> String {
    String(format: "%.2f%%", 100 * Double(part) / Double(max(whole, 1)))
}
print("""
    \(photos) photos through the app's recognizer\(redraw.map { ", redrawn at \($0)" } ?? "")\(portrait ? ", as 3:4 photos" : "")
      squares right: \(percent(squaresRight, photos * 64))
      boards exact: \(exact) of \(photos) (\(percent(exact, photos)))
      boards exact once turned 180 degrees: \(exactTurned) of \(photos) (\(percent(exactTurned, photos)))
      board not found: \(notFound)
    """)
