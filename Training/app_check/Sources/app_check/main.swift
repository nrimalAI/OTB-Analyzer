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
//   --a1-prior P    how much the corner model's choice of a1 counts against
//                   the pieces (BoardOrientation; the app's default is 6), or
//                   "off" to keep the corner model's a1 as it is.
//   --grid-fit on|off  corners from the classical grid fit where it accepts
//                   its own result (the app's default: on).
//   --sides-from-colour on|off  White and Black from the colours of the
//                   pieces where they contradict the piece model (default
//                   off, as in the app).
//   --one-king on|off  exactly one king a side, from the scores the piece
//                   model gave each piece for every kind (the app's default:
//                   on).
//   --flag-below C[,C...]  the app marks a square for the user to check
//                   when its piece's confidence is under 0.85 (AppModel);
//                   count how many of the right and of the wrong pieces that
//                   marks, for each threshold given (default 0.85).
//   --colours FILE  score nothing: write, for each photo, what the piece model
//                   finds, the colour of each piece and the sides the colour
//                   step gives, for `swift_reference.py colours` to compare
//                   with the Python on the same boxes.
//   --grid FILE     run no model: write what the classical grid fit
//                   (GridCorners, the port of grid_corners.py) makes of each
//                   photo to FILE, for `swift_reference.py compare`.
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

/// The photo the way a viewer shows it: turned as its EXIF orientation tag
/// says, which is what the app's camera pipeline hands the recognizer and
/// what `cv2.imread` returns in the Python. Photos without a tag (all of the
/// current test sets) are loaded as they are, undecoded: the recognizer
/// decodes them once (`DecodedImage`). A photo with a tag comes out of
/// ImageIO's thumbnail call as a bitmap already, whose pixels are about a
/// fifth of the bytes away from a full decode turned afterwards; if a test
/// set with tags is ever scored, turn the full decode instead.
func upright(_ url: URL) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
    let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
    guard orientation != 1,
          let width = (properties?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
          let height = (properties?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue
    else { return CGImageSourceCreateImageAtIndex(source, 0, nil) }
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: max(width, height),
    ]
    return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
}

var modelsPath: String?
var listPath: String?
var redraw: Int?
var portrait = false
var verbose = false
var a1Prior: Double? = 6
var a1PriorGiven = false
var gridPath: String?
var gridFit = true
var gridFitGiven = false
var sidesFromColour = false
var sidesFromColourGiven = false
var oneKing: Bool?
var flagBelow: [Float] = [0.85]
var coloursPath: String?
var arguments = Array(CommandLine.arguments.dropFirst())
while !arguments.isEmpty {
    let flag = arguments.removeFirst()
    switch flag {
    case "--models": modelsPath = arguments.isEmpty ? nil : arguments.removeFirst()
    case "--list": listPath = arguments.isEmpty ? nil : arguments.removeFirst()
    case "--redraw": redraw = arguments.isEmpty ? nil : Int(arguments.removeFirst())
    case "--portrait": portrait = true
    case "--grid-fit":
        let value = arguments.isEmpty ? "" : arguments.removeFirst()
        guard value == "on" || value == "off" else { fail("--grid-fit takes on or off") }
        gridFit = value == "on"
        gridFitGiven = true
    case "--grid": gridPath = arguments.isEmpty ? nil : arguments.removeFirst()
    case "--sides-from-colour":
        let value = arguments.isEmpty ? "" : arguments.removeFirst()
        guard value == "on" || value == "off" else { fail("--sides-from-colour takes on or off") }
        sidesFromColour = value == "on"
        sidesFromColourGiven = true
    case "--colours": coloursPath = arguments.isEmpty ? nil : arguments.removeFirst()
    case "--one-king":
        let value = arguments.isEmpty ? "" : arguments.removeFirst()
        guard value == "on" || value == "off" else { fail("--one-king takes on or off") }
        oneKing = value == "on"
    case "--flag-below":
        let values = (arguments.isEmpty ? "" : arguments.removeFirst()).split(separator: ",").map { Float($0) }
        guard !values.isEmpty, values.allSatisfy({ $0 != nil }) else {
            fail("--flag-below takes a number, or several with commas between them")
        }
        flagBelow = values.compactMap { $0 }
    case "--a1-prior":
        let value = arguments.isEmpty ? "" : arguments.removeFirst()
        if value == "off" {
            a1Prior = nil
        } else if let number = Double(value) {
            a1Prior = number
        } else {
            fail("--a1-prior takes a number or off")
        }
        a1PriorGiven = true
    case "--verbose": verbose = true
    default: fail("unknown argument \(flag)")
    }
}
if coloursPath != nil && (gridPath != nil || redraw != nil || portrait) {
    fail("--colours reads the photos as they are: it goes with neither --grid, --redraw nor --portrait")
}
if let gridPath, let listPath {
    let listURL = URL(fileURLWithPath: listPath)
    guard
        let data = try? Data(contentsOf: listURL),
        let list = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { fail("could not read \(listPath) as a JSON object keyed by photo") }
    var out: [String: Any] = [:]
    let started = Date()
    for path in list.keys.sorted() {
        let url = path.hasPrefix("/")
            ? URL(fileURLWithPath: path)
            : listURL.deletingLastPathComponent().appending(path: path)
        guard let image = upright(url) else { fail("could not load \(path)") }
        let found = GridCorners.diagnose(image)
        out[path] = [
            "raw": found.rawCorners.map { $0.map { [Double($0.x), Double($0.y)] } } ?? NSNull(),
            "confidence": found.confidence, "stage": found.stage, "pass": found.pass,
            // (JSON has no NaN, and JSONSerialization raises on one)
            "values": found.values.filter { $0.value.isFinite },
        ] as [String: Any]
        if verbose {
            print("\(url.lastPathComponent): \(found.stage) pass \(found.pass) confidence \(found.confidence)")
        }
    }
    let seconds = Date().timeIntervalSince(started)
    do {
        try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: gridPath))
    } catch {
        fail("could not write \(gridPath): \(error)")
    }
    let accepted = out.values.filter { (($0 as? [String: Any])?["confidence"] as? Double ?? 0) > 0 }.count
    print(String(
        format: "%d photos through the grid fit, %d accepted, %.2f s per photo; written to %@",
        out.count, accepted, seconds / Double(max(out.count, 1)), gridPath))
    exit(0)
}
guard let modelsPath, let listPath else {
    fail("usage: app_check --models DIR --list FILE [--redraw N] [--portrait] [--a1-prior P|off] [--grid-fit on|off] [--grid FILE] [--sides-from-colour on|off] [--one-king on|off] [--flag-below C] [--colours FILE] [--verbose]")
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
    if let oneKing {
        recognizer = try CoreMLBoardRecognizer(
            cornerModel: corners, pieceModel: pieces, a1Prior: a1Prior, gridFit: gridFit,
            sidesFromColour: sidesFromColour, oneKing: oneKing)
    } else {
        recognizer = try CoreMLBoardRecognizer(
            cornerModel: corners, pieceModel: pieces, a1Prior: a1Prior, gridFit: gridFit,
            sidesFromColour: sidesFromColour)
    }
} catch {
    fail("could not load the models in \(modelsPath): \(error)")
}

let listURL = URL(fileURLWithPath: listPath)
guard
    let data = try? Data(contentsOf: listURL),
    let list = try? JSONDecoder().decode([String: String].self, from: data),
    !list.isEmpty
else { fail("could not read \(listPath) as a JSON object of photo -> placement") }

if let coloursPath {
    // What the piece model finds and what the colour step makes of it.
    var out: [String: Any] = [:]
    var changed = 0
    for path in list.keys.sorted() {
        let url = path.hasPrefix("/")
            ? URL(fileURLWithPath: path)
            : listURL.deletingLastPathComponent().appending(path: path)
        guard let image = upright(url) else { fail("could not load \(path)") }
        let pieces: [PieceSideDiagnosis]
        do {
            pieces = try recognizer.diagnosePieceSides(image)
        } catch {
            fail("\(path): \(error)")
        }
        if pieces.contains(where: { $0.label != $0.finalLabel }) {
            changed += 1
        }
        out[path] = pieces.map { piece in
            [
                "cls": piece.label, "confidence": Double(piece.confidence),
                "box": [piece.box.minX, piece.box.minY, piece.box.maxX, piece.box.maxY].map { Double($0) },
                "colour": piece.colour ?? NSNull(),
                "colour_side": piece.colourSide ?? NSNull(),
                "final": piece.finalLabel,
            ] as [String: Any]
        }
    }
    do {
        try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: coloursPath))
    } catch {
        fail("could not write \(coloursPath): \(error)")
    }
    print("\(out.count) photos; the colours changed a side on \(changed); written to \(coloursPath)")
    exit(0)
}

var photos = 0
var squaresRight = 0
var exact = 0
var exactTurned = 0  // right once the board is turned 180 degrees
var notFound = 0
// Pieces the app would mark for the user to check, for each threshold:
// [right, wrong]; and all pieces put on a square: [right, wrong].
var marked = [[Int]](repeating: [0, 0], count: flagBelow.count)
var placed = [0, 0]
for (path, placement) in list.sorted(by: { $0.key < $1.key }) {
    let url = path.hasPrefix("/")
        ? URL(fileURLWithPath: path)
        : listURL.deletingLastPathComponent().appending(path: path)
    guard
        var image = upright(url),
        let expected = FENBuilder.pieces(fromPlacement: placement)
    else { fail("could not load \(path) or its placement") }
    if portrait || redraw != nil {
        // The harness draws the photo itself: it draws the decoded copy, as
        // the recognizer would have (an undecoded JPEG drawn scaled gives
        // other pixels).
        image = DecodedImage.once(image)
    }
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
        for (square, piece) in result.pieces {
            // `squares` runs from a8 to h1.
            let index = (8 - square.rank.value) * 8 + square.file.number - 1
            let wrong = have[index] == want[index] ? 0 : 1
            placed[wrong] += 1
            for (k, threshold) in flagBelow.enumerated() where piece.confidence < threshold {
                marked[k][wrong] += 1
            }
        }
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
    \(photos) photos through the app's recognizer\(redraw.map { ", redrawn at \($0)" } ?? "")\(portrait ? ", as 3:4 photos" : "")\(a1PriorGiven ? ", a1 prior \(a1Prior.map { String($0) } ?? "off")" : "")\(gridFitGiven ? ", grid fit \(gridFit ? "on" : "off")" : "")\(sidesFromColourGiven ? ", sides from colour \(sidesFromColour ? "on" : "off")" : "")\(oneKing.map { ", one king a side \($0 ? "on" : "off")" } ?? "")
      squares right: \(percent(squaresRight, photos * 64))
      boards exact: \(exact) of \(photos) (\(percent(exact, photos)))
      boards exact once turned 180 degrees: \(exactTurned) of \(photos) (\(percent(exactTurned, photos)))
      board not found: \(notFound)
    """)
for (k, threshold) in flagBelow.enumerated() {
    print(String(
        format: "  marked for the user to check (confidence under %@): %d of %d wrong pieces (%@), "
            + "%d of %d right pieces (%@), %.2f a photo",
        "\(threshold)", marked[k][1], placed[1], percent(marked[k][1], placed[1]),
        marked[k][0], placed[0], percent(marked[k][0], placed[0]),
        Double(marked[k][0] + marked[k][1]) / Double(max(photos, 1))))
}
