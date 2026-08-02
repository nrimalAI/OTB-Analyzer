# Chess Analyzer — native iOS app: photo of a physical board → FEN → Stockfish analysis

## Context

There is no code yet — `~/Desktop/chessAnalyzer` is an empty directory. The goal is a simple
iPhone app: point the camera at an over-the-board position, get the position digitized, and
get Stockfish's evaluation and best line.

Three decisions are already made:

| Decision | Choice |
|---|---|
| Recognition | **On-device CoreML** (fully offline, no API, no backend) |
| Platform | **iPhone only**, SwiftUI |
| Distribution | **App Store eventually** |

Toolchain present: Xcode 26.3, Swift 6.2, `xcodegen` at `/usr/local/bin/xcodegen`,
iOS 17.5 and 26.3 simulators.

### Two things to know before we start

**1. Recognition will not be perfect, and the plan is built around that.** Published results
on this exact problem:

- chesscog (Wölflein & Arandjelović, 2021): **88.9%** of boards fully correct on unseen
  images; per-square error 0.17%.
- ChessReD end-to-end (Masouris & van Gemert, VISAPP 2024): **15.3%** of boards fully
  correct — described by the authors as ~7× the prior state of the art.

A 0.2% per-square error rate still means roughly **1 in 8 boards has at least one wrong
square**. So the tap-to-correct editor is not a nicety bolted on the end — it is a
first-class screen, and per-square confidence is surfaced in the UI so the user's eye goes
straight to the squares the model was unsure about. Phase 0 below ships that editor
*before* any ML exists, which also means we have a working, useful app early.

**2. Licensing needs a decision from you, because of App Store intent.** Details and
options are in the Licensing section at the end. Short version: Stockfish is GPL-3, which
means the app itself must be GPL-3 (Lichess does exactly this and ships on the App Store),
and the best training dataset is NonCommercial so it can be used for prototyping but not for
a shipped build. Neither blocks Phase 0–2; both need resolving before submission.

---

## Architecture

Three local Swift packages plus a thin app target, so the hard logic is unit-testable
without launching a simulator.

```
chessAnalyzer/
  project.yml                  # xcodegen — regenerates ChessAnalyzer.xcodeproj
  App/                         # SwiftUI app target (thin; screens + navigation only)
  Packages/
    ChessVision/               # CoreML + Vision + geometry + FEN assembly. No UI.
    ChessAnalysis/             # Stockfish/UCI wrapper. No UI.
    ChessBoardUI/              # Reusable SwiftUI board view + editor. No CoreML, no engine.
  Models/                      # .mlpackage files (git-lfs)
  Training/                    # Python. Dataset prep, training, CoreML export. Not shipped.
```

### Recognition pipeline (`ChessVision`)

Rejected the "warp to top-down, then classify 64 crops" approach: on a **3D** board pieces
occlude their neighbours and lean into adjacent squares after warping. Instead, detect
pieces in the original perspective image and use each piece's *contact point with the board*
to assign it a square. This is the approach the stronger recent projects converge on.

```
UIImage
  │
  ├─► BoardCorners.mlpackage   YOLO-pose, 1 class "board", 4 keypoints (a1,a8,h8,h1)
  │        └─► 4 image-space corner points
  │
  └─► Pieces.mlpackage         YOLO-detect, 12 classes (wK wQ wR wB wN wP bK…bP)
           └─► [bbox, class, confidence]

  homography H: 4 detected corners → unit 8×8 grid
  for each piece: bottomCenter(bbox) ──H──► (file, rank)
  conflict resolution: two pieces in one square → keep higher confidence, demote the other
                       to its next-nearest square if empty, else drop
  ─► BoardState { [Square: Piece?], perSquareConfidence: [Square: Float] }
```

Both models: 640px input, exported as `.mlpackage`, run through `VNCoreMLRequest` with
`MLModelConfiguration.computeUnits = .cpuAndNeuralEngine` (not `.all` — avoids GPU
contention if we later add live camera preview).

Key types:

```swift
public protocol BoardRecognizer: Sendable {           // lets Phase 0 ship a stub
    func recognize(_ image: CGImage) async throws -> RecognitionResult
}
public struct RecognitionResult {
    public let pieces: [Square: (piece: Piece, confidence: Float)]
    public let corners: [CGPoint]        // for the "detected board" overlay
    public let orientationGuess: BoardOrientation
}
```

### What a photo cannot tell you

A FEN is `placement w KQkq - 0 1`. Only the first field comes from the image. The rest:

| Field | How it's handled |
|---|---|
| Side to move | **Asked in the UI.** A two-button segmented control on the review screen. No default is safe. |
| Castling rights | Inferred from placement (K and R on home squares → assume available), shown as four toggles the user can clear. |
| En passant | Defaults to `-`. Effectively never recoverable from a still photo; a rarely-used advanced field. |
| Halfmove / fullmove | `0 1`. Irrelevant to evaluation except for 50-move edge cases. |
| Board orientation | Model guesses from piece distribution; user can flip with one tap. |

### Position validation

Before enabling **Analyze**, validate with `chesskit-swift`'s `Position`: exactly one king
per side, no pawns on ranks 1/8, ≤8 pawns per side, side-not-to-move not already in check.
Invalid → highlight the offending squares in the editor rather than showing an error alert.
This catches most recognition errors automatically.

### Analysis (`ChessAnalysis`)

`chesskit-engine` (MIT wrapper, Stockfish 17 GPL-3, Swift 6 concurrency, async/await).
Stockfish 17 requires both NNUE files present in `Bundle.main` and set via UCI `setoption`
after launch: `nn-1111cefa1111.nnue` (~64 MB) and `nn-37f18f62d772.nnue` (~7 MB). Bundle
them; ~75 MB is acceptable for the simplicity, and moving to on-demand resources later is a
localized change.

```swift
public actor AnalysisEngine {
    public func start() async throws
    public func analyze(fen: String, depth: Int, multiPV: Int) -> AsyncStream<AnalysisUpdate>
    public func stop() async
}
public struct AnalysisUpdate {
    public let depth: Int
    public let score: Score          // .centipawns(Int) | .mate(Int)
    public let lines: [PrincipalVariation]   // SAN, via chesskit-swift
}
```

`multiPV: 3` by default. Convert engine UCI/LAN moves to SAN with `chesskit-swift` before
display.

### Screens

1. **Capture** — `AVCaptureSession` still capture with a board-shaped guide overlay, plus a
   `PhotosPicker` import path (the simulator has no camera, so this is also the dev path).
2. **Review & Correct** — the recognized position on a 2D board. Squares the model was
   unsure about get a subtle confidence tint. Tap a square → piece picker sheet. Side-to-move
   control, orientation flip, castling toggles behind a disclosure. Validation warnings inline.
3. **Analysis** — eval bar, top 3 lines in SAN, arrow overlay for the best move on the board,
   depth slider, and a "back to edit" affordance.

---

## Phases

### Phase 0 — Working app with no ML (do this first)

Everything except recognition. Ships a genuinely usable app and de-risks the engine
integration before ML work starts.

- `project.yml` for xcodegen; app target, iOS 17.0 deployment target, iPhone-only,
  portrait. Verify `chesskit-engine`'s minimum iOS version and raise if needed.
- `Packages/ChessBoardUI` — board view, drag-from-tray piece editor, arrow overlay.
- `Packages/ChessAnalysis` — `chesskit-engine` + `chesskit-swift`; download the two `.nnue`
  files into `App/Resources/`, wire the `setoption` calls, verify the engine returns a
  sane eval for a known position.
- `ChessVision` ships only the `BoardRecognizer` protocol plus `StubRecognizer` (returns the
  starting position) so the full navigation flow is exercisable end to end.
- Capture screen with `PhotosPicker` only; camera added at the end of this phase.

**Done when:** you can hand-place a position on the phone and get Stockfish's evaluation and
best line.

### Phase 1 — Train the models (Python, `Training/`)

Runs entirely outside the app. Ultralytics YOLO (v11 or v26 — v26 is NMS-free, which
simplifies the Swift side; confirm pose-task support before committing).

1. **Data.** Prototype on **ChessReD2K** — the annotated subset of ChessReD with the
   annotations we actually need: piece bounding boxes *and* board corners, for 2,078 images
   across 20 games (1,442 train / 330 val / 306 test), split by game so no position leaks
   across splits. The other ~8,700 ChessReD images have algebraic-notation labels only (no
   boxes), so they are usable for end-to-end evaluation but not for detector training.
   Supplement with the CC BY 4.0 chess datasets on Roboflow Universe and with your own
   photos. `Training/prepare_data.py` converts ChessReD's `annotations.json`
   (`images` / `annotations.pieces` / `categories`) into YOLO label format.
2. **Two models.**
   - `board_corners` — pose task, 1 class, 4 keypoints. Corners are the higher-leverage
     model: a corner error shifts *every* piece. Aim for high precision here first.
   - `pieces` — detect task, 12 classes.
3. **Augmentation** matters more than architecture here: perspective warp, lighting/white
   balance, motion blur, board-material and piece-set variation.
4. **Metrics that reflect the product**, not just mAP: per-square accuracy and
   **full-board exact-match rate** on a held-out set of *your own* phone photos. Target
   ≥99.5% per-square. Treat chesscog's 88.9% board-level figure as the bar to approach.
5. **Export.** `model.export(format="coreml", imgsz=640, quantize=8)` → `.mlpackage`.
   Note Ultralytics CoreML export/validation requires macOS — fine here. Measure the INT8
   vs FP16 accuracy delta and pick deliberately; don't assume INT8 is free.

**Done when:** held-out full-board exact-match is good enough that correction is occasional
rather than routine, and per-image latency on-device is under ~1s.

### Phase 2 — Wire CoreML into the app

- Drop the two `.mlpackage` files into `Models/` (git-lfs), reference from `project.yml`.
- Implement `CoreMLBoardRecognizer: BoardRecognizer` — Vision requests, homography
  (`vImage`/Accelerate or a small hand-rolled 4-point DLT solve), square assignment,
  conflict resolution.
- Feed `perSquareConfidence` into the review screen's tint.
- Replace `StubRecognizer` at the composition root. Nothing else in the app changes — that
  is the point of the protocol.

### Phase 3 — Accuracy and polish

- Collect your own photos in varied conditions; failures here are the training signal.
- Orientation-guess refinement, corner-detection failure UX ("couldn't find the board —
  place pieces manually?"), and a smaller-model / latency pass if needed.

---

## Critical files

| Path | Purpose |
|---|---|
| `project.yml` | xcodegen spec — the single source of project structure |
| `Packages/ChessVision/Sources/ChessVision/BoardRecognizer.swift` | Protocol + result types; the Phase 0 ↔ Phase 2 seam |
| `Packages/ChessVision/Sources/ChessVision/Homography.swift` | 4-point DLT solve, image point → square |
| `Packages/ChessVision/Sources/ChessVision/FENBuilder.swift` | Placement + UI-supplied fields → FEN; validation |
| `Packages/ChessAnalysis/Sources/ChessAnalysis/AnalysisEngine.swift` | Stockfish actor, NNUE setup, UCI ↔ SAN |
| `Packages/ChessBoardUI/Sources/ChessBoardUI/BoardView.swift` | Board, editor, confidence tint, arrows |
| `Training/prepare_data.py` | ChessReD/Roboflow → YOLO format |
| `Training/export_coreml.py` | Export + quantization comparison |

### Reuse rather than rebuild

- **`chesskit-swift`** (github.com/chesskit-app/chesskit-swift) — `Position`, `Board`,
  `Move`, `Square`, `Piece`; bitboard legal-move generation; FEN/SAN/PGN parsing and
  serialization. Do not hand-roll FEN parsing, legality checks, or UCI→SAN conversion.
- **`chesskit-engine`** (github.com/chesskit-app/chesskit-engine) — Stockfish 17 packaged
  for SPM with a Swift 6 async response stream. Do not vendor Stockfish sources directly.
- **Vision** `VNCoreMLRequest` / `VNImageRequestHandler` — handles orientation, scaling, and
  Neural Engine scheduling. Do not call CoreML directly with hand-built `MLMultiArray`s.
- **Ultralytics** one-line CoreML export — do not write a manual coremltools conversion.

---

## Verification

**Phase 0**
- `swift test` in each package: FEN round-trips, castling inference, validation rules
  (9 pawns, missing king, pawn on rank 8) all rejected.
- `xcodegen generate && xcodebuild -scheme ChessAnalyzer -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build`
- Boot the sim, place the Ruy Lopez by hand, analyze. Expect eval near 0.0 and a
  plausible main line — a wildly wrong eval means the NNUE files did not load.
- Golden test: known FEN → engine → assert best move for a position with a forced mate.

**Phase 1**
- `Training/evaluate.py` reports per-square accuracy and full-board exact-match on the
  ChessReD2K test games *and* on your own photo set separately. The gap between those two
  numbers is the real generalization signal.
- Confusion matrix per piece class — bishop/pawn and knight/bishop confusions at a distance
  are the expected failure modes.

**Phase 2**
- Fixture-based `ChessVision` tests: bundle ~20 photos with hand-verified FENs, assert the
  recognizer's output. These run on every build and catch regressions from model swaps.
- On-device timing with Instruments (Core ML template) — confirm Neural Engine residency
  and sub-second inference.
- End-to-end on a real iPhone: photograph a real board, correct any errors, analyze.

---

## Licensing — needs your decision before submission

Not a blocker for Phases 0–2, but it shapes what "App Store eventually" can mean.

1. **Stockfish is GPL-3.** Statically linking it means the app as a whole must be
   distributed under GPL-3, i.e. **open source**. Lichess ships GPL-3 Stockfish on the App
   Store, so the frequently-cited "GPL is incompatible with the App Store" concern has a
   working counter-example. If the app must stay proprietary, Stockfish has to go — and the
   permissively-licensed alternatives are substantially weaker.
2. **Ultralytics YOLO is AGPL-3**, and Ultralytics' position is that models trained with
   their code inherit it. Consistent with (1) if the app is GPL-3. If you need a permissive
   path, RF-DETR (Apache-2.0) or a torchvision detector (BSD) are the alternatives, at the
   cost of a rougher CoreML export path.
3. **ChessReD is CC BY-NC-SA 4.0 — NonCommercial.** This is the sharp one. It is the best
   dataset for this task and fine for prototyping and evaluation, but a model trained on it
   should not ship in a commercially distributed app. **Mitigation:** use ChessReD to prove
   the pipeline in Phase 1, then retrain the shipping model on CC BY 4.0 Roboflow Universe
   datasets plus your own captured/rendered data before submitting. Budget for this — it is
   a real chunk of Phase 3, and it is why "collect your own photos" appears there.

My recommendation: **GPL-3 the app** (resolves 1 and 2 cleanly, matches Lichess precedent)
and **plan the Phase 3 retrain on owned + CC BY data** (resolves 3). If you'd rather keep it
proprietary, say so now — it changes the engine and the detector architecture, not just the
LICENSE file.
