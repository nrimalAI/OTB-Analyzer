# OTB Analyzer

Point your iPhone at an over-the-board chess position, digitize it, and analyze it with
Stockfish — fully on-device.

## Pipeline

```
Photo ──► CoreML board-corner + piece detection ──► FEN ──► Stockfish 17 ──► eval + best lines
                     (Phase 1/2)                    │
                                      tap-to-correct editor (always)
```

Recognition is never perfect (~1 in 8 boards has at least one wrong square even for
state-of-the-art pipelines), so the review/correct screen is a first-class part of the flow
and low-confidence squares are highlighted.

## Project layout

| Path | What |
|---|---|
| `App/` | SwiftUI app target (screens + navigation only) |
| `Packages/ChessVision` | Recognition protocol, FEN assembly, position validation. CoreML in Phase 2. |
| `Packages/ChessAnalysis` | Stockfish/UCI wrapper (`chesskit-engine`), score + SAN line models |
| `Packages/ChessBoardUI` | Reusable board view, piece editor, arrow overlay |
| `Training/` | Python: dataset prep, YOLO training, CoreML export (Phase 1, not shipped) |
| `Scripts/` | Setup scripts |

## Building

Requires Xcode 26+, [xcodegen](https://github.com/yonaskolb/XcodeGen).

```sh
./Scripts/download-nnue.sh    # fetch Stockfish NNUE networks (~70 MB, gitignored)
xcodegen generate
open ChessAnalyzer.xcodeproj
```

Package tests: `swift test` inside each `Packages/*` directory.

## Dependencies

- [chesskit-swift](https://github.com/chesskit-app/chesskit-swift) (MIT) — chess rules, FEN/SAN
- [chesskit-engine](https://github.com/chesskit-app/chesskit-engine) (MIT wrapper) — Stockfish 17 (GPL-3)

## License

GPL-3.0 (see `LICENSE`). The app links Stockfish, which is GPL-3; the app is therefore
distributed under GPL-3 as well.
