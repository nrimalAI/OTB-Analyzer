# Recognition in the app, after the two models

`Packages/ChessVision` turns a photo into a piece placement. The two CoreML
models give four board corners and a list of piece boxes; everything after
that is geometry, and `Training/chess_pipeline.py` is the reference the Swift
must match. This file says what the app-side steps are, how they are tested,
and what they were measured to be worth. Training results are in
`Training/README.md`.

## The steps (`CoreMLBoardRecognizer.recognize`)

1. The photo is letterboxed into a square (`Letterbox`) and given to both models.
2. Corner model: four corners in (a1, a8, h8, h1) order.
3. Piece model: boxes with a class and a confidence.
4. **a1 from the pieces** (`BoardOrientation`, port of `white_side_score` and
   `orient_by_pieces`): of the four rotations of the corners, the one whose
   board looks most like White at ranks 1-2 wins; the corner model's own
   rotation gets a head start of 6 (`BoardOrientation.appPrior`).
5. Each box's bottom centre goes through the corner homography to a square
   (`Homography`, `SquareAssignment`); one piece per square.

## Running the tests

    ./Scripts/swift-test.sh

Plain `swift test` where Xcode (16 or later) is selected. On a Mac with only
the Command Line Tools (the Mac mini) the script adds the search paths that
let `swift test` find Swift Testing there; that is why the tests use Swift
Testing and not XCTest, which the Command Line Tools do not have. CI runs the
same script (`.github/workflows/swift-test.yml`) and fails if the models are
missing or the end-to-end tests did not run.

## How the Swift is checked against the Python

- `Training/swift_reference.py orientation` writes
  `Tests/ChessVisionTests/Fixtures/orientation_cases.json`: 120 random boards
  in perspective, corners in a random rotation, pieces missed, misread, found
  twice or standing beside the board, with what the Python makes of each
  (the four scores, the rotation chosen, the placement). `BoardOrientationTests`
  replays them and requires the same answers. Run it again after changing
  `chess_pipeline.py`; a change in the JSON is a change the Swift must follow.
- `Training/app_check` runs the app's recognizer from the command line over a
  list of photos with known positions (`evaluate.py --dump-list` writes the
  lists) and prints what `evaluate.py` prints. It needs no Xcode.
  `--a1-prior P|off` sets step 4's prior.

## What step 4 is worth

Through `app_check` with the models of pull request #4 (exp3 corners),
2026-11-13, Mac mini:

| | a1 as the model says | prior 0 | prior 3 | **prior 6** |
|---|---|---|---|---|
| ChessReD test, 306 photos: boards exact | 272 | 270 | 271 | **272** |
| ChessReD test: squares right | 99.44% | 99.25% | 99.39% | **99.47%** |
| Fenit board B, 100 photos: squares right | 70.11% | 78.28% | 78.20% | **78.09%** |
| Fenit board B: boards exact | 0 | 0 | 0 | 0 |

On ChessReD, where the corner model mostly knows a1, a low prior lets a few
endgames overrule it wrongly; 6 loses nothing there. On board B, which the
corner model has never seen, the pieces decide whatever the prior. Board B has
no exact board because of the piece model, not the corners.

## Known weak spot: endgames that point the wrong way

Late endgames can have White's pieces on Black's side of the board. The last
photos of ChessReD game G033 are like that, and one of them is the test
fixture `test_004005.jpg` (`7Q/8/1k3p2/4p3/8/p3K3/8/8`): the right way up it
scores only +2.5, turned by 180 degrees -2.5. The exp3 and exp4 corner models
put a1 at the wrong end on these photos, and with the prior of 6 the pieces
cannot overrule them (3.5 against 2.5). The fixture passes with exp3 only
because its imprecise corners lose two of the six pieces, which makes the
score lopsided enough (7.75 against -1.75). With corners precise enough to
read all six, it would stay turned and the test would fail. A corner model
that gets a1 right on these photos, as `main`'s does, is the fix; the app's
flip button is the user's.
