# Recognition in the app, after the two models

`Packages/ChessVision` turns a photo into a piece placement. The two CoreML
models give four board corners and a list of piece boxes; everything after
that is geometry, and `Training/chess_pipeline.py` is the reference the Swift
must match. This file says what the app-side steps are, how they are tested,
and what they were measured to be worth. Training results are in
`Training/README.md`.

## The steps (`CoreMLBoardRecognizer.recognize`)

1. The photo is letterboxed into a square (`Letterbox`) and given to both models.
2. Corner model: four corners in (a1, a8, h8, h1) order. No board, no result.
3. **Corners from the grid** (`GridCorners`, port of `Training/grid_corners.py`
   `find_grid`): a classical fit of the 8×8 grid to the photo itself. Where it
   accepts its own result, its corners replace the model's, turned to the
   rotation nearest the model's (the fit cannot know which corner is a1).
   Where it rejects itself, the model's corners stay. It needs no training, so
   it works on boards no model has seen, where learned corners are a square
   or so off.
4. Piece model: boxes with a class and a confidence.
5. **a1 from the pieces** (`BoardOrientation`, port of `white_side_score` and
   `orient_by_pieces`): of the four rotations of the corners, the one whose
   board looks most like White at ranks 1-2 wins; the corner model's own
   rotation gets a head start of 6 (`BoardOrientation.appPrior`).
6. Each box's bottom centre goes through the corner homography to a square
   (`Homography`, `SquareAssignment`); one piece per square.

Steps 3 and 5 can be switched off (`CoreMLBoardRecognizer(gridFit:a1Prior:)`).

### The grid fit in a paragraph

X-corners (points where four squares meet) are found as saddle points that
pass a ring test; every small parallelogram of neighbouring x-corners proposes
a homography onto an integer lattice, and the one that puts most x-corners on
nodes wins; the photo is warped onto that lattice and the 8×8 window whose
cells alternate best is taken as the board; edge points along its 18 lines
refine the homography; then nine validity checks decide whether to believe it
(`GridCorners.confidence`). Without a hint of where the board is it runs
twice, the second time on the board it found, for more pixels per square. The
files under `Sources/ChessVision/GridCorners/` follow the Python function by
function, and each OpenCV call the Python makes has a Swift routine written to
give the same numbers (`GridImage`, `GridMath`).

Cost: 0.3-0.45 s a photo on the Mac mini (M4) in a release build, about 3.5 s
in a debug build (its loops are written to stay tolerable without
optimisation, but a debug build of the app will feel it). It holds the
photo's pixels in memory once, four bytes each (48 MB for 12 megapixels).

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
- `Training/swift_reference.py grid-cases` writes `Fixtures/grid_cases.json`
  and three drawn boards (`grid_board_*.png`): what `find_grid` makes of the
  drawn boards and of the four fixture photos, and what the OpenCV and NumPy
  calls it is built on make of small inputs (the shrink to the working size,
  the blurs, the saddle response, the dilation, the warp, the sub-pixel
  refinement, the homography fits, percentiles). `GridCornersTests` replays
  them. On the drawn boards, where both sides get the same pixels, the Swift
  corners are the Python's (0.0000 squares apart). On JPEG photos the two
  differ by up to a few hundredths of a square, because ImageIO and OpenCV's
  JPEG decoders differ by one grey level on 3-4% of pixels.
- `Training/app_check` runs the app's recognizer from the command line over a
  list of photos with known positions (`evaluate.py --dump-list` writes the
  lists) and prints what `evaluate.py` prints. It needs no Xcode.
  `--a1-prior P|off` and `--grid-fit on|off` switch the two steps.
- For any list of photos, Python against Swift:
  `swift_reference.py grid --list L.json --out py.json`,
  `app_check --list L.json --grid sw.json`,
  `swift_reference.py compare --python py.json --swift sw.json`.

### How close the Swift grid fit is to the Python

On the three test sets (2026-11-13), against the labelled corners:

| | photos | accepted, Swift / Python | median error, squares, Swift / Python | over half a square, Swift / Python |
|---|---|---|---|---|
| Fenit board B (unseen board) | 98 | 93% / 91% | 0.065 / 0.064 | 1 / 1 |
| ChessReD test | 306 | 94% / 93% | 0.027 / 0.027 | 0 / 0 |
| Fenit A | 270 | 95% / 96% | 0.108 / 0.108 | 1 / 0 |

Where both accept, their corners are a median 0.004-0.011 squares apart. They
disagree on whether to accept about one photo in twenty, in both directions
(34 of 692: 17 accepted by Python only, 17 by Swift only): the fit's decision
is that sensitive to the one-grey-level differences between the JPEG decoders.
Given the same x-corners the Swift lattice fit is identical to the Python's,
so this is the method, not the port. One Python quirk is kept on purpose so
that the two match: `choose_window` indexes a NumPy array with lattice
coordinates that can fall outside it (an IndexError that rejects the photo,
or, for small negative indices, a wrap to the other side); see
`GridWindow.choose`.

## What the two steps are worth

Through `app_check` (the app's own code, Vision doing the resize) on the Mac
mini, 2026-11-13. "Steps off" is `--a1-prior off --grid-fit off`, "both" is the
app's default (grid fit on, prior 6). Boards exact, and in brackets boards
that are exact once turned by 180 degrees (one tap in the app).

| Models | Steps | ChessReD test, 306 | Fenit board B, 100 (unseen) | Fenit A, 286 |
|---|---|---|---|---|
| `main` (yolo11n, 2026-10) | off | 229 = 74.8% | 0; 45.5% of squares; no board in 35 | 0; 72.8% of squares |
| `main` | both | **267 = 87.3%** (267) | 0 (1); 53.7% of squares; no board in 35 | 4 (5); 89.1% of squares |
| exp3 corners (pull request #4) | off | 272 = 88.9% (273) | 0; 70.1% of squares | 21; 95.1% of squares |
| exp3 | a1 from the pieces only | 272 (273) | 0 (1); 78.1% of squares | 20 (21); 93.5% of squares |
| exp3 | both | **282 = 92.2%** (285) | 0 (1); **86.3%** of squares | 20 (21); 93.5% of squares |
| exp4 corners | both | 281 = 91.8% (284) | 0 (1); 86.5% of squares | 20 (21); 93.5% of squares |

What this says:

- The grid fit is worth 10 boards in 306 on ChessReD with exp3's corners and
  38 with `main`'s, whose corners are less precise; on the unseen board it
  adds 8 points of squares. With `main`'s models and both steps all four test
  fixtures are read 64/64.
- No board B photo is fully right with any of these: the piece model misreads
  types and colours there (`Training/README.md`). The corners no longer hold
  it back.
- `main`'s corner model finds no board in 35 of the 100 board B photos, and
  then nothing is read: the grid fit is only used once the model has found a
  board, as in `evaluate.py --grid-corners`. The fit alone accepts 93 of the 100.
  Using it when the model finds nothing (a1 from the pieces alone) would be a
  change to the reference first.
- a1 from the pieces helps where the corner model does not know a1 (board B:
  +8 points of squares) and costs a little where it does (Fenit A, a board
  the model was trained on: -1.6 points; ChessReD: nothing). With the grid
  fit on, priors 0 / 3 / 6 give 280 / 282 / 282 boards on ChessReD, 86.4 /
  86.4 / 86.3% of squares on board B and 92.7 / 93.1 / 93.5% on Fenit A, so 6
  stays.

## Known weak spot: endgames that point the wrong way

Late endgames can have White's pieces on Black's side of the board. The last
photos of ChessReD game G033 are like that, and one of them is the test
fixture `test_004005.jpg` (`7Q/8/1k3p2/4p3/8/p3K3/8/8`): the right way up it
scores only +2.5, turned by 180 degrees -2.5. The exp3 and exp4 corner models
put a1 at the wrong end on these photos, and with the prior of 6 the pieces
cannot overrule them (3.5 against 2.5). So with those corner models and the
grid fit the fixture is read perfectly but turned (52/64), and
`EndToEndFixtureTests` fails; they are the three boards "exact once turned"
in the table. Without the grid fit exp3 passed it only by accident: its
imprecise corners lost two of the six pieces, which made the pieces' vote
lopsided enough (7.75 against -1.75). `main`'s corner model gets a1 right
there. A corner model that knows a1 on these photos is the fix; the app's
flip button is the user's.
