# Recognition in the app, after the two models

`Packages/ChessVision` turns a photo into a piece placement. The two CoreML
models give four board corners and a list of piece boxes; everything after
that is geometry, and `Training/chess_pipeline.py` is the reference the Swift
must match. This file says what the app-side steps are, how they are tested,
and what they were measured to be worth. Training results are in
`Training/README.md`.

## The steps (`CoreMLBoardRecognizer.recognize`)

1. The photo is letterboxed into a square (`Letterbox`) and given to both models.
2. Corner model: four corners in (a1, a8, h8, h1) order, or no board.
3. **Corners from the grid** (`GridCorners`, port of `Training/grid_corners.py`
   `find_grid`): a classical fit of the 8×8 grid to the photo itself. Where it
   accepts its own result, its corners replace the model's, turned to the
   rotation nearest the model's (the fit cannot know which corner is a1).
   Where it rejects itself, the model's corners stay. Where the model found
   no board and the fit accepts, the fit's corners are used alone. Only when
   both fail is there no board. The fit needs no training, so it works on
   boards no model has seen, where learned corners are a square or so off.
4. Piece model: boxes with a class and a confidence.
5. **White and Black from the colours** (`PieceSides`, port of
   `sides_from_colour` and of `evaluate.piece_colours`): the colour of each
   piece is the median CIE Lab of the middle of its box in the photo; the
   pieces are split in two groups by colour (2-means on L*, 4a*, 4b*) and the
   lighter group is White. A piece model that has not seen a piece set (brown
   "white" pieces) can get the kinds right and the sides wrong, while the
   pieces are still two groups of one colour each. The split can be wrong for
   a few pieces (a shadow, a glint), so the model's sides stand unless the
   colours contradict them for more than 15% of the pieces that have a colour
   (a box whose middle lies outside the photo has none); then each of those
   takes its group's side. Kinds and confidences are never changed.
6. **a1 from the pieces** (`BoardOrientation`, port of `white_side_score` and
   `orient_by_pieces`): of the four rotations of the corners, the one whose
   board looks most like White at ranks 1-2 wins; the corner model's own
   rotation gets a head start of 6 (`BoardOrientation.appPrior`). When the
   corners are the fit's alone there is no model opinion, and the pieces
   decide without a head start.
7. Each box's bottom centre goes through the corner homography to a square
   (`Homography`, `SquareAssignment`); one piece per square.

Steps 3, 5 and 6 can be switched off
(`CoreMLBoardRecognizer(a1Prior:gridFit:sidesFromColour:)`).
How the two sets of corners combine is `CoreMLBoardRecognizer.corners(model:in:gridFit:)`,
and it is what `Training/evaluate.py --grid-corners` does.

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

`EndToEndFixtureTests` reads four ChessReD test photos with the models in
`Models/`, twice: as the app does, and with the grid fit switched off. The
second run is the corner model's own test. With the fit on, a corner model
that found no board at all would go unnoticed, because the fit reads these
photos by itself.

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
  refinement, the homography fits, percentiles), six drawn pictures and ten
  sets of x-corners for the lattice fit. `GridCornersTests` replays them. On
  the five drawn pictures with a board (bare, with pieces, partly outside the
  picture, with other chequered things beside it, and a 10×10 board that the
  validity checks refuse), where both sides get the same pixels, the Swift
  gives the Python's corners to a thousandth of a square, the same confidence
  and the same counts at every stage; the lattice fit gives the same nodes
  and homography on all ten sets; the shrink to the working size is OpenCV's
  pixel for pixel.
- Two things keep the two from being identical on every picture, and neither
  can be ported away. On JPEG photos, ImageIO and OpenCV's JPEG decoders
  differ by one grey level on 3-4% of pixels. And on any picture, points that
  are not x-corners (corners of clutter, piece outlines) do not settle in the
  sub-pixel step (`cornerSubPix`): after its 30 rounds they are still moving,
  and where they stop hangs on the last bit of the arithmetic, which OpenCV's
  compiled code and Swift do not share. Such a stray is then kept, moved or
  dropped differently, and one stray x-corner can change which lattice wins.
  The picture with no board among the drawn ones shows it: Python counts 23
  x-corners, Swift 25, and both find no grid.
- `Training/swift_reference.py colour-cases` writes `Fixtures/colour_cases.json`
  and a drawn picture (`colour_pieces.png`): 200 colours in Lab as OpenCV
  gives them, 120 boxes with their patches (in, partly in and beside the
  photo), 60 sets of points split in two (with ties), 74 sets of detections
  with their colours (the model wrong about the side of none, a few, exactly
  15%, or all of the pieces), the drawn picture's brown and black pieces that
  a model calls all black, and the colours of 24 boxes on a fixture photo.
  `PieceSidesTests` replays them: patches, groups and sides must be the
  Python's exactly.
- The colours themselves are the Python's only to within 0.2 in L*, 0.5 in a*
  and 0.35 in b*, on the same pixels. `PieceSides.lab` is the CIE formula
  (with OpenCV's matrix and white point); OpenCV does not compute the formula
  for float pixels, it interpolates in a 33×33×33 table (measured over all
  2^24 colours; L* comes out 0.09 too low on average). On the median of a
  patch of a photo that is about as much as the two JPEG decoders add (0.09
  against 0.04 on average, 0.15 against 0.43 at most, on the 24 patches of a
  fixture). The two groups are usually tens of units apart, so the table was
  not copied; a piece that lies between the two groups can fall the other
  way (2 pieces of 23,523 on the test sets, below).
- `Training/app_check` runs the app's recognizer from the command line over a
  list of photos with known positions (`evaluate.py --dump-list` writes the
  lists) and prints what `evaluate.py` prints. It needs no Xcode.
  `--a1-prior P|off`, `--grid-fit on|off` and `--sides-from-colour on|off`
  switch the steps.
- For any list of photos, Python against Swift:
  `swift_reference.py grid --list L.json --out py.json`,
  `app_check --list L.json --grid sw.json`,
  `swift_reference.py compare --python py.json --swift sw.json`.
  For the colour step, on the boxes the app's piece model finds:
  `app_check --models M --list L.json --colours sw.json`,
  `swift_reference.py colours --swift sw.json --list L.json`.

### How close the Swift grid fit is to the Python

On the three test sets (2026-11-13), against the labelled corners:

| | photos | accepted, Swift / Python | median error, squares, Swift / Python | over half a square, Swift / Python |
|---|---|---|---|---|
| Fenit board B (unseen board) | 98 | 93% / 91% | 0.065 / 0.064 | 1 / 1 |
| ChessReD test | 306 | 94% / 93% | 0.027 / 0.027 | 0 / 0 |
| Fenit A | 270 | 95% / 96% | 0.108 / 0.108 | 1 / 0 |

Where both accept, their corners are a median 0.004-0.011 squares apart. They
disagree on whether to accept about one photo in twenty, in both directions
(34 of 692: 17 accepted by Python only, 17 by Swift only), for the two
reasons above. The fit is that sensitive by itself: Python against Python
with one grey level added to 3.5% of pixels flips the answer on 2 of 100
board B photos and changes the number of x-corners on 72 (measured by the
reviewer of pull request #7). Given the same x-corners the Swift lattice fit
is identical to the Python's, so this is the method, not the port; dropping
x-corners that did not converge is something to try in the Python first. One
Python quirk is kept on purpose so that the two match: `choose_window`
indexes a NumPy array with lattice coordinates that can fall outside it (an
IndexError that rejects the photo, or, for small negative indices, a wrap to
the other side); see `GridWindow.choose`.

## What the two steps are worth

Through `app_check` (the app's own code, Vision doing the resize) on the Mac
mini, 2026-11-13. "Off" is `--a1-prior off --grid-fit off`, "both" is the
app's default (grid fit on, prior 6). Boards exact, and in brackets boards
that are exact once turned by 180 degrees (one tap in the app).

**The models in `Models/` (b546e6a: `main`'s corner model, the baseline piece
weights exported with NMS across classes):**

| Steps | ChessReD test, 306 | Fenit board B, 100 (unseen board) | Fenit A, 286 |
|---|---|---|---|
| off | 238 = 77.8% | 0; 46.6% of squares; no board in 35 | 0; 73.6% of squares |
| a1 from the pieces only | 238 | 0; 46.6% of squares; no board in 35 | 0; 74.0% of squares |
| **both** | **280 = 91.5%** (280) | 0 (1); **82.3%** of squares; no board in 6 | 15 (17); **91.4%** of squares |

**Other corner models, for comparison** (rows marked * were scored before the
fit was used alone; that changes nothing for exp3 and exp4, which find every
board):

| Models | Steps | ChessReD test, 306 | Fenit board B, 100 | Fenit A, 286 |
|---|---|---|---|---|
| `main`'s two models as on `main` | off | 229 = 74.8% | 0; 45.5% of squares; no board in 35 | 0; 72.8% of squares |
| `main`'s two models * | both, fit not used alone | 267 = 87.3% (267) | 0 (1); 53.7% of squares; no board in 35 | 4 (5); 89.1% of squares |
| exp3 corners | off | 272 = 88.9% (273) | 0; 70.1% of squares | 21; 95.1% of squares |
| exp3 corners | a1 from the pieces only | 272 (273) | 0 (1); 78.1% of squares | 20 (21); 93.5% of squares |
| exp3 corners * | both | 282 = 92.2% (285) | 0 (1); 86.3% of squares | 20 (21); 93.5% of squares |
| exp4 corners * | both | 281 = 91.8% (284) | 0 (1); 86.5% of squares | 20 (21); 93.5% of squares |

What this says:

- With the models in `Models/`, the two steps take ChessReD from 238 to 280
  boards exact and the unseen board from 47% to 82% of squares. Most of that
  is the grid fit: exact corners where the model's were imprecise, and, on
  board B, a board at all in 29 of the 35 photos where the corner model found
  none.
- No board B photo is fully right with any of these: the piece model misreads
  types and colours there (`Training/README.md`). The corners no longer hold
  it back.
- a1 from the pieces helps where the corner model does not know a1 (exp3 on
  board B: +8 points of squares) and costs a little where it does (exp3 on
  Fenit A, a board it was trained on: -1.6 points; ChessReD: nothing). With
  the grid fit on and exp3's corners, priors 0 / 3 / 6 give 280 / 282 / 282
  boards on ChessReD, 86.4 / 86.4 / 86.3% of squares on board B and 92.7 /
  93.1 / 93.5% on Fenit A, so 6 stays.

## What the colour step is worth

Through `app_check` on the Mac mini, 2026-11-13, grid fit on and prior 6,
`--sides-from-colour off` against `on`. Boards exact (in brackets: once
turned by 180 degrees) and squares right.

| Piece model | Step | ChessReD test, 306 | Fenit board B, 100 (unseen board) | Fenit A, 286 |
|---|---|---|---|---|
| in `Models/` | off | 280 (280); 99.53% | 0 (1); 82.28% | 15 (17); 91.44% |
| in `Models/` | **on** | 280 (280); 99.53% | **2 (3); 83.81%** | 15 (17); 91.44% |
| `pieces_s_render1` (not in `Models/`) | off | 286 (286); 99.64% | 1 (1); 82.47% | 184 (200); 95.34% |
| `pieces_s_render1` (not in `Models/`) | **on** | 286 (286); 99.64% | **11 (14); 87.08%** | 184 (200); 95.34% |

(`pieces_s_render1` is the training session's piece model trained with
rendered boards, exported for this measurement only, with the corner model in
`Models/`; Fenit A is mostly photos it was trained on.)

- The step is for a piece model that knows the kinds on a new piece set and
  not the sides: with `pieces_s_render1`, board B goes from 1 to 11 boards
  exact. The Python gets 15 there end to end (`Training/README.md`); it reads
  clean 1280-px copies with its own grid fit, the app the photos through
  Vision.
- It costs nothing on the boards the models know: no ChessReD test photo
  changes with either model. That is why it is on by default.
- With the model in `Models/` it changes a side on 38 board B photos (+1.5
  points of squares), on 4 Fenit A photos and on no ChessReD photo.
- Weak spot, the reference's own: there is no smallest distance between the
  two groups, and with six pieces or fewer one disagreement is already over
  15%. The four Fenit A photos are such positions (2, 4, 6 and 9 pieces
  found): on one, two black pieces are split into a "white" and a black one.
  The boards were wrong before and are as wrong after, so the totals do not
  move, but a sparse position whose pieces the model had right could be made
  worse.
- On the same boxes the Swift and the Python agree on the side of every
  piece with the model in `Models/`: 11,749 pieces on the 692 photos of the
  three sets, the step changing a side on the same photos in both (38, 4, 0).
  On `pieces_s_render1`'s boxes they agree on 11,772 of 11,774: one board B
  photo with four near-black pieces and one Fenit A photo each have a piece
  about as far from one group as from the other (54.3 against 55.6 on the
  second), and it falls the other way; on board B that changes the class of
  one piece. Both are OpenCV's Lab table: with the formula on OpenCV's own
  pixels the Python gives the Swift's groups. The JPEG decoders changed no
  side on any of the 1,384 readings.
- Cost: 0.07 s a photo with 32 pieces in a release build, about 1 s in a
  debug build; like the grid fit it reads the photo's pixels once more (the
  rectangle that holds all the pieces).

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
there, which is why it is the one in `Models/` (b546e6a): a new corner model
has to pass this test, and the app's flip button is the user's.
