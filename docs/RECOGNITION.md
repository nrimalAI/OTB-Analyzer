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
4. Piece model: boxes, each with the model's score for every class. The
   class with the highest score is what the piece is taken for, and that
   score is its confidence (see "The scores of a piece" below).
5. **White and Black from the colours, switched off** (`PieceSides`, port of
   `sides_from_colour` and of `evaluate.piece_colours`; see "What the colour
   step is worth" for why it is off): the colour of each
   piece is the median CIE Lab of the middle of its box in the photo; the
   pieces are split in two groups by colour (2-means on L*, 4a*, 4b*) and the
   lighter group is White. A piece model that has not seen a piece set (brown
   "white" pieces) can get the kinds right and the sides wrong, while the
   pieces are still two groups of one colour each. The split can be wrong for
   a few pieces (a shadow, a glint), so the model's sides stand unless the
   colours contradict them for more than 15% of the pieces that have a colour
   (a box whose middle lies outside the photo has none); then each of those
   takes its group's side. Kinds and confidences are never changed.
6. **One king a side** (`OneKingASide`, port of `one_king_a_side`): every
   position has exactly two kings, and a king read as a queen, or a third
   "king", is what a piece model gets wrong on a set it has not seen. A
   board with exactly two kings is left alone, also when both are on one
   side (that is a wrong side, not a wrong kind). With fewer than two, a
   side that has pieces and no king gets one: its piece with the highest
   score for "king", if the model gave it at least 0.02. With more than two,
   a side with several kings keeps the one with the highest score for "king"
   and each of the others becomes its best-scoring other kind, a queen when
   it has none. Sides, confidences and boxes are never changed.
7. **a1 from the pieces** (`BoardOrientation`, port of `white_side_score` and
   `orient_by_pieces`): of the four rotations of the corners, the one whose
   board looks most like White at ranks 1-2 wins; the corner model's own
   rotation gets a head start of 6 (`BoardOrientation.appPrior`). When the
   corners are the fit's alone there is no model opinion, and the pieces
   decide without a head start.
8. Each box's bottom centre goes through the corner homography to a square
   (`Homography`, `SquareAssignment`); one piece per square. A foot that
   falls up to a quarter of a square outside the board is taken back onto
   the rim square (`SquareAssignment.edgeMargin`, the reference's
   `EDGE_MARGIN`); further out it is a captured or spare piece standing
   beside the board and gets no square.

Steps 3, 6 and 7 can be switched off and step 5 on
(`CoreMLBoardRecognizer(a1Prior:gridFit:sidesFromColour:oneKing:)`).
How the two sets of corners combine is `CoreMLBoardRecognizer.corners(model:in:gridFit:)`,
and it is what `Training/evaluate.py --grid-corners` does.

### The scores of a piece

The piece model scores every class for every box. Vision passes those scores
on in a form of its own: the labels of an observation carry the scores
divided by their sum, so that they add up to 1, and the observation's own
confidence is the sum. A label's confidence times the observation's is the
score as the model gave it (`CoreMLBoardRecognizer.classScores`). Nothing in
Vision's documentation says so; `EndToEndFixtureTests.testPieceScoresAreTheModelsOwn`
holds it against the model run without Vision on the same pixels, and finds
the two equal to four decimals for every piece of the four fixtures.

Two things are read from the scores:

- **The confidence of a piece is the model's score for the class it is taken
  for**, as in the Python. Until pull request #12 it was Vision's label
  confidence, the share: about 1.0 for nearly every box, also for one the
  model gave 0.3, since few boxes have a second class with any score at all.
  The confidence decides which of two pieces on one square stays, and it is
  what the app shows: a piece under 0.85 is marked for the user to check
  (`AppModel.lowConfidenceThreshold`). See "What the confidence marks".
- **The score for each kind** (`PieceDetection.kindScores`): the larger of
  the two colours, for step 6. A score under 0.02 is left out: the Python
  has none under 0.02 either (it makes the scores of a piece up from every
  box down to that confidence that overlaps it, `evaluate.kind_scores_of`,
  because ultralytics does not hand out the scores of one box), and the rule
  was measured that way.

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
`Models/`, three times: as the app does, with the grid fit switched off, and
with the colour step switched on. The second run is the corner model's own
test: each of its corners within one square of the grid fit's, and 94% of
the squares of the four photos together. It has no floor of squares a photo,
because with corners most of a square off the count hangs on single pieces
standing on a line (58 or 60 of 64 on one fixture, from the same pixels,
depending on whether CoreML runs on the Neural Engine or the CPU). With the fit on, a corner model
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
  and a drawn picture (`colour_pieces.png`): 200 colours in Lab
  (`evaluate.srgb_to_lab`), 120 boxes with their patches (in, partly in and
  beside the photo), 60 sets of points split in two (with ties), 77 sets of
  detections with their colours (the model wrong about the side of none, a
  few, exactly 15%, or all of the pieces; pieces without a colour; equally
  light groups), each with the sides by colour alone (`colour_sides`) and the
  final classes, the drawn picture's brown and black pieces that a model
  calls all black, and the colours of 24 boxes on a fixture photo.
  `PieceSidesTests` replays them: patches, groups and sides must be the
  Python's exactly, and the colours to nine decimals where both read the same
  pixels (the drawn picture).
- `Training/swift_reference.py king-cases` writes `Fixtures/king_cases.json`:
  275 sets of detections with the scores of each piece for every kind, and
  what `one_king_a_side` makes of them: boards with no, one, two, three and
  four kings, on one side or both, scores on the floor and just under it,
  equal scores, kinds without a score, and each sentence of the rule by
  hand. `OneKingASideTests` replays them and requires the same classes, and
  that side, confidence, box and scores are untouched. These cases are the
  whole contract for this step: on photos the Python and the app are not
  given the same scores (see "The scores of a piece"), so they cannot be
  compared piece for piece, and the board scores differ a little. Two more
  tests place drawn pieces through the recognizer's own code and fail if the
  kings are counted before the sides from the colours or after a1.
- Lab is computed by the formula on both sides, with OpenCV's matrix and
  white point. `cv2.cvtColor` is not used: for float pixels it interpolates
  in a 33×33×33 table and is off from the formula by up to 0.19 in L*, 0.47
  in a* and 0.33 in b* (all 2^24 colours), which moved two pieces of 23,523
  between the groups before the reference changed to the formula (f93479c).
  On a photo the colours still differ by what the two JPEG decoders differ
  (ImageIO's full decode against OpenCV's): nothing at the median, 0.2 at
  the 95th percentile, 0.8 at most, on the medians of 23,510 patches
  (`app_check --colours`, each photo decoded once).
- `Training/app_check` runs the app's recognizer from the command line over a
  list of photos with known positions (`evaluate.py --dump-list` writes the
  lists) and prints what `evaluate.py` prints. It needs no Xcode.
- **Every stage reads one decode of the photo** (`DecodedImage.once`, at the
  entry of the recognizer, of the grid fit and of the colours). An image
  straight from a JPEG file is decoded again each time it is drawn, and how
  it is drawn decides which pixels come out. Against one full decode, the
  letterboxed square differs in about 20% of its bytes (by up to 25 levels
  in a few), a cropped read (the grid fit's second pass, the pieces'
  colours) in 10-20% (up to 13), and a whole read made after a scaled draw
  in up to 28% (up to 42), that last one differently from run to run
  (measured by the reviewer of pull request #9 on Y-LIChess and fixture
  photos; the reviewer of #10 found the cropped reads again, 7-9% of bytes,
  and the run-to-run part not, in a probe without the models). The
  run-to-run part made scores unrepeatable when it was found: two of 200
  Y-LIChess photos went from "no board" to a board and back between
  identical runs of `app_check`. The grid fit itself gives the same answer for the same
  pixels every time. With the copy every stage reads the pixels of one full
  decode, the nearest to what `cv2.imread` gives the Python, in the app, in
  the package's tests and in `app_check` alike. A photo that is a bitmap
  already is not copied: that is what the app passes for every photo it
  redraws upright (`normalizedUprightCGImage`: anything turned or larger
  than 3072 px, so nearly every camera photo). `DecodedImageTests` checks
  the copy. Two limits. Whether an image is a bitmap already is read off its
  file type (`CGImage.utType`, nil for a bitmap made in memory): checked on
  macOS, not yet on iOS, where it decides only the app's rare path of an
  upright photo of 3072 px or less. And `app_check` loads a photo that has
  an EXIF orientation tag through ImageIO's thumbnail call, which is a
  bitmap of its own making, about a fifth of the bytes away from a full
  decode; no current test set has such photos.
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

(Measured before pull request #12, with Vision's share as a piece's
confidence and without the rule for the kings. The app's default path today
is the "on" column of "What one king a side is worth"; the difference to the
"both" rows here is a quarter of a point of squares at most, and Fenit A's
16 boards exact are 32.)

Through `app_check` (the app's own code, Vision doing the resize) on the Mac
mini, 2026-11-13, each photo decoded once (`DecodedImage`). "Off" is
`--a1-prior off --grid-fit off`, "both" is the app's default (grid fit on,
prior 6, colour step off). Boards exact, and in brackets boards that are
exact once turned by 180 degrees (one tap in the app).

**The models in `Models/` (b546e6a: `main`'s corner model, the baseline piece
weights exported with NMS across classes):**

| Steps | ChessReD test, 306 | Fenit board B, 100 (unseen board) | Fenit A, 286 |
|---|---|---|---|
| off | 239 = 78.1% | 0; 46.0% of squares; no board in 36 | 0; 73.6% of squares |
| a1 from the pieces only | 239 | 0; 45.9% of squares; no board in 36 | 0; 73.8% of squares |
| **both** | **279 = 91.2%** (279) | 0 (1); **84.6%** of squares; no board in 4 | 16 (19); **91.4%** of squares |

**Other corner models, for comparison. This table is older: it was measured
before each photo was decoded once**, when the letterbox, the grid fit's two
passes and the colours each got their own decode of the JPEG (see "How the
Swift is checked against the Python"). The models and the grid fit then read
other pixels of the same photos, so these rows are another path, not this
one with noise: the pair in `Models/` with both steps scored 280 of 306, 82.3%
of board B's squares with no board in 6 photos, and 15 (17) on Fenit A that
way. Compare a model with another only through the same binary. (Rows marked
* were scored before the fit was used alone; that changes nothing for exp3
and exp4, which find every board.)

| Models | Steps | ChessReD test, 306 | Fenit board B, 100 | Fenit A, 286 |
|---|---|---|---|---|
| `main`'s two models as on `main` | off | 229 = 74.8% | 0; 45.5% of squares; no board in 35 | 0; 72.8% of squares |
| `main`'s two models * | both, fit not used alone | 267 = 87.3% (267) | 0 (1); 53.7% of squares; no board in 35 | 4 (5); 89.1% of squares |
| exp3 corners | off | 272 = 88.9% (273) | 0; 70.1% of squares | 21; 95.1% of squares |
| exp3 corners | a1 from the pieces only | 272 (273) | 0 (1); 78.1% of squares | 20 (21); 93.5% of squares |
| exp3 corners * | both | 282 = 92.2% (285) | 0 (1); 86.3% of squares | 20 (21); 93.5% of squares |
| exp4 corners * | both | 281 = 91.8% (284) | 0 (1); 86.5% of squares | 20 (21); 93.5% of squares |

What this says:

- With the models in `Models/`, the two steps take ChessReD from 239 to 279
  boards exact and the unseen board from 46% to 85% of squares. Most of that
  is the grid fit: exact corners where the model's were imprecise, and, on
  board B, a board at all in 32 of the 36 photos where the corner model found
  none.
- No board B photo is fully right with any of these: the piece model misreads
  types and colours there (`Training/README.md`). The corners no longer hold
  it back.
- a1 from the pieces helps where the corner model does not know a1 (exp3 on
  board B: +8 points of squares) and costs a little where it does (exp3 on
  Fenit A, a board it was trained on: -1.6 points; ChessReD: nothing; these
  and the next are from the older table's path). With the grid fit on and
  exp3's corners, priors 0 / 3 / 6 give 280 / 282 / 282
  boards on ChessReD, 86.4 / 86.4 / 86.3% of squares on board B and 92.7 /
  93.1 / 93.5% on Fenit A, so 6 stays.

## The second held-out set through the app

`Training/datasets/heldout2`: 1,619 photos of 19 boards that nothing was
trained or tuned on (`Training/README.md`, "Second held-out test set"). Test
only: no step, weight or default here was chosen on it. Through `app_check`,
2026-11-13, each photo decoded once; squares right, boards exact, and photos
in which no board was found.

| Source (photos) | Pair in `Models/`, steps off | Pair in `Models/`, both steps | `pieces_s_render1`, both steps |
|---|---|---|---|
| Y-LIChess (200) | 0%; 0; no board in 200 | 72.4%; 2; 6 | 84.8%; 14; 6 |
| ChessQ (50) | 28.3%; 0; 26 | 64.6%; 1; 10 | 69.8%; 6; 10 |
| chesscog (29) | 32.9%; 0; 10 | 73.3%; 0; 0 | 93.1%; 0; 0 |
| samryan (500) | 54.8%; 0; 72 | 70.9%; 0; 16 | 83.2%; 16; 16 |
| chess-cv games (840) | 53.5%; 0; 32 | 80.6%; 0; 12 | 89.3%; 9; 12 |
| **all (1,619)** | **46.2%; 0; 340** | **75.9%; 3; 44** | **86.3%; 45; 44** |

- The two app-side steps take the pair in `Models/` from 46% to 76% of
  squares on boards nobody has seen, and from no board in 340 photos to 44.
- `pieces_s_render1` (the training session's piece model trained with
  rendered boards, not in `Models/`; same corner model) reads ten points more
  squares on every source and 45 boards fully instead of 3. That is still
  under 3% of unseen boards fully right.
- **Pieces beside the board.** The two "both steps" columns are with
  `SquareAssignment.edgeMargin` at a quarter of a square (pull request #11);
  the "steps off" column was measured with half a square. With half a square
  the same two columns were 75.5%; 2 and 85.4%; 31: captured pieces standing
  next to the board were read as pieces on the rim. The change is nearly all
  Y-LIChess, whose photos have such pieces: 70.1% -> 72.4% with the pair in
  `Models/`, 78.0% -> 84.8% and 0 -> 14 boards exact with
  `pieces_s_render1`. On the three development sets it changes next to
  nothing (pair in `Models/`: ChessReD test 279 of 306 and board B 84.58%
  as before, Fenit A 91.38% -> 91.43%), which is why half a square went
  unnoticed there.
- With the colour step on (measured with the margin at half a square), the
  pair in `Models/` stays at 75.5% (1 board exact) and `pieces_s_render1`
  drops from 85.4% to 82.6% (27): Y-LIChess 78.0% -> 65.5%, samryan 82.9% ->
  79.0%, ChessQ 69.7% -> 68.5%, the other two unchanged. See the next
  section.

## What one king a side is worth

Through `app_check`, 2026-11-13, both steps on, colour step off, margin a
quarter of a square, a piece's confidence the model's score; `--one-king off`
against the default. Squares right; boards exact.

| Set (photos) | Pair in `Models/`: off -> on | `pieces_s_render1`: off -> on |
|---|---|---|
| ChessReD test (306) | 99.53%; 279 -> 99.54%; 280 | 99.64%; 285 -> 99.64%; 285 |
| Fenit board B (100) | 84.58%; 0 -> 84.67%; 1 | 84.31%; 1 -> 84.33%; 1 |
| Fenit A (286) | 91.44%; 16 -> 91.69%; 32 | 95.35%; 185 -> 95.41%; 185 |
| *second held-out set:* | | |
| Y-LIChess (200) | 72.57%; 2 -> 72.49%; 2 | 84.77%; 14 -> 85.02%; 23 |
| ChessQ (50) | 64.56%; 1 -> 64.62%; 1 | 69.78%; 6 -> 69.81%; 6 |
| chesscog (29) | 73.49%; 0 -> 73.71%; 0 | 93.21%; 0 -> 93.59%; 0 |
| samryan (500) | 70.95%; 0 -> 71.12%; 0 | 83.23%; 16 -> 83.32%; 18 |
| chess-cv games (840) | 80.55%; 0 -> 80.62%; 0 | 89.31%; 9 -> 89.62%; 16 |
| **all (1,619)** | **75.98%; 3 -> 76.06%; 3** | **86.34%; 45 -> 86.57%; 63** |

- **On by default.** The three development sets say so for both piece
  models: nothing lost on any of them, Fenit A's boards exact doubled with
  the pair in `Models/` (16 to 32: boards whose only mistake was a king read
as a queen or a bishop, or a queen read as a second king). The
  second held-out set was not used to choose; it agrees, by 0.1 points of
  squares with the pair in `Models/` and 0.2 with `pieces_s_render1`, where
  18 more boards come out fully right (45 to 63 of 1,619).
- It is a small step, as it must be: it can change two pieces of a board at
  most. It pays in boards fully right, and only once the rest of the board
  is right, which is why it shows with `pieces_s_render1` and not with the
  pair in `Models/` on boards nobody has seen.
- **One part goes the other way**: Y-LIChess with the pair in `Models/`,
  72.57% -> 72.49%. Ten photos gain a square each, three lose one, and one
  loses 17 (`wood_0060.jpg`: the model takes nearly every piece for White
  and sees four kings, three of them White; with two of those made something
  else, a1 from the pieces turns the board). The rule changes kinds
  before a1 is read from them, in the Python too, and on a board read that
  badly either can make it worse.
- The Python measured the same direction with its made-up scores
  (`Training/README.md`): `pieces_s_render1` on Y-LIChess games 10 -> 15 of
  102 exact, chess-cv 11 -> 13. The app's numbers are not the Python's
  (other scores, other resize), only the direction is comparable.

## What the confidence marks

A piece's confidence is the model's score since pull request #12, and was
Vision's share before ("The scores of a piece"). For the board that is read
it changes next to nothing: only which of two pieces found on one square
stays. Squares right; boards exact, without the rule for the kings:

| Set | Pair in `Models/`: share -> score | `pieces_s_render1`: share -> score |
|---|---|---|
| ChessReD test | 99.52%; 279 -> 99.53%; 279 | 99.64%; 285 -> 99.64%; 285 |
| Fenit board B | 84.58%; 0 -> 84.58%; 0 | 84.28%; 1 -> 84.31%; 1 |
| Fenit A | 91.43%; 16 -> 91.44%; 16 | 95.34%; 185 -> 95.35%; 185 |
| second held-out set | 75.94%; 3 -> 75.98%; 3 | 86.33%; 45 -> 86.34%; 45 |

What it changes is what the user is shown. The app marks a piece whose
confidence is under 0.85 for the user to check
(`AppModel.lowConfidenceThreshold`, not changed). `app_check` counts how
many of the pieces it put on a square that marks, among those on a wrong
square or of a wrong class and among the right ones:

| Set | Pair in `Models/`: wrong pieces marked | right pieces marked | `pieces_s_render1`: wrong pieces marked | right pieces marked |
|---|---|---|---|---|
| ChessReD test | 10% -> 55% (36 of 66) | 0.05% -> 1.0% | 13% -> 39% (18 of 46) | 0.06% -> 1.1% |
| Fenit board B | 24% -> 76% | 6.5% -> 40% | 16% -> 60% | 3.8% -> 27% |
| Fenit A | 18% -> 64% | 4.7% -> 23% | 3.5% -> 23% | 0.2% -> 12% |
| second held-out set | 15% -> 78% | 5.5% -> 47% | 18% -> 46% | 3.2% -> 11% |

- With the share, the mark said next to nothing: it left more than four in
  five of the wrong pieces on an unseen board unmarked, because a model that is unsure whether
  a piece is there at all is seldom split between two classes.
- With the score, the mark follows what the model knows. On a board like the
  ones it was trained on, one right piece in a hundred is marked and half of
  the few wrong ones. On a board it has not seen, the pair in `Models/`
  marks nearly half of the right pieces too: there it puts 13,000 pieces
  wrong and 16,500 right, and its scores say as much.
- The threshold of 0.85 was set when the number under it was another one.
  It is the owner's to move; `app_check --flag-below C` counts the same for
  any other value. Not tried in the app on a phone.

## What the colour step is worth

Through `app_check` on the Mac mini, 2026-11-13, grid fit on and prior 6,
`--sides-from-colour off` against `on`, each photo decoded once, the rim
margin still at half a square (a quarter changes these sets by a few
hundredths of a point). Boards exact (in brackets: once turned by 180
degrees) and squares right. **The step is off in the app**
(`CoreMLBoardRecognizer(sidesFromColour:)` defaults to false): what follows
is why it was written and why it is off.

| Piece model | Step | ChessReD test, 306 | Fenit board B, 100 (unseen board) | Fenit A, 286 |
|---|---|---|---|---|
| in `Models/` | off | 279 (279); 99.52% | 0 (1); 84.58% | 16 (19); 91.38% |
| in `Models/` | **on** | 279 (279); 99.52% | **2 (3); 85.50%** | 16 (19); 91.38% |
| `pieces_s_render1` (not in `Models/`) | off | 285 (285); 99.64% | 1 (1); 84.27% | 185 (202); 95.29% |
| `pieces_s_render1` (not in `Models/`) | **on** | 285 (285); 99.64% | **11 (13); 88.80%** | 185 (202); 95.29% |

(`pieces_s_render1` is the training session's piece model trained with
rendered boards, exported for this measurement only, with the corner model in
`Models/`; Fenit A is mostly photos it was trained on.)

- The step is for a piece model that knows the kinds on a new piece set and
  not the sides: with `pieces_s_render1`, board B goes from 1 to 11 boards
  exact. The Python gets 15 there end to end (`Training/README.md`); it reads
  clean 1280-px copies with its own grid fit, the app the photos through
  Vision.
- On these three sets it costs nothing: no ChessReD test photo changes with
  either model. With the model in `Models/` it changes a side on 36 board B
  photos (+0.9 points of squares), on 3 Fenit A photos and on no ChessReD
  photo.
- **But its weight (4) and its share (15%) were chosen on board B, and on the
  second held-out set, which nothing is tuned on, it loses.** The training
  session's numbers, in Python (`main`'s corner model, grid fit, prior 6),
  `pieces_s_render1` without and with the step, squares right: Y-LIChess game
  positions 75.2% -> 51.9% (102 photos), Y-LIChess layouts 75.4% -> 74.3%
  (98), ChessQ 70.6% -> 68.6% (50), samryan 79.5% -> 79.2% (500), chesscog
  and chess-cv unchanged; boards exact unchanged everywhere. So the step is
  off by default until the reference has a rule that holds on boards it was
  not tuned on (the training session reworks it on the training sources).
  Through `app_check`, squares right without ->
  with the step: with `pieces_s_render1`, the model the step was made for,
  the app shows the same loss, 78.0% -> 65.5% on the 200 Y-LIChess photos,
  ChessQ 69.7% -> 68.5%, chesscog 92.7% -> 92.7%. With the model in
  `Models/` it gains squares there (Y-LIChess 70.1% -> 71.8%, ChessQ 64.6%
  -> 64.8%, chesscog unchanged) and loses that model's one exact board. So
  it is off because it fails with the piece model that is to replace the
  one in `Models/`, and is worth little with the one that is there.
- The rule has no smallest distance between the two groups, and with six
  pieces or fewer one disagreement is already over 15%. The Fenit A photos
  it acts on are such positions (2 to 9 pieces found): on one, two black
  pieces are split into a "white" and a black one. Asking for two
  disagreeing pieces was tried in the Python and is no better (board B has
  many sparse endgames where the one brown king is what needs correcting).
- On the same boxes the Swift and the Python give every piece the same
  final class: 23,510 pieces on the 692 photos of the three sets, found by
  the model in `Models/` (11,737) and by `pieces_s_render1` (11,773), the
  step changing a side on the same photos in both (36, 3, 0 and 61, 1, 0).
  By colour alone they differ on 2 of the 23,510, on two Fenit A photos with
  `pieces_s_render1`'s boxes where the step does not act: pieces between the
  groups, moved by the JPEG decoders. (Measured with `app_check --colours`
  against the reference's Lab formula. Before each photo was decoded once,
  when the patches were cropped reads of the undecoded photo, it was 23,523
  of 23,523.)
- Cost when on: 0.07 s a photo with 32 pieces in a release build, about 1 s
  in a debug build; like the grid fit it reads the photo's pixels once more
  (the rectangle that holds all the pieces).

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
