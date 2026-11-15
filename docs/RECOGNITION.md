# Recognition in the app, after the two models

`Packages/ChessVision` turns a photo into a piece placement. The two CoreML
models give four board corners and a list of piece boxes; everything after
that is geometry, and `Training/chess_pipeline.py` is the reference the Swift
must match. This file says what the app-side steps are, how they are tested,
and what they were measured to be worth. Training results are in
`Training/README.md`.

**Which models (2026-11-14).** The measurements below are of three pairs
of models: two that have been in `Models/`, and the one that pull request
#19 puts there. Each section says which.

- **The third pair**: `pieces_s_tone1` and `corners_s_canon1`, the pair
  that pull request #19 puts in `Models/`. "What the app reads with the
  third pair" is the section for what the app does once #19 is merged;
  until then, and if #19 is taken back, it is the second pair's.
- **The second pair**: `pieces_s_render3` and the first corner model, in
  `Models/` from pull request #15 until #19.
- **The first pair**: the first piece model and the first corner model,
  until #15. The steps were chosen with it; those sections are kept as the
  record of how.

The tests and the description of the steps hold for any pair.

**Since 2026-11-14 a1 is taken to be a dark square** (step 7). The tables
of the three pairs were measured before that rule and are left as they
were; what the rule changes with the third pair is in "What a dark a1 is
worth".

**Since 2026-11-15 the corner model picks the grid fit's window** (step
3). What that changes, on top of the dark a1, is in "What the model's
choice of window is worth", which is the section for what the app reads
now.

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
   **The model picks the fit's window** (`GridCorners.windowByModel`, port
   of `chess_pipeline.window_by_model`): the fit finds the lines of the grid
   and has to choose which 8×8 squares of it are the board, and a border
   that looks like a rank of squares makes it choose a window one rank or
   file off. So where both found the board, the model's corners are looked
   at in the fit's own board frame: when all four lie within 0.75 squares
   of the same shift by one square (a rank, a file or both), the fit's
   window is moved by it, along the fit's own lines. Nothing is moved when
   the model's corners are nearer the fit's than half a square on average,
   when they are a square and a half or more off on average, or when the
   four do not agree on one shift. The steps after this one (the dark a1, the pieces) see the
   window as moved.
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
   **The side of single pieces, switched off** (`OddSides`, port of
   `odd_sides` and `patch_hidden`; in the reference since 2026-11-15 and
   ported so that it can be measured through the app, `Training/README.md`,
   "The side of single pieces"): the other way to use the same colours.
   The model's own answers say what colour each side is in the photo (the
   median colour of the pieces it calls White and of those it calls Black,
   as L*, 4a*, 4b*), and a piece called Black whose colour is twice as near
   White's as it is to the colour of the other pieces called Black becomes
   White. One way only: a light piece in a shadow is as dark as a black
   one, so a piece called White is never made Black. Nothing changes when
   a side has fewer than two pieces with a colour, when the pieces called
   White are not the lighter, or when the two sides' colours are less than
   three spreads apart (the spread: the median distance of a piece from
   its own side's colour, at least 3). A piece is neither counted nor
   changed when another box whose foot is lower in the picture covers more
   than a fifth of the patch its colour is measured on: what is measured
   there is the piece in front. It comes after the step above and before
   the kings are counted, which are counted by side.
6. **One king a side** (`OneKingASide`, port of `one_king_a_side`): a
   position from a game has exactly two kings, and a king read as a queen,
   or a third "king", is what a piece model gets wrong on a set it has not
   seen. A
   board with exactly two kings is left alone, also when both are on one
   side (that is a wrong side, not a wrong kind). With fewer than two, a
   side that has pieces and no king gets one: its piece with the highest
   score for "king", if the model gave it at least 0.02. With more than two,
   a side with several kings keeps the one with the highest score for "king"
   and each of the others becomes its best-scoring other kind, a queen when
   it has none. Sides, confidences and boxes are never changed; a piece
   whose kind the rule changed is shown to the user with the model's score
   for the kind it became, so that the app marks it.
7. **a1 from the pieces** (`BoardOrientation`, port of `white_side_score` and
   `orient_by_pieces`): of the four rotations of the corners, the one whose
   board looks most like White at ranks 1-2 wins; the corner model's own
   rotation gets a head start of 6 (`BoardOrientation.appPrior`). When the
   corners are the fit's alone there is no model opinion, and the pieces
   decide without a head start.
   **a1 is a dark square** (`GridCorners.squareColours`, port of
   `grid_corners.dark_corner_rotations`): before the pieces are asked, the
   grey of the 64 cells between the corners says which two of the four
   rotations put a dark square on a1, and the pieces choose between those
   two only. They are half a turn apart, so the pieces can still turn a
   board round and can no longer turn it by a quarter, which is what they
   did to boards with few pieces and what the app's flip button does not
   mend. (A reading can still come out a quarter off where the colours
   themselves mislead, with corners a square off the board or a board set
   up with a light square on a1, and where they say nothing, below.) The
   head start still goes to the corners as they
   came, when those are one of the two. The colours say nothing, and all four rotations stay,
   when fewer than 40 cells are seen, when the lighter and the darker cells
   are less than 15 grey levels apart, or when the two colours do not split
   cleanly (each cell is called light or dark; the share of light cells
   among those of a1's colour and among the others must differ by half).
   Each cell is read at four patches near its corners, as the grid fit
   reads it, because a piece usually stands on the middle. **A board set
   up with a light square on a1 is read a quarter turn off**: the rule
   takes the rules of chess at their word.
8. Each box's bottom centre goes through the corner homography to a square
   (`Homography`, `SquareAssignment`); one piece per square. A foot that
   falls up to a quarter of a square outside the board is taken back onto
   the rim square (`SquareAssignment.edgeMargin`, the reference's
   `EDGE_MARGIN`); further out it is a captured or spare piece standing
   beside the board and gets no square.

Steps 3, 6 and 7 can be switched off, the model's choice of window of step
3 and the dark a1 of step 7 each by itself, and the two halves of step 5
each on
(`CoreMLBoardRecognizer(a1Prior:gridFit:sidesFromColour:oneKing:a1Dark:fitWindow:oddSides:)`;
`CoreMLBoardRecognizer.Default` names what the app runs with).
How the two sets of corners combine is
`CoreMLBoardRecognizer.corners(model:in:gridFit:fitWindow:)`, and it is what
`Training/evaluate.py --grid-corners --fit-window` does.

### The scores of a piece

The piece model scores every class for every box. Vision passes those scores
on in a form of its own: the labels of an observation carry the scores
divided by their sum, so that they add up to 1, and the observation's own
confidence is the sum. A label's confidence times the observation's is the
score as the model gave it (`CoreMLBoardRecognizer.classScores`). Nothing in
Vision's documentation says so; `EndToEndFixtureTests.testPieceScoresAreTheModelsOwn`
holds it against the model run without Vision on the same pixels, for
every class of every box: once on the four fixtures as the app reads them,
and once on boxes the model is unsure of (the fixtures drawn at a quarter
of the size, the model's threshold at 0.01), because a box with a single
score says nothing about what the scores are divided by, and a piece model
that reads the fixtures without a doubt has no other on them. The two agree
to five decimals on every score, and the boxes are the same in the same
order: 156 boxes, 94 of them split between two classes, with the third
pair; 153 and 81 with the second; 105 and 70 with the first (on the Mac
mini's Neural Engine; a few more or fewer on the CPU, where the reviewer of
pull request #16 ran them too). The
reviewer of pull request #12 found the same on 479 photos of boards the
models have not seen (5,853 boxes: labels add up to 1, the observation's
confidence is the sum, also where the sum is over 1).

**This is checked on macOS only** (the Mac mini and CI), not on iOS, where
the app runs. Labels that do not add up to 1 are taken to be the scores
themselves and left as they are, so a Vision that handed out plain scores
would be read right as well. To check on iOS: run the package's tests with
an iPhone simulator as the destination; `testPieceScoresAreTheModelsOwn`
fails if Vision does it another way there.

Two things are read from the scores:

- **The confidence of a piece is the model's score for the class it is taken
  for**, as in the Python. Until pull request #12 it was Vision's label
  confidence, the share: about 1.0 for nearly every box, also for one the
  model gave 0.3, since a second class seldom has a score near the first's
  (on the fixtures 1 box of 57 had a second score of 0.02 or more with the
  piece model that was in `Models/` then, none with `pieces_s_render3`; on
  boards that earlier model had not seen about a fifth had, most of them
  small).
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
missing or the end-to-end tests did not run. To run the tests on a
candidate pair without putting it in `Models/`:
`CHESSVISION_TEST_MODELS=/folder ./Scripts/swift-test.sh`, with the folder's
full path (the script runs from `Packages/ChessVision`) and the two packages
in it named `BoardCorners.mlpackage` and `Pieces.mlpackage`. A folder that
is not there, or does not hold them, fails the end-to-end tests; it does not
skip them.

`EndToEndFixtureTests` reads four ChessReD test photos with the models in
`Models/`, three times: as the app does, with the grid fit switched off, and
with the colour step switched on. The second run is the corner model's own
test: each of its corners within one square of the grid fit's, and 94% of
the squares of the four photos together. It has no floor of squares a photo,
because with corners most of a square off the count hangs on single pieces
standing on a line (the first corner model, whose corners are up to 0.72
squares off on one fixture: 58 or 60 of 64 there from the same pixels with
the first piece model, depending on whether CoreML ran on the Neural Engine
or the CPU, and 60 with `pieces_s_render3`; `corners_s_canon1` is within a
quarter of a square on all four and reads 64 of 64). With the fit on, a
corner model that found no board at all would go unnoticed, because the
fit reads these photos by itself.

**Which corner is a1** is checked in every one of these runs, and in the
run on 3:4 photos: no photo may fit its position better once the board
read is turned by a quarter, a half or three quarters. The floors of squares do not see that on an
endgame: the fixture `test_004005.jpg` has six pieces, turned it still has
52 squares of 64 right, and the four photos together 95%. One case is let
through as a known issue, printed by the test run and not failing it:
`corners_s_canon1` reads that fixture turned when it is drawn as a 3:4
photo ("What the app reads with the third pair"). A turn fails in every
other case: with another corner model, another photo, another turn, that
photo turned as the square it is, or turned with a piece wrong besides. A
misreading that is not a turn is for the floors of squares to catch, as
before.

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
- `Training/swift_reference.py dark-cases` writes `Fixtures/dark_cases.json`:
  what `dark_corner_rotations` makes of 20 chequered pictures that both
  sides draw from a few numbers (which corner is dark, the ring of corners
  in each rotation, and each of the three thresholds from both sides: 40
  cells seen and 39, on a board turned in the picture so that its edge
  takes cells one at a time; 15 grey levels and 14; a split of 0.5 and one
  cell less; and cells exactly at the middle grey, which are not light),
  of the five drawn boards of `grid-cases` with their true corners in
  the four rotations, a square along, a square up and half a square off,
  of the drawn table without a board, and of the four fixture photos on the
  corners the Python's grid fit gives them, in the four rotations; and
  which rotation `orient_by_pieces(turns=)` chooses for each of the 120
  cases of `orientation_cases.json` when only rotations 0 and 2, or only 1
  and 3, are allowed. `GridSquareColoursTests` and `BoardOrientationTests`
  replay them: the rotations must be the Python's, the strength of the
  split equal on the drawn pictures (the same pixels) and within two cells
  of 32 on the photos (it is equal there too today). Three more tests
  place pieces on a drawn board through the recognizer's own code: pieces
  that say "a quarter on" get the better of the two rotations the colours
  allow, pieces that agree with the colours and a half turn are left to
  the pieces, and on a picture whose colours say nothing all four
  rotations stay open. One more builds the recognizer as the app does and
  with the rule switched off, and requires that the two place those
  pieces differently: the four fixture photos read the same either way,
  so without it nothing would notice the rule missing from `recognize`.
  Two places where the Swift arithmetic is not NumPy 1.26's to the last
  bit are named in `GridSquareColours.swift`; neither changed an answer
  on the 2,311 photos compared.
- `Training/swift_reference.py window-cases` writes
  `Fixtures/window_cases.json`: what `window_by_model` makes of 220 pairs
  of a fit's corners and a model's, built from a few numbers on four fits
  (a board seen from above, at a slant, from low down, and one whose
  corners run the other way round): the model's corners near the fit's, a
  square on in each of the eight directions, two squares on, one corner
  unlike the other three, the model's corners a rotation on from the
  fit's, tolerances of 0.5 and 0.95, 80 random ones, and each threshold
  from both sides (a mean of 0.4999 and 0.5001 of a square, of 1.4999 and
  1.5001, a corner 0.7499 and 0.7501 from the shift, on each axis and to
  each side); and where 44 windows lie when moved on purpose
  (`evaluate.py --shift-fit`). `GridFitWindowTests` replays them: the
  shift must be the Python's and the corners within a millionth of a
  pixel; a window left alone is the very corners given. No case hangs on
  the last bits (the script refuses one within 0.00002 of a threshold),
  because the two do not share them: the Python gets the fit's
  homography from a singular value decomposition, the Swift by
  elimination. More tests move a window on purpose and require the model
  to move it back, give corners that are no board, and go through the
  recognizer's own code on a drawn board: the fit's lines with the
  model's window where the model has the board a rank on, the fit's
  window where the rule is off, where the model is two ranks on or one of
  its corners does not agree with the other three, and no move where the
  fit stands alone or rejects itself. One builds the recognizer as the
  app does and with the rule switched off and requires the two to differ
  there (`corners(model:in:)`, the method `recognize` calls), since the
  four fixture photos read the same either way; another requires that
  what `app_check --window` is given is what that method does on the
  fixture photos. **No test runs a photo whose window moves through
  `recognize` itself**: the fixtures have none, and the one development
  photo that has (Fenit A's, below) is not in the repository. What stands
  in for it is `app_check`, whose readings change on exactly the photos
  whose window `--window` reports as moved.
- `Training/swift_reference.py odd-cases` writes `Fixtures/odd_cases.json`:
  what `odd_sides` and `patch_hidden` make of 100 sets of detections with
  their colours and of a drawn picture. Forty are built by hand, each
  condition of the rule from both sides: a light piece called Black at
  L* 53.3 and 53.4, where "twice as near" begins; the two sides 8.9 and
  9.1 apart with the spread on its floor, 14.9 and 15.1 with a spread of
  5; White darker than Black, exactly as light, and lighter by a
  thousandth; one and two pieces a side, with and without a colour;
  0.19 and 0.21 of the patch behind a piece in front, a box in front
  whose foot is higher and one whose foot is level; a piece that would
  make a side but stands behind another; a box of no width; other values
  for each of the four numbers; and that a dark piece called White stays
  White. Sixty are random boards whose boxes reach over their neighbours
  (106 pieces behind another in all). The script refuses a case within a
  millionth of a tie, because the two do not share the last bits of a
  distance. `OddSidesTests` replays them: the classes and which pieces
  are behind another must be the Python's, and kind, confidence, box and
  scores untouched. The drawn picture is `colour_pieces.png` with three
  of its brown pieces called Black, its colours read from the PNG. Two
  more tests go through the recognizer's own code on drawn pawns: the
  two cream pawns a model calls Black are White's with the step on and
  not with it off, and the sides are mended before the kings are counted
  (a cream piece called a black king: mended first, White's lesser king
  becomes a queen; counted first, Black's real king would). One builds
  the recognizer as the app does and with the step asked for, and
  requires the first to leave those pawns and the second to mend them.
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
  upright photo of 3072 px or less. And a photo with an EXIF orientation
  tag: `app_check` decodes it once in full and moves the pixels to where
  the tag says, none of them changed. Checked by the reviewer of pull
  request #14 for every tag from 1 to 8: a PNG comes out byte for byte as
  PIL's `exif_transpose` and `cv2.imread` give it, and JPEGs (also grey,
  CMYK, with a Display P3 profile) and PNGs with alpha, a palette or 16
  bits come out as the plain full decode with its pixels moved; on chess-cv
  and samryan photos the difference to PIL and OpenCV is what the JPEG
  decoders differ by (0.07 and 0.02 levels on average, 7 at most). To do it
  again: `app_check --upright PHOTO OUT.png` writes what `app_check` reads.
  Every chess-cv photo of the second held-out set has such a tag (840,
  orientation 6) and two of samryan's
  500. **Until pull request #14 `app_check` loaded those through ImageIO's
  thumbnail call**, a decode of its own making, about a fifth of the bytes
  away from a full decode, while this file said no test set had such
  photos. Comparisons of one model or setting with another were not
  affected (both read the same pixels); the chess-cv figures were a little
  off. Measured on today's defaults, thumbnail -> full decode: the first pair
  (see below) 80.62% -> 80.65% of chess-cv's squares, `pieces_s_render1`
  89.62% -> 89.72% and 16 -> 21 boards exact, one more photo without a
  board (13, not 12); samryan to the square as before. The tables below
  say which of their chess-cv figures are from which path.
- For any list of photos, Python against Swift:
  `swift_reference.py grid --list L.json --out py.json`,
  `app_check --list L.json --grid sw.json`,
  `swift_reference.py compare --python py.json --swift sw.json`.
  For the two colour steps (the sides of the whole board and of single
  pieces), on the boxes the app's piece model finds:
  `app_check --models M --list L.json --colours sw.json`,
  `swift_reference.py colours --swift sw.json --list L.json`.
  For the dark a1, on the corners the app ends with:
  `app_check --models M --list L.json --dark sw.json`,
  `swift_reference.py dark --swift sw.json --list L.json`.
  For the fit's window, on the corners the app's corner model and its
  grid fit give (with `--shift-fit FILES,RANKS`: after the fit's window
  was moved on purpose):
  `app_check --models M --list L.json --window sw.json [--shift-fit 0,1]`,
  `swift_reference.py window --swift sw.json`.

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

## What the app reads with the third pair

`pieces_s_tone1` (a YOLO11s, as `pieces_s_render3` was) and
`corners_s_canon1` (a YOLO11s pose model in place of the first corner
model's YOLO11n; it is trained on corners in an order fixed by the image
and its export turns them back to a1 first, so the app is handed the same
`[1, 17, 8400]` as before). How they were trained and on what is in
`Training/README.md`. Measured on the Mac mini on 2026-11-14 through
`app_check` built from 8a62cd3, the models read from the training session's
export folder, each photo from one full decode. **Nothing here was run on a
phone**: not the readings, not how long the two larger models (37 MB
together, 23 MB before) take there. Squares right; boards exact. A board is
found in every photo of every set, in every column.

| Set (photos) | Both steps and the rule for the kings off | Grid fit only | a1 from the pieces only | Both steps, rule for the kings off | **As the app runs it** |
|---|---|---|---|---|---|
| ChessReD test (306) | 99.74%; 294 | 99.80%; 295 | 99.74%; 294 | 99.80%; 295 | **99.80%; 295** |
| Fenit board B (100; no photo of it was trained on) | 75.86%; 7 | 76.45%; 7 | 92.25%; 13 | 94.27%; 15 | **94.36%; 16** |
| Fenit A (286; trained on these photos) | 99.54%; 255 | 99.49%; 254 | 97.82%; 231 | 97.79%; 230 | **97.79%; 231** |
| *second held-out set:* | | | | | |
| Y-LIChess (200) | 92.28%; 26 | 87.98%; 32 | 91.98%; 29 | 88.20%; 34 | **88.16%; 44** |
| ChessQ (50) | 85.44%; 15 | 86.94%; 15 | 91.81%; 14 | 92.97%; 14 | **93.03%; 14** |
| chesscog (29) | 96.28%; 8 | 96.28%; 8 | 98.17%; 8 | 98.17%; 8 | **98.17%; 8** |
| samryan (500) | 90.73%; 77 | 90.73%; 77 | 95.41%; 87 | 95.41%; 87 | **95.50%; 106** |
| chess-cv games (840) | 97.67%; 187 | 97.67%; 187 | 97.74%; 187 | 97.74%; 187 | **97.99%; 267** |
| **all (1,619)** | 94.46%; 313 | 93.98%; 319 | 96.13%; 325 | 95.70%; 330 | **95.86%; 439** |

(Columns: `--a1-prior off --grid-fit off --one-king off`; `--a1-prior off
--one-king off`; `--grid-fit off --one-king off`; `--one-king off`; no
switch. Counting boards that are right once turned by 180 degrees, one tap
in the app, the last column is ChessReD test 297, board B 19, Fenit A 247,
the second held-out set 451. The last column is to the digit what the
training session measured with its own build; its outputs are kept on the
Mac mini in `Training/runs/app_check/tone1/` and `final_h2/`
(`tone1_canon1_*.txt`). Which sets say what about a board the app has not
met is as for the second pair, below, with one difference: nothing was
trained or tuned on the second held-out set, but the choice to put this
corner model in rested partly on its score there (`Training/README.md`),
so for the third pair it is no longer a set that nothing was chosen on.)

- **Against the goal.** On boards nobody trained on, the app reads 439 of
  1,619 photos fully right (27%) and 16 of board B's 100; the goal is 85%.
  The second pair had 298 and 6, and found no board in 45 photos of the
  second held-out set and 4 of board B; the third finds one in all.
- **The corner model reads endgames of one game turned by 180 degrees.**
  ChessReD game G033 ends with White's pieces on Black's side ("Known weak
  spot", at the end). Of its last frames the third pair reads IMG100 and
  IMG102 turned, every piece on the square it would be on after half a
  turn; drawn as 3:4 photos, which is what a phone takes, it turns IMG101
  and IMG102. IMG101 is the test fixture `test_004005.jpg`: right as the
  square photo it is, turned as a 3:4 photo. The first corner model turns
  none of them. The app's flip button mends it in one tap, and the user has
  to notice. As 3:4 photos over the whole of ChessReD test the third pair
  still reads more boards than the second (282 against 275, 284 counting a
  turn; board B 94.33% of squares and 16 boards against 88.56% and 6).
  `EndToEndFixtureTests.testRecognizesFixturesAsPortraitPhotos` records the
  turned fixture as a known issue and passes.
- **The steps are worth less than they were, because this corner model
  needs them less**: without either it already reads 94% of the squares of
  the second held-out set (the first corner model: 47%). What each still
  does:
  - **a1 from the pieces** is what unseen boards need, and what costs on
    boards the corner model knows. Board B 76% -> 94% of squares, the
    second held-out set 94.0% -> 95.7%. Fenit A 254 -> 230 boards: there
    the corner model has a1 right by itself (without the step no board is
    right only once turned, and one photo of 286 is read turned), and the
    pieces overrule it on 24 boards, 16 by half a turn and 8 by a quarter.
    The head start of 6 for the corner model's a1 was chosen on ChessReD
    test, where no value from 6 up changes anything with this pair. On the
    development sets, head start 0 / 3 / 6 / 9 / 12 / 15 / 20 / 40 / none
    of the pieces: Fenit A 214 / 223 / 231 / 236 / 243 / 247 / 251 / 255 /
    255 boards; board B 94.36% of squares from 0 to 12, then 94.23% /
    93.83% / 92.20% / 76.50% (16 boards up to 20, then 15 and 7); ChessReD
    test 292 / 294 boards at 0 and 3, 295 from 6 up.
  - **A head start of 12 was tried and is not taken; 6 stays.** The
    development sets speak for it: 12 boards gained on Fenit A, no score
    lost on board B or ChessReD test. On the second held-out set, run as a
    test, it loses squares and gains nothing: 95.86% -> 95.71% (183 squares
    lost, 37 gained), 25 photos worse and 5 better, boards exact 439 ->
    440. Every photo that changes is the same board read at another turn,
    no piece is read differently: 24 photos that were the right way up at 6
    are turned at 12, 23 of them by a quarter, which the app's flip button
    (a half turn) does not mend, and 5 go the other way. Boards exact do
    not show it because most of the 24 were a square or more off already.
    - How strong the evidence is: a sign test over photos gives p < 0.001,
      but the photos are not independent. 24 of the 25 are from two
      sources, Y-LIChess (14 worse, 2 better) and samryan (10 and 1), in
      runs of neighbouring positions and repeated set-ups; counted by run
      it is about 8 worse and 3 better. What decides is that there is no
      gain on unseen boards to set against the turned ones.
    - What the changed photos share is few pieces: 35 of the 36 that read
      differently have nine or fewer, as have all 13 that change on Fenit
      A. On those of the test set the corner model's a1 is wrong on 31 and
      right on 5, and at 6 the pieces put 24 of them right; on Fenit A,
      whose photos it was trained on, it is right on 12 of the 13.
      Chesscog and chess-cv read every photo the same at 6 and 12, and
      could not have done otherwise: they have no photo with fewer than
      ten pieces. Board B did not show the cost at 12 and shows it from 15.
    - What this leaves open. The test set was used to reject a change, not
      to pick a value; it is one more look at that set. Values under 6
      were not run on it. 6 is kept because it is what there was, not
      because it is shown to be the best value for unseen boards: the one
      comparison on the test set points downward, and the development sets
      cannot place this number any more (board B reads the same from 0 to
      12, and Fenit A and ChessReD test, both boards the corner model was
      trained on, always want more: 0 and 3 lose boards there). Choosing
      again needs an unseen development board with sparse positions, not
      another look at the test set. A single number may not serve both
      kinds of board at all; something that tells a board the corner model
      knows from one it does not might, or something that asks how much
      the pieces have to say.
    - The values 9 and 15 on the development sets were run with the test;
      0 and 3 after it.
  - **The grid fit** changes nothing at all on samryan, chess-cv and
    chesscog (not one photo reads differently), mends board B (131 squares
    on 11 photos, 2 lost) and ChessQ, and costs on Y-LIChess: 54 photos
    gain 196 squares and 21 lose 680, 18 of them 677, about 38 each, where
    the fit accepts a grid that is not the board (the position read a rank
    off, on 16 of the 18 by the count of the reviewer of pull request #20).
    Over the second held-out set that is 0.4 points of squares lost and 5
    boards gained.
  - **Neither step is changed.** What to do about the grid fit (the
    model's corners where the fit is far from them) belongs in the
    reference first, chosen on board B, Fenit A and ChessReD test and then
    tested; Y-LIChess is part of the test set and shows the cost, it cannot
    be used to tune it away.
- **The rule for the kings** is worth 109 boards of the second held-out set
  (330 -> 439), one on board B and one on Fenit A, none on ChessReD test.
- **What the app marks for the user to check** (confidence under 0.85), and
  what 0.5 and 0.7 would mark: wrong pieces marked / right pieces marked /
  squares marked a photo.

| Set | Under 0.5 | Under 0.7 | Under 0.85 (as now) |
|---|---|---|---|
| ChessReD test | 33% / 0.03% / 0.03 | 42% / 0.2% / 0.07 | 50% / 0.6% / 0.16 |
| Fenit board B | 8.1% / 1.2% / 0.38 | 29% / 4.5% / 1.4 | 57% / 15% / 3.7 |
| Fenit A | 6.3% / 0.2% / 0.07 | 13% / 1.3% / 0.27 | 23% / 9.0% / 1.3 |
| second held-out set | 9.9% / 1.1% / 0.44 | 25% / 3.0% / 1.1 | 49% / 8.2% / 2.7 |

  Much as with the second pair: 2.7 squares a photo on unseen boards, half
  of the wrong pieces among them. The threshold is the owner's to move.

## What the model's choice of window is worth

The rule of step 3: where the corner model and the grid fit both find the
board, the fit's window is moved by one square when all four of the model's
corners lie within 0.75 squares of that same shift. It came from the
reference (`Training/README.md`, "The corner model picks the grid fit's
window": the tolerance was chosen there on board B, in Python). Measured on
the Mac mini on 2026-11-15 with the third pair through `app_check`,
everything else as the app runs it (the dark a1 on), the rule off and on
through the same binary; with it off every photo reads as in the last
column of "What a dark a1 is worth". The development sets were run first;
the second held-out set was then run once, as a test. **It is not a blind
test of this rule on Y-LIChess**: the fault the rule mends, a window a rank
off, was found on Y-LIChess photos of that very set (by the reviewers of
pull requests #20 and #24), and all 18 windows it moves there are those
photos. Its figures for Y-LIChess say that the fault found there is
mended; what is blind in them is that nothing else changed, there or on
the other four parts. Nothing in the rule was set on that set: the
tolerance is the reference's, from board B, and the rule was on by default
before the set was run. Squares right; boards
exact; boards exact or half a turn away; and, of the photos where the model
and the fit both found the board, how many windows the rule moved.

| Set (photos) | Without the rule | **With it, as the app runs it now** | Windows moved |
|---|---|---|---|
| ChessReD test (306) | 99.80%; 295; 297 | **99.80%; 295; 297** | 0 of 288 |
| Fenit board B (100) | 94.48%; 16; 19 | **94.48%; 16; 19** | 0 of 95 |
| Fenit A (286) | 98.09%; 233; 255 | **98.14%; 234; 256** | 1 of 274 |
| *second held-out set:* | | | |
| Y-LIChess (200) | 89.80%; 55; 63 | **94.81%; 60; 68** | 18 of 194 |
| ChessQ (50) | 93.44%; 14; 16 | **93.44%; 14; 16** | 0 of 35 |
| chesscog (29) | 98.17%; 8; 8 | **98.17%; 8; 8** | 0 of 29 |
| samryan (500) | 96.50%; 115; 123 | **96.50%; 115; 123** | 0 of 360 |
| chess-cv games (840) | 97.99%; 267; 267 | **97.99%; 267; 267** | 0 of 744 |
| **all (1,619)** | 96.38%; 459; 477 | **97.00%; 464; 482** | 18 of 1,362 |

- **Only photos whose window moved read differently**: 19 of 2,311, and
  every other photo reads as before, square for square. As 3:4 photos
  ChessReD test and board B read as before too (99.68%; 282; 284 and
  94.45%; 16; 19).
- **The 18 of Y-LIChess** are the photos where the fit took a window one
  rank off ("What a dark a1 is worth", "The 16 the rule made"). All 16
  full boards named there are among them: read at 19 to 24 squares of 64
  before, a rank off and turned by a quarter, they now have 61 to 64
  right, five of them all 64 (`wood_0150`, `0156`, `0158`, `0165`,
  `0178`) and the others one to three squares short. `wood_0148` (Black's
  pieces alone) goes from 43 to 63. In all, 17 photos gain 665 squares.
- **The one that loses squares is mended too.** `wood_0202` is a layout
  with every piece on one half of the board, kinds and sides in no order a
  game has. It goes from 24 squares to 0: before, a rank off and a quarter
  turn off; now on the right squares and half a turn round, 62 of 64 once
  turned, which is one tap in the app. The pieces of such a layout say
  nothing true about where White sits. It is not counted among the boards
  "exact or half a turn away", since two squares are wrong.
- **Fenit A's one** is an endgame of six pieces
  (`7r_8_8_6k1_R6p_6pK_8_8 w.jpg`) on which the app's fit took a window
  one file off: 55 squares before, all 64 now. It is the photo that was
  left a quarter off on Fenit A under the dark a1. The reference, from its
  own reading of the corner model and its own fit, moves no window on the
  development sets; the app moves this one.
- **Readings a quarter off**, counted as in "What a dark a1 is worth".
  By where the pieces land: the second held-out set 19 -> 1 (Y-LIChess 18
  -> 0), Fenit A 1 -> 0. By squares: the second held-out set 10 -> 1
  (Y-LIChess 9 -> 0), Fenit A 1 -> 0. The one left by either count is
  ChessQ `0011`, a photo whose square colours say nothing and on which
  only one of the model and the fit finds the board, so that this rule
  has no say.
- **How near the rule came to moving a window it should not.** It needs
  the mean of the four corners' offsets to round to a square, so to be
  over half a square on a rank or a file. Of the 2,000 photos left alone
  on which both found the board, four have such a mean: two of board B
  (0.85 and 0.62 squares) and two of ChessQ (0.69 and 0.59). On those the
  mean alone would have moved a window that was right, and only the
  corners' disagreement held it: 4, 3, 2 and 2 of the four corners lie
  outside the tolerance of 0.75, the furthest 3.5, 1.7, 1.1 and 1.3
  squares from the shift (the model's worst corner is 4.2, 2.7, 1.5 and
  1.4 squares from the fit's, along a rank or a file). On every other
  photo the mean is under 0.33. **The weak spot**: a corner model that is
  off the same way by more than half a square on average (and less than
  a square and a half), with every corner within 0.75 of the whole
  square, moves a window that was right. That takes every corner at
  least a quarter of a square off that way: 0.26, 0.26, 0.75 and 0.75
  along a rank move a window. No photo of these sets comes near it: on
  the 2,000 left alone, the corner least far off the common way is never
  more than 0.14 squares off it.
- **Windows moved on purpose** (`app_check --window F --shift-fit 0,1`; a
  window is moved back when every corner of the model lies within 0.75
  squares of where the fit had it, so each of the four directions gives
  the same count): board B 80 of 95, Fenit A 273 of 274, ChessReD test 287
  of 288. In Python the reference counts 60 of 92, 277 of 278 and 287 of
  289 (`Training/README.md`): on board B the app's corners of the model and of the fit lie
  nearer each other than the Python's do, for reasons not measured (the
  two resize the photo for the model differently, and their fits differ
  on some photos). With 0.5
  and 0.95 for the tolerance the app's count on board B is 68 and 88
  (Python 46 and 66). On the second held-out set (not to be chosen on,
  and it was not) 0.5 and 0.95 would move the same 18 windows.
- **The Swift against the Python on the same corners** (`app_check
  --window`, `swift_reference.py window`): the same shift on all 2,019
  photos where both found the board, and on the 657 of the development
  sets again with the window moved on purpose; the corners within a
  millionth of a pixel. The Python
  counted 14 windows moved on Y-LIChess from its own corners and its own
  fit, the app 18 from its own: the difference is in what the two are
  given, not in the rule.
- **Against the goal** on boards nobody trained on: 464 of 1,619 photos
  fully right (29%) and still 16 of board B's 100; the goal is 85%.
- **What the app marks** (confidence under 0.85) on the second held-out
  set: 58% of the wrong pieces (1,581 of 2,717; without the rule 52%,
  1,582 of 3,038), 8.3% of the right ones (2,835 of 33,995; 2,798 of
  33,533), 2.7 squares a photo. All of the change is Y-LIChess: 233 of
  450 wrong pieces marked against 234 of 771, 286 of 3,015 right ones
  against 249 of 2,553.
- **Time**: none that can be measured; the rule is two four-point
  homographies. Processor time over a set with the rule against without:
  ChessReD test 65.8 s against 65.2 s, Fenit A 89.6 s against 89.7 s.
  Single runs on a Mac mini that was training; nothing was timed on a
  phone.
- **What it cannot do.** It moves a window by one square, not two. It has
  no say where only one of the two finds the board (257 of the 1,619
  photos of the second held-out set: the fit alone, or the model's
  corners where the fit rejects itself), or where the model is off by a
  rank as well. And it trusts the corner model not to be off the same way
  by more than half a square on average with every corner a quarter of a
  square or more off: true of this one on these sets by a wide margin,
  and to be measured again with every new corner model.

## What a dark a1 is worth

The rule of step 7: the colours of the squares leave two of the four ways
a board can lie, and the pieces choose between those two. It came from the
reference (`Training/README.md`, "a1 is a dark square": chosen there on
board B, Fenit A and ChessReD test). Measured on the Mac mini on 2026-11-14
with the third pair through `app_check`, everything else as the app runs
it, the rule off and on through the same binary (the readings with it off
are, photo for photo, those of the last column of the table above). The
development sets were run first; the second held-out set was then run once,
as a test. Squares right; boards exact; boards exact or half a turn away
(one tap in the app); boards that are wrong as read and exact once turned
by a quarter.

| Set (photos) | Without the rule | **With it, as the app runs it now** |
|---|---|---|
| ChessReD test (306) | 99.80%; 295; 297; 0 | **99.80%; 295; 297; 0** |
| Fenit board B (100) | 94.36%; 16; 19; 0 | **94.48%; 16; 19; 0** |
| Fenit A (286) | 97.79%; 231; 247; 8 | **98.09%; 233; 255; 0** |
| *second held-out set:* | | |
| Y-LIChess (200) | 88.16%; 44; 49; 14 | **89.80%; 55; 63; 0** |
| ChessQ (50) | 93.03%; 14; 15; 1 | **93.44%; 14; 16; 0** |
| chesscog (29) | 98.17%; 8; 8; 0 | **98.17%; 8; 8; 0** |
| samryan (500) | 95.50%; 106; 112; 11 | **96.50%; 115; 123; 0** |
| chess-cv games (840) | 97.99%; 267; 267; 0 | **97.99%; 267; 267; 0** |
| **all (1,619)** | 95.86%; 439; 451; 26 | **96.38%; 459; 477; 0** |

- **No set loses, and no board is left that is right once turned by a
  quarter.** On the second held-out
  set 85 photos read differently: 58 have more squares right (606 squares)
  and 13 fewer (63); the other 14 have as many right as before, read
  another way round. Y-LIChess 27 better and 12 worse, samryan 29 and 1,
  ChessQ 2 and 0; chesscog and chess-cv read every photo as before. 20
  more boards are fully right (11 on Y-LIChess, 9 on samryan: each was a
  quarter off) and 26 more are right or half a turn away. On the
  development sets: Fenit A 11 photos (4 better by 59 squares, 3 worse by
  5: of its 8 boards a quarter off, 2 are now right and 6 half a turn
  away), board B 4 photos (2 better by 8 squares, none worse), ChessReD
  test none. As with the head start, what changes is boards with few
  pieces: 60 of the 85 photos, and 14 of the 15 on the development sets,
  have nine or fewer.
- **The 13 that lose** are 12 Y-LIChess photos and one of samryan. Eight
  of the twelve are full boards (30 to 32 pieces) that were read at 25
  squares of 64 or fewer before and lose one or two more: the corners
  there are not the board's (the grid fit accepts a grid a rank off on
  such photos, "What the app reads with the third pair"), the colours
  find a light a1 as the corners came, and a quarter turn is forced on
  what was wrong already. Three are layouts with every piece on one part
  of the board, a half or a corner of it (19, 17 and 16 squares before;
  0, 0 and 4 after, one of them now exact once turned by half). The last, and samryan's, are
  endgames that were a quarter off and are now half a turn off with one
  square fewer. No board that was fully right is lost.
- **Not every quarter turn is gone, and the rule makes some.** Where the
  colours are clear the pieces can no longer cause one. The colours can:
  with corners one square off the board along a rank or a file, the
  colours of the cells are the other way round, and the rule forces a
  quarter turn on a reading that was the right way round. It would do the
  same on a board set up with a light square on a1 (no set here has one).
  And where the colours say nothing (8 photos of 2,311) all four rotations
  stay open and a quarter turn can come about as before. Counted two
  ways, from the readings (no photo or corner label was looked at):
  - *By squares*: photos with more squares right once the reading is
    turned by a quarter, either way, than as read or turned by half,
    whether or not it is then fully right. The second held-out set 77
    without the rule and 10 with it (Y-LIChess 40 -> 9, ChessQ 4 -> 1,
    samryan 33 -> 0), Fenit A 10 -> 1, board B 4 -> 0, ChessReD test none
    either way. Of the 11 with the rule only two were there without it
    (`wood_0148` of Y-LIChess, `0011` of ChessQ); the rule made the other
    nine. This count is a floor: a reading a square off has about the same
    20 to 25 squares right whichever way it is turned.
  - *By where the pieces land*: the rotation at which most pieces of the
    reading stand on a square that holds the same piece in the position,
    the reading moved by up to one square (counted when at least three
    pieces land and two more than at the other pair of rotations). A
    quarter off: the second held-out set 56 without the rule and 19 with
    it (Y-LIChess 20 -> 18, ChessQ 4 -> 1, samryan 32 -> 0), Fenit A 10 ->
    1, board B 2 -> 0, ChessReD test none.
  - **The 16 the rule made**: of Y-LIChess's 18, 16 are full boards that
    were read the right way round and one rank off without the rule (the
    grid fit accepts a grid a rank off there, "What the app reads with
    the third pair") and are a quarter turn off as well with it: turned
    back and moved one rank, every piece found, or all but one or two,
    stands on its own piece. On all 16 the colours found a light a1 as
    the corners came. Their squares do not show it: 20 to 25 of 64 before,
    19 to 24 after, 357 and 358 in all. The eight full boards that lose
    a square or two, above, are all among them. For the user it is a
    reading that was useless and is now useless and turned. (Since
    2026-11-15 the corner model moves the window on all 16: "What the
    model's choice of window is worth".)
  - The other three: two Y-LIChess layouts, `wood_0148` (Black's pieces
    alone; a quarter off with and without the rule, 48 squares against
    43) and `wood_0202` (half a turn off without the rule, a quarter off
    with it, 32 against 24), both read a rank off as well; and ChessQ
    `0011` (52 against 35), one of the photos whose colours say nothing,
    read the same with and without the rule. Fenit A's one is an endgame
    of which three of six pieces were found, which fit a quarter turn and
    one file off.
  - The reviewer of pull request #23 found `wood_0148`; the reviewer of
    #24 found that the rule turns the boards that are a rank off.
- **Against the goal** on boards nobody trained on: 459 of 1,619 photos
  fully right (28%; 27% without the rule) and still 16 of board B's 100;
  the goal is 85%.
- **What the colours say.** They are unclear on 8 photos of 2,311 (board B
  3, Fenit A 1, ChessQ 4), and there the four rotations stay open as
  before. The corners as they came had a light a1 on 1 photo of Fenit A,
  none of ChessReD test and 30 of board B, and on 192 of the second
  held-out set (samryan 123, Y-LIChess 60, ChessQ 7, chesscog 1, chess-cv
  1): this corner model's a1 is often a quarter off on a board it has not
  seen, and on most of those photos the pieces already turned it right
  without the rule.
- **Half turns are what one tap can still mend**: 18 boards of the second
  held-out set and 22 of Fenit A are right once turned by half. The rule cannot help
  there, nor with the endgames of ChessReD game G033 ("Known weak spot"):
  as 3:4 photos ChessReD test reads as before (99.68%; 282; 284), the
  fixture `test_004005.jpg` among the turned ones, and board B 94.45% (was
  94.33%); 16; 19.
- **The Swift against the Python on the same corners** (`app_check --dark`,
  `swift_reference.py dark`): the same answer on all 2,311 photos, and
  the strength of the split within one cell of 32 on every one of them
  (equal at the 95th percentile of each set).
- **Time**: none that can be measured. The photo is shrunk to the grid
  fit's working size once, for the fit and for the colours. Processor
  time over a whole set, with the rule against without: ChessReD test
  65.5 s against 65.5 s, Fenit A 88.8 s against 88.6 s. Shrinking the
  photo a second time for the colours cost 20.0 s and 28.4 s more on
  those sets (65 and 99 ms a photo) and 14 ms a photo on board B, whose
  photos are smaller. Single runs on a Mac mini that was training at the
  time; nothing was timed on a phone.
- **What the app marks** (confidence under 0.85) on the second held-out
  set with the rule: 52% of the wrong pieces (49% without), 8.3% of the
  right ones and 2.7 squares a photo, as before.
- **What it cannot do, and what was not measured.** A board set up with a
  light square on a1 is read a quarter turn off, and the flip button does
  not mend that; no set here has such a board, so how often people do it
  is not known. Nothing was run on a phone. The head start for the corner
  model's a1 was not looked at again: 12 was not taken because it turned
  sparse boards by a quarter, which the rule now rules out where the
  colours are clear, so that
  question is open again, for the reference first and not for another
  look at the second held-out set.

## What the app read with the second pair

The pair in `Models/` from pull request #15 (2026-11-14) until #19: the
first corner model, and the piece model `pieces_s_render3` (a YOLO11s, 18 MB
where the one before it was a YOLO11n of 5 MB; how it was trained and on
what is in `Training/README.md`). Measured on the Mac mini on 2026-11-14
through `app_check` built from ae9dd12, the merge of #15, each photo from
one full decode. **Nothing here was run on a phone**: not the readings, not
how long the larger model takes there. Squares right; boards exact; photos
in which no board was found. (Where this section says "now", "the app
reads" or "the goal", it speaks of its own time, 2026-11-14 before pull
request #19; the third pair's section above is for the pair that #19
puts in.)

| Set (photos) | Grid fit, a1 from the pieces and the rule for the kings off | Rule for the kings off | **As the app runs it** |
|---|---|---|---|
| ChessReD test (306) | 98.36%; 246 | 99.68%; 288 | **99.68%; 288** |
| Fenit board B (100; no photo of it was trained on) | 46.88%; 0; no board in 36 | 88.45%; 5; 4 | **88.55%; 6; 4** |
| Fenit A (286; the piece model was trained on these photos) | 74.39%; 0 | 95.37%; 184 | **95.32%; 183** |
| *second held-out set:* | | | |
| Y-LIChess (200) | 0%; 0; 200 | 85.60%; 34; 6 | **85.73%; 45; 6** |
| ChessQ (50) | 28.44%; 0; 26 | 71.69%; 14; 10 | **71.66%; 14; 10** |
| chesscog (29) | 32.97%; 0; 10 | 98.28%; 10; 0 | **98.44%; 10; 0** |
| samryan (500) | 55.36%; 0; 72 | 84.72%; 43; 16 | **84.85%; 58; 16** |
| chess-cv games (840) | 54.20%; 0; 33 | 91.74%; 117; 13 | **91.97%; 171; 13** |
| **all (1,619)** | 46.69%; 0; 341 | 88.31%; 218; 45 | **88.49%; 298; 45** |

(The first column is `--a1-prior off --grid-fit off --one-king off`, the
second `--one-king off`. Counting boards that are right once turned by 180
degrees, one tap in the app: board B 7, Fenit A 201, the second held-out
set 302. The last column is, photo for photo, what the training session's
own run of the same build gave; those outputs are kept on the Mac mini in
`Training/runs/app_check/final/` and `final_h2/` (`render3_*.txt`, and
`now_*.txt` for the pair before). The other two columns' outputs are not
kept: the switches above give them again.)

Which of these sets say what about a board the app has not met:

- **The second held-out set is the only blind one.** Nothing was trained
  or chosen on it.
- **Board B** has no photo in any training set, but it is no longer blind:
  the rendered boards that taught this model the side of a piece were made
  from what went wrong on board B, and candidates were compared on it
  (`Training/README.md`).
- **Fenit A and ChessReD** are boards the piece model was trained on (other
  photos of ChessReD's boards; Fenit A's own photos). They show that
  nothing was lost, not what a new board gets.

- **Where this stands against the goal.** On boards nobody trained on, the
  app reads 298 of 1,619 photos fully right (18%) and 6 of board B's 100;
  the goal is 85% of unseen boards. With the pair before it (the first
  pair, below) and the same code it was 3 of 1,619 and 1 of 100; on the
  boards it knows, ChessReD test 280 of 306 and Fenit A 32 of 286 (whose
  photos the first piece model had not been trained on).
- **The grid fit and a1 from the pieces** are worth at least what they were
  with the first pair: ChessReD test 246 -> 288 boards (first pair 239 ->
  279); on the second held-out set 47% -> 88% of squares (first pair 46% ->
  76%), with a board found in all but 45 photos instead of all but 341
  (44 and 340). The corner model is the same one.
- **The rule for the kings is worth more on unseen boards and nothing on
  the others**: 80 more boards fully right on the second held-out set (218
  -> 298, where the first pair had 3 with and without it), one more on
  board B (as with the first pair), none on ChessReD test (first pair: one)
  and one fewer on Fenit A (first pair: 16 more). The rule mends a board
  whose only mistake is a king; on unseen boards such boards come with a
  better piece model, and on the boards this model knows it leaves none to
  mend.
- **What the rule costs.** Boards are lost only on positions set up
  without two kings: on Fenit A two boards are lost and one gained, 14
  squares lost and 5 gained, every loss on its 13 such positions, and the
  three boards Y-LIChess loses are among its 50. No board with two kings in
  its position was lost on any set. Single squares are lost on ordinary
  positions too: on the second held-out set about 100 photos lose a square
  while about 265 gain one or more, and ChessQ ends a square down.
- **What the app marks for the user to check** (a piece whose confidence is
  under 0.85, `AppModel.lowConfidenceThreshold`), and what 0.5 and 0.7 would
  mark: wrong pieces marked / right pieces marked / squares marked a photo.

| Set | Under 0.5 | Under 0.7 | Under 0.85 (as now) |
|---|---|---|---|
| ChessReD test | 22% / 0.02% / 0.03 | 35% / 0.2% / 0.08 | 38% / 0.7% / 0.20 |
| Fenit board B | 6.6% / 1.7% / 0.44 | 24% / 5.9% / 1.6 | 54% / 22% / 4.6 |
| Fenit A | 5.0% / 0.3% / 0.11 | 10% / 1.3% / 0.30 | 22% / 11% / 1.6 |
| second held-out set | 7.8% / 1.3% / 0.45 | 20% / 3.3% / 1.1 | 40% / 8.8% / 2.7 |

  The new model is surer of itself, right and wrong: on unseen boards the
  mark now falls on 2.7 squares a photo where it fell on 11.3 with the
  first pair, and catches 40% of the wrong pieces where it caught 79%. The
  threshold is still the owner's to move ("What the confidence marks",
  below, has the first pair's table). Nothing of this was tried on a phone.

## The older measurements: the first pair

The sections from here on were written before pull request #15 and are
kept as the record of how each step was chosen. Where a table does not say
otherwise they were measured with the pair that was in `Models/` from
b546e6a until #15, called **the first pair** below: the first corner model
(`main`'s original, a YOLO11n, in `Models/` until pull request #19), and
the first piece model, a YOLO11n
with `main`'s baseline weights, exported with NMS across classes. (It is
not the very first content of `Models/`: `main`'s original export of the
same weights, with NMS for each class by itself, is the row "`main`'s two
models as on `main`" in the next section, and exp3's corners were in
`Models/` for a time.) `pieces_s_render1` was a candidate piece model; it
was in `Models/` on the branch for a few minutes on 2026-11-13 (773e729,
out again in b32117e) and is never what "the first pair" means.
`pieces_s_render3`, two generations on, was the piece model of the second
pair. What the app reads with the second and the third pair is in the two
sections above. The steps are
the same code for both; what they are worth depends on the piece model, so
the sections below say how each step was chosen, and the sections above
say what the later pairs read. Where a section below says "today", "now" or "the
app's default path", it speaks of its own time. ("Known weak spot" at the
end is about corner models and stands as it is.)

## What the two steps are worth

(Measured before pull request #12, with Vision's share as a piece's
confidence and without the rule for the kings. With those two, the first
pair's figures are the "on" column of "What one king a side is worth"; the
difference to the "both" rows here is a quarter of a point of squares at
most, and Fenit A's 16 boards exact are 32. What the app reads with the
later pairs is in the two sections above.)

Through `app_check` (the app's own code, Vision doing the resize) on the Mac
mini, 2026-11-13, each photo decoded once (`DecodedImage`). "Off" is
`--a1-prior off --grid-fit off`, "both" is the app's default (grid fit on,
prior 6, colour step off). Boards exact, and in brackets boards that are
exact once turned by 180 degrees (one tap in the app).

**The first pair (b546e6a: `main`'s corner model, the baseline piece
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
one with noise: the first pair with both steps scored 280 of 306, 82.3%
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

- With the first pair, the two steps take ChessReD from 239 to 279
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

| Source (photos) | First pair, steps off | First pair, both steps | `pieces_s_render1`, both steps |
|---|---|---|---|
| Y-LIChess (200) | 0%; 0; no board in 200 | 72.4%; 2; 6 | 84.8%; 14; 6 |
| ChessQ (50) | 28.3%; 0; 26 | 64.6%; 1; 10 | 69.8%; 6; 10 |
| chesscog (29) | 32.9%; 0; 10 | 73.3%; 0; 0 | 93.1%; 0; 0 |
| samryan (500) | 54.8%; 0; 72 | 70.9%; 0; 16 | 83.2%; 16; 16 |
| chess-cv games (840) | 53.5%; 0; 32 | 80.6%; 0; 12 | 89.3%; 9; 12 |
| **all (1,619)** | **46.2%; 0; 340** | **75.9%; 3; 44** | **86.3%; 45; 44** |

(This table and the points under it are from before pull requests #12 and
#14: Vision's share as a piece's confidence, no rule for the kings, and
chess-cv's photos, which carry an EXIF orientation, read through ImageIO's
thumbnail call and not from one full decode, whatever the line above the
table says of each photo. The chess-cv row and the totals are therefore a
little off: by a tenth of a point of squares, a few boards exact and one
photo without a board (44 is 45 from the full decode), going by what the
change of path did to the first pair's figures. Those are in "What one
king a side is worth"; the later pairs' are in the two sections above.)

- The two app-side steps take the first pair from 46% to 76% of
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
  Y-LIChess, whose photos have such pieces: 70.1% -> 72.4% with the first
  pair, 78.0% -> 84.8% and 0 -> 14 boards exact with `pieces_s_render1`.
  On the three development sets it changes next to
  nothing (first pair: ChessReD test 279 of 306 and board B 84.58%
  as before, Fenit A 91.38% -> 91.43%), which is why half a square went
  unnoticed there.
- With the colour step on (measured with the margin at half a square), the
  first pair stays at 75.5% (1 board exact) and `pieces_s_render1`
  drops from 85.4% to 82.6% (27): Y-LIChess 78.0% -> 65.5%, samryan 82.9% ->
  79.0%, ChessQ 69.7% -> 68.5%, the other two unchanged. See the next
  section.

## What one king a side is worth

Through `app_check`, 2026-11-13, both steps on, colour step off, margin a
quarter of a square, a piece's confidence the model's score; `--one-king off`
against the default. Squares right; boards exact.

| Set (photos) | First pair: off -> on | `pieces_s_render1`: off -> on |
|---|---|---|
| ChessReD test (306) | 99.53%; 279 -> 99.54%; 280 | 99.64%; 285 -> 99.64%; 285 |
| Fenit board B (100) | 84.58%; 0 -> 84.67%; 1 | 84.31%; 1 -> 84.33%; 1 |
| Fenit A (286) | 91.44%; 16 -> 91.69%; 32 | 95.35%; 185 -> 95.41%; 185 |
| *second held-out set:* | | |
| Y-LIChess (200) | 72.57%; 2 -> 72.49%; 2 | 84.77%; 14 -> 85.02%; 23 |
| ChessQ (50) | 64.56%; 1 -> 64.62%; 1 | 69.78%; 6 -> 69.81%; 6 |
| chesscog (29) | 73.49%; 0 -> 73.71%; 0 | 93.21%; 0 -> 93.59%; 0 |
| samryan (500) | 70.95%; 0 -> 71.12%; 0 | 83.23%; 16 -> 83.32%; 18 |
| chess-cv games (840) | 80.60%; 0 -> 80.65%; 0 | 89.43%; 13 -> 89.72%; 21 |
| **all (1,619)** | **76.00%; 3 -> 76.08%; 3** | **86.40%; 49 -> 86.62%; 68** |

(chess-cv and samryan as re-scored for pull request #14, their photos with
an EXIF orientation turned from one full decode. Through the thumbnail
call, as first measured, chess-cv was 80.55%; 0 -> 80.62%; 0 and 89.31%; 9
-> 89.62%; 16, and the totals 75.98%; 3 -> 76.07%; 3 and 86.34%; 45 ->
86.56%; 63 (given at first as 76.06% and 86.57%, from the rounded parts;
the totals here are from the squares counted). No board is found in 45
photos in every column; through the thumbnail call it was 44. The outputs
for every photo are kept outside the repository, on the Mac mini, in
`Training/runs/app_check/one_king/` and `exif_full_decode/`.)

- **On by default.** The three development sets said so when it was
  chosen (with `pieces_s_render3`, later, Fenit A loses a board: "What the
  app read with the second pair"): no set loses in total with either piece model, and the
  first pair gains on all three, Fenit A's boards exact doubling (16 to 32: boards whose only
  mistake was a king read as a queen or a bishop, or a queen read as a
  second king). With `pieces_s_render1` the development sets hardly move
  (no photo of ChessReD test changes, board B gains seven squares and loses
  six, Fenit A's 185 boards exact are two gained and two lost). The second
  held-out set was not used to choose; it agrees, by 0.1 points of squares
  with the first pair and 0.2 with `pieces_s_render1`, where 19 more boards
  come out fully right (49 to 68 of 1,619).
- Single photos do lose: with the first pair ten Fenit A photos lose
  11 squares between them while fifty gain 56.
- **Positions without two kings lose.** The rule takes every position to
  have one king a side. 13 of Fenit A's 286 positions and 50 of Y-LIChess's
  200 are set up without (pieces laid out, not games), and there the rule
  can only do harm: the three boards that were exact and are not any more
  (two on Fenit A, one on Y-LIChess, all with `pieces_s_render1`) are such
  positions. The app is for positions from games.
- It is a small step: it changes a piece or two of a board (more only where
  the model saw four or more kings). It pays in boards fully right, and only
  once the rest of the board is right, which is why it shows with
  `pieces_s_render1` and not with the first pair on boards nobody has
  seen.
- **One part goes the other way**: Y-LIChess with the first pair,
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

| Set | First pair: share -> score | `pieces_s_render1`: share -> score |
|---|---|---|
| ChessReD test | 99.52%; 279 -> 99.53%; 279 | 99.64%; 285 -> 99.64%; 285 |
| Fenit board B | 84.58%; 0 -> 84.58%; 0 | 84.28%; 1 -> 84.31%; 1 |
| Fenit A | 91.43%; 16 -> 91.44%; 16 | 95.34%; 185 -> 95.35%; 185 |
| second held-out set | 75.94%; 3 -> 75.98%; 3 | 86.33%; 45 -> 86.34%; 45 |

(Second held-out set, here and in the next table: chess-cv through the
thumbnail call, both halves alike; see "How the Swift is checked against
the Python".)

What it changes is what the user is shown. The app marks a piece whose
confidence is under 0.85 for the user to check
(`AppModel.lowConfidenceThreshold`, not changed). `app_check` counts how
many of the pieces it put on a square that marks, among those on a wrong
square or of a wrong class and among the right ones:

| Set | First pair: wrong pieces marked | right pieces marked | `pieces_s_render1`: wrong pieces marked | right pieces marked |
|---|---|---|---|---|
| ChessReD test | 10% -> 55% (36 of 66) | 0.05% -> 1.0% | 13% -> 39% (18 of 46) | 0.06% -> 1.1% |
| Fenit board B | 24% -> 76% | 6.5% -> 40% | 16% -> 60% | 3.8% -> 27% |
| Fenit A | 18% -> 64% | 4.7% -> 23% | 3.5% -> 23% | 0.2% -> 12% |
| second held-out set | 15% -> 78% | 5.5% -> 47% | 18% -> 46% | 3.2% -> 11% |

(The "share" halves of both tables cannot be measured with the code as it
is now: their board scores are those of the commit before pull request #12,
their marks came from a build with a temporary switch back to the share.)

- With the share, the mark said next to nothing: it left more than four in
  five of the wrong pieces of the second held-out set unmarked (three in
  four on board B), because a model that is unsure whether a piece is there
  at all is seldom split between two classes.
- With the score, the mark follows what the model knows. With the first
  pair, on a board like the ones it was trained on, one right piece in a
  hundred was marked and half of the few wrong ones. On a board it had not
  seen, it marked nearly half of the right pieces too: there it put 13,000
  pieces wrong and 16,500 right, and its scores said as much.
- A piece whose kind the rule for the kings changed is shown with the
  model's score for the kind it became (a queen at 0.93 that became a king
  on a king score of 0.05 is shown at 0.05), so it is marked. The board
  read is the same either way.
- **The threshold of 0.85 was set when the number under it was another
  one, and it is the owner's to move.** With it, the first pair marked 11
  squares a photo on the second held-out set where it had marked 2, enough
  for the app's line under the board to say "N squares to double-check" on
  most photos of an unfamiliar board (with the second and with the third
  pair it is 2.7 a photo: see their sections). What other thresholds marked,
  on the app's default path (rule for the kings on); wrong pieces marked /
  right pieces marked / squares marked a photo
  (`app_check --flag-below 0.5,0.7,0.85`):

| Set | Under 0.5 | Under 0.7 | Under 0.85 (as now) |
|---|---|---|---|
| *first pair:* | | | |
| ChessReD test | 27% / 0.1% / 0.08 | 44% / 0.3% / 0.15 | 53% / 1.1% / 0.33 |
| Fenit board B | 25% / 8.2% / 2.2 | 53% / 19% / 4.7 | 77% / 40% / 8.1 |
| Fenit A | 18% / 5.6% / 1.2 | 40% / 12% / 2.6 | 63% / 24% / 4.5 |
| second held-out set | 33% / 10% / 3.7 | 56% / 24% / 7.0 | 79% / 48% / 11.3 |
| *`pieces_s_render1`:* | | | |
| ChessReD test | 15% / 0.05% / 0.03 | 30% / 0.3% / 0.10 | 39% / 1.1% / 0.27 |
| Fenit board B | 15% / 2.2% / 1.1 | 32% / 10% / 2.9 | 62% / 28% / 6.4 |
| Fenit A | 4.0% / 0.1% / 0.07 | 10% / 1.2% / 0.29 | 23% / 12% / 1.7 |
| second held-out set | 9.8% / 1.9% / 0.70 | 25% / 4.9% / 1.8 | 46% / 12% / 3.8 |

  With the share at 0.85 the same sets had 0.03, 1.9, 1.1 and 1.8 squares
  marked a photo with the first pair, and 0.03, 1.3, 0.08 and 1.3
  with `pieces_s_render1` (the last of each for the second held-out set
  with chess-cv through the thumbnail call). No threshold marks most of the wrong pieces and
  few of the right ones: on a board the model has not seen, its scores for
  right and wrong pieces overlap. Nothing of this was tried in the app on a
  phone.

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
| first | off | 279 (279); 99.52% | 0 (1); 84.58% | 16 (19); 91.38% |
| first | **on** | 279 (279); 99.52% | **2 (3); 85.50%** | 16 (19); 91.38% |
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
  either model. With the first piece model it changes a side on 36 board B
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
  ChessQ 69.7% -> 68.5%, samryan 82.9% -> 79.0%, chesscog and chess-cv
  unchanged. With the first piece model it gains on three parts and loses
  on one (Y-LIChess 70.1% -> 71.8%, ChessQ 64.6% -> 64.8%, chess-cv 80.5%
  -> 81.0%, chesscog unchanged, samryan 70.4% -> 68.8%), which comes to
  nothing over the whole set (75.5% both ways), and it loses one of that
  model's two exact boards. (All of these from before pull request #14:
  chess-cv through the thumbnail call.) So it is off
  because it failed with `pieces_s_render1`, the model it was made for, and
  was worth nothing with the first piece model. It has not been measured
  with `pieces_s_render3` or `pieces_s_tone1`.
- The rule has no smallest distance between the two groups, and with six
  pieces or fewer one disagreement is already over 15%. The Fenit A photos
  it acts on are such positions (2 to 9 pieces found): on one, two black
  pieces are split into a "white" and a black one. Asking for two
  disagreeing pieces was tried in the Python and is no better (board B has
  many sparse endgames where the one brown king is what needs correcting).
- On the same boxes the Swift and the Python give every piece the same
  final class: 23,510 pieces on the 692 photos of the three sets, found by
  the first piece model (11,737) and by `pieces_s_render1` (11,773), the
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
lopsided enough (7.75 against -1.75). The first corner model gets a1 right
there, which is why it stayed in `Models/` (b546e6a) while exp3 and exp4
were tried.

`corners_s_canon1`, the corner model of the third pair, gets a1 right on
the fixture as the square photo it is, and wrong on the same photo drawn
as a 3:4 photo and on two neighbouring frames of the game (G033 IMG100
and IMG102 square; IMG101 and IMG102 as 3:4 photos). The head start does
not settle it. IMG100 and IMG102 score better turned whatever it is (the
training session found them turned at 0, 3, 6 and 12): there the pieces
are wrong and the corner model does not correct them. The fixture as a 3:4
photo scores +2.5 the right way up and the corner model has a1 at the
wrong end, so it is read right with a head start under 5 and turned from 5
up (the reviewer of pull request #20 tried 0, 3, 4, 5, 6 and 12). A smaller
head start is not on offer on what the development sets say: it loses
boards on Fenit A and on ChessReD test, both boards the corner model was
trained on, and those sets can no longer place this number alone; a larger
one loses squares on the second held-out set ("What the app reads with the
third pair"). Pull request #19 puts this corner model in
`Models/` all the same, because the pair reads far more boards, also as
3:4 photos, and the owner was told of this cost with the way to take the
corner model out again. The tests say so each time they
run: the fixture as a 3:4 photo is a known issue of
`testRecognizesFixturesAsPortraitPhotos`. A corner model that gets these
frames right can have that exception taken out; until then the app's flip
button is the user's.
