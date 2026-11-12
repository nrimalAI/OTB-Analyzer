# OTB Analyzer ("Chess Scanner")

Native iPhone app: photograph a physical chess board, two on-device CoreML YOLO models find the board corners and the pieces, the app builds a FEN, the user corrects any wrong squares, and on-device Stockfish 17 analyzes the position. Fully offline. GPL-3.

Read before doing anything substantial:

- `docs/PLAN.md`: the original plan, architecture, and the licensing constraints.
- `Training/README.md`: the training workflow, data sources, accuracy findings and dead ends.
- `README.md`: project layout and how to build.

## Two machines

The owner works on this repo from two computers. Claude's own memory does not sync between them, so anything a future session needs must live in the repo or in the project memory below.

- **MacBook (app side):** the app, design, Xcode builds. Its disk is nearly full: **no dataset downloads and no training here.**
- **Mac mini (training side):** all dataset downloads, training, evaluation and CoreML export. Usually driven over Remote Control.

## Project memory (automatic)

Project memory lives in its own private repo, `nrimalAI/otb_analyzer_notes`, not in this one. A hook (`.claude/hooks/memory.sh`, wired in `.claude/settings.json`) clones it into `.notes/` at the top of each checkout and worktree. `.notes/` is git-ignored here, so notes never enter this repo's history.

- `.notes/app.md`: the MacBook side. Sessions on the MacBook write it.
- `.notes/training.md`: the Mac mini side. Sessions on the Mac mini write it.

Which side a machine is comes from `git config claude-notes.side` (`app` or `training`). The hook sets it the first time: a Mac mini is `training`, anything else `app`. The hook's first line at session start says which side you are; write only that side's file and treat the other as read-only.

What the hook does:

- **Session start:** fast-forwards the current branch of this repo, syncs `.notes/`, and prints both files.
- **Before a prompt:** at most every 5 minutes, syncs `.notes/` and prints whichever file changed. It does not pull code mid-session.
- **When you stop:** if any file in this repo is newer than your side's file, it asks you once to update it. Then it commits `.notes/` and pushes, rebasing first if the other machine pushed in between. A conflict or a failed push is reported, not hidden.

How to keep it:

- Rewrite **State** in place so it is always true now. Add to **Decisions** and **Open questions** only what a future session needs. Add one dated line to **Log** per finished piece of work; keep the last 15 or so.
- To tell the other machine something, put it under **For the other side** in your own file; its sessions see it on their next start or prompt.
- Long jobs (downloads, training): record under **State** what is running, where its log and checkpoints are, and how to check on or resume it.
- Keep the file under about 60 lines and 4,000 characters; the hook cuts what it prints to fit. Detail belongs in `docs/` or `Training/README.md`, with a pointer from the notes.
- No secrets in the notes. Do not commit or push `.notes/` yourself. If nothing worth recording changed, `touch` your side's file.
- If the hook reports a notes conflict, resolve it as its message says, keeping the facts from both versions.

## Fixed decisions

- iPhone only, portrait, SwiftUI, iOS 17.0+. No network calls, analytics or third-party UI dependencies.
- The app is GPL-3 (Stockfish). `AboutView` carries the source offer and must stay reachable.
- ChessReD is NonCommercial: fine for personal builds, but a model shipped on the App Store must be trained without it (`build_dataset.py merge --shippable`).
- Project structure is defined in `project.yml` (xcodegen); never hand-edit the `.xcodeproj`. `*.nnue` files and the `.xcodeproj` are git-ignored: run `./Scripts/download-nnue.sh` and `xcodegen generate` in a fresh checkout.
- `Training/chess_pipeline.py` is the reference the Swift code in `Packages/ChessVision` must match.
- Don't replace the models in `Models/`, push, or merge to `main` without the owner's go-ahead.
