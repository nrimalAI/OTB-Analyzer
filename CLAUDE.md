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
- Don't replace the models in `Models/`, push, or merge to `main` without the owner's go-ahead. **The owner has given the Mac mini a standing go-ahead for the retraining work: see the next section.**

## Autonomous retraining on the Mac mini (owner's standing go-ahead, 2026-11-12)

The owner is abroad from 2026-11-13 with only the MacBook. They want the Mac mini to finish the retraining end to end with no human in the loop. Their words: "fully autonomous and what it thinks is best"; "finish all of the work it can end to end by itself without human intervention"; on the target, "85% of boards fully correct on an unseen board ... but try even higher, no true fixed bar".

**The goal.** Models that read boards they were never trained on: at least 85% of boards fully correct on a board held out of training, and higher if it can be had, without getting worse on the ChessReD test split (baseline 97.9% per-square, 73.5% exact). Phone-sized models only: they run on-device.

**What Mac mini sessions may do without asking:**

- Train, evaluate, and decide the next experiment. Download public datasets. Install free tools the work needs (e.g. Blender, to render boards with exact corner labels). Fix and extend the scripts under `Training/`.
- Commit and push to `data/multi-source-datasets`.
- When a model is clearly better: export it, replace `Models/*.mlpackage`, open a pull request to `main`, and merge it yourself.

**Before replacing the models or merging, all of these must hold:**

- The new models beat the ones in `Models/` on the held-out board by more than the noise (`evaluate.py` prints the interval), and are not worse on the ChessReD test split beyond noise. Scores for old and new, on both, are in the pull request and in `Training/README.md`.
- The model interfaces are unchanged (corners: raw `[1,17,8400]`, no NMS; pieces: NMS pipeline with the same class labels), scored through the exported `.mlpackage`, letterboxed.
- `origin/main` is merged into the branch first (it has the letterbox fix in `Packages/ChessVision`), and `swift test` in `Packages/ChessVision` passes with the new models in place.
- The weights of every model that reaches `main` are kept under `Training/baseline/` with a line in its README, so any step can be undone. Merge commits only: never force-push, never rewrite `main`.
- **Every pull request is reviewed with Fable before it is merged, and the review's changes are applied** (owner, 2026-11-12: "always review a PR and apply changes with fable before merging"). Open the pull request, have a Fable reviewer (`claude-fable-5-2`; `/code-review` on the pull request, or a reviewer subagent on that model) read the whole diff, check each finding against the code, apply the ones that hold up, re-run the tests and re-score if the change could affect a score, push, and only then merge. Say in the pull request what the review found and what was done about each finding. This applies to every pull request, not only ones that replace models.

**Limits that still apply:**

- Spend no money: no cloud GPUs, no paid APIs or datasets.
- No secrets in the repo or the notes. Do not change system settings that need the owner's password.
- **App code (owner, 2026-11-13, boarding a flight with the MacBook off: "do everything on the mac mini").** Mac mini sessions may now change code under `Packages/ChessVision`, and under `App/` where a change needs it. This replaces the earlier rule that left all app code to the MacBook. It was given for the two steps measured in `Training/README.md`: board corners from the classical grid fit, and a1 from the detected pieces. `Training/chess_pipeline.py` stays the reference the Swift code must match; add tests for what you add; keep the fixed decisions above (iOS 17, no network, no third-party dependencies).
- `swift test` must still pass before anything is merged, and the Mac mini cannot run it while Xcode's licence is unaccepted (that needs the owner's password). Find a way that needs no password and costs nothing: the Command Line Tools through `DEVELOPER_DIR`, or a GitHub Actions macOS job (the repo is public). Failing that, leave the pull request open and ask the MacBook through the notes; it tests when the owner is next online.
- Keep at least 20 GB of disk free. Never delete a downloaded source dataset to make room without recording why.
- A model trained on ChessReD is for personal builds only; say in each pull request which sources the model was trained on.

**How to work alone:**

- Do not stop to wait for a go-ahead. After each run: score it, record it, decide the next step, start it. When one idea is exhausted, pick the next lever yourself (more varied boards, rendered data, loss and augmentation settings, model size within phone limits).
- Nobody is there to prompt you when a run ends in the night. Run under a self-paced loop (`/loop`) or a scheduled check, so the session wakes, sees the finished run, and carries on.
- Start long jobs detached from the session (so closing or restarting a session does not kill them) and record under **State** in `.notes/training.md` what is running, where its log and checkpoints are, and how to resume it.
- `.notes/training.md` is the owner's only window: keep **State** true, and put anything only the owner can do at the top under **Needs the owner**, then carry on with other work.
- A new session must be able to pick up from the notes and `Training/README.md` alone. Write them for that reader.
