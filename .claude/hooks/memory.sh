#!/bin/bash
# Project memory hook for Claude Code. The protocol is described in CLAUDE.md.
#
# The project's notes live in their own private repo, nrimalAI/otb_analyzer_notes,
# which this hook clones into .notes/ at the top of each checkout (git-ignored):
#
#   app.md       written by sessions on the MacBook (app, design, Xcode)
#   training.md  written by sessions on the Mac mini (datasets, training)
#
# Both machines work in the same code repo, so the side is a property of the
# machine: `git config claude-notes.side app|training`, set automatically the
# first time (a Mac mini is the training side, any other machine the app side).
#
#   start   session start: fast-forward the code, sync the notes, print both files
#   prompt  before a prompt, at most every 5 minutes: sync the notes and print
#           whichever file changed since this session last saw it
#   stop    when Claude stops: ask for a notes update if work is newer than this
#           side's file, then commit and push the notes
#
# It never touches the code repo's history, and it never fails a session:
# every error path exits 0. Adapted from the hook in nrimalAI/voice_solver.

set -u
MODE="${1:-}"
INPUT="$(cat 2>/dev/null || true)"
export GIT_TERMINAL_PROMPT=0

RECHECK_SECONDS=300
OUTPUT_BUDGET=8200  # hook output over 10,000 characters is replaced by a preview

# Run a command with a time limit (macOS has no `timeout`).
limit() { perl -e 'alarm shift; exec @ARGV' "$@"; }

field() {  # a top-level field of the hook's JSON input
  printf '%s' "$INPUT" | perl -MJSON::PP -e '
    binmode(STDOUT, ":utf8"); local $/;
    my $d = eval { decode_json(<STDIN>) }; exit 0 unless ref($d) eq "HASH";
    my $v = $d->{$ARGV[0]};
    print ref($v) ? ($$v ? "true" : "false") : ($v // "")' "$1" 2>/dev/null
}

json() { perl -MJSON::PP -e 'print JSON::PP->new->canonical->encode({@ARGV}), "\n"' "$@"; }

mtime() {
  if [ "$(uname)" = "Darwin" ]; then stat -f %m "$1" 2>/dev/null || echo 0
  else stat -c %Y "$1" 2>/dev/null || echo 0; fi
}

side_of() {  # which half of the project this machine is; fails outside the repo
  local name side
  name="$(git -C "$1" remote get-url origin 2>/dev/null)" || return 1
  name="${name##*/}"
  name="$(printf '%s' "${name%.git}" | tr 'A-Z' 'a-z')"
  [ "$name" = otb-analyzer ] || return 1
  side="$(git -C "$1" config --get claude-notes.side 2>/dev/null)"
  case "$side" in app|training) echo "$side"; return 0 ;; esac
  # Not set yet: decide from the hardware once and remember it.
  if system_profiler SPHardwareDataType 2>/dev/null | grep -q 'Model Name: Mac mini'; then
    side=training
  else
    side=app
  fi
  git -C "$1" config claude-notes.side "$side" >/dev/null 2>&1
  echo "$side"
}

# The checkout this session is working in. The hook input's cwd comes first:
# in a worktree session CLAUDE_PROJECT_DIR can point at the main checkout.
ROOT="" SIDE=""
for dir in "$(field cwd)" "${CLAUDE_PROJECT_DIR:-}" "$PWD"; do
  [ -n "$dir" ] && [ -d "$dir" ] || continue
  top="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)" || continue
  if [ "$(basename "$top")" = ".notes" ]; then  # inside the notes clone: step out
    top="$(git -C "$(dirname "$top")" rev-parse --show-toplevel 2>/dev/null)" || continue
  fi
  SIDE="$(side_of "$top")" || continue
  ROOT="$top"
  break
done
[ -n "$ROOT" ] || exit 0
cd "$ROOT" 2>/dev/null || exit 0
GIT_DIR_PATH="$(git rev-parse --absolute-git-dir 2>/dev/null)" || exit 0

NOTES="$ROOT/.notes"
OWN="$SIDE.md"
if [ "$SIDE" = app ]; then OTHER="training.md"; else OTHER="app.md"; fi

# State is kept per session, so two sessions in one checkout do not hide
# changes from each other.
SESSION="$(field session_id | tr -cd 'A-Za-z0-9_-')"
STATE="$GIT_DIR_PATH/claude-notes/${SESSION:-default}"
mkdir -p "$STATE" 2>/dev/null || exit 0

WARN=""
warn() { WARN="${WARN:+$WARN }$1"; }
CONFLICT=0
CONFLICT_MSG="The project notes in .notes/ have local changes that conflict with newer notes on GitHub, so they are not pushed. Fix it now: run git -C .notes pull --rebase origin main, edit the conflicted file so the facts from both versions are kept, then run git -C .notes add -A and git -C .notes -c core.editor=true rebase --continue. Do not push; this hook pushes at the next sync."

busy() {  # a merge or rebase is under way in the code repo; leave git alone
  [ -d "$GIT_DIR_PATH/rebase-merge" ] || [ -d "$GIT_DIR_PATH/rebase-apply" ] ||
    [ -f "$GIT_DIR_PATH/MERGE_HEAD" ]
}

# Fast-forward the code to GitHub's copy of the current branch. Only at session
# start: files must not change under a session or a job that is already running.
pull_code() {
  local before b
  before="$(git rev-parse HEAD 2>/dev/null)" || return 0
  [ -f "$STATE/head" ] || echo "$before" >"$STATE/head"
  busy && return 0
  b="$(git symbolic-ref --short -q HEAD)" || return 0
  git rev-parse -q --verify "refs/remotes/origin/$b" >/dev/null 2>&1 || return 0
  limit 15 git fetch -q origin "$b" >/dev/null 2>&1 || return 0  # offline: the notes sync says so
  git merge-base --is-ancestor "origin/$b" HEAD 2>/dev/null && return 0  # nothing new
  if git merge -q --ff-only "origin/$b" >/dev/null 2>&1; then
    # Commits that arrived by pulling are not this session's work.
    [ "$(cat "$STATE/head")" = "$before" ] && git rev-parse HEAD >"$STATE/head"
  else
    warn "The code on branch $b could not be fast-forwarded to GitHub's copy: the two have diverged, or local changes are in the way. Nothing was changed."
  fi
}

n() { git -C "$NOTES" "$@"; }

notes_url() {  # the notes repo sits next to this repo on the same host
  local url
  url="$(git remote get-url origin 2>/dev/null)" || return 1
  case "$url" in
    *.git) echo "${url%/*}/otb_analyzer_notes.git" ;;
    *) echo "${url%/*}/otb_analyzer_notes" ;;
  esac
}

ensure_notes() {  # make sure .notes/ is a clone of the notes repo
  local exclude url tmp
  # Keep the clone out of `git status` in every worktree, whatever .gitignore says.
  exclude="$(git rev-parse --git-common-dir 2>/dev/null)/info/exclude"
  if ! grep -qxF '/.notes*' "$exclude" 2>/dev/null; then
    mkdir -p "$(dirname "$exclude")" 2>/dev/null && echo '/.notes*' >>"$exclude"
  fi
  [ -d "$NOTES/.git" ] && return 0
  if [ -e "$NOTES" ]; then
    warn "$NOTES exists but is not a clone of the notes repo. Move it away so the hook can clone the notes there."
    return 1
  fi
  url="$(notes_url)" || return 1
  rm -rf "$ROOT"/.notes.tmp.* 2>/dev/null
  tmp="$ROOT/.notes.tmp.$$"
  if limit 20 git clone -q "$url" "$tmp" >/dev/null 2>&1 && mv "$tmp" "$NOTES" 2>/dev/null; then
    return 0
  fi
  rm -rf "$tmp" 2>/dev/null
  warn "The project notes could not be cloned from $url into .notes/. Check the network and the GitHub sign-in on this machine (gh auth status, gh auth setup-git)."
  return 1
}

notes_busy() {
  local g
  g="$(n rev-parse --absolute-git-dir 2>/dev/null)" || return 1
  [ -d "$g/rebase-merge" ] || [ -d "$g/rebase-apply" ] || [ -f "$g/MERGE_HEAD" ]
}

SECRET_RE='sk-ant-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----'

# Commit local note edits, take in GitHub's copy, push. $1 = retry to try the
# push a second time when another machine pushed in between.
sync_notes() {
  local try
  if notes_busy; then CONFLICT=1; return 1; fi
  if [ -n "$(n status --porcelain -- '*.md' 2>/dev/null)" ]; then
    if grep -rEqs --include='*.md' --exclude-dir=.git -e "$SECRET_RE" "$NOTES"; then
      warn "The notes were not committed: something that looks like a key or token is in .notes/. Remove it; secrets never go in the notes."
      return 1
    fi
    if ! { n add -A -- '*.md' && n commit -q -m "Update $SIDE notes ($(hostname -s 2>/dev/null || echo unknown))"; } >/dev/null 2>&1; then
      warn "The notes could not be committed in .notes/."
      return 1
    fi
  fi
  for try in 1 2; do
    if ! limit 10 git -C "$NOTES" fetch -q origin main >/dev/null 2>&1; then
      if [ -n "$(n rev-list origin/main..HEAD 2>/dev/null)" ]; then
        warn "GitHub could not be reached: the notes shown may be out of date, and this machine has notes that are not pushed yet."
      else
        warn "GitHub could not be reached: the notes shown may be out of date."
      fi
      return 1
    fi
    if [ -n "$(n rev-list HEAD..origin/main 2>/dev/null)" ]; then
      if ! limit 10 git -C "$NOTES" rebase -q --autostash origin/main >/dev/null 2>&1; then
        n rebase --abort >/dev/null 2>&1
        CONFLICT=1
        return 1
      fi
    fi
    [ -z "$(n rev-list origin/main..HEAD 2>/dev/null)" ] && return 0
    limit 15 git -C "$NOTES" push -q origin HEAD:main >/dev/null 2>&1 && return 0
    [ "${1:-}" = retry ] || break
  done
  warn "The notes are committed in .notes/ but could not be pushed. The next sync tries again."
  return 1
}

note_hash() { if [ -f "$NOTES/$1" ]; then git hash-object "$NOTES/$1" 2>/dev/null; else echo none; fi; }
note_size() { if [ -f "$NOTES/$1" ]; then wc -c <"$NOTES/$1" | tr -d ' '; else echo 0; fi; }

emit() {  # file, title, most characters to print (whole lines only)
  [ -f "$NOTES/$1" ] || return 0
  echo
  echo "===== $2 (.notes/$1) ====="
  perl -e '
    my ($max, $path, $name) = @ARGV; my $used = 0;
    open(my $fh, "<", $path) or exit 0;
    while (my $line = <$fh>) {
      if ($used + length($line) > $max) {
        print "[Cut here to fit. Read .notes/$name for the rest.]\n"; last;
      }
      print $line; $used += length($line);
    }' "$3" "$NOTES/$1" "$1"
  note_hash "$1" >"$STATE/shown.$1"
}

emit_both() {  # print the files named in $1, this side's first, within the budget
  local own_size other_size other_cap own_cap
  own_size=0; other_size=0
  case " $1 " in *" $OWN "*) own_size="$(note_size "$OWN")" ;; esac
  case " $1 " in *" $OTHER "*) other_size="$(note_size "$OTHER")" ;; esac
  # The other side is guaranteed 3,300 characters; this side gets the rest.
  other_cap=$((other_size < 3300 ? other_size : 3300))
  own_cap=$((OUTPUT_BUDGET - other_cap))
  [ "$own_size" -lt "$own_cap" ] && own_cap="$own_size"
  other_cap=$((OUTPUT_BUDGET - own_cap))
  case " $1 " in *" $OWN "*) emit "$OWN" "Notes for this side, $SIDE" "$own_cap" ;; esac
  case " $1 " in *" $OTHER "*) emit "$OTHER" "Notes for the other side, ${OTHER%.md}: read-only here" "$other_cap" ;; esac
}

trouble() {  # everything that went wrong, as one line
  local msg=""
  [ "$CONFLICT" = 1 ] && msg="$CONFLICT_MSG"
  [ -n "$WARN" ] && msg="${msg:+$msg }$WARN"
  echo "$msg"
}

case "$MODE" in
start)
  # Drop the state of sessions that ended long ago.
  find "$GIT_DIR_PATH/claude-notes" -mindepth 1 -maxdepth 1 -type d -mtime +14 \
    -exec rm -rf {} + 2>/dev/null
  mkdir -p "$STATE" 2>/dev/null || exit 0
  pull_code
  echo "Project memory for OTB Analyzer, loaded by the memory hook. Read it before starting work."
  echo "This machine is the $SIDE side. Keep .notes/$OWN current as CLAUDE.md describes; the hook commits and pushes .notes/ for you."
  if ensure_notes; then
    sync_notes
    emit_both "$OWN $OTHER"
  fi
  msg="$(trouble)"
  if [ -n "$msg" ]; then echo; echo "Warning: $msg"; fi
  echo "$msg" >"$STATE/warned"
  date +%s >"$STATE/checked"
  ;;

prompt)
  now="$(date +%s)"
  last="$(cat "$STATE/checked" 2>/dev/null || echo 0)"
  [ $((now - last)) -lt "$RECHECK_SECONDS" ] && exit 0
  echo "$now" >"$STATE/checked"
  if ensure_notes; then
    sync_notes
    changed=""
    for f in "$OWN" "$OTHER"; do
      [ -f "$NOTES/$f" ] || continue
      [ "$(note_hash "$f")" != "$(cat "$STATE/shown.$f" 2>/dev/null)" ] && changed="$changed $f"
    done
    if [ -n "$changed" ]; then
      echo "The project memory changed since you last read it. Read it before starting."
      emit_both "$changed"
    fi
  fi
  msg="$(trouble)"  # say a problem once, not on every check
  if [ -n "$msg" ] && [ "$msg" != "$(cat "$STATE/warned" 2>/dev/null)" ]; then
    echo; echo "Warning: $msg"
  fi
  echo "$msg" >"$STATE/warned"
  ;;

stop)
  busy && exit 0
  head="$(git rev-parse HEAD 2>/dev/null)" || exit 0
  last_head="$(cat "$STATE/head" 2>/dev/null || echo "$head")"
  active="$(field stop_hook_active)"

  if ! ensure_notes; then
    if [ "$WARN" != "$(cat "$STATE/warned" 2>/dev/null)" ]; then json systemMessage "$WARN"; fi
    echo "$WARN" >"$STATE/warned"
    exit 0
  fi

  # Newest change to the code repo: uncommitted files, plus files in commits
  # made since this hook last passed. .notes/ is ignored, so it never counts.
  newest=0
  note() { local m; m="$(mtime "$1")"; [ "$m" -gt "$newest" ] && newest="$m"; }
  skip_next=0
  while IFS= read -r -d '' entry; do
    if [ "$skip_next" = 1 ]; then skip_next=0; continue; fi
    case "${entry:0:2}" in R*|C*) skip_next=1 ;; esac  # next field is the old name
    note "${entry:3}"
  done < <(git status --porcelain -z --untracked-files=all 2>/dev/null)
  if [ "$head" != "$last_head" ] && git cat-file -e "$last_head" 2>/dev/null; then
    while IFS= read -r -d '' path; do note "$path"; done \
      < <(git diff --name-only -z "$last_head" "$head" 2>/dev/null)
  fi

  key="$head:$newest"
  if [ "$newest" -gt "$(mtime "$NOTES/$OWN")" ] && [ "$active" != true ] &&
    [ "$key" != "$(cat "$STATE/asked" 2>/dev/null)" ]; then
    echo "$key" >"$STATE/asked"  # ask once per set of changes, never in a loop
    json decision block reason "Project memory is older than the work in this repo. Before stopping, update .notes/$OWN as CLAUDE.md describes: current state, any decision made, open questions, anything the other side needs to know, and one dated log line. Keep it short and do not commit or push it yourself; this hook does that when you stop. If nothing worth recording changed, run: touch .notes/$OWN"
    exit 0
  fi

  # What this session leaves in its own file counts as seen by it; anything the
  # sync merges in from GitHub after this point does not.
  note_hash "$OWN" >"$STATE/shown.$OWN"
  sync_notes retry
  git rev-parse HEAD >"$STATE/head" 2>/dev/null

  if [ "$CONFLICT" = 1 ] && [ "$active" != true ] &&
    [ "$(n rev-parse HEAD 2>/dev/null)" != "$(cat "$STATE/asked-conflict" 2>/dev/null)" ]; then
    n rev-parse HEAD >"$STATE/asked-conflict" 2>/dev/null
    json decision block reason "$CONFLICT_MSG"
    exit 0
  fi
  msg="$(trouble)"
  if [ -n "$msg" ] && [ "$msg" != "$(cat "$STATE/warned" 2>/dev/null)" ]; then
    json systemMessage "$msg"
  fi
  echo "$msg" >"$STATE/warned"
  ;;
esac
exit 0
