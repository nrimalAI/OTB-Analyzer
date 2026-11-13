#!/bin/bash
# Runs the experiments listed in runs/queue.txt one after the other (two at
# once only slow each other on 16 GB), and picks up where it was after a
# power cut or a restart.
#
#   nohup bash queue.sh > /dev/null 2>&1 &        start it detached
#
# runs/queue.txt      one run per line:  <run name> <train.py arguments>
#                     Lines starting with # and empty lines are skipped. Edit
#                     it at any time; the top line is taken next.
# runs/queue.current  the line that is running now. If it is there when the
#                     script starts, that run was cut off: it is continued
#                     from its last.pt (or started again if there is none).
# runs/queue.done     lines that finished; runs/queue.failed: lines that ended
#                     with an error or were stopped by hand (not retried).
# runs/queue.log      what happened and when.
#
# Only one copy runs: a second one exits at once. To stop a run and move on,
# kill its train process (pkill -f "name <run name>"); to stop everything,
# kill this script first. The loop ends when queue.txt is empty.
cd "$(dirname "$0")" || exit 1
mkdir -p runs
log() { echo "$(date '+%F %T') $*" >> runs/queue.log; }

# One copy at a time. mkdir either makes the lock or fails, so two copies
# started together cannot both pass; a lock left by a copy that died (a power
# cut) is taken over.
if ! mkdir runs/queue.lock 2>/dev/null; then
  other="$(cat runs/queue.lock/pid 2>/dev/null)"
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null \
     && ps -p "$other" -o command= | grep -q "queue.sh"; then
    exit 0
  fi
fi
echo $$ > runs/queue.lock/pid
trap 'rm -rf runs/queue.lock' EXIT

# Take a line out of queue.txt (the first exact match only). The line is
# passed through the environment: awk -v would read backslashes as escapes.
drop_line() {
  l="$1" awk 'done || $0 != ENVIRON["l"] { print; next } { done = 1 }' runs/queue.txt > runs/queue.tmp \
    && mv runs/queue.tmp runs/queue.txt
}

# Training processes, ours or anyone's (two at once only slow each other).
# The patterns are those of real command lines, so that a shell which merely
# mentions a run does not hold the queue up.
training() {
  pgrep -f "[pP]ython -u train\.py (corners|pieces|resume) |[pP]ython -u -c import sys; from ultralytics" >/dev/null
}

# The state of a run, from its last.pt.  resume: a checkpoint to continue
# from;  done: it finished (the final checkpoint carries no optimizer);
# fresh: the file does not load.
state_of() {
  ./venv/bin/python - "$1" <<'EOF' 2>/dev/null || echo fresh
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("resume" if c.get("optimizer") is not None and c.get("epoch", -1) >= 0 else "done")
EOF
}

while true; do
  while training; do sleep 30; done
  if [ -s runs/queue.current ]; then
    line="$(cat runs/queue.current)"
    cut_off=1
    # If the cut came between writing queue.current and dropping the line.
    [ -f runs/queue.txt ] && grep -qxF -- "$line" runs/queue.txt && drop_line "$line"
  else
    line="$(grep -v -E '^[[:space:]]*(#|$)' runs/queue.txt 2>/dev/null | head -n 1)"
    [ -z "$line" ] && { log "queue empty, stopping"; exit 0; }
    echo "$line" > runs/queue.current
    drop_line "$line"
    cut_off=0
  fi
  name="${line%% *}"; args="${line#* }"
  case "$args" in corners*) dir="runs/pose/$name" ;; *) dir="runs/detect/$name" ;; esac

  state=fresh
  [ "$cut_off" = 1 ] && [ -f "$dir/weights/last.pt" ] && state="$(state_of "$dir/weights/last.pt")"
  if [ "$state" = done ]; then
    log "$name had already finished"
    rc=0
  elif [ "$state" = resume ]; then
    log "continue $name from $dir/weights/last.pt"
    # Through train.py, which sets up what the run needs beyond its saved
    # settings (canonical_corners.py). --name is not read there: it puts the
    # run's name on the command line, so pkill -f "name <run name>" finds it.
    caffeinate -i ./venv/bin/python -u train.py resume --model "$dir/weights/last.pt" \
      --name "$name" >> "runs/$name.log" 2>&1
    rc=$?
  elif [ "$cut_off" = 0 ] && [ -d "$dir" ]; then
    # A run of that name already exists (and later lines may use its
    # weights): do not touch it.
    log "$name: $dir exists already, not started; give the run another name"
    rc=1
  else
    # The folder of a cut-off run that never wrote a usable checkpoint would
    # make ultralytics pick another name; keep it out of the way.
    [ -d "$dir" ] && mv "$dir" "$dir.cut-$(date +%s)"
    log "start $name: $args"
    # shellcheck disable=SC2086
    caffeinate -i ./venv/bin/python -u train.py $args --device mps --name "$name" > "runs/$name.log" 2>&1
    rc=$?
  fi
  log "end $name (exit $rc)"
  if [ "$rc" = 0 ]; then echo "$line" >> runs/queue.done; else echo "$line" >> runs/queue.failed; fi
  rm -f runs/queue.current
done
