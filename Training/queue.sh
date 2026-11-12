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

if [ -f runs/queue.pid ] && kill -0 "$(cat runs/queue.pid)" 2>/dev/null \
   && ps -p "$(cat runs/queue.pid)" -o command= | grep -q "queue.sh"; then
  exit 0
fi
echo $$ > runs/queue.pid

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
  while pgrep -f "train.py (corners|pieces)|queue-resume" >/dev/null; do sleep 30; done
  if [ -s runs/queue.current ]; then
    line="$(cat runs/queue.current)"
    cut_off=1
  else
    line="$(grep -v -E '^\s*(#|$)' runs/queue.txt 2>/dev/null | head -n 1)"
    [ -z "$line" ] && { log "queue empty, stopping"; rm -f runs/queue.pid; exit 0; }
    # Drop that line from the file (first exact match only).
    awk -v l="$line" 'done || $0 != l { print; next } { done = 1 }' runs/queue.txt > runs/queue.tmp && mv runs/queue.tmp runs/queue.txt
    echo "$line" > runs/queue.current
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
    # "queue-resume" in the command line is what the wait above looks for.
    caffeinate -i ./venv/bin/python -u -c 'import sys; from ultralytics import YOLO; YOLO(sys.argv[1]).train(resume=True)  # queue-resume' \
      "$dir/weights/last.pt" >> "runs/$name.log" 2>&1
    rc=$?
  else
    # A folder left by a run that never wrote a usable checkpoint would make
    # ultralytics pick another name; keep it out of the way.
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
