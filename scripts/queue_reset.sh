#!/bin/bash
# Clear queue status so jobs become claimable again.
#
#   bash scripts/queue_reset.sh <job_id> [<job_id>...]   # specific jobs
#   bash scripts/queue_reset.sh --failed                 # every parked failure
#   bash scripts/queue_reset.sh --stale                  # .running with a dead heartbeat
#   bash scripts/queue_reset.sh --all                    # everything except .done
#   bash scripts/queue_reset.sh --all --include-done     # everything, forces full re-run
#
# A .done job is only re-runnable with --include-done, and even then the worker will
# skip it if its run directory still holds a complete run_manifest.json -- delete the
# run directory too if you really mean to redo the work.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_DIR="$REPO_ROOT/jobs/status"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
STALE_SEC="${STALE_SEC:-$((HEARTBEAT_SEC * 3))}"

[[ -d "$STATUS_DIR" ]] || { echo "no status dir: $STATUS_DIR"; exit 0; }
[[ $# -gt 0 ]] || { sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 1; }

include_done=0
for a in "$@"; do [[ "$a" == "--include-done" ]] && include_done=1; done

removed=0
drop() { # drop <path>
  [[ -e "$1" ]] || return 0
  echo "  removed $(basename "$1")"
  rm -f "$1"; removed=$((removed+1))
}

shopt -s nullglob
for arg in "$@"; do
  case "$arg" in
    --include-done) ;;
    --failed) for f in "$STATUS_DIR"/*.failed; do drop "$f"; done ;;
    --stale)
      now=$(date +%s)
      for f in "$STATUS_DIR"/*.running; do
        mtime=$(stat -c %Y "$f" 2>/dev/null || echo "$now")
        age=$(( now - mtime ))
        if (( age > STALE_SEC )); then
          echo "  stale by ${age}s:"; drop "$f"
        else
          echo "  keeping $(basename "$f") (heartbeat ${age}s old -- looks alive)"
        fi
      done ;;
    --all)
      for f in "$STATUS_DIR"/*.failed "$STATUS_DIR"/*.running; do drop "$f"; done
      (( include_done )) && for f in "$STATUS_DIR"/*.done; do drop "$f"; done ;;
    --*) echo "unknown option: $arg"; exit 1 ;;
    *)
      drop "$STATUS_DIR/$arg.failed"
      drop "$STATUS_DIR/$arg.running"
      (( include_done )) && drop "$STATUS_DIR/$arg.done"
      ;;
  esac
done
shopt -u nullglob

echo "cleared $removed status file(s)"
