#!/bin/bash
# Launch one worker per GPU under tmux, plus a status pane.
#
#   bash scripts/queue_launch.sh [job_file]
#
# Workers exit cleanly when the list drains, so re-running this after appending
# lines to the job file picks up only the new work.
#
# One worker per GPU and no more: a second process on the same device would corrupt
# the peak-memory and throughput measurements that the cost table (B6) depends on.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JOBFILE="${1:-jobs/campaign.jobs}"
SESSION="${SESSION:-asrq}"
GPUS="${GPUS:-0 1}"

[[ -f "$JOBFILE" ]] || { echo "No job file: $JOBFILE"; exit 1; }
command -v tmux >/dev/null || { echo "tmux is required"; exit 1; }
command -v jq   >/dev/null || { echo "jq is required"; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p logs/queue jobs/status

first=1
for gpu in $GPUS; do
  cmd="cd '$REPO_ROOT' && GPU=$gpu bash scripts/queue_worker.sh '$JOBFILE' 2>&1 | tee -a logs/queue/worker${gpu}.log"
  if (( first )); then
    tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$cmd"
    first=0
  else
    tmux new-window -t "$SESSION" -n "gpu$gpu" "$cmd"
  fi
done

tmux new-window -t "$SESSION" -n status \
  "watch -n 30 'bash $REPO_ROOT/scripts/queue_status.sh $JOBFILE'"

echo "Launched session '$SESSION' with workers on GPUs: $GPUS"
echo "Attach:  tmux attach -t $SESSION"
echo "Status:  bash scripts/queue_status.sh $JOBFILE"
