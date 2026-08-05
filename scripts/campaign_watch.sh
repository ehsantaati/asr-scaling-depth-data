#!/bin/bash
# Read-only campaign monitor. Reports; never launches, kills or deletes.
#
#   bash scripts/campaign_watch.sh                    # plain status, no LLM
#   bash scripts/campaign_watch.sh --claude           # + Claude Code triage
#   bash scripts/campaign_watch.sh --claude --jobs jobs/campaign.jobs
#
# Suitable for cron:
#   */30 * * * * cd /home/ehsan/asr-scaling-depth-data && bash scripts/campaign_watch.sh --claude >> logs/watch.log 2>&1
#
# The --allowedTools allowlist below is the safety mechanism: the agent is given
# read-only commands only, so it cannot start or stop a job even if it concludes it
# should. Campaign runs are hours of GPU time and their checkpoints do not survive the
# job, so an agent that can restart things can quietly destroy evidence.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JOBFILE="jobs/campaign.jobs"
USE_CLAUDE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --claude) USE_CLAUDE=1; shift ;;
    --jobs)   JOBFILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

echo "===== campaign watch $(date -Is) ====="

echo "--- queue ---"
[[ -f "$JOBFILE" ]] && bash scripts/queue_status.sh "$JOBFILE" || echo "  no job file: $JOBFILE"

echo "--- gpus ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'

echo "--- disk ---"
df -h /home/ehsan | tail -1 | awk '{print "  free "$4"  used "$3}'
du -sh ~/.cache/huggingface/datasets/*/ 2>/dev/null | sed 's/^/  cache /'

echo "--- completed runs ---"
found=0
while IFS= read -r m; do
  found=1
  d="$(dirname "$m")"
  jq -r --arg d "$d" '
    "  \($d | split("/") | .[-1])  steps=\(.schedule.steps_completed // "?")/\(.schedule.max_steps // "?")" +
    "  seed=\(.seeding.seed // "?")" +
    (.results // {} | to_entries | map(select(.value.wer_fixed != null)
      | "  \(.key)=\((.value.wer_fixed*10000|round)/100)%(n=\(.value.denominator))") | join(""))' "$m" 2>/dev/null
done < <(find outputs -name run_manifest.json 2>/dev/null | sort)
(( found )) || echo "  none yet"

echo "--- artifact contract ---"
fails=0
while IFS= read -r m; do
  d="$(dirname "$m")"
  if ! bash scripts/check_artifacts.sh "$d" >/dev/null 2>&1; then
    echo "  FAILS: $d"; fails=$((fails+1))
  fi
done < <(find outputs -name run_manifest.json 2>/dev/null | sort)
(( fails == 0 )) && echo "  all complete runs pass"

echo "--- recent failures ---"
shopt -s nullglob
for f in jobs/status/*.failed; do
  jq -r --arg id "$(basename "$f" .failed)" '"  \($id): attempt \(.attempts) — \(.reason)"' "$f" 2>/dev/null
done
shopt -u nullglob

if (( USE_CLAUDE )); then
  echo "--- triage ---"
  command -v claude >/dev/null || { echo "  claude CLI not installed"; exit 0; }
  claude -p "You are monitoring a running ASR training campaign. Read CLAUDE.md first.

Report concisely:
1. Overall progress: jobs done / running / failed / pending.
2. Any run whose loss trajectory, WER or denominator looks anomalous — read
   metrics.jsonl and eval/*/metrics.json, do not just trust check_artifacts.sh.
3. Any failed job, its likely cause from logs/queue/<job>.gpu<N>.log, and whether it
   looks retryable or systematic.
4. Whether free disk is sufficient for the current phase.

If nothing is wrong, say so in one line. Do NOT start, stop, re-queue or delete
anything — you have read-only tools and that is deliberate." \
    --allowedTools \
      "Read" "Grep" "Glob" \
      "Bash(bash scripts/queue_status.sh:*)" \
      "Bash(bash scripts/check_artifacts.sh:*)" \
      "Bash(nvidia-smi:*)" \
      "Bash(df:*)" "Bash(du:*)" \
      "Bash(jq:*)" \
    --permission-mode acceptEdits 2>&1 | sed 's/^/  /'
fi
