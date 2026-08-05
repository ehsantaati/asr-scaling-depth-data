#!/bin/bash
# Campaign progress: one line per job, plus live step/ETA for running jobs.
#
#   bash scripts/queue_status.sh [job_file]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JOBFILE="${1:-jobs/campaign.jobs}"
STATUS_DIR="$REPO_ROOT/jobs/status"

[[ -f "$JOBFILE" ]] || { echo "No job file: $JOBFILE"; exit 1; }

printf '%-34s %-9s %s\n' "JOB" "STATE" "DETAIL"
printf '%s\n' "--------------------------------------------------------------------------------"

n_done=0; n_run=0; n_fail=0; n_todo=0
while IFS=$'\t' read -r id cfg seed rn odir extra; do
  [[ -z "${id:-}" || "$id" == \#* ]] && continue
  state="pending"; detail=""

  if [[ -f "$STATUS_DIR/$id.done" ]]; then
    state="done"; n_done=$((n_done+1))
    detail="$(jq -r '"\(.wall_s // "?")s"' "$STATUS_DIR/$id.done" 2>/dev/null)"
    rd="$(jq -r '.run_dir // ""' "$STATUS_DIR/$id.done" 2>/dev/null)"
    if [[ -n "$rd" && -f "$rd/run_manifest.json" ]]; then
      wer="$(jq -r '[.results | to_entries[] | select(.value.wer_fixed != null)
                    | "\(.key)=\(.value.wer_fixed*100 | .*100 | round / 100)"] | join(" ")' \
             "$rd/run_manifest.json" 2>/dev/null)"
      [[ -n "$wer" && "$wer" != "null" ]] && detail="$detail  $wer"
    fi
  elif [[ -f "$STATUS_DIR/$id.running" ]]; then
    state="running"; n_run=$((n_run+1))
    detail="$(jq -r '"gpu\(.gpu) pid\(.pid) attempt\(.attempt)"' "$STATUS_DIR/$id.running" 2>/dev/null)"
    rd="$(jq -r '.run_dir // ""' "$STATUS_DIR/$id.running" 2>/dev/null)"
    [[ -z "$rd" ]] && rd="$(ls -d "$odir/${rn}_frac_"*"_subset_"* 2>/dev/null | head -1)"
    if [[ -n "$rd" && -f "$rd/progress.json" ]]; then
      detail="$detail  $(jq -r '"step \(.step)/\(.max_steps) eta \(.eta_s // "?")s"' "$rd/progress.json" 2>/dev/null)"
    fi
  elif [[ -f "$STATUS_DIR/$id.failed" ]]; then
    state="FAILED"; n_fail=$((n_fail+1))
    detail="$(jq -r '"attempt \(.attempts): \(.reason)"' "$STATUS_DIR/$id.failed" 2>/dev/null)"
  else
    n_todo=$((n_todo+1))
  fi

  printf '%-34s %-9s %s\n' "$id" "$state" "$detail"
done < "$JOBFILE"

printf '%s\n' "--------------------------------------------------------------------------------"
printf 'done=%d  running=%d  failed=%d  pending=%d\n' "$n_done" "$n_run" "$n_fail" "$n_todo"
