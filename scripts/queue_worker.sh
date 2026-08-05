#!/bin/bash
# One worker == one GPU. Claims jobs from a priority-ordered list under flock, runs
# them, and records status so the campaign survives crashes and reboots.
#
#   GPU=0 bash scripts/queue_worker.sh jobs/campaign.jobs
#
# Job list format (tab-separated, '#' comments, file order == priority order):
#   job_id <TAB> config_path <TAB> seed <TAB> run_name <TAB> output_dir <TAB> extra_args
# extra_args of '-' means none. The sentinel --INFERENCE routes a job to
# inference.py instead of train.py (vanilla baselines).
#
# The list is hand-enumerated, one line per (config, seed). There is deliberately no
# cross-product loop anywhere in this runner: that is what keeps the asymmetric seed
# design (LoRA 3-5 seeds, full FT 2-3 on claim-bearing configs only) explicit and
# auditable. Adding a seed means adding a line.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

JOBFILE="${1:-jobs/campaign.jobs}"
GPU="${GPU:-0}"
PY="$(resolve_py)"
QDIR="$REPO_ROOT/jobs"
STATUS_DIR="$QDIR/status"
LOCKFILE="$QDIR/queue.lock"
LOGDIR="$REPO_ROOT/logs/queue"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
STALE_MIN="${STALE_MIN:-20}"        # a .running file older than this == orphaned
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"

mkdir -p "$STATUS_DIR" "$LOGDIR"
touch "$LOCKFILE"

[[ -f "$JOBFILE" ]] || { echo "No job file: $JOBFILE"; exit 1; }
[[ -x "$PY" ]] || { echo "No interpreter: $PY"; exit 1; }
command -v jq >/dev/null || { echo "jq is required"; exit 1; }

# stderr, not stdout: claim_next runs inside a command substitution, so anything on
# stdout there would be parsed as part of the claimed job line.
log() { echo "[$(date -Is)] [gpu$GPU] $*" >&2; }

# --- status helpers ---------------------------------------------------------
# Always write via a temp file + mv so a reader never sees a partial status.
write_status() {  # write_status <job_id> <state> <json>
  local id="$1" state="$2" json="$3"
  local tmp="$STATUS_DIR/.$id.$$.tmp"
  printf '%s\n' "$json" > "$tmp"
  mv -f "$tmp" "$STATUS_DIR/$id.$state"
}

clear_status() { rm -f "$STATUS_DIR/$1".{running,done,failed}; }

attempts_of() {
  local f="$STATUS_DIR/$1.failed"
  [[ -f "$f" ]] && jq -r '.attempts // 0' "$f" 2>/dev/null || echo 0
}

run_dir_for() {  # run_dir_for <output_dir> <run_name> -> glob-resolved run directory
  local odir="$1" rn="$2"
  # train.py writes <output_dir>/<run_name>_frac_<f>_subset_<i>
  local hit
  hit=$(ls -d "$odir/${rn}_frac_"*"_subset_"* 2>/dev/null | head -1)
  printf '%s' "$hit"
}

manifest_complete() {  # manifest_complete <run_dir>
  local rd="$1"
  [[ -n "$rd" && -f "$rd/run_manifest.json" ]] || return 1
  jq -e '.status == "complete"' "$rd/run_manifest.json" >/dev/null 2>&1
}

# --- orphan recovery --------------------------------------------------------
# Three distinct end states are required: done, failed, and running-but-orphaned.
# An orphan is a .running file whose pid is gone (worker killed / OOM) or whose
# heartbeat has gone stale (host rebooted). It is requeued up to MAX_ATTEMPTS, then
# parked as .failed for a human.
sweep_orphans() {
  shopt -s nullglob
  for f in "$STATUS_DIR"/*.running; do
    local id host pid stale=0
    id="$(basename "$f" .running)"
    host="$(jq -r '.host // ""' "$f" 2>/dev/null)"
    pid="$(jq -r '.pid // 0' "$f" 2>/dev/null)"

    if [[ "$host" == "$(hostname)" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 0 )); then
      kill -0 "$pid" 2>/dev/null || stale=1
    fi
    if [[ -n "$(find "$f" -mmin "+$STALE_MIN" 2>/dev/null)" ]]; then
      stale=1
    fi

    if (( stale )); then
      local n; n=$(( $(jq -r '.attempt // 1' "$f" 2>/dev/null) ))
      log "orphan detected: $id (pid=$pid, host=$host) -> failed{orphaned}"
      rm -f "$f"
      write_status "$id" failed "$(jq -nc --arg id "$id" --argjson n "$n" \
        '{finished:now|todate, exit:null, attempts:$n, reason:"orphaned"}')"
    fi
  done
  shopt -u nullglob
}

# --- atomic claim -----------------------------------------------------------
# The lock is held only for the scan+claim (milliseconds), never during training.
claim_next() {
  exec 9>"$LOCKFILE"
  flock 9

  sweep_orphans

  local id cfg seed rn odir extra
  while IFS=$'\t' read -r id cfg seed rn odir extra; do
    [[ -z "${id:-}" || "$id" == \#* ]] && continue

    [[ -e "$STATUS_DIR/$id.done" || -e "$STATUS_DIR/$id.running" ]] && continue

    local attempt=1
    if [[ -e "$STATUS_DIR/$id.failed" ]]; then
      local n; n=$(attempts_of "$id")
      (( n >= MAX_ATTEMPTS )) && continue     # give up; leave for inspection
      attempt=$(( n + 1 ))
      rm -f "$STATUS_DIR/$id.failed"
    fi

    # Idempotent re-entry: a complete manifest outranks any status file, so a run
    # whose status was lost is recognised as done instead of being repeated.
    local rd; rd="$(run_dir_for "$odir" "$rn")"
    if manifest_complete "$rd"; then
      write_status "$id" done "$(jq -nc --arg rd "$rd" \
        '{finished:now|todate, exit:0, run_dir:$rd, reason:"pre-existing complete manifest"}')"
      continue
    fi

    write_status "$id" running "$(jq -nc --arg h "$(hostname)" --argjson p $$ \
      --argjson g "$GPU" --argjson a "$attempt" --arg rd "$rd" \
      '{host:$h, pid:$p, gpu:$g, attempt:$a, started:(now|todate), run_dir:$rd}')"
    # The attempt number travels with the claim. Recomputing it in run_job would
    # always read 0, because the .failed file has just been removed above -- which
    # is how the retry counter silently never advanced and a failing job looped
    # forever.
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$cfg" "$seed" "$rn" "$odir" "$extra" "$attempt"
    break
  done < "$JOBFILE"

  flock -u 9
  exec 9>&-
}

# --- run one job ------------------------------------------------------------
run_job() {
  local id="$1" cfg="$2" seed="$3" rn="$4" odir="$5" extra="$6" attempt="${7:-1}"
  local logfile="$LOGDIR/${id}.gpu${GPU}.log"
  local t0 t1 rc
  t0=$(date +%s)

  [[ "$extra" == "-" ]] && extra=""

  local script="train.py"
  if [[ "$extra" == *"--INFERENCE"* ]]; then
    script="inference.py"
    extra="${extra//--INFERENCE/}"
  fi

  log "START $id  ($script, config=$cfg, seed=$seed) -> $logfile"

  # Stale partial run directories are removed before a retry: without a complete
  # manifest they are indistinguishable from a finished run to downstream analysis.
  local rd; rd="$(run_dir_for "$odir" "$rn")"
  if [[ -n "$rd" ]] && ! manifest_complete "$rd"; then
    log "removing incomplete run dir $rd before retry"
    rm -rf "$rd"
  fi

  local -a cmd
  if [[ "$script" == "train.py" ]]; then
    cmd=("$PY" train.py --config_path "$cfg" --seed "$seed" --run_name "$rn" --output_dir "$odir")
  else
    cmd=("$PY" inference.py --config_path "$cfg" --output_dir "$odir")
  fi
  # shellcheck disable=SC2206
  [[ -n "$extra" ]] && cmd+=($extra)

  CUDA_VISIBLE_DEVICES="$GPU" \
  QUEUE_JOB_ID="$id" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  TOKENIZERS_PARALLELISM=false \
    "${cmd[@]}" > "$logfile" 2>&1 &
  local child=$!

  # Heartbeat: touch the status file so sweep_orphans can tell a live run from a
  # host that went away mid-job.
  ( while kill -0 "$child" 2>/dev/null; do touch "$STATUS_DIR/$id.running" 2>/dev/null; sleep "$HEARTBEAT_SEC"; done ) &
  local hb=$!

  wait "$child"; rc=$?
  kill "$hb" 2>/dev/null; wait "$hb" 2>/dev/null

  t1=$(date +%s)
  rd="$(run_dir_for "$odir" "$rn")"
  rm -f "$STATUS_DIR/$id.running"

  if (( rc == 0 )) && manifest_complete "$rd"; then
    write_status "$id" done "$(jq -nc --arg rd "$rd" --argjson w "$((t1-t0))" \
      '{finished:now|todate, exit:0, run_dir:$rd, wall_s:$w}')"
    log "DONE  $id in $((t1-t0))s"
  else
    local reason="exit=$rc"
    (( rc == 0 )) && reason="exited 0 but run_manifest.json is not complete"
    write_status "$id" failed "$(jq -nc --arg r "$reason" --arg l "$logfile" \
      --argjson e "$rc" --argjson n "$attempt" \
      '{finished:now|todate, exit:$e, attempts:$n, reason:$r, log:$l}')"
    log "FAIL  $id attempt $attempt/$MAX_ATTEMPTS ($reason) -- see $logfile"
  fi
}

# --- main loop --------------------------------------------------------------
log "worker starting on GPU $GPU, job file $JOBFILE"
while true; do
  job="$(claim_next)"
  [[ -z "$job" ]] && break
  IFS=$'\t' read -r id cfg seed rn odir extra attempt <<< "$job"
  run_job "$id" "$cfg" "$seed" "$rn" "$odir" "$extra" "$attempt"
done
log "no claimable jobs left; worker exiting"
