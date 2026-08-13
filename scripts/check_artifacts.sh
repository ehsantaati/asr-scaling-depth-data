#!/bin/bash
# Artifact contract for a completed run. Exits non-zero unless every artifact the
# revision plan depends on is present and well-formed.
#
# The point is to fail here, on one small run, rather than after 40 expensive ones.
# The original campaign's checkpoints were deleted, so a missing diagnostic cannot be
# recovered after the fact.
#
#   bash scripts/check_artifacts.sh <run_dir>
#   bash scripts/check_artifacts.sh --seed-divergence <run_dir_a> <run_dir_b>
#   bash scripts/check_artifacts.sh --baseline-parity  <run_dir> <baseline_dir>
set -uo pipefail

PASS=0; FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
check(){ if eval "$1" >/dev/null 2>&1; then ok "$2"; else bad "$2"; fi; }

command -v jq >/dev/null || { echo "jq is required"; exit 1; }

# ---------------------------------------------------------------- seed divergence
# The check that validates the whole campaign: if --seed does not change the loss
# trajectory, then multi-seed runs are measuring nothing and B1 is worthless.
seed_divergence() {
  local A="$1" B="$2"
  echo "== seed divergence: $A  vs  $B"
  for d in "$A" "$B"; do
    [[ -f "$d/metrics.jsonl" ]] || { bad "metrics.jsonl missing in $d"; return; }
  done
  local sa sb
  sa=$(jq -r 'select(.kind=="train" and .loss!=null) | "\(.step):\(.loss)"' "$A/metrics.jsonl" | head -5 | tr '\n' ' ')
  sb=$(jq -r 'select(.kind=="train" and .loss!=null) | "\(.step):\(.loss)"' "$B/metrics.jsonl" | head -5 | tr '\n' ' ')
  echo "     A: $sa"
  echo "     B: $sb"
  if [[ "$sa" == "$sb" ]]; then
    bad "train losses are IDENTICAL in the first logged steps -- --seed is not reaching the training loop"
  else
    ok "train losses diverge within the first logged steps (seed affects the run)"
  fi

  # Data order: under data_order_seed_mode=auto at fraction 1.0 the two runs should
  # ALSO see a different sample order; under "fixed" they must see the same one.
  local ma mb mode fa fb
  ma="$A/run_manifest.json"; mb="$B/run_manifest.json"
  if [[ -f "$ma" && -f "$mb" ]]; then
    mode=$(jq -r '.seeding.data_order_seed_mode' "$ma")
    fa=$(jq -rc '.seeding.first_train_samples' "$ma")
    fb=$(jq -rc '.seeding.first_train_samples' "$mb")
    local sa_seed sb_seed
    sa_seed=$(jq -r '.seeding.effective_shuffle_seed' "$ma")
    sb_seed=$(jq -r '.seeding.effective_shuffle_seed' "$mb")
    echo "     mode=$mode  shuffle_seed A=$sa_seed B=$sb_seed"
    if [[ "$sa_seed" == "$sb_seed" ]]; then
      check "[[ '$fa' == '$fb' ]]" "same shuffle_seed => identical first training samples"
    else
      check "[[ '$fa' != '$fb' ]]" "different shuffle_seed => different first training samples"
    fi
  fi
}

# ---------------------------------------------------------------- baseline parity
# Fine-tuned and baseline WERs are only comparable if they were produced by the same
# decode spec. Compare the spec blocks field by field.
baseline_parity() {
  local RUN="$1" BASE="$2"
  echo "== baseline parity: $RUN  vs  $BASE"
  local rs bs
  rs=$(jq -Sc '.decode_spec | del(.batch_size, .path)' "$RUN/results.json" 2>/dev/null)
  bs=$(jq -Sc '.decode_spec | del(.batch_size, .path)' "$BASE/results.json" 2>/dev/null)
  [[ -z "$rs" || "$rs" == "null" ]] && { bad "no decode_spec in $RUN/results.json"; return; }
  [[ -z "$bs" || "$bs" == "null" ]] && { bad "no decode_spec in $BASE/results.json"; return; }
  echo "     run : $rs"
  echo "     base: $bs"
  if [[ "$rs" == "$bs" ]]; then
    ok "decode specs are identical (language forcing, beams, dtype, normalizer)"
  else
    bad "decode specs differ -- baseline and fine-tuned WERs are NOT comparable"
    # Name the offending fields; "they differ" is not actionable at 3am.
    jq -n --argjson a "$rs" --argjson b "$bs" '
      (($a|keys) + ($b|keys) | unique)[]
      | . as $k
      | select(($a[$k] // null) != ($b[$k] // null))
      | "       differs: \($k)  run=\($a[$k] // "<missing>")  base=\($b[$k] // "<missing>")"' -r
  fi

  # And they must have been scored on the same eval sets.
  local rk bk
  rk=$(jq -r '.metrics | keys | join(",")' "$RUN/results.json")
  bk=$(jq -r '.metrics | keys | join(",")' "$BASE/results.json")
  check "[[ '$rk' == '$bk' ]]" "same eval sets scored: $rk"
}

# ---------------------------------------------------------------- single-run audit
audit_run() {
  local RUN="$1"
  echo "== artifact audit: $RUN"
  [[ -d "$RUN" ]] || { echo "no such run directory"; exit 2; }

  local M="$RUN/run_manifest.json"

  # --- manifest -------------------------------------------------------------
  check "[[ -f '$M' ]]"                                   "run_manifest.json exists"
  check "jq -e '.status==\"complete\"' '$M'"              "manifest status == complete"
  check "jq -e '.provenance.git.commit != null' '$M'"     "git commit recorded"
  check "jq -e '.provenance.config_sha256 != null' '$M'"  "resolved-config hash recorded"
  check "jq -e '.adaptation.target_modules.trainable_parameter_names | length > 0' '$M'" \
                                                          "resolved target modules non-empty"
  check "jq -e '.adaptation.target_modules.encoder_parameters_trainable | length == 0' '$M'" \
                                                          "no encoder parameter is trainable"
  check "jq -e '.optimization.optim != null and .optimization.lr_scheduler_type != null and .optimization.max_grad_norm != null' '$M'" \
                                                          "optimizer/scheduler/clipping recorded (R1-7.1)"
  check "jq -e '.schedule.max_steps > 0 and .schedule.steps_completed > 0' '$M'" \
                                                          "step counts recorded (R2-6a)"
  check "jq -e '.seeding.training_args_seed == .seeding.seed' '$M'" \
                                                          "TrainingArguments.seed == config seed (the seed-plumbing fix)"
  check "jq -e '.model.untie_procedure != null' '$M'"     "embedding-untie procedure recorded (R1-7.3)"
  check "jq -e '.decoding.force_language == true and .decoding.num_beams != null' '$M'" \
                                                          "decoding settings recorded (R1-7.1)"

  # --- curves ---------------------------------------------------------------
  local J="$RUN/metrics.jsonl"
  check "[[ -f '$J' ]]"                                   "metrics.jsonl exists"
  if [[ -f "$J" ]]; then
    local ntr nev n0
    ntr=$(jq -s '[.[] | select(.kind=="train" and .loss!=null and .grad_norm!=null and .learning_rate!=null)] | length' "$J")
    nev=$(jq -s '[.[] | select(.kind=="eval" and .eval_loss!=null)] | length' "$J")
    n0=$(jq -s  '[.[] | select(.kind=="eval" and .step==0)] | length' "$J")
    check "(( $ntr >= 20 ))"                              "train rows with loss+grad_norm+lr: $ntr (>=20)"
    # How many evals a run can log is set by its length, not by a constant: a run logs
    # floor(max_steps / eval_steps) evals plus the step-0 anchor. A flat floor of 5
    # fails short runs that behaved perfectly -- SPGISpeech frac10 is 979 steps at
    # eval_steps 250, so 4 is the correct and complete number. Derive the expectation
    # instead, and keep a floor of 2 (step 0 plus one) so a run that logged almost
    # nothing still fails.
    local sc es want
    sc=$(jq -r '.schedule.steps_completed // 0' "$M" 2>/dev/null)
    es=$(jq -r '.provenance.config_resolved.eval_steps // 0' "$M" 2>/dev/null)
    if (( sc > 0 && es > 0 )); then
      # steps_completed, not max_steps: a fixed-budget run stops a little short when the
      # post-filter stream runs dry (finding A8), and the evals it logged follow the
      # steps it actually took.
      want=$(( sc / es + 1 ))
      (( want < 2 )) && want=2
    else
      want=5                                              # manifest predates the fields
    fi
    check "(( $nev >= want ))"                            "eval rows with eval_loss: $nev (>=$want for steps=$sc/eval_steps=$es)"
    check "(( $n0 >= 1 ))"                                "step-0 eval anchor present"
  fi

  # --- cost -----------------------------------------------------------------
  local C="$RUN/cost.json"
  check "[[ -f '$C' ]]"                                   "cost.json exists"
  if [[ -f "$C" ]]; then
    check "jq -e '.train_wall_s > 0' '$C'"                "train wall-clock > 0"
    check "jq -e '.peak_mem_alloc_bytes > 0' '$C'"        "peak GPU memory recorded (R1-6.2)"
    check "jq -e '.train_samples_per_second > 0' '$C'"    "throughput recorded"
    check "jq -e '.total_flos_proxy != null' '$C'"        "compute proxy recorded"
    check "jq -e '.fingerprint.gpus | length > 0' '$C'"   "hardware fingerprint populated"
    check "jq -e '.inference_wall_s >= 0 and .eval_wall_s >= 0' '$C'" \
                                                          "wall-clock split into train/eval/inference"
  fi

  # --- eval sets (in-domain + OOD), dual-path -------------------------------
  local n_sets=0 n_ood=0
  if [[ -d "$RUN/eval" ]]; then
    for d in "$RUN"/eval/*/; do
      local name mj; name=$(basename "$d"); mj="$d/metrics.json"
      [[ -f "$mj" ]] || { bad "$name: metrics.json missing"; continue; }
      n_sets=$((n_sets+1))
      jq -e '.kind=="ood"' "$mj" >/dev/null 2>&1 && n_ood=$((n_ood+1))

      check "jq -e '.wer_fixed != null and .wer_fixed > 0 and .wer_fixed < 1' '$mj'" \
                                                          "$name: wer_fixed finite and in (0,1)"
      check "jq -e '.denominator > 0' '$mj'"              "$name: denominator reported"
      check "jq -e '.n_samples_from_dataset != null and .n_dropped_empty_ref != null' '$mj'" \
                                                          "$name: filtered-sample counts reported"
      check "jq -e '.spec.num_beams == 1 and .spec.force_language == true and .spec.dtype == \"fp16\"' '$mj'" \
                                                          "$name: decode spec is the fixed path"
      # Amendment: the legacy path must have been scored too, with its own denominator.
      check "jq -e 'has(\"wer_legacy\") and .wer_legacy != null' '$mj'" \
                                                          "$name: wer_legacy present (old-vs-new delta)"
      check "jq -e '.denominator_legacy != null' '$mj'"   "$name: denominator_legacy present"
      check "jq -e '.delta_wer_legacy_minus_fixed != null' '$mj'" \
                                                          "$name: legacy-vs-fixed delta recorded"
      check "[[ -s '$d/predictions.jsonl' ]]"             "$name: per-utterance predictions.jsonl (action Z1)"

      jq -r '"       -> \(.kind // "?")  wer_fixed=\(.wer_fixed)  n=\(.denominator)  |  wer_legacy=\(.wer_legacy)  n=\(.denominator_legacy)  |  delta=\(.delta_wer_legacy_minus_fixed)"' "$mj"
    done
  fi
  check "(( $n_sets >= 1 ))"                              "at least one eval set scored ($n_sets)"
  check "(( $n_ood >= 2 ))"                               "LibriSpeech clean+other OOD scored ($n_ood) (action B7)"

  # --- best-vs-final (action B8) --------------------------------------------
  local B="$RUN/best_vs_final.json"
  if jq -e '.run_best_vs_final == true' "$M" >/dev/null 2>&1 \
     || jq -e '.provenance.config_resolved.run_best_vs_final == true' "$M" >/dev/null 2>&1; then
    check "[[ -f '$B' ]]"                                 "best_vs_final.json exists"
    if [[ -f "$B" ]]; then
      check "jq -e '.best_step != null and .wer_best != null and .wer_final != null' '$B'" \
                                                          "best-vs-final has best_step, wer_best, wer_final"
      jq -r '"       -> best_step=\(.best_step) wer_best=\(.wer_best) wer_final=\(.wer_final) delta=\(.delta_wer_final_minus_best)"' "$B"
    fi
  fi

  # --- storage discipline ---------------------------------------------------
  check "[[ ! -d '$RUN/ckpt' ]]"                          "intermediate checkpoints deleted"
  local full_models
  full_models=$(find "$RUN" -maxdepth 2 \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) 2>/dev/null | wc -l)
  check "(( $full_models == 0 ))"                         "no full model weights persisted (encoder is frozen)"

  local mode; mode=$(jq -r '.artifacts.weights.save_mode // "?"' "$M" 2>/dev/null)
  case "$mode" in
    adapter)
      check "[[ -f '$RUN/adapter/adapter_model.safetensors' ]]" "LoRA adapter saved"
      local ab; ab=$(jq -r '.artifacts.weights.bytes // 0' "$M")
      # An adapter's size is set by how many modules carry one, how big those modules
      # are, and at what rank -- so a flat cap is wrong along the very axis this study
      # varies. Two corrections have been needed here:
      #   1. A flat 100 MB failed L5/L6 (168 MiB is correct at r=64).
      #   2. A uniform per-module budget then failed L0, because proj_out is not a
      #      typical module: it is 1024 x 51865, so LoRA on it alone is ~13.5 MB at
      #      r=64 -- more than a flat per-module allowance grants for the whole run.
      # Budget the two module classes separately, both linear in rank:
      #   proj_out         r x (1024 + 51865) x 4 B  ~= 211,000 B per rank unit
      #   decoder modules  measured (95.4 MB - 13.5 MB)/120 at r=64 ~= 10,656 B per unit
      # That reproduces the observed sizes almost exactly (L4 95.4 MB, L5 177.2 MB), so
      # 2x headroom is genuine headroom. The cap stays far below a merged
      # whisper-medium (1.5 GiB fp16), which is the failure this check exists to catch.
      local nmod rank cap
      nmod=$(jq -r '.adaptation.target_modules.n_lora_layers // 0' "$M")
      rank=$(jq -r '.adaptation.lora.r // 0' "$M")
      if (( nmod > 0 && rank > 0 )); then
        cap=$(( 2 * rank * (211000 + (nmod - 1) * 10656) ))
      else
        cap=$(( 400 * 1048576 ))   # manifest predates those fields: flat fallback
      fi
      check "(( ab < cap ))" \
        "adapter within depth budget ($(( ab / 1048576 )) MiB < $(( cap / 1048576 )) MiB, ${nmod} modules @ r=${rank})"
      ;;
    trainable)
      check "[[ -f '$RUN/final.safetensors' ]]"           "trainable-only state dict saved"
      local tb; tb=$(jq -r '.artifacts.weights.bytes // 0' "$M")
      check "(( tb < 1073741824 ))"                       "trainable state dict < 1 GiB ($(( tb / 1048576 )) MiB)"
      ;;
    *) bad "unexpected save_mode: $mode" ;;
  esac

  local du_mib; du_mib=$(du -sm "$RUN" 2>/dev/null | cut -f1)
  echo "       run directory size: ${du_mib} MiB"

  # --- results.json ---------------------------------------------------------
  check "jq -e '.config_path != null' '$RUN/results.json'" "results.json records config_path"
  check "jq -e '.decode_spec != null' '$RUN/results.json'" "results.json records the decode spec"
}

# ---------------------------------------------------------------- dispatch
case "${1:-}" in
  --seed-divergence) seed_divergence "$2" "$3" ;;
  --baseline-parity) baseline_parity "$2" "$3" ;;
  "" ) echo "usage: $0 <run_dir> | --seed-divergence <a> <b> | --baseline-parity <run> <baseline>"; exit 1 ;;
  * )  audit_run "$1" ;;
esac

echo
printf 'passed=%d  failed=%d\n' "$PASS" "$FAIL"
(( FAIL == 0 )) || exit 1
