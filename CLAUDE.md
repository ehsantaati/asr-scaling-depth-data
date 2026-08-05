# Working in this repository

This repo runs the instrumented rerun campaign for a journal major-revision of
"Data and Capacity Trade-offs in In-Domain Parameter-Efficient Adaptation of Whisper".
The plan lives in `instructions/revision_plan_context.md` (gitignored, local only);
the code audit that produced the current design is `instructions/revision_code_audit.pdf`.

Read this before changing anything under `train.py`, `decoding.py`, `data/`,
`configs/rev/`, `jobs/` or `scripts/queue_*`.

## The one fact that shapes everything

**The original campaign's checkpoints were deleted.** Every checkpoint-dependent
analysis therefore has to run *inside* the training job, before its weights go away.
That is why `train.py` does OOD evaluation, best-val-vs-final comparison and dual-path
scoring in-process rather than leaving them for later, and why a job that exits 0
without a complete `run_manifest.json` counts as a **failure**.

If you find yourself thinking "we can compute that afterwards from the saved model" —
we cannot. Add it to the in-job path.

## Invariants

1. **One job = one (config, seed) = one run directory.** Rerun configs set
   `data_fractions: [<single value>]` and `num_subsets: 1`.
2. **`run_manifest.json` with `status: "complete"` is the authority** on whether a run
   finished. It outranks any file in `jobs/status/`. The queue relies on this for
   idempotent restarts — do not write it earlier in the job.
3. **The encoder is frozen throughout.** `set_trainable_parameters` raises if any
   encoder parameter is trainable, and `save_trainable_state` raises if one would be
   persisted. Do not relax either guard; the entire study is scoped to decoder
   adaptation depth.
4. **Never persist a full model.** LoRA runs save adapters (before merge), full-FT runs
   save a trainable-only fp16 state dict. `check_artifacts.sh` fails the run otherwise.
5. **Seeds must reach the training loop.** `Seq2SeqTrainingArguments` must receive
   `seed=config.seed`. Without it `Trainer.__init__` calls `set_seed(42)` and every
   "different seed" produces an identical run — this was a real bug and it silently
   invalidated the original multi-seed results.
6. **The asymmetric seed design is deliberate.** LoRA gets 3-5 seeds, full FT 2-3 and
   only on claim-bearing configs. `jobs/campaign.jobs` is hand-enumerated, one line per
   (config, seed). Never replace it with a nested loop — that turns it into a symmetric
   grid and changes what the paper claims.
7. **Both scoring paths run.** Every eval set is scored by the fixed decode path
   (`decoding.transcribe_and_score`) and by the legacy `inference.run_inference_map`,
   recorded as `wer_fixed` / `wer_legacy` with both denominators. `run_inference_map`
   is frozen: it exists to measure the old-vs-new delta, so do not "fix" it.
8. **Report denominators.** The legacy path silently dropped 22% of the GigaSpeech test
   set. Any WER must be accompanied by the number of utterances it averages over.

## Environment

- Host: `.venv/bin/python`. Container: `/opt/venv/bin/python`.
  `scripts/_common.sh:resolve_py` picks correctly in both — use it rather than
  hardcoding a path. The host `.venv` is visible at `/app/.venv` inside the container
  and must never be selected there.
- Docker is **rootless** on this machine: container root maps to the host user. Do not
  add a `user:` mapping to `docker-compose.yml` (see the comment there).
- Build context for `docker/Dockerfile` is the **repo root**, not `docker/`.
  Use `docker compose -f docker/docker-compose.yml build` or `scripts/docker_build.sh`.

## Data

- Corpora are **cached, never streamed**. Streaming measured ~2 MB/s against a link
  that does ~94 MB/s, kept nothing, and made a single full-data pass ~25 h — repeated
  per seed. Use `scripts/prefetch_datasets.py` before starting a phase.
- `datasets.load_dataset` builds *every* split of a config, so requesting one test split
  can materialise tens of GB of unrelated training data. Check afterwards and delete
  what is not needed (LibriSpeech train splits were 58 GB of pure waste).
- **LibriSpeech test-clean/other must stay cached for the whole campaign** — it is the
  OOD set in every B1/B2 job. It is only ~1.4 GB.
- Storage is tight, so corpora are processed one at a time and deleted between phases.
  **Only delete a corpus after `check_artifacts.sh` passes on every job for it.**
  `Z1` bootstrap reads `predictions.jsonl` from run directories, so it survives deletion.

## Running things

```bash
bash scripts/prefetch_datasets.py --config <cfg>   # cache a phase's corpora first
bash scripts/queue_launch.sh jobs/campaign.jobs    # tmux, one worker per GPU
bash scripts/queue_status.sh jobs/campaign.jobs    # progress
bash scripts/queue_reset.sh --stale                # clear dead .running files
bash scripts/check_artifacts.sh <run_dir>          # artifact contract, 56 checks
```

One worker per GPU, never two: a second process on the same device corrupts the peak
memory and throughput numbers the cost table (action B6) depends on.

## For agents working in this repo

Default to **read-only**. Reporting, analysis and triage are welcome; starting,
stopping and deleting are not.

- **Do not** launch, kill or re-queue training jobs unless explicitly asked in that
  message. A campaign run is hours of GPU time and its checkpoints do not survive.
- **Do not** delete anything under `outputs/`, `~/.cache/huggingface`, or
  `/home/ehsan/data/`. Losing data is how this revision became necessary.
- **Do not** edit a config while runs using it are in flight — a change between seeds
  breaks the comparison the seeds exist to make.
- **Do not** retry past `MAX_ATTEMPTS`. A job that failed twice is usually systematic;
  the parked `.failed` state is there so a human looks at it.
- Green checks are not the same as good results. `check_artifacts.sh` verifies that
  artifacts *exist and are well-formed*, not that the numbers are sane. Read the loss
  trajectory, the WERs and the denominators before declaring a run good.
