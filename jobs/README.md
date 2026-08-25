# Job queue

`campaign.jobs` is the campaign's ground truth: one tab-separated line per
**(config, seed)** pair, in priority order (B1 > B7 > Z3 > B3 > B6 > B5 > B4 > B10 > B11;
B6/B7 carry no lines of their own because they execute inside the B1/B2 jobs).

Batches after B1 have their own job files rather than lines in `campaign.jobs` --
`b1_{voxpopuli,gigaspeech,spgispeech}.jobs`, `b3_*.jobs`, `b4_*.jobs`, `b5_voxpopuli.jobs`,
`b10_intermediate.jobs`, `b11_gigaspeech.jobs`. Launch one at a time; workers exit when
the list drains.

One line per **(config, seed, subset)** where a cell is replicated over data subsets:
B11 names its partition through `subset_index` in the config rather than looping inside
the job, because `run_dir_for()` below resolves a single run directory per job.

```
job_id <TAB> config_path <TAB> seed <TAB> run_name <TAB> output_dir <TAB> extra_args
```

`extra_args` of `-` means none. The sentinel `--INFERENCE` routes a job to
`inference.py` instead of `train.py` (vanilla baselines).

The list is **hand-enumerated**. There is deliberately no cross-product loop in the
runner, because the campaign's seed design is asymmetric: LoRA gets 3-5 seeds per
claim-bearing config, full FT gets 2-3 and only on the configs carrying contested
claims (GigaSpeech L4-L6; SPGISpeech/VoxPopuli L5), and shallow L0-L3 stay
single-seed. A nested loop would quietly turn that into a symmetric grid. Adding a
seed means adding a line.

`status/` holds runtime state (gitignored): `<job_id>.{running,done,failed}`.
A run directory containing `run_manifest.json` with `status == "complete"` outranks
any status file, so a lost status file never causes a completed run to be repeated.

Usage:

    bash scripts/queue_launch.sh jobs/campaign.jobs   # tmux, one worker per GPU
    bash scripts/queue_status.sh jobs/campaign.jobs
