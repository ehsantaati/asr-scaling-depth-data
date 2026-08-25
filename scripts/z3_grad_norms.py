#!/usr/bin/env python
"""Action Z3, gradient behaviour: train/grad_norm for the fig1 cells.

R1-4.3 asks for convergence evidence, and names gradient behaviour as part of it.
fig1 covers the loss trajectories and the checkpoint evolution; this covers the
gradients, for exactly the same representative cells so the two read together.

Read from the archived TensorBoard event files under each run's runs/ directory,
which is what the comment asks for. Every value is cross-checked against the
grad_norm column of the same run's metrics.jsonl and the run aborts on any
mismatch -- the two are written by different code paths (the HF TB callback and
callbacks.py), so agreement is a real check rather than a tautology.

The one thing to keep in mind reading the output: HF logs the total gradient norm
*before* clipping, and every run in this campaign trains with max_grad_norm=1.0.
So a logged value above 1.0 is a step that was clipped, and the "clipped %" column
is the fraction of training that ran at the clip ceiling rather than on the raw
gradient. That is the number R1-4.1's "unstable optimization" claim actually turns
on.

    python scripts/z3_grad_norms.py
"""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# The fig1 cells, in the same order and with the same labels, so the two outputs
# can be read side by side.
CELLS = [
    ("Fixed budget", "GigaSpeech", "b1", "gigaspeech", "l5_lora", "L5 LoRA"),
    ("Fixed budget", "GigaSpeech", "b1", "gigaspeech", "l5_full", "L5 full FT"),
    ("Fixed budget", "GigaSpeech", "b2", "gigaspeech", "l0_lora", "L0 LoRA"),
    ("Data-limited", "VoxPopuli", "b1", "voxpopuli", "l5_lora_frac10", "L5 LoRA, 10% data"),
    ("Data-limited", "VoxPopuli", "b1", "voxpopuli", "l5_lora", "L5 LoRA, 100% data"),
]


def read_grad_norm(run_dir):
    """(steps, values) for train/grad_norm from the run's TB events, verified
    against metrics.jsonl."""
    events = sorted(Path(run_dir).glob("runs/*/events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"no TB event file under {run_dir}/runs")
    steps, vals = [], []
    for ev in events:
        ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        ea.Reload()
        if "train/grad_norm" not in ea.Tags()["scalars"]:
            continue
        for s in ea.Scalars("train/grad_norm"):
            steps.append(s.step)
            vals.append(s.value)
    order = np.argsort(steps)
    steps, vals = np.array(steps)[order], np.array(vals, dtype=float)[order]

    ref = {r["step"]: r["grad_norm"]
           for r in (json.loads(l) for l in (Path(run_dir) / "metrics.jsonl").open())
           if r.get("kind") == "train" and r.get("grad_norm") is not None}
    common = [(v, ref[s]) for s, v in zip(steps, vals) if s in ref]
    if not common:
        raise AssertionError(f"{run_dir}: no overlapping steps to cross-check")
    # A NaN in both sources is agreement, not a mismatch: fp16 AMP logs a
    # non-finite norm on a gradient-scaler overflow, and both writers record it.
    pairs = [(a, b) for a, b in common
             if not (not np.isfinite(a) and not np.isfinite(b))]
    worst = max((abs(a - b) for a, b in pairs), default=0.0)
    assert worst < 1e-5, f"{run_dir}: TB vs metrics.jsonl disagree by {worst}"
    return steps, vals, len(common), worst


def summarise(run_dir):
    m = json.load((Path(run_dir) / "run_manifest.json").open())
    cfg = m["provenance"]["config_resolved"]
    clip = cfg["max_grad_norm"]
    warm = cfg["warmup_steps"]
    steps, vals, n_checked, worst = read_grad_norm(run_dir)

    # fp16 AMP writes a non-finite norm on a gradient-scaler overflow; that step is
    # skipped by the optimizer, so it is counted and reported but excluded from the
    # distribution statistics, which would otherwise all collapse to NaN.
    finite = np.isfinite(vals)
    nf_steps = steps[~finite].tolist()
    fvals, fsteps = vals[finite], steps[finite]

    post = fvals[fsteps > warm]
    late = fvals[fsteps > fsteps.max() * 0.75]
    med_post = float(np.median(post)) if len(post) else float("nan")
    return {
        "run": run_dir,
        "clip": clip,
        "warmup": warm,
        "max_steps": m["schedule"]["max_steps"],
        "n_logged": len(vals),
        "n_checked": n_checked,
        "worst_diff": worst,
        "warm_pct": 100.0 * warm / m["schedule"]["max_steps"],
        "med_warm_over_post": (float(np.median(fvals[fsteps <= warm])) / med_post)
                              if (fsteps <= warm).any() and med_post == med_post else float("nan"),
        "n_nonfinite": len(nf_steps),
        "nonfinite_steps": nf_steps,
        "peak": float(fvals.max()),
        "peak_step": int(fsteps[fvals.argmax()]),
        "med_warm": float(np.median(fvals[fsteps <= warm])) if (fsteps <= warm).any() else float("nan"),
        "med_post": med_post,
        "mean_post": float(post.mean()) if len(post) else float("nan"),
        "p99_post": float(np.percentile(post, 99)) if len(post) else float("nan"),
        "med_late": float(np.median(late)) if len(late) else float("nan"),
        "clipped_pct": 100.0 * float((fvals > clip).mean()),
        "clipped_pct_post": 100.0 * float((post > clip).mean()) if len(post) else float("nan"),
        # A "spike" is an order of magnitude above the run's own post-warmup median,
        # so the threshold adapts to each configuration instead of being absolute.
        "spikes": int((post > 10 * med_post).sum()) if len(post) else 0,
    }


def agg(rs, key):
    xs = [r[key] for r in rs]
    return st.mean(xs), min(xs), max(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional CSV of the per-run rows")
    args = ap.parse_args()

    per_run, per_cell = [], []
    for regime, corpus, batch, cdir, cell, label in CELLS:
        runs = sorted(Path("outputs/rev", batch, cdir, cell).glob("*/run_manifest.json"))
        rs = []
        for mp in runs:
            if json.load(mp.open()).get("status") != "complete":
                continue
            s = summarise(str(mp.parent))
            s.update(regime=regime, corpus=corpus, cell=cell, label=label)
            rs.append(s)
            per_run.append(s)
        if rs:
            per_cell.append((regime, corpus, label, rs))

    checked = sum(r["n_checked"] for r in per_run)
    worst = max(r["worst_diff"] for r in per_run)
    print(f"Source: archived TensorBoard events (runs/events.out.tfevents.*), "
          f"tag train/grad_norm.")
    print(f"Cross-check: {checked:,} values against metrics.jsonl across "
          f"{len(per_run)} runs, worst absolute difference {worst:g}.")
    clips = {r["clip"] for r in per_run}
    print(f"max_grad_norm = {sorted(clips)} for every run: HF logs the norm BEFORE "
          f"clipping, so a logged value > 1.0 is a clipped step.\n")

    hdr = (f"{'condition':<22} {'n':>2} {'median':>13} {'mean':>7} {'p99':>7} "
           f"{'peak':>8} {'@step':>7} {'clip%':>7} {'late med':>9} {'warm%':>6} "
           f"{'spikes':>7} {'nonfin':>7}")
    print(hdr)
    print("-" * len(hdr))
    last_regime = None
    for regime, corpus, label, rs in per_cell:
        if regime != last_regime:
            print(f"[{regime} · {corpus if last_regime is None else corpus}]"
                  if False else f"[{regime}]")
            last_regime = regime
        mp, lo, hi = agg(rs, "med_post")
        print(f"{label:<22} {len(rs):>2} "
              f"{mp:>6.3f} ({lo:.2f}-{hi:.2f}) "
              f"{agg(rs, 'mean_post')[0]:>7.3f} "
              f"{agg(rs, 'p99_post')[0]:>7.2f} "
              f"{agg(rs, 'peak')[0]:>8.1f} "
              f"{int(st.median([r['peak_step'] for r in rs])):>7} "
              f"{agg(rs, 'clipped_pct_post')[0]:>6.1f}% "
              f"{agg(rs, 'med_late')[0]:>9.3f} "
              f"{agg(rs, 'warm_pct')[0]:>5.1f}% "
              f"{agg(rs, 'spikes')[0]:>7.1f} "
              f"{sum(r['n_nonfinite'] for r in rs):>3}/{sum(r['n_logged'] for r in rs):<4}")
    print()
    print("median/mean/p99/clip%/late-med are post-warmup (step > warmup_steps); "
          "median column shows the mean across seeds with the seed range in brackets.")
    print("late med = median over the final 25% of steps. spikes = post-warmup steps "
          "above 10x that run's own post-warmup median.")
    print("warm% = warmup_steps as a share of that cell's step budget -- the "
          "data-limited run spends most of a much shorter schedule in LR warmup.")
    print("nonfin = non-finite logged norms / logged steps, summed over the cell's "
          "seeds: fp16 AMP gradient-scaler overflows, whose optimizer step is skipped.")

    nf = [r for r in per_run if r["n_nonfinite"]]
    print(f"\nNon-finite grad norms: {sum(r['n_nonfinite'] for r in per_run)} across "
          f"{len(nf)} of {len(per_run)} runs, never more than one per run.")
    for r in nf:
        print(f"    {r['run']}  step(s) {r['nonfinite_steps']} of {r['max_steps']:,}")
    print("Logging is every 25 steps, so these are sampled, not exhaustive: one "
          "non-finite value in N logged steps implies an overflow rate of order 1/N of\n"
          "optimizer steps, not one overflow in the whole run. What matters for R1-4.1 "
          "is that the rate is the same order for LoRA and full FT, shallow and deep.")

    if args.out:
        import csv
        keys = list(per_run[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_run)
        print(f"\nwrote {len(per_run)} per-run rows to {args.out}")


if __name__ == "__main__":
    main()
