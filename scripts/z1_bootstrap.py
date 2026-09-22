#!/usr/bin/env python
"""Action Z1: bootstrap confidence intervals over per-utterance predictions.

Answers R1-2.2 ("several reported WER differences are very small and may not be
statistically meaningful"), and supports R1-2.3, R2-3 and R3-3.

What this measures, and what it does not
----------------------------------------
This resamples the *evaluation set*, so it quantifies how much a WER would move if
the test set had been a different draw of utterances from the same distribution. It
is NOT a substitute for seed variance, which measures how much a WER moves if
training had been rerun. The two are complementary and B1 already supplies the
second: report both, per the plan.

The paired design matters. For a comparison the same resampled utterance indices are
applied to both systems, so the shared difficulty of the drawn utterances cancels and
what is left is the difference between systems. An unpaired interval on each system
separately would be far wider and would understate the evidence.

WER is a ratio of totals, not a mean of per-utterance rates: sum(edits)/sum(ref
words). Every resample therefore re-aggregates both numerator and denominator; taking
a mean of per-utterance WERs would silently reweight short utterances and give a
different number from the one in the manifest.

Two prediction formats are read:
  * reruns  -- eval/<set>/predictions.jsonl, one {idx, prediction, reference} per line
  * original campaign -- <run>/predictions.json, {"predictions": {set: [...]},
    "references": {set: [...]}} with the two lists positionally aligned
Both are already normalised: recomputing WER from them reproduces `wer_fixed` in
run_manifest.json exactly (verified to 4 dp on GigaSpeech L5 LoRA, 9.3723).

Usage:
    python scripts/z1_bootstrap.py --scan outputs/rev/b1 outputs/rev/b3
    python scripts/z1_bootstrap.py --scan outputs/rev/b1 --compare-only
    python scripts/z1_bootstrap.py --legacy outputs/voxpopuli --eval-set voxpopuli-en
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Comparisons the review actually contests. Each is (label, cell_a, cell_b) where a
# cell is the directory name under outputs/rev/<batch>/<corpus>/. Anything not present
# for a corpus is skipped rather than erroring, so the same table serves all three.
CONTESTED = [
    # B10 added L2/L3, which locates the GigaSpeech method crossover instead of
    # merely asserting it: full FT is ahead at L2/L3 and behind at L4/L5, so the
    # ordering flips inside the L3->L4 interval. Those two rows are the evidence.
    # B13 completed the full-FT depth curve on SPGISpeech, so the shallow method
    # comparison and the full-FT L2->L3 step exist on more than one corpus and are
    # worth their own intervals. L0 fires on VoxPopuli and SPGISpeech; GigaSpeech
    # gains it when b13 group c lands.
    ("LoRA vs full FT @ L0", "l0_lora", "l0_full"),
    ("full FT: L2 vs L3", "l2_full", "l3_full"),
    ("LoRA vs full FT @ L2", "l2_lora", "l2_full"),
    ("LoRA vs full FT @ L3", "l3_lora", "l3_full"),
    ("LoRA: L2 vs L3", "l2_lora", "l3_lora"),
    ("LoRA: L3 vs L4", "l3_lora", "l4_lora"),
    ("full FT: L3 vs L4", "l3_full", "l4_full"),
    ("LoRA vs full FT @ L4", "l4_lora", "l4_full"),
    ("LoRA vs full FT @ L5", "l5_lora", "l5_full"),
    ("full FT: L4 vs L5", "l4_full", "l5_full"),
    ("full FT: L5 vs L6", "l5_full", "l6_full"),
    # B13b restored the L6 LoRA cells (the 2026-08-06 drop rested on a false premise).
    # This is the LoRA-side counterpart of the full-FT L5->L6 null, and it is the row
    # that turns "the depth axis saturates at L5 for both methods" from an inference
    # into a measurement. NOTE voxpopuli l6_lora contains one pathological run
    # (s1234, decode repetition loops) -- read that corpus's interval accordingly.
    ("LoRA: L5 vs L6", "l5_lora", "l6_lora"),
    ("LoRA: L4 vs L5", "l4_lora", "l5_lora"),
    ("LoRA: full data vs 10%", "l5_lora", "l5_lora_frac10"),
    # B11 (GigaSpeech only): the data-scaling curve R1-5.3 asks for, at L4. Two kinds of
    # row, and they must not be conflated when reported:
    #   * 100% vs each reduced fraction -- the scaling effect itself;
    #   * subset vs subset at a fixed fraction -- how much of that effect is just which
    #     10%/20% of the pool the run happened to see. The manuscript's instability
    #     numbers are this second quantity, so it needs its own intervals.
    # Every row pairs on run name, and each B11 cell holds exactly one run (s42), so
    # these pair against the s42 member of the n=5 100% anchor.
    ("L4 LoRA: 100% vs 50%", "l4_lora", "l4_lora_frac50_sub0"),
    ("L4 LoRA: 100% vs 20% sub0", "l4_lora", "l4_lora_frac20_sub0"),
    ("L4 LoRA: 100% vs 20% sub1", "l4_lora", "l4_lora_frac20_sub1"),
    ("L4 LoRA: 100% vs 10% sub0", "l4_lora", "l4_lora_frac10_sub0"),
    ("L4 LoRA: 100% vs 10% sub1", "l4_lora", "l4_lora_frac10_sub1"),
    ("L4 LoRA: 100% vs 10% sub2", "l4_lora", "l4_lora_frac10_sub2"),
    ("L4 LoRA @10%: sub0 vs sub1", "l4_lora_frac10_sub0", "l4_lora_frac10_sub1"),
    ("L4 LoRA @10%: sub0 vs sub2", "l4_lora_frac10_sub0", "l4_lora_frac10_sub2"),
    ("L4 LoRA @10%: sub1 vs sub2", "l4_lora_frac10_sub1", "l4_lora_frac10_sub2"),
    ("L4 LoRA @20%: sub0 vs sub1", "l4_lora_frac20_sub0", "l4_lora_frac20_sub1"),
    ("L4 LoRA: 50% vs 20% sub0", "l4_lora_frac50_sub0", "l4_lora_frac20_sub0"),
    ("L4 LoRA: 20% sub0 vs 10% sub0", "l4_lora_frac20_sub0", "l4_lora_frac10_sub0"),
]


# --- pooled comparisons ----------------------------------------------------------
# CONTESTED pairs one cell against one cell. That is the wrong shape for a data-scaling
# ladder, where a "fraction" is not a cell but a SET of subset cells, and quoting any
# single cross-pairing is arbitrary: the six ways of pairing B11's 20% subsets against
# its 10% subsets span 0.231 pp and disagree on sign (+0.026 to -0.205). Reporting one of
# them as "20% vs 10%" is how a spans-zero result gets claimed for a step that in fact
# resolves -- which is exactly what happened, and is why this exists.
#
# Each side is averaged over its member cells on the SAME resampled utterances, so the
# draw still cancels and the statistic stays paired.
#
# READ THE INTERVAL CORRECTLY. This is an evaluation-set bootstrap: it answers "would
# this ordering survive a different test set", NOT "would it survive a different draw of
# training subsets". The second question is answered by the subset spread, which for
# B11 at 10% is SD 0.089 pp -- larger than the 20%->10% step itself. Report both.
POOLED = [
    # (label, corpus, [cells_a], [cells_b])
    ("L4 LoRA: 100% vs 50%  (pooled)", "gigaspeech", ["l4_lora"], ["l4_lora_frac50_sub0"]),
    ("L4 LoRA: 50% vs 20%   (pooled)", "gigaspeech", ["l4_lora_frac50_sub0"],
     ["l4_lora_frac20_sub0", "l4_lora_frac20_sub1"]),
    ("L4 LoRA: 20% vs 10%   (pooled)", "gigaspeech",
     ["l4_lora_frac20_sub0", "l4_lora_frac20_sub1"],
     ["l4_lora_frac10_sub0", "l4_lora_frac10_sub1", "l4_lora_frac10_sub2"]),
    ("L4 LoRA: 100% vs 20%  (pooled)", "gigaspeech", ["l4_lora"],
     ["l4_lora_frac20_sub0", "l4_lora_frac20_sub1"]),
    ("L4 LoRA: 100% vs 10%  (pooled)", "gigaspeech", ["l4_lora"],
     ["l4_lora_frac10_sub0", "l4_lora_frac10_sub1", "l4_lora_frac10_sub2"]),
]


def report_pooled(cells, n_boot, rng, out):
    """Fraction-vs-fraction comparisons, each side averaged over its subset cells."""
    printed = False
    for label, corpus, cells_a, cells_b in POOLED:
        keys_a = [(corpus, c) for c in cells_a]
        keys_b = [(corpus, c) for c in cells_b]
        if any(k not in cells for k in keys_a + keys_b):
            continue
        names = set.intersection(*[set(cells[k]) for k in keys_a + keys_b])
        for name in sorted(names):
            ga = [r for k in keys_a for r in cells[k][name]]
            gb = [r for k in keys_b for r in cells[k][name]]
            n_utt = len(ga[0][0])
            if any(len(r[0]) != n_utt for r in ga + gb):
                continue
            if not printed:
                print(f"\n{'pooled comparison':<46}{'delta pp':>10}{'95% CI':>22}"
                      f"{'P(sign flips)':>15}")
                print("-" * 93)
                print(f"  [{corpus}]")
                printed = True
            da, db = bootstrap([ga, gb], n_utt, n_boot, rng)
            delta = da - db
            obs = (float(np.mean([wer(e, l) for e, l in ga]))
                   - float(np.mean([wer(e, l) for e, l in gb])))
            lo, hi = ci(delta)
            p = float(np.mean(delta >= 0) if obs < 0 else np.mean(delta <= 0))
            flag = "" if (lo > 0) == (hi > 0) else "   <- spans zero"
            print(f"    {label:<42}{obs * 100:>10.3f}"
                  f"{'[' + format(lo * 100, '.3f') + ', ' + format(hi * 100, '.3f') + ']':>22}"
                  f"{p:>15.4f}{flag}")
            out["comparisons"].append({
                "corpus": corpus, "comparison": label, "eval_set": name,
                "cell_a": "+".join(cells_a), "cell_b": "+".join(cells_b),
                "pooled": True, "delta_wer": obs, "ci95": [lo, hi], "p_sign_flip": p,
                "n_utterances": n_utt, "n_runs_a": len(ga), "n_runs_b": len(gb),
            })


def utt_counts(refs, hyps):
    """Per-utterance (edit distance, reference word count) over whitespace tokens."""
    if len(refs) != len(hyps):
        raise ValueError(f"length mismatch: {len(refs)} refs vs {len(hyps)} hyps")
    edits = np.empty(len(refs), dtype=np.int64)
    lens = np.empty(len(refs), dtype=np.int64)
    for i, (r, h) in enumerate(zip(refs, hyps)):
        rt, ht = r.split(), h.split()
        edits[i] = Levenshtein.distance(rt, ht)
        lens[i] = len(rt)
    return edits, lens


def wer(edits, lens):
    total = lens.sum()
    return float(edits.sum()) / float(total) if total else float("nan")


def load_rerun(run_dir, eval_set):
    p = Path(run_dir) / "eval" / eval_set / "predictions.jsonl"
    if not p.exists():
        return None
    refs, hyps = [], []
    with p.open() as f:
        for line in f:
            row = json.loads(line)
            refs.append(row["reference"])
            hyps.append(row["prediction"])
    return utt_counts(refs, hyps)


def load_legacy(run_dir, eval_set):
    """Original-campaign predictions, which come in two shapes.

    VoxPopuli runs scored more than one set, so predictions/references are dicts
    keyed by set name (voxpopuli-en, voxpopuli-en-accented). GigaSpeech and
    SPGISpeech runs scored one set and store plain lists. Accept both; for the flat
    form the eval_set argument only labels the output.
    """
    p = Path(run_dir) / "predictions.json"
    if not p.exists():
        p = Path(run_dir) / "inference_results" / "predictions.json"
    if not p.exists():
        return None
    d = json.load(p.open())
    refs, hyps = d.get("references"), d.get("predictions")
    if isinstance(refs, dict):
        if eval_set not in refs:
            return None
        refs, hyps = refs[eval_set], hyps[eval_set]
    if not isinstance(refs, list) or not isinstance(hyps, list) or not refs:
        return None
    return utt_counts(refs, hyps)


def in_domain_set(run_dir):
    """The in-domain eval set name, read from the manifest rather than guessed."""
    m = Path(run_dir) / "run_manifest.json"
    if not m.exists():
        return None
    results = json.load(m.open()).get("results", {})
    for name, r in results.items():
        if r.get("kind") == "in_domain":
            return name
    return None


CHUNK = 250  # resamples held in memory at once


def bootstrap(groups, n_utt, n_boot, rng):
    """Bootstrap WER distributions for several groups of runs, paired.

    `groups` is a list of run-lists. Every group is evaluated on the *same* resampled
    indices, which is what makes a difference between two groups a paired statistic.
    Returns one (n_boot,) array per group, each already averaged over that group's
    seeds.

    Done in chunks because the index matrix is n_boot x n_utt: at 10k resamples over
    GigaSpeech's 19,898 utterances that is 1.6 GB before any indexing, and each
    take() would allocate another. Chunking keeps peak memory at CHUNK/n_boot of that
    with no change to the result.
    """
    outs = [np.empty(n_boot, dtype=np.float64) for _ in groups]
    done = 0
    while done < n_boot:
        m = min(CHUNK, n_boot - done)
        idx = rng.integers(0, n_utt, size=(m, n_utt))
        for gi, runs in enumerate(groups):
            acc = np.zeros(m, dtype=np.float64)
            for edits, lens in runs:
                e = edits[idx].sum(axis=1)
                l = lens[idx].sum(axis=1)
                acc += e / np.maximum(l, 1)
            outs[gi][done:done + m] = acc / len(runs)
        done += m
        del idx
    return outs


def ci(v, alpha=0.05):
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# Report at most this many seeds per cell so every cell's CI is computed over the
# same n as its mean/SD in the W4 table, even though the campaign itself deliberately
# ran some LoRA cells (gigaspeech/l4_lora, gigaspeech/l5_lora, spgispeech/l5_lora,
# voxpopuli/l6_lora) with 5 seeds -- see CLAUDE.md "The asymmetric seed design is
# deliberate." The extra seeds' runs and predictions are untouched on disk; this only
# caps what gets pooled into the bootstrap. The same seeds are kept across every eval
# set of a cell, per the paper's convention that in-domain and OOD results use the
# same retained seeds.
#
# Kept seeds are the lowest-valued N -- EXCEPT voxpopuli/l6_lora, where two of the five
# seeds are the documented failures (CAVEAT_L6_LORA in w4_tables.py / footnote c in the
# manuscript table): s1234 (9.332 WER) decodes with repetition loops, s12345 (7.433)
# has a training-loss excursion. The lowest-valued rule would silently keep s1234 --
# one of the two runs footnote c says to exclude -- so this cell is seed-selected by
# hand to match the paper's own "reported over three of five seeds" convention instead.
FORCE_SEEDS = {
    ("voxpopuli", "l6_lora"): {123456, 123, 42},  # excludes 1234, 12345 -- see above
}
MAX_SEEDS_PER_CELL = 3


def scan_cells(roots, eval_set_of=in_domain_set):
    """{(corpus, cell): {eval_set: [(edits, lens), ...]}} over every complete run."""
    cells = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(dict)  # (corpus, cell) -> {seed: run_dir}, lowest kept
    for root in roots:
        for manifest in sorted(Path(root).glob("*/*/*/run_manifest.json")):
            run = manifest.parent
            m = json.load(manifest.open())
            if m.get("status") != "complete":
                continue
            corpus, cell = run.parts[-3], run.parts[-2]
            seeds[(corpus, cell)][m["seeding"]["seed"]] = run
    kept = {}
    for key, s in seeds.items():
        if key in FORCE_SEEDS:
            kept[key] = {run for seed, run in s.items() if seed in FORCE_SEEDS[key]}
        else:
            kept[key] = {run for _, run in sorted(s.items())[:MAX_SEEDS_PER_CELL]}
    for root in roots:
        for manifest in sorted(Path(root).glob("*/*/*/run_manifest.json")):
            run = manifest.parent
            if json.load(manifest.open()).get("status") != "complete":
                continue
            corpus, cell = run.parts[-3], run.parts[-2]
            if run not in kept[(corpus, cell)]:
                continue
            name = eval_set_of(run)
            if not name:
                continue
            counts = load_rerun(run, name)
            if counts is not None:
                cells[(corpus, cell)][name].append(counts)
    return cells


def report_cells(cells, n_boot, rng, out):
    print(f"\n{'corpus/cell':<34}{'n':>3}{'WER %':>9}{'95% CI (eval-set)':>24}{'utts':>9}")
    print("-" * 79)
    for (corpus, cell) in sorted(cells):
        for name, runs in sorted(cells[(corpus, cell)].items()):
            n_utt = len(runs[0][0])
            (dist,) = bootstrap([runs], n_utt, n_boot, rng)
            obs = float(np.mean([wer(e, l) for e, l in runs]))
            lo, hi = ci(dist)
            print(f"{corpus + '/' + cell:<34}{len(runs):>3}{obs * 100:>9.3f}"
                  f"{'[' + format(lo * 100, '.3f') + ', ' + format(hi * 100, '.3f') + ']':>24}{n_utt:>9}")
            out["cells"].append({
                "corpus": corpus, "cell": cell, "eval_set": name, "n_runs": len(runs),
                "n_utterances": n_utt, "wer": obs, "ci95": [lo, hi],
            })


def report_comparisons(cells, n_boot, rng, out):
    print(f"\n{'comparison':<46}{'delta pp':>10}{'95% CI':>22}{'P(sign flips)':>15}")
    print("-" * 93)
    corpora = sorted({c for c, _ in cells})
    for corpus in corpora:
        printed = False
        for label, a, b in CONTESTED:
            ka, kb = (corpus, a), (corpus, b)
            if ka not in cells or kb not in cells:
                continue
            shared = set(cells[ka]) & set(cells[kb])
            for name in sorted(shared):
                ra, rb = cells[ka][name], cells[kb][name]
                n_utt = len(ra[0][0])
                if len(rb[0][0]) != n_utt:
                    continue                      # different eval sets; not pairable
                if not printed:
                    print(f"  [{corpus}]")
                    printed = True
                # Same resampled utterances drive both sides: the draw cancels.
                da, db = bootstrap([ra, rb], n_utt, n_boot, rng)
                delta = da - db
                obs = (float(np.mean([wer(e, l) for e, l in ra]))
                       - float(np.mean([wer(e, l) for e, l in rb])))
                lo, hi = ci(delta)
                # Share of resamples landing on the other side of zero: how often the
                # observed ordering would reverse on a different evaluation set.
                p = float(np.mean(delta >= 0) if obs < 0 else np.mean(delta <= 0))
                flag = "" if (lo > 0) == (hi > 0) else "   <- spans zero"
                print(f"    {label:<42}{obs * 100:>10.3f}"
                      f"{'[' + format(lo * 100, '.3f') + ', ' + format(hi * 100, '.3f') + ']':>22}"
                      f"{p:>15.4f}{flag}")
                out["comparisons"].append({
                    "corpus": corpus, "comparison": label, "eval_set": name,
                    "cell_a": a, "cell_b": b, "delta_wer": obs, "ci95": [lo, hi],
                    "p_sign_flip": p, "n_utterances": n_utt,
                    "n_runs_a": len(ra), "n_runs_b": len(rb),
                })


# Superseded original runs. `data_scaling/` was rerun on 2026-04-01 over the 2026-03-09
# `data_scaling_old/`, and the manuscript reports the newer set: its SPGISpeech 10%
# instability figure of 8.21 matches 008/0081/data_scaling/..._frac_0.1_subset_2
# (8.226), where the older twin gives 5.415. Note that all_exps_final.csv still carries
# the *older* values under the newer paths for 39 rows -- see PROGRESS §3.10. Bootstrap
# the runs the paper actually reports, not the ones it superseded.
SUPERSEDED = ("data_scaling_old", "_old", "repeat_inference")


def report_legacy(root, eval_set, n_boot, rng, out, include_superseded=False):
    """Single-seed original cells: they stay single-seed but gain an interval."""
    runs = sorted({p.parent for p in Path(root).rglob("predictions.json")})
    if not include_superseded:
        skipped = [r for r in runs if any(s in str(r) for s in SUPERSEDED)]
        runs = [r for r in runs if r not in set(skipped)]
        if skipped:
            print(f"skipping {len(skipped)} superseded run dirs "
                  f"({', '.join(SUPERSEDED)}); pass --include-superseded to keep them")
    print(f"\n{'original run':<62}{'WER %':>9}{'95% CI (eval-set)':>24}")
    print("-" * 95)
    for run in runs:
        counts = load_legacy(run, eval_set)
        if counts is None:
            continue
        edits, lens = counts
        n_utt = len(edits)
        (dist,) = bootstrap([[counts]], n_utt, n_boot, rng)
        obs = wer(edits, lens)
        lo, hi = ci(dist)
        label = str(run).replace("outputs/", "")
        print(f"{label[-60:]:<62}{obs * 100:>9.3f}"
              f"{'[' + format(lo * 100, '.3f') + ', ' + format(hi * 100, '.3f') + ']':>24}")
        out["legacy"].append({
            "run": str(run), "eval_set": eval_set, "n_utterances": n_utt,
            "wer": obs, "ci95": [lo, hi],
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="*", default=["outputs/rev/b1", "outputs/rev/b3"],
                    help="rerun roots to scan for completed runs")
    ap.add_argument("--legacy", help="original-campaign root, e.g. outputs/voxpopuli")
    ap.add_argument("--eval-set", help="eval set name, required with --legacy")
    ap.add_argument("--compare-only", action="store_true", help="skip per-cell intervals")
    ap.add_argument("--include-superseded", action="store_true",
                    help="also bootstrap data_scaling_old / repeat_inference runs")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42, help="bootstrap RNG seed")
    ap.add_argument("--out", default="outputs/rev/z1/bootstrap.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = {"n_boot": args.n_boot, "rng_seed": args.seed,
           "cells": [], "comparisons": [], "legacy": []}

    if args.legacy:
        if not args.eval_set:
            raise SystemExit("--legacy requires --eval-set")
        report_legacy(args.legacy, args.eval_set, args.n_boot, rng, out,
                      include_superseded=args.include_superseded)
    else:
        cells = scan_cells(args.scan)
        if not cells:
            raise SystemExit(f"no complete runs under {args.scan}")
        if not args.compare_only:
            report_cells(cells, args.n_boot, rng, out)
        report_comparisons(cells, args.n_boot, rng, out)
        report_pooled(cells, args.n_boot, rng, out)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")
    print("Evaluation-set uncertainty only. Report alongside the seed SD from B1 —\n"
          "they answer different questions and neither replaces the other.")


if __name__ == "__main__":
    main()
