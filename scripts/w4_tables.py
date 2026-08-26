#!/usr/bin/env python
"""Action W4: the complete numerical results table, assembled from run artifacts.

R1-2.4 and R3-7 ask for "mean WER, variability, number of runs, trainable parameter
count, and optimization budget for each principal configuration". R1-2.1 asks that n be
reported everywhere. This builds that table from what the runs actually wrote.

Sources, in order of authority:
  run_manifest.json  WER per eval set with its denominator, seeds, step counts, targets
  cost.json          peak memory, throughput, wall-clock, trainable parameter count
  outputs/rev/z1/bootstrap.json   evaluation-set 95% CIs

It deliberately does NOT read outputs/all_exps_final.csv. That file disagrees with the
results.json at the path it cites for 45 rows, carrying superseded 2026-03-09 values
under 2026-04-01 paths (PROGRESS §3.10). Any table built from it would reproduce
retired numbers under current labels.

Two conventions worth stating in the paper:

  * Percent-trainable uses the BASE model total, 816,967,680 parameters. PEFT reports
    LoRA runs against base+adapter (840,799,808), which understates LoRA's share and
    makes it non-comparable with full FT's denominator. The base-model convention
    reproduces the manuscript's own figures -- L5 full 55.8%, L6 full 62.4%, LoRA L5
    5.4% -- against its "55-62%" and "below 6%".
  * n is printed for every cell and cells with n < 3 are flagged. An SD from two runs
    is barely an estimate and should not be presented as though it were one.

    python scripts/w4_tables.py                  # markdown to stdout
    python scripts/w4_tables.py --csv w4.csv     # also write a flat CSV
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE_PARAMS = 816_967_680  # Whisper-medium, excluding any LoRA adapter

IN_DOMAIN = {
    "voxpopuli": "voxpopuli-en",
    "gigaspeech": "speechcolab-gigaspeech-m",
    "spgispeech": "spgispeech_2",
}
OOD = ("openslr-librispeech-asr-clean", "openslr-librispeech-asr-other")
BATCHES = ("b1", "b2", "b3", "b4", "b5", "b10", "b11", "b13")


def load_runs(root="outputs/rev"):
    """{(batch, corpus, cell): [ {manifest, cost}, ... ]} over complete runs."""
    cells = defaultdict(list)
    for batch in BATCHES:
        for mp in sorted(Path(root, batch).glob("*/*/*/run_manifest.json")):
            m = json.load(mp.open())
            if m.get("status") != "complete":
                continue
            corpus, cell = mp.parts[-4], mp.parts[-3]
            cp = mp.parent / "cost.json"
            cost = json.load(cp.open()) if cp.exists() else {}
            cells[(batch, corpus, cell)].append({"m": m, "c": cost, "dir": mp.parent})
    return cells


def load_cis(path="outputs/rev/z1/bootstrap.json"):
    p = Path(path)
    if not p.exists():
        return {}
    return {(c["corpus"], c["cell"]): c["ci95"] for c in json.load(p.open())["cells"]}


def trainable_params(run):
    """Trainable parameters, LoRA and full FT alike.

    cost.json carries n_params only for full FT (the saved state dict is the trainable
    set). LoRA saves an adapter, so count its tensors from the adapter file instead --
    that IS the trainable set for a LoRA run.
    """
    w = run["c"].get("weights", {})
    if w.get("n_params"):
        return int(w["n_params"])
    adapter = run["dir"] / "adapter" / "adapter_model.safetensors"
    if adapter.exists():
        from safetensors import safe_open
        n = 0
        with safe_open(str(adapter), framework="pt") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                p = 1
                for d in shape:
                    p *= d
                n += p
        return n
    return None


def summarise(cells, cis):
    rows = []
    for (batch, corpus, cell), runs in sorted(cells.items()):
        ind = IN_DOMAIN[corpus]
        res = [r["m"]["results"] for r in runs]
        if ind not in res[0]:
            continue
        wers = [x[ind]["wer_fixed"] * 100 for x in res]
        sched = runs[0]["m"]["schedule"]
        lora = runs[0]["m"]["adaptation"].get("lora") or {}
        npar = trainable_params(runs[0])
        ci = cis.get((corpus, cell))
        rows.append({
            "batch": batch,
            "corpus": corpus,
            "cell": cell,
            "method": runs[0]["m"]["adaptation"]["method"],
            "rank": lora.get("r") or "",
            "lr": runs[0]["m"]["optimization"]["learning_rate"],
            "fraction": sched.get("fraction"),
            "n": len(runs),
            "wer": st.mean(wers),
            "sd": st.stdev(wers) if len(wers) > 1 else None,
            "ci_lo": ci[0] * 100 if ci else None,
            "ci_hi": ci[1] * 100 if ci else None,
            "n_utts": res[0][ind]["denominator"],
            "ood_clean": st.mean([x[OOD[0]]["wer_fixed"] * 100 for x in res]),
            "ood_other": st.mean([x[OOD[1]]["wer_fixed"] * 100 for x in res]),
            "params": npar,
            "pct": (100 * npar / BASE_PARAMS) if npar else None,
            "peak_gib": st.mean([r["c"]["peak_mem_alloc_gib"] for r in runs
                                 if "peak_mem_alloc_gib" in r["c"]] or [0]) or None,
            "samples_s": st.mean([r["c"]["train_samples_per_second"] for r in runs
                                  if "train_samples_per_second" in r["c"]] or [0]) or None,
            "train_h": st.mean([r["c"]["train_wall_excl_eval_s"] for r in runs
                                if "train_wall_excl_eval_s" in r["c"]] or [0]) / 3600 or None,
            "max_steps": sched.get("max_steps"),
            "steps": sched.get("steps_completed"),
            "epochs": sched.get("epochs_consumed"),
        })
    return rows


def fmt(v, spec=".3f", dash="—"):
    return dash if v is None else format(v, spec)


def markdown(rows):
    out = []
    out.append("## Complete numerical results (W4 — R1-2.4, R3-7)\n")
    out.append(f"Trainable % is against the base model, {BASE_PARAMS:,} parameters. "
               "WER is the fixed decode path; the in-domain denominator differs per "
               "corpus and is given as *n utts*. CIs are evaluation-set bootstrap "
               "(10,000 paired resamples) and are **not** a substitute for the seed SD "
               "beside them.\n")
    out.append("**†** marks cells with n < 3, where the SD is an indication, not an "
               "estimate.\n")
    for corpus in ("voxpopuli", "spgispeech", "gigaspeech"):
        sub = [r for r in rows if r["corpus"] == corpus]
        if not sub:
            continue
        out.append(f"\n### {corpus}\n")
        out.append("| Config | n | WER % | SD | 95% CI | n utts | LS-clean | LS-other | "
                   "Trainable | % | Peak GiB | samp/s | Steps | Epochs |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sub, key=lambda x: (x["batch"], x["cell"])):
            flag = "†" if r["n"] < 3 else ""
            ci = (f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]" if r["ci_lo"] is not None else "—")
            out.append(
                f"| `{r['cell']}` | {r['n']}{flag} | {fmt(r['wer'])} | {fmt(r['sd'])} | {ci} | "
                f"{r['n_utts']:,} | {fmt(r['ood_clean'])} | {fmt(r['ood_other'])} | "
                f"{r['params']/1e6:.1f}M | {fmt(r['pct'], '.2f')} | {fmt(r['peak_gib'], '.2f')} | "
                f"{fmt(r['samples_s'], '.1f')} | {r['steps']:,} | {fmt(r['epochs'], '.3f')} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/rev")
    ap.add_argument("--csv", help="also write a flat CSV of every column")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    # There are several bootstrap files (b1+b3 only, +b5, all batches). Whichever is
    # passed must cover the batches in BATCHES or cells silently print "—" for CI,
    # which reads as "not computed" when it really means "not scanned".
    ap.add_argument("--cis", default="outputs/rev/z1/bootstrap_all.json",
                    help="Z1 bootstrap json supplying the 95% CIs")
    args = ap.parse_args()

    cis = load_cis(args.cis)
    rows = summarise(load_runs(args.root), cis)
    missing = [f"{r['corpus']}/{r['cell']}" for r in rows if r["ci_lo"] is None]
    if missing:
        print(f"WARNING: no bootstrap CI for {len(missing)} cells "
              f"(not in {args.cis}): {', '.join(missing)}")
    md = markdown(rows)
    if args.out:
        Path(args.out).write_text(md + "\n")
        print(f"wrote {args.out} ({len(rows)} cells)")
    else:
        print(md)
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
