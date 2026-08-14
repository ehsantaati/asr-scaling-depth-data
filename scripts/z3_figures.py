#!/usr/bin/env python
"""Action Z3: diagnostic figures from the rerun logs.

R1-4.2 asks for training and validation loss curves for representative shallow and
deep configurations; R1-4.3 for convergence evidence -- gradient behaviour,
generalization gaps, checkpoint evolution. R1-4.1 says the instability and memorization
explanations are "plausible but not directly demonstrated". These figures are the
demonstration.

Four figures, each answering a specific comment:

  fig1_loss_curves      train + validation loss with seed-spread bands   R1-4.2
  fig2_generalization   final train loss against test WER                R1-4.1, R1-4.3
  fig3_depth_wer        WER against depth, per corpus, with seed SD      R1-2.1, R1-2.3
  fig4_ood_tradeoff     in-domain gain against out-of-domain cost        R1-5.5, R3-10

Design decisions that are not arbitrary:

  * Bands are min-max across seeds, not +/- SD. With n=2 or 3 an SD band implies a
    distributional claim the sample cannot support; a min-max envelope says exactly
    what was observed and nothing more.
  * Method is encoded by colour AND line style, so the figures survive greyscale
    printing and colour-vision deficiency. Every series is also directly labelled.
  * No dual axes anywhere. Where two quantities of different scale must be compared
    (train loss vs WER) they get a scatter, not a twin axis.
  * The vanilla baseline is drawn as a reference rule wherever it exists, because
    several configurations are *worse* than no adaptation and that is invisible
    without it.

    python scripts/z3_figures.py --out figures/
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-3 of the validated palette; validated for this use with
# scripts/validate_palette.js (light, adjacent): worst CVD dE 9.2, normal-vision 27.6.
# Aqua sits below 3:1 on a white surface, so every series is directly labelled --
# that is the documented relief, not an oversight.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"
GRID = "#e6e5e1"

METHOD = {
    "lora": {"c": BLUE, "ls": "-", "label": "LoRA"},
    "full": {"c": ORANGE, "ls": "--", "label": "Full fine-tuning"},
}
IN_DOMAIN = {
    "voxpopuli": "voxpopuli-en",
    "gigaspeech": "speechcolab-gigaspeech-m",
    "spgispeech": "spgispeech_2",
}
CORPUS_LABEL = {"voxpopuli": "VoxPopuli", "spgispeech": "SPGISpeech", "gigaspeech": "GigaSpeech"}
DEPTH_ORDER = ["l0", "l4", "l5", "l6"]


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def load_cell(batch, corpus, cell):
    """[(manifest, metrics_rows)] for every complete run in a cell."""
    out = []
    for mp in sorted(Path("outputs/rev", batch, corpus, cell).glob("*/run_manifest.json")):
        m = json.load(mp.open())
        if m.get("status") != "complete":
            continue
        rows = [json.loads(l) for l in (mp.parent / "metrics.jsonl").open()]
        out.append((m, rows))
    return out


def series(runs, kind, key):
    """Common step grid across seeds, plus per-seed values interpolated onto it."""
    curves = []
    for _, rows in runs:
        pts = [(r["step"], r[key]) for r in rows if r.get("kind") == kind and r.get(key) is not None]
        if pts:
            curves.append(np.array(pts, dtype=float))
    if not curves:
        return None, None
    lo = max(c[0, 0] for c in curves)
    hi = min(c[-1, 0] for c in curves)
    grid = np.linspace(lo, hi, 300)
    vals = np.vstack([np.interp(grid, c[:, 0], c[:, 1]) for c in curves])
    return grid, vals


def band(ax, x, vals, colour, ls, label):
    """Mean line with a min-max envelope across seeds."""
    ax.fill_between(x, vals.min(axis=0), vals.max(axis=0), color=colour, alpha=0.16, lw=0)
    ax.plot(x, vals.mean(axis=0), color=colour, ls=ls, lw=1.8, label=label, solid_capstyle="round")


def fig1_loss_curves(out):
    """R1-4.2: train and validation loss, shallow vs deep, both methods.

    Two panels rather than train/val side by side. The first 500 steps drop the loss
    from ~5.5 to ~0.2, which on a linear axis flattens everything after it -- including
    the LoRA-vs-full separation that R1-4.1 turns on. So: left panel shows the whole
    trajectory on a log axis, right panel zooms the plateau where the methods actually
    differ. Same data, two questions.
    """
    corpus = "gigaspeech"          # where deep FT degrades -- the contested case
    cells = [("b2", "l0_lora", "lora", "L0", AQUA, "-"),
             ("b1", "l5_lora", "lora", "L5", BLUE, "-"),
             ("b1", "l5_full", "full", "L5", ORANGE, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    loaded = []
    for batch, cell, method, depth, colour, ls in cells:
        runs = load_cell(batch, corpus, cell)
        if runs:
            loaded.append((f"{depth} {METHOD[method]['label']}", runs, colour, ls))

    # Left: whole trajectory, log y so the collapse and the plateau both read.
    ax = axes[0]
    for lbl, runs, colour, ls in loaded:
        x, vals = series(runs, "train", "loss")
        if x is None:
            continue
        band(ax, x, vals, colour, ls, f"{lbl} (n={len(runs)})" if len(runs) > 1 else lbl)
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Training loss (log scale)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_title("Whole trajectory", color=INK, loc="left", fontsize=9.5)

    # Right: the plateau, where the methods separate. Skip the first 15% of steps.
    ax = axes[1]
    ends = []
    for lbl, runs, colour, ls in loaded:
        x, vals = series(runs, "train", "loss")
        if x is None:
            continue
        keep = x >= x[0] + 0.15 * (x[-1] - x[0])
        band(ax, x[keep], vals[:, keep], colour, ls, lbl)
        ends.append((vals[:, keep].mean(axis=0)[-1], lbl, colour))
    # Stagger direct labels so they cannot collide, in value order.
    ends.sort()
    for i, (y, lbl, colour) in enumerate(ends):
        ax.annotate(lbl, xy=(1.01, y), xycoords=("axes fraction", "data"),
                    xytext=(2, (i - (len(ends) - 1) / 2) * 11), textcoords="offset points",
                    color=colour, fontsize=7.5, va="center")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Training loss")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Plateau, after the initial collapse", color=INK, loc="left", fontsize=9.5)

    fig.suptitle("GigaSpeech: loss trajectories, bands span the observed seed range",
                 fontsize=10, color=INK, y=1.03)
    fig.subplots_adjust(right=0.84)
    fig.savefig(out / "fig1_loss_curves.png")
    fig.savefig(out / "fig1_loss_curves.pdf")
    plt.close(fig)


def cell_summary():
    """Every cell: final train loss, in-domain WER, OOD WER, method, depth."""
    rows = []
    for batch in ("b1", "b2", "b3", "b4", "b5"):
        root = Path("outputs/rev", batch)
        if not root.exists():
            continue
        for cdir in sorted(root.iterdir()):
            corpus = cdir.name
            if corpus not in IN_DOMAIN:
                continue
            for celldir in sorted(cdir.iterdir()):
                runs = load_cell(batch, corpus, celldir.name)
                if not runs:
                    continue
                ind = IN_DOMAIN[corpus]
                res = [m["results"] for m, _ in runs]
                if ind not in res[0]:
                    continue
                tl = []
                for m, _ in runs:
                    c = json.load((Path(m["run"]["output_dir"]) / "cost.json").open())
                    v = c.get("hf_train_metrics", {}).get("train_loss")
                    if v is not None:
                        tl.append(v)
                rows.append({
                    "batch": batch, "corpus": corpus, "cell": celldir.name,
                    "method": runs[0][0]["adaptation"]["method"],
                    "n": len(runs),
                    "wer": st.mean([r[ind]["wer_fixed"] * 100 for r in res]),
                    "sd": st.stdev([r[ind]["wer_fixed"] * 100 for r in res]) if len(res) > 1 else 0.0,
                    "ood": st.mean([r["openslr-librispeech-asr-clean"]["wer_fixed"] * 100 for r in res]),
                    "train_loss": st.mean(tl) if tl else None,
                })
    return rows


def baselines():
    out = {}
    for p in Path("outputs/rev/b2").glob("baseline_*/inference_results/results.json"):
        corpus = p.parts[-3].replace("baseline_", "")
        m = json.load(p.open())["metrics"]
        key = IN_DOMAIN.get(corpus)
        if key and key in m:
            out[corpus] = (m[key]["wer"] * 100,
                           m["openslr-librispeech-asr-clean"]["wer"] * 100)
    return out


def fig2_generalization(rows, out):
    """R1-4.1/4.3: lower training loss with worse test WER is the gap, drawn."""
    sub = [r for r in rows if r["corpus"] == "gigaspeech" and r["train_loss"]
           and r["cell"] in ("l4_lora", "l5_lora", "l4_full", "l5_full", "l6_full")]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for r in sub:
        m = METHOD[r["method"]]
        ax.errorbar(r["train_loss"], r["wer"], yerr=r["sd"] if r["sd"] else np.nan, fmt="o",
                    ms=8, mfc=m["c"], mec="white", mew=1.4, ecolor=m["c"],
                    elinewidth=1.4, capsize=3, zorder=3)
        # Left-hand cluster labels to the right, right-hand ones to the left, so
        # neither runs off the axis or into its neighbour.
        left = r["train_loss"] < 0.185
        ax.annotate(r["cell"].replace("_", " ").upper(),
                    xy=(r["train_loss"], r["wer"]),
                    xytext=(9 if left else -9, 0), textcoords="offset points",
                    ha="left" if left else "right", va="center",
                    fontsize=7.5, color=INK2)
    for meth in ("lora", "full"):
        pts = sorted([(r["train_loss"], r["wer"]) for r in sub if r["method"] == meth])
        if len(pts) > 1:
            ax.plot(*zip(*pts), color=METHOD[meth]["c"], ls=METHOD[meth]["ls"],
                    lw=1.8, alpha=0.7, label=METHOD[meth]["label"], zorder=2)
    ax.margins(x=0.16, y=0.10)
    ax.set_xlabel("Final training loss  (lower = fits training data better)")
    ax.set_ylabel("Test WER %  (lower = generalizes better)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    ax.set_title("GigaSpeech: full fine-tuning fits better and generalizes worse",
                 color=INK, loc="left")
    fig.savefig(out / "fig2_generalization.png")
    fig.savefig(out / "fig2_generalization.pdf")
    plt.close(fig)


def fig3_depth_wer(rows, out):
    """R1-2.1/2.3: the depth axis with the seed spread actually shown."""
    base = baselines()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), sharey=False)
    for ax, corpus in zip(axes, ("voxpopuli", "spgispeech", "gigaspeech")):
        for meth in ("lora", "full"):
            pts = []
            for d in DEPTH_ORDER:
                cand = [r for r in rows if r["corpus"] == corpus and r["method"] == meth
                        and r["cell"] in (f"{d}_{meth}",)]
                if cand:
                    pts.append((DEPTH_ORDER.index(d), cand[0]["wer"], cand[0]["sd"], cand[0]["n"]))
            if not pts:
                continue
            m = METHOD[meth]
            xs, ys, es, ns = zip(*pts)
            ax.errorbar(xs, ys, yerr=[e if e else np.nan for e in es], fmt="o-", ms=7,
                        color=m["c"], ls=m["ls"], lw=1.8, mfc=m["c"], mec="white",
                        mew=1.2, ecolor=m["c"], elinewidth=1.4, capsize=3,
                        label=m["label"], zorder=3)
            short = "LoRA" if meth == "lora" else "Full FT"
            ax.annotate(short, xy=(xs[-1], ys[-1]), xytext=(5, 0),
                        textcoords="offset points", color=m["c"], fontsize=7.5, va="center")
        if corpus in base:
            ax.axhline(base[corpus][0], color=MUTED, ls=":", lw=1.4, zorder=1)
            ax.annotate("no adaptation", xy=(0, base[corpus][0]), xytext=(0, 4),
                        textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(DEPTH_ORDER)))
        ax.set_xticklabels([d.upper() for d in DEPTH_ORDER])
        ax.set_title(CORPUS_LABEL[corpus], color=INK, loc="left")
        ax.set_xlabel("Adaptation depth (decoder)")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.margins(x=0.42)
    axes[0].set_ylabel("In-domain WER %")
    axes[0].legend(loc="upper right")
    fig.suptitle("Error bars are the seed SD; dotted rule is the un-adapted model",
                 fontsize=9, color=INK2, y=1.03)
    fig.savefig(out / "fig3_depth_wer.png")
    fig.savefig(out / "fig3_depth_wer.pdf")
    plt.close(fig)


def fig4_ood_tradeoff(rows, out):
    """R1-5.5/R3-10: what in-domain adaptation costs out of domain.

    Every cell is plotted, but only the ones carrying the argument are labelled --
    the L0 anchors (two of which are *worse* in-domain than no adaptation at all) and
    the L5 pair per corpus. Labelling all twenty collides into unreadability and
    breaks the selective-direct-label rule for no gain.
    """
    base = baselines()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labelled = []
    for r in rows:
        if r["corpus"] not in base or r["batch"] in ("b3", "b4"):
            continue
        b_in, b_ood = base[r["corpus"]]
        gain = b_in - r["wer"]           # positive = better in-domain
        cost = r["ood"] - b_ood          # positive = worse out of domain
        m = METHOD.get(r["method"], METHOD["lora"])
        key = (r["cell"].startswith("l0_"), r["cell"] in ("l5_lora", "l5_full"),
               r["cell"].startswith("enc_"))
        notable = key[0] or key[1]
        ax.scatter(gain, cost, s=64 if notable else 34,
                   color=m["c"], edgecolor="white", linewidth=1.2,
                   alpha=1.0 if notable else 0.45, zorder=3 if notable else 2,
                   marker="o" if r["method"] == "lora" else "s")
        if notable:
            tag = f"{CORPUS_LABEL[r['corpus']][:4]} {r['cell'].split('_')[0].upper()}"
            labelled.append((gain, cost, tag, m["c"]))

    # Small uniform offset: a label far from its marker is worse than a slight
    # overlap, because the reader cannot tell which point it belongs to.
    for x, y, tag, colour in labelled:
        ax.annotate(tag, xy=(x, y), xytext=(7, 6), textcoords="offset points",
                    fontsize=7, color=INK2, ha="left", va="bottom")

    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
    ax.margins(x=0.16, y=0.16)
    for meth, mk in (("lora", "o"), ("full", "s")):
        ax.scatter([], [], s=64, color=METHOD[meth]["c"], marker=mk,
                   edgecolor="white", linewidth=1.2, label=METHOD[meth]["label"])
    ax.legend(loc="lower right", borderaxespad=0.8)
    ax.annotate("worse in-domain than\nno adaptation at all", xy=(0.015, 0.985),
                xycoords="axes fraction", fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("In-domain WER gained vs no adaptation  (pp)")
    ax.set_ylabel("LibriSpeech-clean WER lost  (pp)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Adaptation buys in-domain accuracy and spends it out of domain",
                 color=INK, loc="left", fontsize=9.5, pad=10)
    fig.savefig(out / "fig4_ood_tradeoff.png")
    fig.savefig(out / "fig4_ood_tradeoff.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    rows = cell_summary()
    fig1_loss_curves(out)
    fig2_generalization(rows, out)
    fig3_depth_wer(rows, out)
    fig4_ood_tradeoff(rows, out)
    print(f"wrote 4 figures (png + pdf) to {out}/ from {len(rows)} cells")


if __name__ == "__main__":
    main()
