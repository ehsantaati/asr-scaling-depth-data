#!/usr/bin/env python
"""Action Z3: diagnostic figures from the rerun logs.

R1-4.2 asks for training and validation loss curves for representative shallow and
deep configurations; R1-4.3 for convergence evidence -- gradient behaviour,
generalization gaps, checkpoint evolution. R1-4.1 says the instability and memorization
explanations are "plausible but not directly demonstrated". These figures are the
demonstration.

Four figures, each answering a specific comment:

  fig1_loss_curves      train + held-out curves, 2x2: {fixed budget,
                        data-limited} x {training, held-out}               R1-4.2,
                                                                          R1-4.1, R1-4.3
  fig2_generalization   final train loss against test WER                R1-4.1, R1-4.3
  fig3_depth_wer        WER against depth, per corpus, with seed SD      R1-2.1, R1-2.3
  fig4_ood_tradeoff     in-domain gain against out-of-domain cost        R1-5.5, R3-10

Design decisions that are not arbitrary:

  * Bands are min-max across seeds, not +/- SD. With n=2 or 3 an SD band implies a
    distributional claim the sample cannot support; a min-max envelope says exactly
    what was observed and nothing more.
  * On the curve figure each seed is smoothed over a short step window *before* the
    envelope is taken. Mini-batch logging noise at the plateau (~0.010) is larger
    than the seed spread there (~0.004), so a raw envelope would be a picture of
    batch variance wearing a seed-spread label.
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


def _smooth(y, n):
    """Centred moving average over n logging points, reflect-padded at the ends.

    Logging noise on the training loss is ~0.010 at the plateau while the *seed*
    spread there is ~0.004, so an envelope taken over raw points would be a picture
    of mini-batch variance wearing a seed-spread label. Each seed is smoothed first
    and the envelope is taken across the smoothed curves, so the band means what the
    caption says it means.
    """
    if n < 3 or len(y) < n:
        return y
    pad = n // 2
    padded = np.concatenate([y[pad:0:-1], y, y[-2:-pad - 2:-1]])
    return np.convolve(padded, np.ones(n) / n, mode="valid")[:len(y)]


def series(runs, kind, key, smooth_frac=0.0):
    """Common step grid across seeds, plus per-seed values interpolated onto it."""
    curves = []
    for _, rows in runs:
        pts = [(r["step"], r[key]) for r in rows if r.get("kind") == kind and r.get(key) is not None]
        if pts:
            curves.append(np.array(pts, dtype=float))
    if not curves:
        return None, None
    if smooth_frac:
        n = max(3, int(smooth_frac * len(curves[0]))) | 1
        curves = [np.column_stack([c[:, 0], _smooth(c[:, 1], n)]) for c in curves]
    lo = max(c[0, 0] for c in curves)
    hi = min(c[-1, 0] for c in curves)
    grid = np.linspace(lo, hi, 300)
    vals = np.vstack([np.interp(grid, c[:, 0], c[:, 1]) for c in curves])
    return grid, vals


def band(ax, x, vals, colour, ls, label, lw=1.8, z=3):
    """Mean line with a min-max envelope across seeds."""
    if len(vals) > 1:
        ax.fill_between(x, vals.min(axis=0), vals.max(axis=0), color=colour,
                        alpha=0.16, lw=0, zorder=z - 1)
    ax.plot(x, vals.mean(axis=0), color=colour, ls=ls, lw=lw, label=label,
            solid_capstyle="round", zorder=z)


# ---------------------------------------------------------------------------
# fig1: the R1-4.2 panel grid.
#
# R1-4.2 asks for "training and validation loss curves for representative shallow
# and deep configurations under both data-limited and fixed-budget regimes". That
# is four things crossed -- {train, held-out} x {shallow, deep} x {LoRA, full FT}
# x {fixed budget, data-limited} -- so the figure is a 2x2: regime on rows,
# quantity on columns.
#
# Row 1 (fixed budget, GigaSpeech) carries R1-4.1 and R1-4.3 as well: deep full FT
# sits *below* LoRA on training loss and *above* it on held-out loss and final WER,
# which is the generalization gap drawn rather than asserted, and its held-out
# curve turns upward while its training loss is still falling, which is the
# "unstable / memorizing optimization" claim made visible.
#
# Row 2 (data-limited, VoxPopuli @10%) shows the same configuration under a tenth
# of the budget, against its own full-data run as the reference. Both rows share a
# per-row x-axis so the left and right panels of a row are read off the same step.
# ---------------------------------------------------------------------------

FIG1_ROWS = [
    {
        "regime": "Fixed-budget regime",
        "corpus": "gigaspeech",
        "denom": 19898,
        "max_steps": 42504,
        "smooth_train": 0.025,
        "smooth_eval": 0.03,
        # Zoom the inset from the best checkpoint onward: everything to the right of
        # this line is budget spent after held-out performance stopped improving.
        "zoom_from": 4250,
        "marker": {"step": 4250, "cell": "l5_full", "title": "best checkpoint",
                   "seed_agreement": True},
        "note_train": "still falling",
        "note_held": "no further gain",
        "curves": [
            # L0 is an order of magnitude above the others and is kept out of the
            # zoom: including it sets the inset's y-range and squashes the L5
            # LoRA-vs-full separation that the inset exists to show.
            {"batch": "b2", "cell": "l0_lora", "label": "L0 LoRA", "c": AQUA, "ls": ":",
             "in_zoom": False},
            {"batch": "b1", "cell": "l5_lora", "label": "L5 LoRA", "c": BLUE, "ls": "-"},
            {"batch": "b1", "cell": "l5_full", "label": "L5 full FT", "c": ORANGE, "ls": "--"},
        ],
    },
    {
        "regime": "Data-limited regime",
        "corpus": "voxpopuli",
        "denom": 1830,
        "max_steps": 11405,
        "smooth_train": 0.05,
        "smooth_eval": 0.0,
        "zoom_from": 500,
        "marker": {"step": 1140, "cell": "l5_lora_frac10", "title": "10% data budget ends",
                   "seed_agreement": False},
        "note_train": "10% run ends",
        "note_held": "10% run ends",
        # Both series are LoRA, so colour cannot separate them here without breaking
        # the colour==method rule the other figures use. The varying factor is the
        # data fraction, so it gets the line style, and the full-data reference is
        # drawn in the same neutral grey the other figures use for reference series.
        "curves": [
            {"batch": "b1", "cell": "l5_lora_frac10", "label": "L5 LoRA, 10% data", "c": BLUE, "ls": "-", "z": 5},
            {"batch": "b1", "cell": "l5_lora", "label": "L5 LoRA, 100% data (reference)", "c": INK2, "ls": "-.", "z": 2},
        ],
    },
]


def _held_out(runs, smooth_frac):
    """Held-out trajectory: eval loss if it was logged, else checkpoint WER.

    Returns (x, vals, source, ylabel). The fallback reads the B8 best-vs-final
    records, which hold WER at the selected checkpoint and at the final step -- two
    points, not a trajectory, so it is drawn as markers and reported as such rather
    than passed off as a curve.
    """
    x, vals = series(runs, "eval", "eval_loss", smooth_frac=smooth_frac)
    if x is not None:
        n_val = runs[0][0]["provenance"]["config_resolved"]["val_dataset_args"]["max_samples"]
        return x, vals, "eval_loss", f"Held-out loss, log scale\n(validation split, {n_val} utts)"
    pts = []
    for m, _ in runs:
        p = Path(m["run"]["output_dir"]) / "best_vs_final.json"
        if not p.exists():
            continue
        b = json.load(p.open())
        pts.append(np.array([[b["best_step"], b["wer_best"] * 100],
                             [m["schedule"]["steps_completed"], b["wer_final"] * 100]]))
    if not pts:
        return None, None, "missing", None
    grid = np.linspace(max(p[0, 0] for p in pts), min(p[-1, 0] for p in pts), 2)
    vals = np.vstack([np.interp(grid, p[:, 0], p[:, 1]) for p in pts])
    return grid, vals, "checkpoint_wer", "Held-out WER %, log scale\n(B8 checkpoint records)"


def _plateau_inset(ax, drawn, x_from, x_to, note, log_y=False):
    """A linear-scale zoom of the plateau, parked in the empty upper-right corner.

    The log axis the whole trajectory needs makes the plateau unreadable: the first
    ~500 steps take the loss from ~5.5 to ~0.2, and the separation that carries
    R1-4.1 -- full FT below LoRA on training loss, above it on held-out loss -- is a
    few percent on top of that. Zooming it is the only way both facts fit in one
    panel without a second row of axes.
    """
    axin = ax.inset_axes([0.44, 0.36, 0.54, 0.40])
    lo, hi = np.inf, -np.inf
    for x, vals, colour, ls, z in drawn:
        keep = (x >= x_from) & (x <= x_to)
        if keep.sum() < 2:
            continue
        band(axin, x[keep], vals[:, keep], colour, ls, None, lw=1.3, z=z)
        lo, hi = min(lo, vals[:, keep].min()), max(hi, vals[:, keep].max())
    if not np.isfinite(lo):
        axin.remove()
        return None
    pad = 0.10 * (hi - lo) or 0.01
    axin.set_ylim(lo - pad, hi + pad)
    axin.set_xlim(x_from, x_to)
    if log_y:
        axin.set_yscale("log")
    axin.tick_params(labelsize=6, length=2, pad=1.5)
    axin.ticklabel_format(axis="x", style="sci", scilimits=(3, 3), useMathText=True)
    axin.xaxis.get_offset_text().set_fontsize(6)
    for sp in ("top", "right"):
        axin.spines[sp].set_visible(False)
    axin.set_facecolor("white")
    axin.grid(axis="y")
    axin.set_axisbelow(True)
    # One right-aligned line: the inset's own x-axis already gives the step range,
    # and a left-aligned two-line title collides with the marker tag.
    axin.set_title(f"zoom, linear scale · {note}",
                   fontsize=6.2, color=INK2, loc="right", pad=3)
    return axin


def _best_step_agreement(mk, corpus):
    """How many seeds of the marked cell actually selected the marked checkpoint.

    Read from the B8 records rather than hardcoded, so the label stays true if the
    cell is ever reseeded.
    """
    hit = tot = 0
    for spec_batch in ("b1", "b2", "b3", "b4", "b5"):
        d = Path("outputs/rev", spec_batch, corpus, mk["cell"])
        if not d.exists():
            continue
        for bp in sorted(d.glob("*/best_vs_final.json")):
            tot += 1
            hit += int(json.load(bp.open())["best_step"] == mk["step"])
    return hit, tot


def fig1_loss_curves(out):
    """R1-4.2 in full: train and held-out curves, both regimes, both methods.

    Layout is a 2x2 -- regime on rows, quantity on columns -- because R1-4.2 asks
    for "training and validation loss curves for representative shallow and deep
    configurations under both data-limited and fixed-budget regimes", which is
    exactly that cross.

    Row 1 (fixed budget, GigaSpeech) also carries R1-4.1 and R1-4.3: deep full FT
    sits *below* LoRA on training loss and *above* it on held-out loss and final
    WER, so the generalization gap is drawn rather than asserted; and its held-out
    curve turns upward at ~10% of the budget while its training loss is still
    falling, which is the "unstable / memorizing optimization" claim made visible.

    Row 2 (data-limited, VoxPopuli at 10%) runs the same configuration on a tenth
    of the data against its own full-data run as the reference.

    Returns the provenance records so main() can print what each curve was built
    from -- run directories, n, and which held-out quantity the right column used.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
    report = []

    for ri, row in enumerate(FIG1_ROWS):
        ax_tr, ax_ho = axes[ri, 0], axes[ri, 1]
        loaded = []
        for spec in row["curves"]:
            runs = load_cell(spec["batch"], row["corpus"], spec["cell"])
            if runs:
                loaded.append((spec, runs))

        ind = IN_DOMAIN[row["corpus"]]
        ho_sources, ho_label = set(), None
        ends, drawn_tr, drawn_ho = [], [], []
        train_lo = None

        for spec, runs in loaded:
            n = len(runs)
            # A missing band on an n=1 series looks exactly like a very tight seed
            # spread. Name it in the legend rather than leaving it to be inferred
            # from the absence of shading.
            lbl = f"{spec['label']}  (n=1, single seed \u2014 no band)" if n == 1 \
                else f"{spec['label']}  (n={n})"

            z = spec.get("z", 3)
            x, vals = series(runs, "train", "loss", smooth_frac=row["smooth_train"])
            if x is not None:
                band(ax_tr, x, vals, spec["c"], spec["ls"], lbl, z=z)
                train_lo = min(train_lo, vals.min()) if train_lo else vals.min()
                if spec.get("in_zoom", True):
                    drawn_tr.append((x, vals, spec["c"], spec["ls"], z))

            hx, hvals, src, ylab = _held_out(runs, row["smooth_eval"])
            ho_sources.add(src)
            wer = st.mean([m["results"][ind]["wer_fixed"] * 100 for m, _ in runs])
            if hx is not None:
                if ylab:
                    ho_label = ylab
                if src == "checkpoint_wer":
                    ax_ho.plot(hx, hvals.mean(axis=0), color=spec["c"], ls=spec["ls"],
                               lw=1.8, marker="o", ms=5, mec="white", mew=1.2, label=lbl)
                else:
                    band(ax_ho, hx, hvals, spec["c"], spec["ls"], lbl, z=z)
                    if spec.get("in_zoom", True):
                        drawn_ho.append((hx, hvals, spec["c"], spec["ls"], z))
                ends.append((hx[-1], float(hvals.mean(axis=0)[-1]), wer, spec))

            report.append({
                "regime": row["regime"], "corpus": row["corpus"], "cell": spec["cell"],
                "label": spec["label"], "n": n, "held_out": src, "wer": wer,
                "runs": [m["run"]["output_dir"] for m, _ in runs],
            })

        if len(ho_sources - {"missing"}) > 1:
            print(f"  WARNING: {row['regime']} mixes held-out sources: {sorted(ho_sources)}")

        ax_tr.set_yscale("log")
        ax_tr.set_ylabel("Training loss, log scale")
        # Open white space under the plateau so the row legend has somewhere to sit
        # that is not on top of a curve. Costs nothing on a log axis.
        if train_lo is not None:
            ax_tr.set_ylim(bottom=train_lo / 3.0)
        ax_ho.set_yscale("log")
        ax_ho.set_ylabel(ho_label or "Held-out")
        for ax in (ax_tr, ax_ho):
            ax.set_xlim(0, row["max_steps"])
            ax.set_xlabel("Optimization step")
            ax.grid(axis="y")
            ax.set_axisbelow(True)

        # Plateau zooms. These are added before the annotations so the direct labels
        # and the best-checkpoint tag can be placed clear of them.
        ins_tr = _plateau_inset(ax_tr, drawn_tr, row["zoom_from"], row["max_steps"],
                                row["note_train"])
        ins_ho = _plateau_inset(ax_ho, drawn_ho, row["zoom_from"], row["max_steps"],
                                row["note_held"])

        # Final in-domain WER, printed at the end of each held-out curve. Placed in
        # the right margin with a leader and staggered in value order: on GigaSpeech
        # the LoRA and full-FT curves end ~0.03 apart on a log axis and inline
        # labels would overlap.
        ends.sort(key=lambda e: e[1])
        for i, (ex, ey, wer, spec) in enumerate(ends):
            dy = (i - (len(ends) - 1) / 2) * 15
            ax_ho.annotate(f"{spec['label']}\nfinal WER {wer:.3f}",
                           xy=(ex, ey), xycoords="data",
                           xytext=(16, dy), textcoords="offset points",
                           fontsize=7, color=spec["c"], va="center", ha="left",
                           annotation_clip=False,
                           arrowprops=dict(arrowstyle="-", color=spec["c"],
                                           lw=0.7, alpha=0.6, shrinkA=0, shrinkB=2))

        # The best-checkpoint marker. Deep full FT on GigaSpeech peaks at 10% of the
        # budget: to the right of this line the training loss keeps falling while
        # held-out performance has already turned. Drawn in both panels of the row,
        # and in both insets, so the divergence is read off one step value.
        mk = row["marker"]
        if mk:
            mcol = next((c["c"] for c in row["curves"] if c["cell"] == mk["cell"]), MUTED)
            tag = [mk["title"], f"step {mk['step']:,} of {row['max_steps']:,}"]
            if mk.get("seed_agreement"):
                # Say "k of n seeds" rather than "the best checkpoint": on GigaSpeech
                # deep FT the val-loss-selected checkpoint is step 4250 in 2 of 3
                # seeds and step 34000 in the third. This figure goes into a
                # rebuttal; an unqualified claim there is a gift to a reviewer.
                hit, tot = _best_step_agreement(mk, row["corpus"])
                tag.append(f"({hit} of {tot} seeds)")
            for ax in (ax_tr, ax_ho):
                ax.axvline(mk["step"], color=mcol, ls=(0, (3, 3)), lw=1.1, alpha=0.9, zorder=1)
                # Upper-left band: left of the inset, above every curve at this step.
                ax.annotate("\n".join(tag), xy=(mk["step"], 0.97),
                            xycoords=("data", "axes fraction"),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=6.8, color=mcol, va="top", ha="left", linespacing=1.35)
            for axin in (ins_tr, ins_ho):
                if axin is not None:
                    axin.axvline(mk["step"], color=mcol, ls=(0, (3, 3)), lw=1.0, alpha=0.9, zorder=1)

        head = (f"{row['regime']} · {CORPUS_LABEL[row['corpus']]}\n"
                f"in-domain test set, n = {row['denom']:,} utterances")
        ax_tr.set_title(f"{head}", color=INK, loc="left", fontsize=8.5, pad=6)
        ax_ho.set_title(f"{head}", color=INK, loc="left", fontsize=8.5, pad=6)
        # One legend per row, in the left panel: the same series appear in both
        # columns, so repeating it would only cost space.
        ax_tr.legend(loc="lower left", fontsize=7.5, borderaxespad=0.6,
                     handlelength=2.6, labelspacing=0.35)

    fig.suptitle("Training and held-out trajectories under both regimes "
                 "(left: training loss · right: held-out performance)",
                 fontsize=11, color=INK, y=0.985)
    caption = [
        "Shaded bands span the observed range across SEEDS (min–max), not bootstrap "
        "confidence intervals; n per condition is given in the row legend.",
        "Per-seed curves are smoothed over a short step window before the envelope is "
        "taken, so the band shows seed spread rather than mini-batch logging noise.",
    ]
    singles = [r["label"] for r in report if r["n"] == 1]
    if singles:
        caption.append(
            f"{', '.join(singles)} ran at a single seed (42) and carries NO band: its "
            "line is one run, and its seed spread is unmeasured, not small.")
    fig.text(0.5, 0.005, "\n".join(caption),
             ha="center", va="bottom", fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.075, right=0.79, top=0.885, bottom=0.11,
                        hspace=0.52, wspace=0.60)
    fig.savefig(out / "fig1_loss_curves.png")
    fig.savefig(out / "fig1_loss_curves.pdf")
    plt.close(fig)
    return report


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


HELD_OUT_DESC = {
    "eval_loss": "eval loss (logged in metrics.jsonl)",
    "checkpoint_wer": "checkpoint WER (B8 best_vs_final.json)",
    "missing": "NEITHER -- no eval loss and no B8 record",
}


def print_fig1_provenance(report):
    """What every fig1 curve was built from. R1-4.2 goes into a rebuttal, so the
    source runs, the seed count and the held-out quantity have to be checkable
    without re-reading the script."""
    print()
    print("fig1_loss_curves -- provenance per curve")
    print("=" * 78)
    for r in report:
        print(f"{r['regime']} / {CORPUS_LABEL[r['corpus']]} / {r['label']}")
        print(f"    cell            {r['cell']}")
        print(f"    n (seeds)       {r['n']}")
        print(f"    right panel     {HELD_OUT_DESC[r['held_out']]}")
        print(f"    final WER %     {r['wer']:.3f}")
        print(f"    run dirs        ({len(r['runs'])})")
        for d in r["runs"]:
            print(f"        {d}")
    missing = [r for r in report if r["held_out"] == "missing"]
    fallback = [r for r in report if r["held_out"] == "checkpoint_wer"]
    print("-" * 78)
    if fallback:
        print("HELD-OUT FALLBACK -- eval loss was NOT logged, checkpoint WER used instead:")
        for r in fallback:
            print(f"    {r['regime']} / {r['cell']}")
    if missing:
        print("HELD-OUT MISSING -- no eval loss and no B8 record, curve omitted:")
        for r in missing:
            print(f"    {r['regime']} / {r['cell']}")
    if not fallback and not missing:
        print("Held-out source: eval loss was logged for every condition; no substitutions.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    rows = cell_summary()
    fig1_report = fig1_loss_curves(out)
    fig2_generalization(rows, out)
    fig3_depth_wer(rows, out)
    fig4_ood_tradeoff(rows, out)
    print(f"wrote 4 figures (png + pdf) to {out}/ from {len(rows)} cells")
    print_fig1_provenance(fig1_report)


if __name__ == "__main__":
    main()
