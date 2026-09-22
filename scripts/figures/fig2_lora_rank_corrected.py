"""Regenerate the LoRA-rank-vs-WER figure with corrected vanilla baselines.

The original figure's vanilla/unadapted baseline WER values were produced by a
now-fixed evaluation bug (empty EnglishTextNormalizer spelling map + no forced
language + a dropped-utterance issue on GigaSpeech; see CLAUDE.md invariants 7-8
and commit 8196d9d). This script keeps the original LoRA-rank sweep (depth L0,
ranks 8-512, 5 seeds/rank, from outputs/<dataset>/002/002x/) but replaces the
vanilla-baseline and Full-FT reference lines with the corrected `wer_fixed`
values from the rerun campaign (outputs/rev/).

The Full-FT reference is depth-matched to the LoRA sweep: L0 full fine-tuning
trains only `proj_out` + the final decoder layer norm (see
configs/rev/b13/gigaspeech/l0_full.yaml), the same depth as the LoRA rank sweep.

Data sources (see script output / task report for exact values):
  - LoRA sweep:  outputs/{gigaspeech,spgispeech_2,voxpopuli}/002/00{20..26}/
                 seed_*_frac_1.0_subset_0[_old]/results.json  (field: metrics.wer,
                 or metrics.voxpopuli-en.wer for VoxPopuli)
  - Baseline:    outputs/rev/b2/baseline_{gigaspeech,spgispeech,voxpopuli}/
                 inference_results/results.json  (field: wer_fixed)
  - Full-FT:     outputs/rev/b13/{gigaspeech,spgispeech}/l0_full/ and
                 outputs/rev/b3/voxpopuli/l0_full/
                 s{42,123,1234}_frac_1_subset_0/run_manifest.json (field: wer_fixed)
"""

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures"

RANKS = [8, 16, 32, 64, 128, 256, 512]
RANK_TO_EXP = {8: "0020", 16: "0021", 32: "0022", 64: "0023", 128: "0024", 256: "0025", 512: "0026"}
SEED_DIR_RE = re.compile(r"seed_(\d+)_frac_1\.0_subset_0(_old)?$")

# (old-pipeline output dir name, panel title, baseline dir suffix, run_manifest wer key)
DATASETS = [
    ("spgispeech_2", "SPGISpeech 2.0", "spgispeech", "spgispeech_2"),
    ("voxpopuli", "VoxPopuli", "voxpopuli", "voxpopuli-en"),
    ("gigaspeech", "GigaSpeech", "gigaspeech", "speechcolab-gigaspeech-m"),
]


def _extract_wer(metrics: dict, out_dir_name: str) -> float | None:
    """Mirror the field lookup used by the original notebook's rank-sweep cells."""
    wer = metrics.get("wer")
    if wer is not None:
        return wer
    if out_dir_name == "voxpopuli":
        return metrics.get("voxpopuli-en", {}).get("wer")
    lookup = "speechcolab-gigaspeech-m" if out_dir_name == "gigaspeech" else out_dir_name
    return metrics.get(lookup, {}).get("wer")


def load_lora_sweep(out_dir_name: str) -> dict[int, tuple[float, float, int]]:
    """Per-rank (mean_wer_pct, sd_wer_pct, n_seeds) for the L0 rank-sensitivity sweep."""
    sweep = {}
    for rank, exp_id in RANK_TO_EXP.items():
        base = REPO_ROOT / "outputs" / out_dir_name / "002" / exp_id
        wers = []
        for d in sorted(base.iterdir()) if base.exists() else []:
            if not d.is_dir() or not SEED_DIR_RE.match(d.name):
                continue
            results_file = d / "results.json"
            if not results_file.exists():
                continue
            data = json.loads(results_file.read_text())
            wer = _extract_wer(data.get("metrics", {}), out_dir_name)
            if wer is not None:
                wers.append(wer * 100)
        if not wers:
            continue
        mean = statistics.mean(wers)
        sd = statistics.stdev(wers) if len(wers) > 1 else 0.0
        sweep[rank] = (mean, sd, len(wers))
    return sweep


def load_baseline_wer(baseline_suffix: str, wer_key: str) -> tuple[float, int]:
    """Corrected vanilla-baseline WER (%) and utterance denominator."""
    path = (
        REPO_ROOT
        / "outputs"
        / "rev"
        / "b2"
        / f"baseline_{baseline_suffix}"
        / "inference_results"
        / "results.json"
    )
    data = json.loads(path.read_text())
    entry = data["metrics"][wer_key]
    return entry["wer_fixed"] * 100, entry["denominator"]


# L0 full-FT (proj_out + final decoder layer norm) lives in different batches
# per dataset: b13 for gigaspeech/spgispeech, b3 for voxpopuli.
FULL_FT_L0_BATCH = {"gigaspeech": "b13", "spgispeech": "b13", "voxpopuli": "b3"}


def load_full_ft_wer(baseline_suffix: str, wer_key: str) -> tuple[float, float, int]:
    """Full-FT reference mean/SD/n WER (%) at depth L0 (proj_out only), seeds 42/123/1234."""
    batch = FULL_FT_L0_BATCH[baseline_suffix]
    wers = []
    for seed in (42, 123, 1234):
        path = (
            REPO_ROOT
            / "outputs"
            / "rev"
            / batch
            / baseline_suffix
            / "l0_full"
            / f"s{seed}_frac_1_subset_0"
            / "run_manifest.json"
        )
        data = json.loads(path.read_text())
        assert data["status"] == "complete", f"{path} is not complete"
        wers.append(data["results"][wer_key]["wer_fixed"] * 100)
    return statistics.mean(wers), statistics.stdev(wers), len(wers)


def plot_panel(ax, out_dir_name, title, baseline_suffix, wer_key, color, legend_loc="best"):
    sweep = load_lora_sweep(out_dir_name)
    baseline_wer, baseline_n = load_baseline_wer(baseline_suffix, wer_key)
    full_ft_mean, full_ft_sd, full_ft_n = load_full_ft_wer(baseline_suffix, wer_key)

    ranks_present = [r for r in RANKS if r in sweep]
    means = [sweep[r][0] for r in ranks_present]
    sds = [sweep[r][1] for r in ranks_present]
    x = list(range(len(ranks_present)))

    ax.plot(x, means, marker="o", color=color, linewidth=2, markersize=6, label="LoRA (Mean WER)", zorder=3)
    lower = [m - s for m, s in zip(means, sds)]
    upper = [m + s for m, s in zip(means, sds)]
    ax.fill_between(x, lower, upper, color=color, alpha=0.2, zorder=2)

    ax.axhline(baseline_wer, color="red", linestyle="--", alpha=0.8, label=f"Baseline: {baseline_wer:.2f}%", zorder=1)
    ax.axhline(full_ft_mean, color="black", linestyle=":", alpha=0.8, label=f"Full FT: {full_ft_mean:.2f}%", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks_present])
    ax.set_xlabel("LoRA Rank (r)")
    ax.set_ylabel("WER (%)")
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=9, frameon=True)

    all_vals = means + lower + upper + [baseline_wer, full_ft_mean]
    span = max(all_vals) - min(all_vals)
    pad = max(span * 0.12, 0.3)
    ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)

    return {
        "ranks": ranks_present,
        "means": means,
        "sds": sds,
        "n_seeds": [sweep[r][2] for r in ranks_present],
        "baseline_wer": baseline_wer,
        "baseline_n": baseline_n,
        "full_ft_mean": full_ft_mean,
        "full_ft_sd": full_ft_sd,
        "full_ft_n": full_ft_n,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(11, 9))
    fig.suptitle("Data–Capacity Trade-offs in Whisper", fontsize=14)

    gs = fig.add_gridspec(2, 4)
    ax_spgi = fig.add_subplot(gs[0, 0:2])
    ax_vox = fig.add_subplot(gs[0, 2:4])
    ax_giga = fig.add_subplot(gs[1, 1:3])

    colors = {"spgispeech_2": "royalblue", "voxpopuli": "darkgreen", "gigaspeech": "darkorange"}

    summary = {}
    summary["SPGISpeech 2.0"] = plot_panel(
        ax_spgi, "spgispeech_2", "(a) SPGISpeech 2.0", "spgispeech", "spgispeech_2", colors["spgispeech_2"]
    )
    summary["VoxPopuli"] = plot_panel(
        ax_vox, "voxpopuli", "(b) VoxPopuli", "voxpopuli", "voxpopuli-en", colors["voxpopuli"]
    )
    summary["GigaSpeech"] = plot_panel(
        ax_giga,
        "gigaspeech",
        "(c) GigaSpeech",
        "gigaspeech",
        "speechcolab-gigaspeech-m",
        colors["gigaspeech"],
        legend_loc="upper center",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    pdf_path = OUT_DIR / "fig2_lora_rank_corrected.pdf"
    png_path = OUT_DIR / "fig2_lora_rank_corrected.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print()
    for name, s in summary.items():
        print(f"=== {name} ===")
        print(f"  Corrected baseline WER: {s['baseline_wer']:.4f}% (n={s['baseline_n']})")
        print(f"  Full-FT (L0, proj_out) WER: {s['full_ft_mean']:.4f}% +/- {s['full_ft_sd']:.4f} (n={s['full_ft_n']})")
        for r, m, sd, n in zip(s["ranks"], s["means"], s["sds"], s["n_seeds"]):
            print(f"  r={r}: mean={m:.4f}% sd={sd:.4f} n={n}")


if __name__ == "__main__":
    main()
