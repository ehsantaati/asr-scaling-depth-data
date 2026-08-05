#!/usr/bin/env python
"""Generate B1 configs and job-list lines from an explicit cell table.

Two reasons this is generated rather than hand-written:

  * The target-module lists run to 240+ entries at L5/L6. A typo in one silently
    adapts nothing (LoRA) or nothing-in-that-layer (full FT), and the run still
    "succeeds" -- exactly the class of error the campaign cannot afford.
  * The layer-wise definitions must be identical across datasets and methods, or the
    depth axis is not comparable. Deriving them from one table guarantees that.

Layer definitions follow exps/ToDo.md:
    L0  proj_out
    L1  + output-side norms                (LoRA cannot wrap LayerNorm -- skipped)
    L2  + last 1 decoder block
    L3  + last 2 decoder blocks
    L4  + upper half of decoder blocks     (12-23)
    L5  + all decoder blocks               (0-23)
    L6  + embed_tokens, embed_positions

Usage:
    python scripts/gen_campaign_configs.py --phase voxpopuli --out configs/rev/b1
    python scripts/gen_campaign_configs.py --phase voxpopuli --jobs        # print job lines
"""

import argparse
from pathlib import Path

N_LAYERS = 24
LORA_MODS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
    "encoder_attn.q_proj", "encoder_attn.k_proj", "encoder_attn.v_proj", "encoder_attn.out_proj",
    "fc1", "fc2",
]

# Decoder blocks adapted at each depth, deepest-first (output side first).
LAYER_SETS = {
    "L0": [],
    "L2": list(range(23, 22, -1)),
    "L3": list(range(23, 21, -1)),
    "L4": list(range(23, 11, -1)),
    "L5": list(range(23, -1, -1)),
    "L6": list(range(23, -1, -1)),
}

DATASETS = {
    "voxpopuli":   {"name": "voxpopuli-en",            "declared_train": 182482},
    "spgispeech":  {"name": "spgispeech_2",            "declared_train": 512724},
    "gigaspeech":  {"name": "speechcolab-gigaspeech-m", "declared_train": 680072},
}

# --- the cell table -------------------------------------------------------------
# Seed counts are tiered by what is at stake, not applied flat:
#   tier 1 = contested crossovers   LoRA 5 seeds / FT 3
#   tier 2 = supporting cells       LoRA 3 seeds / FT 2
# Tier 1 is where claims turn on 0.3-0.7 WER (SPGISpeech L5 4.19 vs 4.86;
# GigaSpeech 9.74 -> 10.25 at L4->L5 against LoRA 9.43 at L6); a 3-seed SD cannot
# settle those. Tier 2 claims are instability magnitudes of 1-2 WER points and
# plateau shapes, which 3 seeds already settle.
SEEDS = {1: {"lora": [42, 123, 1234, 12345, 123456], "full": [42, 123, 1234]},
         2: {"lora": [42, 123, 1234],                "full": [42, 123]}}

# Methods are per-cell, not universal:
#   * The data-scaling analysis (manuscript sec 4.3, Figures 4-8) is LoRA-only by
#     design -- the instability numbers being defended (8.21, 8.45 at 10%) are LoRA
#     numbers. Adding FT seeds there would be new experiments answering a question
#     no reviewer asked.
#   * SPGISpeech's FT-vs-LoRA claim lives at L5 (tier 1). L4/L6 on SPGISpeech carry
#     no contested crossover, so they are LoRA-only too.
CELLS = [
    # (dataset,     depth, fraction, tier, methods)
    ("gigaspeech",  "L4", 1.0, 1, ("lora", "full")),
    ("gigaspeech",  "L5", 1.0, 1, ("lora", "full")),
    ("gigaspeech",  "L6", 1.0, 1, ("lora", "full")),
    ("spgispeech",  "L5", 1.0, 1, ("lora", "full")),
    ("spgispeech",  "L4", 1.0, 2, ("lora",)),
    ("spgispeech",  "L6", 1.0, 2, ("lora",)),
    ("voxpopuli",   "L4", 1.0, 2, ("lora", "full")),
    ("voxpopuli",   "L5", 1.0, 2, ("lora", "full")),
    ("voxpopuli",   "L6", 1.0, 2, ("lora", "full")),
    # Data-scale cells carrying claims: L6 @ 10% (SPGISpeech/VoxPopuli instability)
    # and L5 @ 10% (Figure 8 regime comparison, whose crossover rests on 0.30 WER on
    # VoxPopuli -- squarely in the territory the reseeding campaign exists to protect).
    ("spgispeech",  "L6", 0.1, 2, ("lora",)),
    ("voxpopuli",   "L6", 0.1, 2, ("lora",)),
    ("voxpopuli",   "L5", 0.1, 2, ("lora",)),
]


def full_ft_targets(depth):
    t = ["proj_out", "model.decoder.layer_norm"]
    if depth == "L6":
        t += ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
    t += [f"model.decoder.layers.{i}" for i in LAYER_SETS[depth]]
    return t


def lora_targets(depth):
    # LoRA cannot wrap LayerNorm or the embeddings, so those are absent by
    # construction. This asymmetry with full FT is real and is stated in W1.
    t = ["proj_out"]
    t += [f"model.decoder.layers.{i}.{m}" for i in LAYER_SETS[depth] for m in LORA_MODS]
    return t


def init_blocks(depth):
    layers = LAYER_SETS[depth]
    if not layers:
        return [["proj_out"]]
    return [["proj_out", f"layers.{layers[0]}"]] + [[f"layers.{i}"] for i in layers[1:]]


def render(dataset, depth, fraction, method, out_dir):
    ds = DATASETS[dataset]
    name = ds["name"]
    is_lora = method == "lora"
    tag = f"{depth.lower()}_{method}" + ("" if fraction == 1.0 else f"_frac{int(fraction*100)}")
    # fraction 1.0 -> fixed-budget (num_epochs 0, steps from the FULL dataset);
    # fraction < 1 -> the data-limited cells cited in the regime comparison.
    num_epochs = 0 if fraction == 1.0 else 1

    L = [f"""# B1 {dataset} {depth} {method.upper()} @ {fraction:g} data -- GENERATED by
# scripts/gen_campaign_configs.py; edit the cell table there, not this file.
model_id: "openai/whisper-medium"
language: "en"
task: "transcribe"

train_sets:
  - name: "{name}"
train_dataset_args:
  max_audio_duration_secs: 30

val_sets:
  - name: "{name}"
val_dataset_args:
  max_audio_duration_secs: 30
  max_samples: 256          # fixed deterministic probe; keeps eval <6% of run time

eval_sets:
  - name: "{name}"
eval_dataset_args:
  max_audio_duration_secs: 30
  max_samples: -1

ood_eval_sets:
  - name: "openslr-librispeech-asr-clean"
  - name: "openslr-librispeech-asr-other"
ood_eval_dataset_args:
  max_audio_duration_secs: 30
  max_samples: -1

data_fractions: [{fraction:g}]
num_subsets: 1
"""]

    if is_lora:
        L.append("\nlora_config:\n  r: 64\n  lora_alpha: 128\n  lora_dropout: 0.1\n  init_blocks:\n")
        for b in init_blocks(depth):
            L.append("    - [" + ", ".join(f'"{x}"' for x in b) + "]\n")
        targets = lora_targets(depth)
    else:
        targets = full_ft_targets(depth)

    L.append("\ntarget_modules:\n")
    L += [f'  - "{t}"\n' for t in targets]

    L.append(f"""
output_dir: "outputs/rev/b1/{dataset}/{tag}"
num_epochs: {num_epochs}
batch_size: 16
grad_accum_steps: 1
learning_rate: 1e-5
warmup_ratio: 0.01
weight_decay: {0.1 if not is_lora else 0.0}
fp16: true

data_seed: 42
data_order_seed_mode: "auto"
dataloader_num_workers: 8
strict_reproducibility: true

logging_steps: 25
eval_steps: 250
eval_on_start: true
eval_batch_size: 16

# Action B8 applies to deep full FT only: those are the configs argued to be
# "harder to optimize", so their ranking must be shown to survive checkpoint
# selection (R2-6c).
checkpoint_fraction: {0.1 if (not is_lora and depth in ("L5", "L6")) else 0.0}
run_best_vs_final: {str(not is_lora and depth in ("L5", "L6")).lower()}
best_vs_final_max_samples: 2000

score_legacy_path: true
decode_num_beams: 1
decode_dtype: "fp16"
decode_force_language: true
save_mode: "auto"
log_dataset_metadata: false
""")

    path = Path(out_dir) / dataset / f"{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(L))
    return path, len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(DATASETS) + ["all"])
    ap.add_argument("--out", default="configs/rev/b1")
    ap.add_argument("--jobs", action="store_true", help="print job-list lines instead")
    args = ap.parse_args()

    cells = [c for c in CELLS if args.phase == "all" or c[0] == args.phase]
    # Tier 1 first: if the schedule slips, what gets dropped is Tier 2.
    cells.sort(key=lambda c: (c[3], c[0], c[1], -c[2]))

    lines, total = [], 0
    for dataset, depth, fraction, tier, methods in cells:
        for method in methods:
            path, n_targets = render(dataset, depth, fraction, method, args.out)
            seeds = SEEDS[tier][method]
            total += len(seeds)
            tag = path.stem
            if not args.jobs:
                print(f"  {path}  ({n_targets} target modules, tier {tier}, {len(seeds)} seeds)")
            for s in seeds:
                lines.append(
                    f"{dataset}_{tag}_s{s}\t{path}\t{s}\ts{s}\t"
                    f"outputs/rev/b1/{dataset}/{tag}\t-"
                )

    if args.jobs:
        print(f"# B1 {args.phase} -- generated by scripts/gen_campaign_configs.py")
        print("# tier 1 (contested crossovers) first: schedule slippage drops tier 2.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} cells -> {total} runs")


if __name__ == "__main__":
    main()
