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

# The campaign-wide learning rate. Every B1 cell holds it fixed by construction, which
# is exactly what R1-3.1-3.4 / R2-2 / R3-2 object to -- see B3_CELLS.
LR_BASE = 1e-5

# The campaign-wide LoRA rank. Chosen on L0 and held fixed everywhere, which is what
# R1-5.4 and R3-4 object to -- see B4_CELLS.
RANK_BASE = 64


def lr_tag(lr):
    """1e-5 -> 'lr1e5', 3e-05 -> 'lr3e5', 1e-04 -> 'lr1e4'. Filename-safe."""
    m, e = f"{lr:.0e}".split("e")
    return f"lr{m}e{abs(int(e))}"


def lr_yaml(lr):
    """1e-5, not python's 1e-05. Keeps B1 configs byte-identical to what already ran."""
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e-{abs(int(e))}"

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
#
# There is no L6 LoRA cell, and this is not an oversight (2026-08-06). L6 is defined
# as L5 plus embed_tokens and embed_positions; full_ft_targets() adds them, but
# lora_targets() cannot -- LORA_MODS wraps attention and FFN projections only, and
# LoRA has nowhere to put an embedding matrix. L5 and L6 therefore resolve to the
# same 241 modules, and l6_lora.yaml was byte-identical to l5_lora.yaml apart from
# output_dir. The cells were enumerated separately until the queue was already
# running, which would have spent ~14 runs recomputing L5 under the name L6.
# The honest form of the result is that LoRA saturates at L5 because the parameters
# that distinguish L6 are not LoRA-adaptable; that belongs in W1 next to the existing
# note on LayerNorm. Do not re-add an L6 LoRA cell without first giving L6 LoRA a
# distinct meaning (an embedding adapter, or embeddings trainable alongside LoRA) --
# and note that such a variant is a new experiment, not comparable to the original
# paper's numbers.
CELLS = [
    # (dataset,     depth, fraction, tier, methods)
    ("gigaspeech",  "L4", 1.0, 1, ("lora", "full")),
    ("gigaspeech",  "L5", 1.0, 1, ("lora", "full")),
    ("gigaspeech",  "L6", 1.0, 1, ("full",)),
    ("spgispeech",  "L5", 1.0, 1, ("lora", "full")),
    ("spgispeech",  "L4", 1.0, 2, ("lora",)),
    ("voxpopuli",   "L4", 1.0, 2, ("lora", "full")),
    ("voxpopuli",   "L5", 1.0, 2, ("lora", "full")),
    ("voxpopuli",   "L6", 1.0, 2, ("full",)),
    # Data-scale cells carrying claims: the 10% instability numbers being defended
    # (8.21, 8.45) and Figure 8's regime comparison, whose crossover rests on 0.30 WER
    # on VoxPopuli -- squarely in the territory the reseeding campaign exists to
    # protect. Both were written as "L6 @ 10%"; at LoRA that is L5 @ 10%, so they are
    # enumerated under their true depth. The runs are the same runs, correctly named.
    ("spgispeech",  "L5", 0.1, 2, ("lora",)),
    ("voxpopuli",   "L5", 0.1, 2, ("lora",)),
]


# --- B3: learning-rate sensitivity ----------------------------------------------
# B1 holds the learning rate at LR_BASE by construction and therefore *cannot* answer
# the strongest methodological objection in the review: R1-3.1-3.4, R2-2 and R3-2 all
# ask whether deep full FT is intrinsically harder to optimize or merely under-tuned at
# 1e-5. The plan calls this non-negotiable.
#
# Single seed (42) per cell, deliberately. B1 measured the seed SD directly -- 0.03-0.11
# WER pp across every cell -- while the effects under test here are the 0.40-0.52 pp
# depth and method gaps. One seed resolves an effect an order of magnitude larger than
# the noise; spending 3 seeds per LR would buy precision nobody is asking for. Quote the
# measured SD when reporting these, not an assumption.
#
# Arms:
#   deep FT  L5/L6 x {3e-5, 1e-4} on both cached corpora. GigaSpeech is where deep FT
#            degrades (9.49 -> 9.89 -> 9.96) so it is the corpus the claim lives on;
#            VoxPopuli is the contrast, where the same budget produces no degradation
#            (6.26 -> 5.86 -> 5.86). If the gap closes at higher LR the effect was
#            under-tuning; if it does not, it is intrinsic (R1-3.3).
#   shallow  L0 x {1e-5, 1e-4} on VoxPopuli. R1-3.4 asks for shallow *and* deep. Put on
#            VoxPopuli because a GigaSpeech pass costs ~4x the wall-clock for the same
#            answer, and the 1e-5 arm doubles as B2's VoxPopuli L0 anchor.
#   LoRA     L5 x {3e-5, 1e-4} on GigaSpeech. R3-2 objects that FT and LoRA share one
#            learning rate, making the comparison possibly unfair; LoRA usually wants a
#            higher LR. Without this arm the FT-vs-LoRA crossover stays confounded.
B3_CELLS = [
    # (dataset,    depth, method, lr)
    ("gigaspeech", "L5", "full", 3e-5),
    ("gigaspeech", "L5", "full", 1e-4),
    ("gigaspeech", "L6", "full", 3e-5),
    ("gigaspeech", "L6", "full", 1e-4),
    ("gigaspeech", "L5", "lora", 3e-5),
    ("gigaspeech", "L5", "lora", 1e-4),
    ("voxpopuli",  "L5", "full", 3e-5),
    ("voxpopuli",  "L5", "full", 1e-4),
    ("voxpopuli",  "L6", "full", 3e-5),
    ("voxpopuli",  "L6", "full", 1e-4),
    ("voxpopuli",  "L0", "full", LR_BASE),   # anchor; also B2's VoxPopuli L0
    ("voxpopuli",  "L0", "full", 1e-4),
]
B3_SEED = 42


# --- B5: encoder adaptation ------------------------------------------------------
# R1-5.2: "a limited encoder-only or combined encoder-decoder adaptation experiment
# would materially strengthen the argument that decoder-side adaptation captures the
# dominant domain-specific gains". The manuscript asserts that; nothing in the study
# tests it, because the encoder is frozen everywhere by construction.
#
# These are the only runs in the campaign OUTSIDE the frozen-encoder scope. They set
# allow_encoder_adaptation: true, which is a deliberate scoped exception to the guards
# in train.set_trainable_parameters and utils.save_trainable_state -- see configs.py.
# Report them separately from everything else; they do not belong in the depth tables.
#
# The Whisper encoder has 24 layers and only 6 LoRA-wrappable modules each
# (self_attn q/k/v/out, fc1, fc2) -- there is no cross-attention, unlike the decoder's
# 10. So an encoder layer is cheaper to adapt than a decoder layer, and the arms below
# are matched on layer count rather than parameter count.
#
# init_blocks entries are fully qualified here, unlike B1's. train.py matches them by
# substring (`any(target in name ...)`), so a bare "layers.23" would hit encoder and
# decoder alike once both carry LoRA, silently collapsing two init blocks into one.
# B1 was safe only because its encoder modules were never wrapped.
ENC_MODS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.out_proj", "fc1", "fc2"]

B5_CELLS = [
    # (tag,                    encoder layers,          include decoder L5?)
    ("enc_top12_lora",         list(range(23, 11, -1)), False),
    ("enc_all_lora",           list(range(23, -1, -1)), False),
    ("enc_top12_dec_l5_lora",  list(range(23, 11, -1)), True),
]
B5_SEED = 42
B5_DATASET = "voxpopuli"   # cheapest corpus; the question is qualitative


def b5_targets(enc_layers, with_decoder):
    t = [f"model.encoder.layers.{i}.{m}" for i in enc_layers for m in ENC_MODS]
    if with_decoder:
        t += lora_targets("L5")
    return t


def b5_init_blocks(enc_layers, with_decoder):
    blocks = [[f"model.encoder.layers.{i}"] for i in enc_layers]
    if with_decoder:
        dec = LAYER_SETS["L5"]
        blocks += [["proj_out", f"model.decoder.layers.{dec[0]}"]]
        blocks += [[f"model.decoder.layers.{i}"] for i in dec[1:]]
    return blocks


# --- B4: LoRA rank sensitivity ----------------------------------------------------
# R1-5.4: "the LoRA rank is selected only for L0 and then fixed for all deeper
# configurations ... provide at least a limited rank-sensitivity analysis for a deeper
# configuration". R3-4 asks the same. r=64 at L5 already exists in B1, so this adds the
# two flanking ranks at the depth where the method claims actually live.
#
# Corpora: GigaSpeech, where LoRA *beats* full FT at L5, and SPGISpeech, where it
# loses. Both are tier-1 cells, and putting the check on both means the answer cannot
# be an artifact of which side LoRA happens to be on. VoxPopuli is omitted -- it would
# cost another ~6 h to re-ask a question these two already answer.
B4_CELLS = [
    # (dataset,    depth, method, rank)
    ("gigaspeech", "L5", "lora", 32),
    ("gigaspeech", "L5", "lora", 128),
    ("spgispeech", "L5", "lora", 32),
    ("spgispeech", "L5", "lora", 128),
    # Added after B5: a budget control, not a rank result. B5's encoder-only arms train
    # 14.2M and 28.3M parameters against decoder L5's 44.3M, so "decoder beats encoder"
    # was confounded with budget. Decoder L5 at r=32 is 22.1M, which together with the
    # existing r=64 point *brackets* enc_all_lora at 28.3M. If enc_all lands between the
    # two decoder points, budget explains the gap and adaptation site does not separate
    # them; if the smaller decoder point still wins, the site effect is real. VoxPopuli
    # because that is the only corpus B5 ran on -- a cross-corpus control would not
    # control anything.
    ("voxpopuli",  "L5", "lora", 32),
]

# --- B2: shallow anchors ----------------------------------------------------------
# One L0 rerun per corpus so the OOD and cost tables span shallow-to-deep
# (R1-2.1, R1-5.5, R3-10). LoRA, not full FT: the manuscript's L0 cells are LoRA
# (11-16 lora rows per corpus in all_exps_final.csv against 1 standard), so a full-FT
# L0 would not be the anchor those tables need.
#
# Note this supersedes an earlier claim in PROGRESS: B3's VoxPopuli L0 arm is *full*
# FT, produced for the shallow learning-rate question, and does NOT double as this
# anchor. All three corpora are needed here.
B2_CELLS = [
    ("gigaspeech", "L0", "lora"),
    ("spgispeech", "L0", "lora"),
    ("voxpopuli",  "L0", "lora"),
]
B4_SEED = B2_SEED = 42

# --- B10: intermediate depths ------------------------------------------------------
# R1-5.3 asks for intermediate configurations so the depth axis is not just its
# endpoints. L2 and L3 were dropped from the multi-seed design to save compute, which
# left the depth curve with a gap between L0 (proj_out only) and L4 (upper half of the
# decoder) -- exactly the region where the shallow-end damage measured at L0 has to turn
# into the gains measured at L4.
#
# Seeded as TIER 2, reusing SEEDS[2]: LoRA 3 seeds, full FT 2. That asymmetry is the
# campaign's existing design, not a shortcut -- LoRA carries an extra noise source in
# its initialisation, while full FT's are data order, dropout and fp16, which measured
# smaller. It also means these rows are supported exactly as well as VoxPopuli's L4/L5/L6
# full-FT rows, which are already n=2; no new n=1 row enters the results table, which is
# what R1-2.1 is asking about.
#
# Method coverage mirrors what each corpus already has rather than filling the grid:
# SPGISpeech has full FT at L5 only, so full FT at L2/L3 there would create a method
# comparison at a depth with no counterpart elsewhere in that corpus.
B10_CELLS = [
    # (dataset,     depth, methods)
    ("voxpopuli",   "L2", ("lora", "full")),
    ("voxpopuli",   "L3", ("lora", "full")),
    ("spgispeech",  "L2", ("lora",)),
    ("spgispeech",  "L3", ("lora",)),
    ("gigaspeech",  "L2", ("lora", "full")),
    ("gigaspeech",  "L3", ("lora", "full")),
]
B10_TIER = 2


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


def render(dataset, depth, fraction, method, out_dir, lr=LR_BASE, batch="b1",
           rank=RANK_BASE, tag_override=None, targets_override=None,
           init_blocks_override=None, extra_yaml=""):
    ds = DATASETS[dataset]
    name = ds["name"]
    is_lora = method == "lora"
    tag = f"{depth.lower()}_{method}" + ("" if fraction == 1.0 else f"_frac{int(fraction*100)}")
    # Only non-baseline learning rates are tagged, so every B1 path stays byte-identical
    # to what the completed phases already wrote.
    if lr != LR_BASE:
        tag += f"_{lr_tag(lr)}"
    if rank != RANK_BASE:
        tag += f"_r{rank}"
    if tag_override:
        tag = tag_override
    # fraction 1.0 -> fixed-budget (num_epochs 0, steps from the FULL dataset);
    # fraction < 1 -> the data-limited cells cited in the regime comparison.
    num_epochs = 0 if fraction == 1.0 else 1

    lr_note = "" if lr == LR_BASE else f", lr={lr_yaml(lr)}"
    L = [f"""# {batch.upper()} {dataset} {depth} {method.upper()} @ {fraction:g} data{lr_note} -- GENERATED by
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
        # lora_alpha tracks r so the scaling factor alpha/r stays at 2 across ranks.
        # Holding alpha fixed instead would vary effective update magnitude along with
        # capacity, and B4 could not then attribute anything to rank (R1-5.4).
        L.append(f"\nlora_config:\n  r: {rank}\n  lora_alpha: {2 * rank}\n"
                 f"  lora_dropout: 0.1\n  init_blocks:\n")
        for b in (init_blocks_override or init_blocks(depth)):
            L.append("    - [" + ", ".join(f'"{x}"' for x in b) + "]\n")
        targets = targets_override or lora_targets(depth)
    else:
        targets = targets_override or full_ft_targets(depth)

    L.append("\ntarget_modules:\n")
    L += [f'  - "{t}"\n' for t in targets]

    L.append(f"""
output_dir: "outputs/rev/{batch}/{dataset}/{tag}"
num_epochs: {num_epochs}
batch_size: 16
grad_accum_steps: 1
learning_rate: {lr_yaml(lr)}
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

    if extra_yaml:
        L.append(extra_yaml)

    path = Path(out_dir) / dataset / f"{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(L))
    return path, len(targets)


def emit_b1(phase, out_dir, as_jobs):
    cells = [c for c in CELLS if phase == "all" or c[0] == phase]
    # Tier 1 first: if the schedule slips, what gets dropped is Tier 2.
    cells.sort(key=lambda c: (c[3], c[0], c[1], -c[2]))

    lines, total = [], 0
    for dataset, depth, fraction, tier, methods in cells:
        for method in methods:
            path, n_targets = render(dataset, depth, fraction, method, out_dir)
            seeds = SEEDS[tier][method]
            total += len(seeds)
            tag = path.stem
            if not as_jobs:
                print(f"  {path}  ({n_targets} target modules, tier {tier}, {len(seeds)} seeds)")
            for s in seeds:
                lines.append(
                    f"{dataset}_{tag}_s{s}\t{path}\t{s}\ts{s}\t"
                    f"outputs/rev/b1/{dataset}/{tag}\t-"
                )

    if as_jobs:
        print(f"# B1 {phase} -- generated by scripts/gen_campaign_configs.py")
        print("# tier 1 (contested crossovers) first: schedule slippage drops tier 2.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} cells -> {total} runs")


def emit_b3(phase, out_dir, as_jobs):
    cells = [c for c in B3_CELLS if phase == "all" or c[0] == phase]
    # Deep FT first: it carries the objection. If the schedule slips, the shallow and
    # LoRA arms are what get dropped, not the claim under dispute.
    order = {("full", "L5"): 0, ("full", "L6"): 1, ("lora", "L5"): 2, ("full", "L0"): 3}
    cells.sort(key=lambda c: (order.get((c[2], c[1]), 9), c[0], c[3]))

    lines = []
    for dataset, depth, method, lr in cells:
        path, n_targets = render(dataset, depth, 1.0, method, out_dir, lr=lr, batch="b3")
        tag = path.stem
        if not as_jobs:
            print(f"  {path}  ({n_targets} target modules, lr={lr:g}, seed {B3_SEED})")
        lines.append(
            f"{dataset}_{tag}_s{B3_SEED}\t{path}\t{B3_SEED}\ts{B3_SEED}\t"
            f"outputs/rev/b3/{dataset}/{tag}\t-"
        )

    if as_jobs:
        print(f"# B3 {phase} learning-rate sensitivity -- generated by scripts/gen_campaign_configs.py")
        print("# Single seed 42: B1 measured seed SD at 0.03-0.11 WER pp, the effects here are 0.40-0.52.")
        print("# Deep FT first -- it carries R1-3.1-3.4 / R2-2 / R3-2.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} cells -> {len(cells)} runs (1 seed each)")


def emit_b5(phase, out_dir, as_jobs):
    """B5 goes through render() with explicit targets: encoder modules have no place in
    the depth ladder, so they cannot be derived from a LAYER_SETS entry."""
    if phase not in (B5_DATASET, "all"):
        print(f"# B5 is {B5_DATASET}-only; nothing for phase {phase}")
        return
    extra = ("\n# Scoped exception: this run adapts encoder layers on purpose (R1-5.2),\n"
             "# and is therefore OUTSIDE the frozen-encoder scope every other claim in the\n"
             "# study assumes. Report it separately from the depth tables.\n"
             "allow_encoder_adaptation: true\n")
    lines = []
    for tag, enc_layers, with_dec in B5_CELLS:
        path, n_targets = render(
            B5_DATASET, "L5", 1.0, "lora", out_dir, batch="b5", tag_override=tag,
            targets_override=b5_targets(enc_layers, with_dec),
            init_blocks_override=b5_init_blocks(enc_layers, with_dec),
            extra_yaml=extra,
        )
        if not as_jobs:
            print(f"  {path}  ({n_targets} target modules, "
                  f"{len(enc_layers)} encoder layers, decoder L5: {with_dec})")
        lines.append(f"{B5_DATASET}_{tag}_s{B5_SEED}\t{path}\t{B5_SEED}\ts{B5_SEED}\t"
                     f"outputs/rev/b5/{B5_DATASET}/{tag}\t-")
    if as_jobs:
        print("# B5 encoder adaptation -- generated by scripts/gen_campaign_configs.py")
        print("# OUTSIDE the frozen-encoder scope; report separately from the depth tables.")
        print("# Reference arm is the existing decoder-only outputs/rev/b1/voxpopuli/l5_lora.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(B5_CELLS)} cells -> {len(B5_CELLS)} runs (1 seed each)")


def emit_b10(phase, out_dir, as_jobs):
    """Intermediate depths, seeded on the same tier-2 schedule as the rest."""
    cells = [c for c in B10_CELLS if phase == "all" or c[0] == phase]
    lines, total = [], 0
    for dataset, depth, methods in cells:
        for method in methods:
            path, n_targets = render(dataset, depth, 1.0, method, out_dir, batch="b10")
            seeds = SEEDS[B10_TIER][method]
            total += len(seeds)
            tag = path.stem
            if not as_jobs:
                print(f"  {path}  ({n_targets} target modules, {len(seeds)} seeds)")
            for sd in seeds:
                lines.append(f"{dataset}_{tag}_s{sd}\t{path}\t{sd}\ts{sd}\t"
                             f"outputs/rev/b10/{dataset}/{tag}\t-")
    if as_jobs:
        print("# B10 intermediate depths L2/L3 (R1-5.3) -- generated by scripts/gen_campaign_configs.py")
        print("# Tier-2 seeding, same as the rest of the campaign: LoRA 3 seeds, full FT 2.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} cells -> {total} runs")


def emit_simple(cells, batch, phase, out_dir, as_jobs, seed, header):
    """B2/B4: one run per cell, single seed, no tier logic."""
    sel = [c for c in cells if phase == "all" or c[0] == phase]
    lines = []
    for c in sel:
        dataset, depth, method = c[0], c[1], c[2]
        rank = c[3] if len(c) > 3 else RANK_BASE
        path, n_targets = render(dataset, depth, 1.0, method, out_dir,
                                 batch=batch, rank=rank)
        tag = path.stem
        if not as_jobs:
            print(f"  {path}  ({n_targets} target modules, r={rank}, seed {seed})")
        lines.append(f"{dataset}_{tag}_s{seed}\t{path}\t{seed}\ts{seed}\t"
                     f"outputs/rev/{batch}/{dataset}/{tag}\t-")
    if as_jobs:
        print(f"# {batch.upper()} {phase} -- generated by scripts/gen_campaign_configs.py")
        for h in header:
            print(f"# {h}")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(sel)} cells -> {len(sel)} runs (1 seed each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(DATASETS) + ["all"])
    ap.add_argument("--batch", default="b1", choices=["b1", "b2", "b3", "b4", "b5", "b10"])
    ap.add_argument("--out", default=None, help="default: configs/rev/<batch>")
    ap.add_argument("--jobs", action="store_true", help="print job-list lines instead")
    args = ap.parse_args()

    out_dir = args.out or f"configs/rev/{args.batch}"
    if args.batch == "b1":
        emit_b1(args.phase, out_dir, args.jobs)
    elif args.batch == "b3":
        emit_b3(args.phase, out_dir, args.jobs)
    elif args.batch == "b10":
        emit_b10(args.phase, out_dir, args.jobs)
    elif args.batch == "b5":
        emit_b5(args.phase, out_dir, args.jobs)
    elif args.batch == "b4":
        emit_simple(B4_CELLS, "b4", args.phase, out_dir, args.jobs, B4_SEED,
                    ["LoRA rank sensitivity at L5 (R1-5.4, R3-4). r=64 is the B1 baseline.",
                     "lora_alpha tracks r so alpha/r stays at 2 and only capacity varies."])
    else:
        emit_simple(B2_CELLS, "b2", args.phase, out_dir, args.jobs, B2_SEED,
                    ["Shallow L0 anchors, LoRA (R1-2.1, R1-5.5, R3-10).",
                     "Vanilla baselines are inference-only: see configs/rev/b2/baseline_*.yaml."])


if __name__ == "__main__":
    main()
