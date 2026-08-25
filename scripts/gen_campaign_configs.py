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

# Methods are per-cell in B1, not universal:
#   * The data-scaling analysis (manuscript sec 4.3, Figures 4-8) is LoRA-only by
#     design -- the instability numbers being defended (8.21, 8.45 at 10%) are LoRA
#     numbers. Adding FT seeds there would be new experiments answering a question
#     no reviewer asked.
#   * SPGISpeech's FT-vs-LoRA claim lives at L5 (tier 1), so B1 left L4 there
#     LoRA-only. SUPERSEDED by B13, which fills the depth grid to both methods on
#     every corpus; the tables below are kept as the record of what B1 itself ran.
#
# SUPERSEDED 2026-08-25. The paragraph that stood here said there is no L6 LoRA cell
# because "LoRA has nowhere to put an embedding matrix", and that L5 and L6 resolve to
# the same 241 modules. That premise is wrong and the conclusion drawn from it -- that
# LoRA saturates at L5 because the distinguishing parameters are not LoRA-adaptable --
# must not be written into the paper. PEFT wraps nn.Embedding as lora_embedding_A/B,
# and every original L6 LoRA train.log on all three corpora shows exactly
# embed_tokens.lora_embedding_A/B. The 2026-08-06 cells were not vacuous; the generated
# configs were, because lora_targets() could not express them. Fixed 2026-08-24 and
# verified on openai/whisper-medium: L5 = 482 trainable tensors / 44.28M params,
# L6 = 484 / 47.66M, no encoder leak. B13 runs the cells.
#
# What IS true, and belongs in W1 next to the LayerNorm note: L6 is not a symmetric
# cross-method comparison. full_ft_targets() takes embed_tokens AND embed_positions;
# lora_targets() takes embed_tokens only (matching the original). But that asymmetry
# is not an L6 property -- full FT carries model.decoder.layer_norm plus every block's
# internal norms and biases at EVERY depth, and LoRA carries none of them at any depth.
# The increment L6 adds to the mismatch is embed_positions alone: 448 x 1024 = 458,752
# params, 0.86% of the L5->L6 step and 0.06% of the base model. So report L6 WITHIN
# method, each arm against its own L5, and keep L5 as the cross-method headline --
# because L5 is the deepest depth where both methods span the same decoder-block set
# and is where the tier-1 seed budget went, NOT because L6 is uniquely confounded.
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
# Three seeds for BOTH methods, which deviates from the tier-2 schedule (LoRA 3 /
# full FT 2) used elsewhere. The deviation is deliberate and must be stated in the
# paper's seed-allocation paragraph, or the tiering stops reading as a stated design.
#
# Why 3/3 rather than 3/2: an SD from two runs is barely an estimate, and the paper
# already concedes that for VoxPopuli's full-FT cells. Seeding L2/L3 full FT at n=2
# would add four more rows carrying that caveat. These are the shallowest and fastest
# configurations in the campaign, so the third seed costs 4 runs and ~10 h of wall
# clock -- cheap enough that widening a known weak spot to save it would be a poor
# trade. Never split L2 and L3 against each other: asymmetry between adjacent
# supporting depths is the one allocation with no rationale behind it.
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
B10_SEEDS = {"lora": [42, 123, 1234], "full": [42, 123, 1234]}


# --- B11: GigaSpeech data-scaling curve ---------------------------------------------
# R1-5.3's remaining half. B10 made the *depth* axis dense at fraction 1.0; the comment
# also asks for depth resolution inside the *data-scaling* sweep, and B1 gives GigaSpeech
# none at all -- every GigaSpeech cell there is fraction 1.0. So the one corpus carrying
# the contested LoRA-vs-FT crossover is also the one corpus whose layer-wise threshold is
# never re-checked at reduced data, while VoxPopuli and SPGISpeech both have a 10% cell.
#
# Depth is L4, not L5. The 100% endpoint then already exists at n=5
# (b1/gigaspeech/l4_lora, 9.439 +/- 0.032), so these three runs complete a four-point
# curve whose anchor is multi-seed. Note that this puts GigaSpeech's scaling curve at a
# different depth from VoxPopuli's (L5 @ 10%): defensible, because each curve sits next
# to its own n>=3 same-depth anchor, but it has to be *stated* in W1/W2 rather than left
# for a reviewer to notice.
#
# Single seed (42), justified the way B3's was and by measurement, not assertion:
# GigaSpeech LoRA seed SD is 0.03-0.04 pp (see PROGRESS 3.4) against data-scaling effects
# of whole WER points. Quote the measured SD when reporting these.
#
# Fractions are shares of the 680,064-segment post-filter training pool (~747 h), NOT of
# GigaSpeech M (~1,000 h) -- see PROGRESS 3.3. "10%" here is 68,006 segments = ~7.5% of
# M. Label the figure axis against the effective pool; getting this wrong by a third is
# exactly the error 3.3 exists to prevent.
#
# Subset replication follows the ORIGINAL manuscript exactly: 3 subsets at 0.1,
# 2 at 0.2, 1 at 0.5. That schedule is not a guess -- it is what outputs/all_exps_final.csv
# records for the original GigaSpeech data-scaling runs at L0 (exp 002/0023) and L5
# (exp 007/0071), and identically for VoxPopuli and SPGISpeech. Since L4 is being slotted
# into a figure alongside those L0 and L5 curves, it must carry the same replication or
# the three depths are not comparable: the manuscript's 10% error bars are a spread over
# SUBSETS, and an L4 point with one subset would have no comparable bar to show.
#
# Note what the replication is and is not. These are six distinct data subsets at one
# seed, not seed repeats -- the original campaign was single-seed throughout, and
# data_order_seed_mode "auto" pins the shuffle to data_seed at fraction < 1 so that
# subset identity never gets confounded with seed noise (configs.py says this explicitly).
# So report the 0.1 and 0.2 spreads as subset-to-subset variation. The seed-to-seed SD is
# a separate, already-measured quantity (0.03-0.04 pp on GigaSpeech LoRA); do not present
# one as the other.
#
# One job per (fraction, subset), never one job with num_subsets: 3. The original ran
# subsets inside a single job; the rerun cannot, because invariant 1 is one job = one run
# directory and queue_worker.sh's run_dir_for() globs `_subset_*` and takes the first hit,
# so a 3-subset job would be marked complete on subset 0 alone and the other two would
# never be checked. Hence configs.TrainConfig.subset_index, which names the partition and
# leaves num_subsets at 1.
#
# The fractions are NOT a nested series, and that is worth stating before a reviewer
# asks. PartitionedDataset selects by modulo stride over the seed-42-shuffled stream, so
# subset i at 10% is every 10th sample from offset i, at 20% every 5th, at 50% every 2nd.
# 10%/sub0 sits inside both 20%/sub0 and 50%/sub0, but 20%/sub0 is not inside 50%/sub0.
# Every subset is still an unbiased sample of the same pool, so no point is biased; what
# non-nesting costs is a little extra between-point sampling noise. This is exactly the
# mechanism the original campaign's own curves ran under, which is why it is kept.
#
# Ordering is longest-job-first (LPT), and that is load balancing rather than priority:
# workers claim in file order, so the 4.5 h run starts immediately instead of landing
# last on an otherwise-drained queue. Expect ~14.6 GPU-h, ~8 h wall on two GPUs.
B11_CELLS = [
    # (dataset,     depth, fraction, method, subset)
    ("gigaspeech",  "L4", 0.5, "lora", 0),
    ("gigaspeech",  "L4", 0.2, "lora", 0),
    ("gigaspeech",  "L4", 0.2, "lora", 1),
    ("gigaspeech",  "L4", 0.1, "lora", 0),
    ("gigaspeech",  "L4", 0.1, "lora", 1),
    ("gigaspeech",  "L4", 0.1, "lora", 2),
]
B11_SEED = 42


# --- B12: the data-scaling section, rerun -------------------------------------------
# The original paper's data-scaling programme, reproduced in the fixed pipeline. Three
# reasons this is worth ~430 GPU-h rather than a prose limitation:
#
#   1. VoxPopuli's 36 fractional runs HAVE NO ARTIFACTS. Not stale -- absent: 0 of 36
#      result_paths in all_exps_final.csv exist on disk, so 0 have predictions.json and
#      none can be given a bootstrap CI. That whole section of the paper currently rests
#      on numbers in the one file PROGRESS forbids building tables from. SPGISpeech is
#      33 of 39 backed, GigaSpeech 12 of 12.
#   2. The reported runs are FIXED-BUDGET, and nothing in the rerun campaign reproduces
#      that. B1's frac10 cells and B11 are all data-limited (1 epoch over the subset),
#      i.e. ~10x less optimisation than the cells the manuscript reports.
#   3. L6 LoRA carries the headline instability numbers (8.21 / 8.45) and was dropped on
#      a false premise -- see lora_targets().
#
# GRID. Faithful to the original enumeration, read off all_exps_final.csv and confirmed
# against the run directories that still exist:
#     voxpopuli    L0 L2 L3 L4 L5 L6   0.1x3  0.2x2  0.5x1
#     spgispeech   L0                  0.05x5 0.1x3  0.2x2  0.5x1
#                  L2 L3 L4 L5         0.1x3  0.2x2  0.5x1
#                  L6                  0.1x3         0.5x1     <- no 0.2 in the original
#     gigaspeech   L0 L5               0.1x3  0.2x2  0.5x1
# 87 cells. Two irregularities are the original's, not ours, and are reproduced rather
# than tidied: SPGISpeech L0 gets a 0.05 rung nothing else has, and SPGISpeech L6 has no
# 0.2 rung. Adding SPGI L6 @0.2 would cost 4 runs (~12 GPU-h) and make that corpus
# rectangular; it is deliberately NOT done here, because "we ran what they ran" is the
# claim being made. GigaSpeech intermediate depths are likewise absent because the
# original never ran them -- B11 already covers GigaSpeech L4 on the data-limited side.
#
# ONE DEVIATION, and it is an improvement: both regimes get the SAME 87-cell grid, so
# the regime comparison is exactly paired. The original's two arms were run at different
# times over different, both-irregular grids (the March data_scaling_old dirs are
# data-limited, the April data_scaling dirs are fixed-budget), which is why the paper's
# regime comparison is hard to read. Paired arms cost nothing extra and settle it.
#
# SEEDS. One seed (42) per cell, as the original. Replication here is over SUBSETS, not
# seeds -- that is what the manuscript's error bars are, and B11 measured why it matters:
# subset-to-subset gaps at 10% reached 0.176 pp against a 10%->100% effect of 0.215-0.391.
#
# TWO CELLS ARE DELIBERATELY ABSENT. voxpopuli/spgispeech L5 @10% sub0 data-limited at
# seed 42 already exist as b1/<corpus>/l5_lora_frac10 s42 -- byte-identical configs, so
# rerunning them would only add duplicate cells to W4. Use the B1 runs for those points.
B12_FRACS_STD = {0.1: 3, 0.2: 2, 0.5: 1}
B12_GRID = {
    "voxpopuli":  {d: B12_FRACS_STD for d in ("L0", "L2", "L3", "L4", "L5", "L6")},
    "spgispeech": dict({"L0": {0.05: 5, 0.1: 3, 0.2: 2, 0.5: 1},
                        "L6": {0.1: 3, 0.5: 1}},
                       **{d: B12_FRACS_STD for d in ("L2", "L3", "L4", "L5")}),
    "gigaspeech": {d: B12_FRACS_STD for d in ("L0", "L5")},
}
B12_REGIMES = ("fixed_budget", "data_limited")
B12_SEED = 42
# (corpus, depth, fraction, subset, regime) already covered by an identical B1 run.
B12_SKIP = {("voxpopuli", "L5", 0.1, 0, "data_limited"),
            ("spgispeech", "L5", 0.1, 0, "data_limited")}
# Corpus order is priority, and it is evidence-driven: VoxPopuli has zero artifact
# backing, SPGISpeech is missing 6 of 39, GigaSpeech is complete. If the schedule slips,
# what gets dropped is the corpus that already has its numbers.
B12_CORPUS_ORDER = ("voxpopuli", "spgispeech", "gigaspeech")


def b12_cells():
    """Flatten the grid. Fixed-budget first within a corpus: it is the arm the
    manuscript actually reports, so a slipped schedule drops the other one."""
    out = []
    for corpus in B12_CORPUS_ORDER:
        for regime in B12_REGIMES:
            for depth, fracs in sorted(B12_GRID[corpus].items()):
                for frac, n in sorted(fracs.items(), reverse=True):
                    for sub in range(n):
                        if (corpus, depth, frac, sub, regime) in B12_SKIP:
                            continue
                        out.append((corpus, depth, frac, sub, regime))
    return out


# --- B13: complete the depth grid to both methods on every corpus -------------------
# B1 and B10 allocated method coverage by which claim was contested, which left the
# depth grid ragged: SPGISpeech had full FT at L5 only, GigaSpeech and SPGISpeech had no
# L0 full FT, and no corpus had L6 LoRA (dropped 2026-08-06 on a premise now known to be
# wrong -- see the note above CELLS). B13 fills every hole so the table reads
# {L0,L2,L3,L4,L5,L6} x {LoRA, full FT} x 3 corpora with no absent cells.
#
# This is a deliberate departure from the tiered allocation the paper describes, and the
# seed-allocation paragraph must be rewritten to say so: the grid was completed after the
# fact, rather than scoped to contested claims. Do not leave the old description standing
# beside these cells.
#
# Three seeds for both methods, as in B10. A two-run SD is barely an estimate and the
# paper already concedes that for VoxPopuli's B1 full-FT cells; widening that weak spot
# across nine new cells to save nine runs would be a poor trade.
#
# On L6: these cells make L6 a two-method row, but the cross-method comparison there is
# NOT symmetric (full FT adds embed_positions and every LayerNorm, LoRA adds neither).
# Report L6 within method -- each arm against its own L5 -- and keep L5 as the
# cross-method headline. The full-FT L5->L6 step is already measured as a null on
# VoxPopuli (+0.001 pp [-0.031, +0.032]) and -0.069 [-0.109, -0.032] on GigaSpeech;
# the LoRA arm is what makes "the depth axis saturates at L5 for both methods" a
# measured statement instead of an inference, and it also gives the original
# manuscript's reported L6 LoRA numbers their first rerun counterpart.
# Run in three groups, each its own job file, in this order. The order is by how much
# of the grid each group repairs per GPU-hour, not by corpus:
#
#   a  SPGISpeech full FT at L0/L2/L3/L4/L6. The single largest hole -- SPGISpeech is
#      the only corpus whose method comparison exists at one depth. Cheapest runs in the
#      campaign (~1.6-2.1 h), so this buys five two-method rows for ~33 GPU-h.
#   b  L6 LoRA on all three corpora. Gives the manuscript's reported L6 LoRA numbers
#      their first rerun counterpart, and makes "the depth axis saturates at L5 for both
#      methods" measured rather than inferred. GigaSpeech dominates the cost here.
#   c  The L0 row: GigaSpeech L0 full FT (the last absent cell) plus the seed top-ups
#      below. Last because L0 is an anchor, not a claim-bearing depth -- if the schedule
#      slips this is what gets dropped, and the grid is still complete at n>=1.
B13_CELLS = [
    # (group, dataset,   depth, methods) -- configs generated under configs/rev/b13
    ("a", "spgispeech",  "L0", ("full",)),
    ("a", "spgispeech",  "L2", ("full",)),
    ("a", "spgispeech",  "L3", ("full",)),
    ("a", "spgispeech",  "L4", ("full",)),
    ("a", "spgispeech",  "L6", ("full",)),
    ("b", "voxpopuli",   "L6", ("lora",)),
    ("b", "gigaspeech",  "L6", ("lora",)),
    ("b", "spgispeech",  "L6", ("lora",)),
    ("c", "gigaspeech",  "L0", ("full",)),
]
B13_SEEDS = {"lora": [42, 123, 1234], "full": [42, 123, 1234]}
B13_GROUPS = {
    "a": "SPGISpeech full FT across depths (L0/L2/L3/L4/L6)",
    "b": "L6 LoRA on all three corpora",
    "c": "GigaSpeech L0 full FT, and the n=1 L0 seed top-ups",
}

# Four cells exist already but at n=1, which would leave the completed grid with an
# entire row carrying no SD while every other row has one. These are seed top-ups on the
# EXISTING cell, not new cells: they reuse the b2/b3 config and write into the same
# output_dir, so the cell stays one cell rather than splitting across batches.
# (dataset, existing config path, existing output_dir, method, seeds to add)
B13_TOPUP = [   # all group "c"
    ("voxpopuli",  "configs/rev/b2/voxpopuli/l0_lora.yaml",
     "outputs/rev/b2/voxpopuli/l0_lora",  "lora", [123, 1234]),
    ("voxpopuli",  "configs/rev/b3/voxpopuli/l0_full.yaml",
     "outputs/rev/b3/voxpopuli/l0_full",  "full", [123, 1234]),
    ("gigaspeech", "configs/rev/b2/gigaspeech/l0_lora.yaml",
     "outputs/rev/b2/gigaspeech/l0_lora", "lora", [123, 1234]),
    ("spgispeech", "configs/rev/b2/spgispeech/l0_lora.yaml",
     "outputs/rev/b2/spgispeech/l0_lora", "lora", [123, 1234]),
]


def full_ft_targets(depth):
    t = ["proj_out", "model.decoder.layer_norm"]
    if depth == "L6":
        t += ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
    t += [f"model.decoder.layers.{i}" for i in LAYER_SETS[depth]]
    return t


def lora_targets(depth):
    # LoRA cannot wrap LayerNorm, so norms are absent by construction. This asymmetry
    # with full FT is real and is stated in W1.
    #
    # It CAN wrap an embedding, and at L6 the original campaign did. Corrected
    # 2026-08-24 -- the 2026-08-06 decision to drop the L6 LoRA cells rested on the
    # premise that "LoRA has nowhere to put an embedding matrix", which is wrong: PEFT
    # wraps nn.Embedding as lora_embedding_A/B, and every original L6 LoRA train.log on
    # all three corpora shows exactly `embed_tokens.lora_embedding_A/B`. The cells were
    # not vacuous; the generated configs were, because lora_targets() could not express
    # what the original ran. Verified on openai/whisper-medium: L5 = 482 trainable
    # tensors / 44.28M params, L6 = 484 / 47.66M, no encoder leak.
    #
    # embed_tokens ONLY, not embed_positions -- full_ft_targets() adds both, the original
    # LoRA runs added just the one, and reproducing the original is the point here.
    #
    # Whisper ties proj_out to embed_tokens, so adapting both puts a LoRA on each side of
    # a tied weight and PEFT emits a tie warning. The original did this too. Do not
    # "fix" it: it is part of what the manuscript's L6 numbers mean, and W3 already owes
    # a sentence on the embedding tie-breaking procedure.
    t = ["proj_out"]
    if depth == "L6":
        t.append("model.decoder.embed_tokens")
    t += [f"model.decoder.layers.{i}.{m}" for i in LAYER_SETS[depth] for m in LORA_MODS]
    return t


def init_blocks(depth):
    layers = LAYER_SETS[depth]
    # embed_tokens rides in the first block with proj_out because the two are tied
    # weights; re-initialising them in separate seed-reset stages would be arbitrary.
    head = ["proj_out"] + (["embed_tokens"] if depth == "L6" else [])
    if not layers:
        return [head]
    return [head + [f"layers.{layers[0]}"]] + [[f"layers.{i}"] for i in layers[1:]]


def render(dataset, depth, fraction, method, out_dir, lr=LR_BASE, batch="b1",
           rank=RANK_BASE, tag_override=None, targets_override=None,
           init_blocks_override=None, extra_yaml="", subset=None, regime=None):
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
    # Only tagged when a caller names a partition explicitly, so every config written
    # before subset_index existed keeps its path and regenerates byte-identically.
    if subset is not None:
        tag += f"_sub{subset}"
    # Regime is in the path because the two arms are the same (depth, fraction, subset)
    # and differ only in step budget -- without it they would collide on one output_dir.
    if regime is not None:
        tag += {"fixed_budget": "_fb", "data_limited": "_dl"}[regime]
    if tag_override:
        tag = tag_override
    # The two regimes the manuscript compares, made explicit rather than inferred:
    #   "fixed_budget"  num_epochs 0 -> max_steps from the FULL pool at EVERY fraction,
    #                   so a 10% run sees the same optimisation budget and simply repeats
    #                   its subset ~10x. This is what the original's reported (April) runs
    #                   did -- verified in their train.logs, e.g. spgispeech L5 @10%:
    #                   "max_steps: 9797 ... Dataset Len: 15675".
    #   "data_limited"  num_epochs 1 -> max_steps from the SUBSET, one pass. This is what
    #                   the original's superseded (March) `data_scaling_old` runs did, and
    #                   what B1's frac10 cells and B11 do.
    # regime=None keeps the pre-2026-08-24 rule so every config written before this
    # parameter existed regenerates byte-identically.
    if regime == "fixed_budget":
        num_epochs = 0
    elif regime == "data_limited":
        num_epochs = 1
    else:
        num_epochs = 0 if fraction == 1.0 else 1
    # Emitted only when named, for the same byte-identity reason as the tag above.
    subset_yaml = "" if subset is None else f"subset_index: {subset}\n"

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
{subset_yaml}"""]

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
            seeds = B10_SEEDS[method]
            total += len(seeds)
            tag = path.stem
            if not as_jobs:
                print(f"  {path}  ({n_targets} target modules, {len(seeds)} seeds)")
            for sd in seeds:
                lines.append(f"{dataset}_{tag}_s{sd}\t{path}\t{sd}\ts{sd}\t"
                             f"outputs/rev/b10/{dataset}/{tag}\t-")
    if as_jobs:
        print("# B10 intermediate depths L2/L3 (R1-5.3) -- generated by scripts/gen_campaign_configs.py")
        print("# Three seeds for both methods -- see B10_SEEDS for why this deviates from tier 2.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} cells -> {total} runs")


def emit_b11(phase, out_dir, as_jobs):
    """GigaSpeech data-scaling curve: one depth, three fractions, six subsets, one seed.

    Goes through render() with fraction < 1.0, which sets num_epochs: 1 -- so the step
    budget comes from the partition (0.1 -> ~4,250 steps) rather than from the
    full-pool fixed budget the 1.0 cells use, and data_order_seed_mode "auto" pins the
    shuffle seed to data_seed so the subset is identical across any future seeds. That
    is the same mechanism the existing l5_lora_frac10 cells ran under, which is what
    makes these three points comparable to the 100% anchor.
    """
    cells = [c for c in B11_CELLS if phase == "all" or c[0] == phase]
    if not cells:
        print(f"# B11 is gigaspeech-only; nothing for phase {phase}")
        return
    lines = []
    for dataset, depth, fraction, method, subset in cells:
        path, n_targets = render(dataset, depth, fraction, method, out_dir,
                                 batch="b11", subset=subset)
        tag = path.stem
        if not as_jobs:
            print(f"  {path}  ({n_targets} target modules, fraction {fraction:g}, "
                  f"subset {subset}, seed {B11_SEED})")
        lines.append(f"{dataset}_{tag}_s{B11_SEED}\t{path}\t{B11_SEED}\ts{B11_SEED}\t"
                     f"outputs/rev/b11/{dataset}/{tag}\t-")
    if as_jobs:
        print("# B11 GigaSpeech data-scaling curve (R1-5.3) -- generated by scripts/gen_campaign_configs.py")
        print("# L4 LoRA at 0.1/0.2/0.5; the 1.0 endpoint already exists at n=5 in b1/gigaspeech/l4_lora.")
        print("# Subsets 3/2/1 at 0.1/0.2/0.5 -- the ORIGINAL manuscript's replication schedule,")
        print("# read off outputs/all_exps_final.csv (L0 exp 002/0023, L5 exp 007/0071).")
        print("# One job per (fraction, subset): see configs.TrainConfig.subset_index for why.")
        print("# Fractions are shares of the 680,064-segment post-filter pool (~747 h), not of GigaSpeech M.")
        print("# Longest job first: load balancing across two workers, not priority.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} cells -> {len(cells)} runs (1 seed each)")


def emit_b12(phase, out_dir, as_jobs):
    """The original data-scaling grid under both regimes, one job per cell."""
    cells = [c for c in b12_cells() if phase == "all" or c[0] == phase]
    if not cells:
        print(f"# nothing for phase {phase}")
        return
    lines, seen = [], set()
    for corpus, depth, frac, sub, regime in cells:
        key = (corpus, depth, frac, sub, regime)
        if key in seen:
            raise AssertionError(f"duplicate B12 cell {key}")
        seen.add(key)
        path, n_targets = render(corpus, depth, frac, "lora", out_dir, batch="b12",
                                 subset=sub, regime=regime)
        tag = path.stem
        if not as_jobs:
            print(f"  {path}  ({n_targets} targets, {regime})")
        lines.append(f"{corpus}_{tag}_s{B12_SEED}\t{path}\t{B12_SEED}\ts{B12_SEED}\t"
                     f"outputs/rev/b12/{corpus}/{tag}\t-")
    if as_jobs:
        print(f"# B12 {phase} -- the original data-scaling grid, rerun. Generated by scripts/gen_campaign_configs.py")
        print("# 87 cells x 2 regimes (fixed_budget `_fb`, data_limited `_dl`), minus 2 already in B1.")
        print("# `_fb` reproduces the manuscript's REPORTED runs: full-pool step budget at every")
        print("# fraction, so a 10% run repeats its subset ~10x. `_dl` is one pass over the subset.")
        print("# Single seed 42; replication is over SUBSETS, as in the original.")
        print("# Corpus order is priority: VoxPopuli has 0 of 36 original artifacts, SPGISpeech")
        print("# 33 of 39, GigaSpeech 12 of 12. Slippage should drop the corpus that has its numbers.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} cells -> {len(cells)} runs (1 seed each)")


def emit_b13(phase, out_dir, as_jobs, group="all"):
    """Fill the depth grid to both methods everywhere, in three ordered groups.

    Two kinds of line. New cells go through render() as usual and land under
    configs/rev/b13. Top-ups point at the config and output_dir the cell ALREADY has in
    b2/b3 -- rendering a duplicate under b13 would split one cell across two batches and
    every downstream scan (w4_tables, z1_bootstrap, z3) groups by (batch, corpus, cell).
    """
    cells = [c for c in B13_CELLS
             if (phase == "all" or c[1] == phase) and (group == "all" or c[0] == group)]
    topups = [t for t in B13_TOPUP
              if (phase == "all" or t[0] == phase) and group in ("all", "c")]
    lines, total = [], 0
    for _grp, dataset, depth, methods in cells:
        for method in methods:
            path, n_targets = render(dataset, depth, 1.0, method, out_dir, batch="b13")
            seeds = B13_SEEDS[method]
            total += len(seeds)
            tag = path.stem
            if not as_jobs:
                print(f"  [{_grp}] {path}  ({n_targets} target modules, {len(seeds)} seeds)")
            for sd in seeds:
                lines.append(f"{dataset}_{tag}_s{sd}\t{path}\t{sd}\ts{sd}\t"
                             f"outputs/rev/b13/{dataset}/{tag}\t-")
    for dataset, cfg, outdir, method, seeds in topups:
        tag = Path(cfg).stem
        total += len(seeds)
        if not as_jobs:
            print(f"  [c] {cfg}  (top-up, {len(seeds)} seeds -> {outdir})")
        for sd in seeds:
            lines.append(f"{dataset}_{tag}_s{sd}\t{cfg}\t{sd}\ts{sd}\t{outdir}\t-")
    if as_jobs:
        label = "all groups" if group == "all" else f"group {group} -- {B13_GROUPS[group]}"
        print(f"# B13 depth-grid completion, {label}")
        print("# Generated by scripts/gen_campaign_configs.py. Three seeds for both")
        print("# methods, as in B10. Run groups in order a -> b -> c.")
        if group in ("all", "b"):
            print("# L6 becomes a two-method ROW but is NOT a symmetric cross-method")
            print("# comparison: full FT takes embed_positions and every LayerNorm, LoRA")
            print("# takes neither. Report L6 within method, each arm against its own L5;")
            print("# L5 stays the cross-method headline.")
        if group in ("all", "c"):
            print("# The l0_lora/l0_full lines are seed top-ups on EXISTING b2/b3 cells and")
            print("# write into the existing output_dir on purpose. Drop them if n=1 at L0")
            print("# is acceptable -- the grid is complete without them.")
        print("# job_id\tconfig\tseed\trun_name\toutput_dir\textra")
        print("\n".join(lines))
    else:
        print(f"\n  {len(cells)} new cells + {len(topups)} top-ups -> {total} runs")


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
    ap.add_argument("--batch", default="b1", choices=["b1", "b2", "b3", "b4", "b5", "b10", "b11", "b12", "b13"])
    ap.add_argument("--out", default=None, help="default: configs/rev/<batch>")
    ap.add_argument("--jobs", action="store_true", help="print job-list lines instead")
    ap.add_argument("--group", default="all", choices=["all", "a", "b", "c"],
                    help="B13 only: which ordered group to emit")
    args = ap.parse_args()

    out_dir = args.out or f"configs/rev/{args.batch}"
    if args.batch == "b1":
        emit_b1(args.phase, out_dir, args.jobs)
    elif args.batch == "b3":
        emit_b3(args.phase, out_dir, args.jobs)
    elif args.batch == "b10":
        emit_b10(args.phase, out_dir, args.jobs)
    elif args.batch == "b11":
        emit_b11(args.phase, out_dir, args.jobs)
    elif args.batch == "b12":
        emit_b12(args.phase, out_dir, args.jobs)
    elif args.batch == "b13":
        emit_b13(args.phase, out_dir, args.jobs, args.group)
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
