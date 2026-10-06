# Experiment specification

Completed run artifacts are authoritative for executed budgets. Use
`schedule.max_steps` and `schedule.steps_completed` from `run_manifest.json`
or `cost.json` whenever they are available.

## Shared setup

- Model: `openai/whisper-medium`; English transcription.
- Training datasets: SPGISpeech 2.0, VoxPopuli-en, and GigaSpeech-M.
- OOD datasets: LibriSpeech test-clean and test-other.
- Main scopes: L0, L2, L3, L4, L5, L6; L1 was not evaluated.
- Main LoRA setting: rank 64, alpha 128, dropout 0.1.
- Main learning rate: 1e-5; AdamW; linear schedule; 0.01 warmup; weight decay 0.1; batch size 16; gradient accumulation 1; FP16.
- Decoding: greedy English transcription, maximum 200 new tokens.
- Fractions use deterministic shuffled-stream partitioning in `data/partitioning.py`.

## Experiment families

| Family | Datasets | Methods/settings | Regime and replication |
|---|---|---|---|
| Decoder depth | All three training datasets | LoRA and full FT; L0/L2/L3/L4/L5/L6; encoder frozen | Full-data controlled comparison; use completed manifests for exact steps and seeds |
| LoRA rank selection | All three training datasets | L0 LoRA; ranks 8, 16, 32, 64, 128, 256, 512; alpha = 2r | Fixed full-data budget; five seeds per rank |
| Learning-rate sensitivity | GigaSpeech and VoxPopuli | Reported L0/L5/L6 full FT and GigaSpeech L5 LoRA cells | 1e-5, 3e-5, 1e-4; higher-rate runs are single-seed where specified |
| Convergence/checkpoints | GigaSpeech and VoxPopuli | Representative LoRA/full-FT cells | Fixed-budget and data-limited; use recorded metrics/checkpoints |
| L5 rank sensitivity | GigaSpeech and SPGISpeech | L5 LoRA ranks 32/64/128 and full-FT reference | Fixed full-data budget; replication follows the manuscript |
| Encoder control | VoxPopuli | Encoder top-12, encoder all-24, decoder L5, encoder+decoder | Full-data targeted control |
| Data-limited scaling | All three training datasets | LoRA across reported depths | 10%, 20%, 50%, 100%; steps scale with data |
| Fixed-budget scaling | All three training datasets | LoRA across reported depths | 10%, 20%, 50%, 100%; full-data budget held constant |
| Efficiency | All three training datasets | LoRA r=64 and full FT | Use recorded cost artifacts for time, throughput, memory, and parameters |
| Bootstrap | Reported evaluation sets | Paired WER bootstrap | 10,000 resamples, seed 42; implemented in `scripts/z1_bootstrap.py` |
