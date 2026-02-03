|ID|Experiment_ID|Target Modules|SpgiSpeech_2|GigaSpeech|Notes|
|---|---|---|---|---|---|
| - | 000 | - | ✅ | ✅ | Whisper-medium baseline vanilla Inference (Base 1)|
| - | 001 | proj_out | ✅ | ✅ | Standard fine-tuning (Base 2)|
| L0 | 002* | proj_out | ✅ | ✅ | Lora Rank=64, lora_alpha=128|
| L1 | 003* | proj_out + output-side norms | ❌| ❌ | Skipped due to LoRA not supporting layer norm|
| L2 | 004* | last 1 decoder block + output-side norms + proj_out | ✅ | ✅ | Repat experiment 0041
| L3 | 005* | last 2 decoder blocks + output-side norms + proj_out | ✅ |✅ | Repeat experiment 0051 to compare identical LORA initialisation per additional layer
| L4 | 006* | half decoder blocks + output-side norms + proj_out | ✅ | ⏳ |
| L5 | 007* | Entire decoder blocks + output-side norms + proj_out | ❌| ❌ |
| L6 | 008* | embed_tokens + embed_positions + entire decoder + norms + proj_out | ❌ | ❌ |
