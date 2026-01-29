|ID|Experiment_ID|Target Modules|Status|Notes|
|---|---|---|---|---|
| - | 000 | - | ✅ | Whisper-medium baseline vanilla Inference (Base 1)|
| - | 001 | proj_out | ✅ | Standard fine-tuning (Base 2)|
| L0 | 002* | proj_out | ✅ | Lora Rank=64, lora_alpha=128|
| L1 | 003* | proj_out + output-side norms | ⏳ | |
| L2 | 004* | last 1 decoder block + output-side norms + proj_out | ❌ | |
| L3 | 005* | last 2 decoder blocks + output-side norms + proj_out | ❌ | |
| L4 | 006* | last 4 decoder blocks + output-side norms + proj_out | ❌ | |
| L5 | 007* | All decoder | ❌| |
