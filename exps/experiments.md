# Experiment Tracking

| ID | Config File  | Type | Target Modules | Data Scale | Description | Trianable Parameters (%) |WER(%)|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 000 | `000_inference.yaml` | Inference | - | - | Inference with Whisper-medium vanilla model | 0 |13.23|
| 001 | `001_tr_s_1.yaml` | Standard | `['proj_out']` | `1.0` | Training with standard modules (proj_out) | 6.50 |3.124|
| 002 | `002_tr_l_1.yaml` | LoRA | `['proj_out']` | `1.0` | Training with LoRA modules (proj_out) | 0.052 |5.043
| 003 | `003_tr_s_1.yaml` | Standard | `['proj_out']`<br>`['decoder.layer_norm']`<br>`['decoder.layers.11']` | `1.0` | Training with LoRA modules (proj_out) | 8.56 |-|

> [!NOTE]
> To execute the experiments, run the following commands:
> ```bash
> ./exps/run_exp.sh ID
> ```