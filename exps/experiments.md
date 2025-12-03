# Experiment Tracking

| ID | Config File  | Type | Target Modules | Data Scale | Description | Command |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 000 | `000_inference.yaml` | Inference | - | - | Inference configuration | `./exps/run_inf.sh 000` |
| 001 | `001_tr_s_1.yaml` | Standard | `['proj_out']` | `1.0` | Baseline training configuration | `./exps/run_exp.sh 001` |


> [!NOTE]
> Please rename your config files to match the "Config File" column (e.g., `train_config.yaml` -> `001_baseline.yaml`).
