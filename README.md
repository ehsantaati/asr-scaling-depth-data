# ASR Depth-Data Scaling
This repository accompanies the paper **Data and Capacity Trade-offs in In-Domain Parameter-Efficient Adaptation of Whisper**. The work presents a systematic empirical study of how adaptation depth, training data availability, and optimization budget jointly influence performance in end-to-end automatic speech recognition (ASR). The manuscript is currently under review.

Using Whisper-based models, we evaluate layer-wise decoder fine-tuning under both full-parameter and parameter-efficient (LoRA) settings across three datasets: SPGISpeech 2.0, VoxPopuli, and GigaSpeech. The study reveals that performance is not governed by data or model capacity alone, but by their interaction with optimization, where deeper adaptation improves performance but becomes unstable under limited data, and parameter-efficient methods can achieve competitive results with significantly fewer trainable parameters.

The findings provide practical guidance for efficient and robust domain adaptation of ASR systems, particularly in scenarios with constrained data and computational resources.

## Method Overview

<img src="imgs/overview.png" width="700"/>

*Layer-wise Whisper's decoder adaptation under varying data and optimization regimes.*

## Setup

Open `.env` and add your Hugging Face token:
```
HF_TOKEN=hf_...
```

## Configuration Templates

The `configs/` directory contains templates that demonstrate all available configuration options. These templates serve as starting points for various experiments.

* `configs/template_full_ft.yaml`: Standard configuration for full model fine-tuning. Includes settings for dataset selection, data scaling, training hyperparameters, and layer-specific regularization.
* `configs/template_lora_ft.yaml`: Configuration for Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA). Includes options to define LoRA rank, alpha, dropout, block initialization, and specific target projections.
* `configs/template_inference.yaml`: Configuration dedicated to model evaluation. Includes settings for evaluation datasets, fast inference optimizations, and mixed precision.

### Compute-Matched Training

The codebase supports compute-matched training when scaling datasets. By setting `num_epochs: 0` in the training configuration, the script calculates `max_steps` based on the size of the full, unpartitioned dataset.

This ensures that when training on a smaller fraction of the dataset, the model still trains for the exact same number of steps (compute) as one epoch on 100% of the data. For example, training on a 10% data fraction with `num_epochs: 0` results in effectively training on that 10% for 10 epochs.

## Usage

You can run experiments and inference using the provided bash scripts. They simplify execution and allow you to specify the target GPU device.

### Standard Training

To run a single training experiment, use `run_exp.sh`:
```bash
./scripts/run_exp.sh configs/template_full_ft.yaml [device_id]
```

### Inference Only

To run inference using an inference configuration, use `run_inf.sh`:
```bash
./scripts/run_inf.sh configs/template_inference.yaml [device_id]
```

### Multi-Seed Training

To run experiments across multiple random seeds, use `run_ms_exp.sh`:
```bash
./scripts/run_ms_exp.sh configs/template_full_ft.yaml [device_id]
```
