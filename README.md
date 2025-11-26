# asr-data-scaling


## Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your Hugging Face token:
   ```
   HF_TOKEN=hf_...
   ```

## Usage

To run the training script:

```bash
python train.py --config_path configs/debug_config.yaml
```
