import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import transformers
import simple_parsing
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig

# Add current directory to path to allow imports from train.py
sys.path.append(str(Path(__file__).parent))

from configs import InferenceConfig
from utils import prepare_dataset
from data import registry
import data.configs

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Run inference with Whisper model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML training config file")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint (optional). If not provided, uses the model_id from config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (optional). Defaults to config.output_dir/inference_results")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference. If not provided, uses config.batch_size or defaults to 1.")
    
    args = parser.parse_args()

    # 1. Load Configuration
    logging.info(f"Loading config from {args.config_path}")
    # We use simple_parsing to load the yaml into the InferenceConfig dataclass
    try:
        config = simple_parsing.load(InferenceConfig, args.config_path)
    except AttributeError:
        # Fallback if simple_parsing.load is not directly available (older versions)
        import yaml
        with open(args.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        from simple_parsing.helpers.serialization import load_yaml
        config = load_yaml(InferenceConfig, Path(args.config_path))

    # Determine batch size
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        # Use eval_batch_size from BaseConfig as default for inference
        batch_size = config.eval_batch_size if hasattr(config, "eval_batch_size") else 1
    logging.info(f"Using batch size: {batch_size}")

    # Register datasets
    registry.register_datasets(data.configs.ALL_CONFIGS)

    # 2. Determine Model and Output Paths
    if args.checkpoint_path:
        model_path = args.checkpoint_path
        logging.info(f"Using checkpoint from CLI: {model_path}")
    elif config.checkpoint_path:
        model_path = config.checkpoint_path
        logging.info(f"Using checkpoint from config: {model_path}")
    else:
        model_path = config.model_id
        logging.info(f"Using vanilla model: {model_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.output_dir / "inference_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved to: {output_dir}")

    # 3. Load Model and Processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(config.model_id, language=config.language, task=config.task)
    
    # Load model
    # Check if it's a PEFT model or full model
    # If checkpoint_path is a directory containing adapter_config.json, it's PEFT
    is_peft = False
    if model_path and (Path(model_path) / "adapter_config.json").exists():
        is_peft = True
        logging.info("Detected PEFT adapter checkpoint.")
        # Load base model first
        model = WhisperForConditionalGeneration.from_pretrained(config.model_id)
        # Load adapters
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_path)

    if config.fp16:
        logging.info("Converting model to fp16")
        model = model.half()

    model.to(device)
    
    # 4. Prepare Evaluation Datasets
    logging.info("Preparing evaluation datasets...")
    eval_dataset = prepare_dataset(
        config.get_eval_sets(), config.eval_dataset_args
    )


    # 5. Run Inference with Pipeline
    metrics, predictions, references = run_inference_pipeline(
        model,
        processor,
        eval_dataset,
        device,
        batch_size=batch_size,
        language=config.language
    )
    
    
    # 6. Save Results
    save_inference_results(
        output_dir,
        metrics,
        predictions,
        references,
        model_path,
        args.config_path
    )


def save_inference_results(
    output_dir,
    metrics,
    predictions,
    references,
    model_path,
    config_path=None
):
    import json
    import logging
    
    # Ensure output_dir is Path
    output_dir = Path(output_dir)
    
    results = {
        "model_path": str(model_path),
        "metrics": metrics,
        "config_path": str(config_path) if config_path else None
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logging.info(f"Saved results to {results_file}")
    if "wer" in metrics:
        logging.info(f"WER: {metrics['wer']}")

    # Also save predictions for inspection
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, "w") as f:
        json.dump({"predictions": predictions, "references": references}, f, indent=2)
    logging.info(f"Saved predictions to {predictions_file}")


def run_inference_pipeline(
    model,
    processor,
    dataset,
    device,
    batch_size=1,
    language="en",
):
    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    import evaluate
    from tqdm import tqdm
    
    logging.info(f"Starting inference with pipeline (chunk_length_s=30) and batch_size={batch_size}...")

    # Initialize pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=device,
        batch_size=batch_size,
        ignore_warning=True,
        num_workers=1,
    )

    # Collect references first to avoid multiprocessing side-effect issues
    logging.info("Collecting references...")
    references = []
    for sample in dataset:
        references.append(sample.text)

    def input_generator():
        for sample in dataset:
             yield {"raw": sample.audio, "sampling_rate": sample.sample_rate}

    generated_text = []

    # Run pipeline
    # Note: len(dataset) works for mapped datasets and SizedIterableDataset, but not generic IterableDataset.
    try:
        total_samples = len(dataset)
    except TypeError:
        total_samples = None
        
    for out in tqdm(pipe(input_generator(), return_timestamps=False, batch_size=batch_size, generate_kwargs={"language": language}), total=total_samples, desc="Inference"):
        generated_text.append(out["text"])

    # Normalization
    normalizer = EnglishTextNormalizer({})
    predictions = [normalizer(t) for t in generated_text]
    references = [normalizer(r) for r in references]
    
    # Compute metrics
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=predictions, references=references)
    logging.info(f"Final Inference WER: {wer}")

    return {"wer": wer}, predictions, references

if __name__ == "__main__":
    main()
