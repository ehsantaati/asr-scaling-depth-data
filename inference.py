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
from utils import prepare_dataset, run_inference
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
    # Get raw dataset items (no dataloaders yet)
    data_opts = config.get_eval_sets()
    data_args = config.eval_dataset_args
    # Manually load dataset as iterable
    data_sets = []
    for ds_config in data_opts:
        if ds_config.name not in registry.DATASET_MAP:
             registry.register_datasets([ds_config])
        ds = registry.create_dataset(ds_config.name, data_args, verbose=True)
        data_sets.append(ds)
    
    if len(data_sets) > 1:
        eval_dataset = data.datasets.InterleaveDataset(data_sets)
    else:
        eval_dataset = data_sets[0]
        
    if config.eval_dataset_args.max_samples != -1:
        eval_dataset = data.datasets.Range(eval_dataset, config.eval_dataset_args.max_samples)


    # 5. Run Inference with Pipeline
    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    import evaluate
    from tqdm import tqdm

    logging.info(f"Starting inference with pipeline (chunk_length_s=30) and batch_size={batch_size}...")

    # Initialize pipeline
    # Note: 'device' argument for pipeline expects int (e.g., 0) or str (e.g., "cuda:0") or device object
    # If model is already on device, we might not need to pass device, but pipeline is safer with it.
    
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
    
    # We need to iterate the dataset and feed it to the pipeline.
    # The pipeline accepts an iterator of dicts with "audio" key (containing array/sampling_rate) or path.
    # Our dataset yields VoiceSample objects which have .audio (numpy array) and .text (str).
    
    def data_generator(dataset):
        for sample in dataset:
            # Pipeline expects dict with "raw" audio or "array" and "sampling_rate"
            yield {
                "raw": sample.audio, 
                "sampling_rate": sample.sample_rate
            }

    # Collect references separately since generator is consumed
    references = []
    # We'll need a way to correspond predictions to references. 
    # The pipeline is an iterator, so we can iterate both.
    # But pipeline(generator) yields results in order.
    
    # Let's iterate dataset once to yield to pipeline AND store reference.
    # NOTE: This assumes sequential execution. If pipeline is async/multithreaded prefetching, 
    # we must ensure order is preserved. HF pipeline preserves order.
    
    # Better approach:
    # Create a list of references as we yield to the pipeline? 
    # No, we can't yield and append to list in the same generator easily if pipeline consumes it eagerly.
    # But we can wrap the generator.

    # Collect references first to avoid multiprocessing side-effect issues
    logging.info("Collecting references...")
    references = []
    # Since we might be using a Range dataset or just want to be safe, let's just collect them.
    # If the dataset is huge and streaming, this doubles the I/O cost, but it guarantees correctness.
    # For cached datasets (streaming=False), this is instant.
    dataset_audios = []
    for sample in eval_dataset:
        references.append(sample.text)
        # We can also pre-collect audios if memory permits, but that might be OOM for large sets.
        # Generating a lightweight iterable for the pipeline is better.

    def input_generator():
        for sample in eval_dataset:
            yield {"raw": sample.audio, "sampling_rate": sample.sample_rate}

    generated_text = []

    # Run pipeline
    # We iterate over the output of the pipeline
    # Generate_kwargs can be used to pass language if needed, e.g. generate_kwargs={"language": "en"}
    # Note: len(eval_dataset) might be slow if not cached, but for Range/GenericDataset it should be O(1)
    total_samples = len(eval_dataset)
    for out in tqdm(pipe(input_generator(), return_timestamps=False, batch_size=batch_size, generate_kwargs={"language": "en"}), total=total_samples, desc="Inference"):
        generated_text.append(out["text"])

    # Normalization
    normalizer = EnglishTextNormalizer({})
    predictions = [normalizer(t) for t in generated_text]
    references = [normalizer(r) for r in references]
    
    # Compute metrics
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=predictions, references=references)
    metrics = {"wer": wer}
    
    logging.info(f"Final Inference WER: {wer}")

    # 6. Save Results
    results = {
        "model_path": model_path,
        "is_peft": is_peft,
        "metrics": metrics,
        "config_path": args.config_path
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logging.info(f"Saved results to {results_file}")
    logging.info(f"WER: {metrics['wer']}")

    # Also save predictions for inspection
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, "w") as f:
        json.dump({"predictions": predictions, "references": references}, f, indent=2)
    logging.info(f"Saved predictions to {predictions_file}")

if __name__ == "__main__":
    main()
