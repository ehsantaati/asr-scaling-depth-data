import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import simple_parsing
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel
import evaluate
import datasets as hf_datasets
from tqdm import tqdm

# Add current directory to path to allow imports from train.py
sys.path.append(str(Path(__file__).parent))

from configs import InferenceConfig
from utils import prepare_dataset
from data import registry, text_proc, filtering, datasets as data_datasets
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

    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (optional). Defaults to config.output_dir/inference_results")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference. If not provided, uses config.batch_size or defaults to 1.")

    parser.add_argument("--use_fast_inference", action="store_true", help="Use optimized inference for short audio (<30s).")
    parser.add_argument("--num_inference_workers", type=int, default=None, help="Number of workers for fast inference.")
    
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
    
    if args.use_fast_inference:
        config.use_fast_inference = True
    if args.num_inference_workers is not None:
        config.num_inference_workers = args.num_inference_workers

    # Register datasets
    registry.register_datasets(data.configs.ALL_CONFIGS)

    # 2. Determine Model and Output Paths
    model_path = config.model_id
    logging.info(f"Using model and processor from: {model_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.output_dir / "inference_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler to logging
    file_handler = logging.FileHandler(output_dir / "inference.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Results will be saved to: {output_dir}")

    # 3. Load Model and Processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(model_path, language=config.language, task=config.task)
    
    # Load model
    # We assume the model at model_path is the full merged model
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


    # 5. Run Inference
    if config.use_fast_inference:
        metrics, predictions, references = run_inference_map(
             model,
             processor,
             eval_dataset,
             batch_size=batch_size,
             language=config.language,
             num_workers=config.num_inference_workers
        )
    else:
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
        args.config_path,
        dataset_name=f"{'+'.join([d['name'] for d in config.eval_sets])}_{config.eval_dataset_args.split}"
    )


def save_inference_results(
    output_dir,
    metrics,
    predictions,
    references,
    model_path,
    config_path=None,
    dataset_name=None
):
    # Ensure output_dir is Path
    output_dir = Path(output_dir)
    
    results = {
        "model_path": str(model_path),
        "metrics": metrics,
        "config_path": str(config_path) if config_path else None,
        "dataset_name": dataset_name,
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

def run_inference_map(
    model,
    processor,
    dataset,
    batch_size=16,
    language="en",
    num_workers=4
):
    logging.info(f"Starting inference with map (fast mode) and batch_size={batch_size}, num_workers={num_workers}...")
    
    # 1. Unwrap the dataset to get the underlying HF dataset
    # We expect dataset to be a prepare_dataset result, which might be Range or GenericDataset or InterleaveDataset.
    # For now, let's assume simple structure and try to find the GenericDataset or HF dataset.
    
    hf_ds = None
    
    
    # Support for Range dataset limit
    max_samples = None
    
    # Helper to unwrap and find range limit
    def unwrap_finding_limit(ds):
        nonlocal max_samples
        
        if isinstance(ds, data_datasets.Range):
            if max_samples is None:
                max_samples = ds._length
                logging.info(f"Found dataset limit (Range): {max_samples}")
            return unwrap_finding_limit(ds._dataset)
        
        if isinstance(ds, data_datasets.VoiceDataset):
             return unwrap_finding_limit(ds._dataset)
        
        if hasattr(ds, "_dataset"):
            return unwrap_finding_limit(ds._dataset)
        
        return ds

    # Unwrap GenericDataset or Wrapper to get HF dataset
    inner = unwrap_finding_limit(dataset)
    
    # Needs explicit import for isinstance check on data.datasets above, ensuring import is correct.
    # We added 'import data.datasets as data_datasets' ? No, 'from data import registry, ...'
    # Actually 'data.datasets' available if imported 'data.datasets' or 'from data import datasets'
    # Let's fix import first.
    
    hf_ds = None
    if isinstance(inner, hf_datasets.Dataset) or isinstance(inner, hf_datasets.IterableDataset):
        hf_ds = inner
    else:
         logging.warning(f"Could not unwrap to HF dataset, found {type(inner)}. Fallback might fail.")
         hf_ds = inner
    
    # Apply limit if found
    if max_samples is not None and hf_ds is not None:
        logging.info(f"Applying dataset limit to HF dataset: .take({max_samples})")
        hf_ds = hf_ds.take(max_samples)
             
            
    if hf_ds is None:
         raise ValueError("Could not extract underlying Hugging Face dataset for .map() operations.")
         
    # 2. Map function
    def map_to_pred(batch):
        # Batch is a dict of lists
        audio_arrays = [x['array'] for x in batch["audio"]]
        sampling_rates = [x['sampling_rate'] for x in batch["audio"]]

        # Check audio length
        for i, (audio, sr) in enumerate(zip(audio_arrays, sampling_rates)):
            duration = len(audio) / sr
            if duration >= 30.0:
                logging.warning(f"Audio sample {i} in batch has duration {duration:.2f}s, which is >= 30s. Fast inference might fail or truncate.")
        
        # Processor expects single item or list. 
        # Check sampling rate consistency
        assert all(sr == 16000 for sr in sampling_rates), "All sampling rates must be 16000"
        
        input_features = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding="max_length").input_features
        
        # Normalize reference text
        
        # We must replicate text normalization or ensure hf_ds has 'text' field.
        # Spgispeech_2 has 'transcript' and 'raw_transcript'.
        # GenericDataset configures `transcript_field`.
        
        # The prompt code assumes `batch['text']`.
        # We need to know the text column.
        # We can try to guess or use the standard `text` if available.
        
        ref_texts = []
        ref_texts = []
        if "transcript" in batch: 
            raw_texts = batch["transcript"]
        elif "text" in batch:
            raw_texts = batch["text"]
        else:
            raise ValueError(f"Batch samples must contain 'transcript' or 'text' fields. Available keys: {list(batch.keys())}")
        
        # Apply preprocessing to references
        clean_refs = []
        for t in raw_texts:
            clean_refs.append(text_proc.format_asr_text(t))
        
        raw_texts = clean_refs
            
        # Normalize
        # We can use processor.tokenizer._normalize if available or normalizer
        # In current code:
        # Let's simple use the normalizer later on the full list to be consistent with pipeline flow.
        # But we need to store it.
        batch["reference"] = raw_texts 
        
        # Check model dtype
        input_features = input_features.to(device=model.device, dtype=model.dtype)
        
        # Generate
        with torch.no_grad():
             generated_ids = model.generate(input_features)
        
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        batch["prediction"] = transcriptions
        return batch

    # 3. Apply map
    # Ensure model is on GPU
    # map_to_pred moves input to device, so model should be there.
    
    # Enforce 30s limit for fast inference
    logging.info("Fast inference enabled: Filtering out audio samples > 30.0s and empty/invalid text")
    
    class FilterTracker:
        def __init__(self):
            self.total = 0
            self.passed = 0
            self.normalizer = EnglishTextNormalizer({})
            
        def __call__(self, sample):
            self.total += 1
            keep = filtering.is_valid_sample(sample, max_duration=30.0)
            
            if keep:
                # Additional check: normalized text must be non-empty
                # Extract text similar to how map_to_pred does it
                raw_text = sample.get("transcript") or sample.get("text")
                if raw_text:
                    try:
                        # is_valid_sample already checks if format_asr_text is non-empty, 
                        # but we need to check if *normalized* is non-empty.
                        formatted = text_proc.format_asr_text(raw_text)
                        normalized = self.normalizer(formatted)
                        if len(normalized) == 0:
                            keep = False
                    except Exception:
                        keep = False
                else:
                    keep = False

            if keep:
                self.passed += 1
            return keep

    tracker = FilterTracker()
    hf_ds = hf_ds.filter(tracker)


    logging.info("Running inference using dataset.map...")
    
    map_kwargs = {
        "batched": True,
        "batch_size": batch_size,
        "remove_columns": hf_ds.column_names
    }
    
    if not isinstance(hf_ds, hf_datasets.IterableDataset):
        if num_workers is not None and num_workers > 0:
            map_kwargs["num_proc"] = num_workers

    result_ds = hf_ds.map(map_to_pred, **map_kwargs)
    
    # 4. Extract results
    if isinstance(result_ds, hf_datasets.IterableDataset):
        predictions = []
        references = []
        total_samples = None
        try:
             total_samples = len(dataset)
        except:
             pass
             
        for sample in tqdm(result_ds, total=total_samples, desc="Fast Inference"):
             predictions.append(sample["prediction"])
             references.append(sample["reference"])
    else:
        predictions = result_ds["prediction"]
        references = result_ds["reference"]
        
    # Log filtering statistics
    if tracker.total > 0:
        filtered = tracker.total - tracker.passed
        logging.info(f"Filtering Summary: Processed {tracker.total} samples. Kept {tracker.passed}, Filtered {filtered} ({filtered/tracker.total:.2%})")
    else:
         logging.warning("Filtering Summary: No samples processed by filter (cached or empty?)")
    
    # Normalization (Post-processing)
    normalizer = EnglishTextNormalizer({})
    
    predictions_norm = [normalizer(t) for t in predictions]
    references_norm = [normalizer(r) for r in references]
    
    # Compute metrics
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=predictions_norm, references=references_norm)
    logging.info(f"Final Inference WER: {wer}")

    return {"wer": wer}, predictions_norm, references_norm

if __name__ == "__main__":
    main()
