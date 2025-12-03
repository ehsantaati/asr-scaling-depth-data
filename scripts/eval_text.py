import argparse
import json
import logging
import sys
from pathlib import Path

import evaluate
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Calculate WER from predictions JSON")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the predictions JSON file")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        logging.error(f"File not found: {json_path}")
        sys.exit(1)

    logging.info(f"Loading predictions from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "predictions" not in data or "references" not in data:
        logging.error("JSON must contain 'predictions' and 'references' keys")
        sys.exit(1)

    predictions = data["predictions"]
    references = data["references"]

    if len(predictions) != len(references):
        logging.error(f"Number of predictions ({len(predictions)}) does not match number of references ({len(references)})")
        sys.exit(1)

    logging.info(f"Loaded {len(predictions)} samples")

    # Normalization
    logging.info("Normalizing text...")
    normalizer = EnglishTextNormalizer({})
    
    norm_predictions = [normalizer(p) for p in predictions]
    norm_references = [normalizer(r) for r in references]

    # Calculate WER
    logging.info("Calculating WER...")
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=norm_predictions, references=norm_references)

    logging.info(f"WER: {wer}")
    print(f"WER: {wer}")

if __name__ == "__main__":
    main()
