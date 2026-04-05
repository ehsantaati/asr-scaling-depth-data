import datasets
from tqdm.auto import tqdm
import argparse
import sys
import json
import os

DATASET_CONFIGS = {
    "spgispeech_2": {
        "id": "spgispeech_2",
        "path": "/mnt/asr-data-scaling/data/local_datasets/spgispeech_2",
        "name": None,
        "split": "train"
    },
    "voxpopuli-en": {
        "id": "voxpopuli-en",
        "path": "facebook/voxpopuli",
        "name": "en",
        "split": "train"
    },
    "speechcolab-gigaspeech-m": {
        "id": "speechcolab-gigaspeech-m",
        "path": "speechcolab/gigaspeech",
        "name": "m",
        "split": "train"
    }
}

def calculate_durations(dataset_id):
    if dataset_id not in DATASET_CONFIGS:
        print(f"Error: Dataset '{dataset_id}' not found.")
        print(f"Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
        sys.exit(1)
        
    config = DATASET_CONFIGS[dataset_id]
    max_duration = 30.0

    print(f"\n{'='*50}")
    print(f"Loading {dataset_id} ...")
    
    # Load the dataset
    ds = datasets.load_dataset(
        config["path"], 
        name=config["name"],
        split=config["split"], 
        trust_remote_code=True,
        streaming=True # Use streaming to avoid downloading massive amounts of data at once if it's not local
    )
    
    total_samples = 0
    total_duration_secs = 0.0
    
    filtered_samples = 0
    filtered_duration_secs = 0.0
    
    print(f"Calculating durations for {dataset_id}...")
    
    for sample in tqdm(ds, desc=f"Processing {dataset_id}"):
        audio = sample["audio"]
        
        # Streaming returns audio as a dictionary
        array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        
        duration = len(array) / sampling_rate
        
        total_samples += 1
        total_duration_secs += duration
        
        # Filter condition used in datasets.py
        if duration <= max_duration:
            filtered_samples += 1
            filtered_duration_secs += duration

    # Compile the result
    result = {
        "dataset_name": dataset_id,
        "before_filtering": {
            "total_samples": total_samples,
            "total_duration_hours": float(f"{total_duration_secs / 3600:.4f}")
        },
        "after_filtering_30s": {
            "total_samples": filtered_samples,
            "total_duration_hours": float(f"{filtered_duration_secs / 3600:.4f}")
        }
    }
    
    print(f"\n--- Results for {dataset_id} ---")
    print(json.dumps(result, indent=4))
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset durations before and after 30s filtering.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="all",
        help=f"The dataset to process. Choices: {', '.join(DATASET_CONFIGS.keys())}, or 'all'"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="duration_results.json",
        help="Path where the output JSON will be saved."
    )
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.dataset == "all":
        datasets_to_run = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_run = [args.dataset]
        
    for ds_id in datasets_to_run:
        result = calculate_durations(ds_id)
        all_results[ds_id] = result
        
    # Write aggregated results to the output JSON file
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as f:
        json.dumps(all_results, indent=4)
        json.dump(all_results, f, indent=4)
        
    print(f"\n{'-'*50}")
    print(f"Results successfully written to {output_path}")
