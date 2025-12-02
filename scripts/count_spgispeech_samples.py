import json
import os
from datasets import load_dataset
from tqdm import tqdm

def count_samples():
    subsets = ["S", "M", "L"]
    results = {}

    for subset in subsets:
        print(f"Processing subset: {subset}")
        try:
            # Load dataset in streaming mode
            ds = load_dataset(
                "kensho/spgispeech", 
                subset, 
                streaming=True, 
                trust_remote_code=True
            )
            
            subset_counts = {}
            for split, split_ds in ds.items():
                print(f"  Counting samples in split: {split}")
                # Count samples by iterating
                count = sum(1 for _ in tqdm(split_ds, desc=f"{subset}-{split}"))
                subset_counts[split] = count
                print(f"    Count: {count}")
            
            results[subset] = subset_counts
            
        except Exception as e:
            print(f"Error processing subset {subset}: {e}")
            results[subset] = {"error": str(e)}

    # Save results to JSON
    output_file = "spgispeech_samples.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    count_samples()
