"""
Inspect libriheavy dataset structure to understand how to extract text and audio
"""
from datasets import load_dataset

print("Loading libriheavy dataset (streaming)...")
dataset = load_dataset("pkufool/libriheavy", split="validation", streaming=True)

print("\nInspecting first sample...")
sample = next(iter(dataset))

print("\nAvailable fields:")
for key, value in sample.items():
    print(f"  {key}: {type(value).__name__}")
    if isinstance(value, (str, int, float)):
        print(f"    Value: {value}")
    elif isinstance(value, dict):
        print(f"    Keys: {list(value.keys())}")
    elif isinstance(value, list) and len(value) > 0:
        print(f"    Length: {len(value)}, First item type: {type(value[0]).__name__}")

print("\n" + "="*60)
print("Full sample structure:")
print("="*60)
import json
print(json.dumps(sample, indent=2, default=str))
