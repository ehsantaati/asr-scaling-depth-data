"""
Check how many unique audio files are needed for libriheavy validation/test
"""
import gzip
import json
from pathlib import Path
from collections import Counter

data_dir = Path("/Users/ehsantaati/Documents/GitHub/asr-data-scaling/data/libriheavy")

def get_audio_files(jsonl_gz_path):
    """Extract unique audio file paths from a cuts file"""
    audio_files = set()
    with gzip.open(jsonl_gz_path, 'rt') as f:
        for line in f:
            data = json.loads(line)
            if 'recording' in data and 'sources' in data['recording']:
                for source in data['recording']['sources']:
                    if 'source' in source:
                        audio_files.add(source['source'])
    return audio_files

# Get audio files for each split
val_files = get_audio_files(data_dir / "libriheavy_cuts_dev.jsonl.gz")
test_clean_files = get_audio_files(data_dir / "libriheavy_cuts_test_clean.jsonl.gz")
test_other_files = get_audio_files(data_dir / "libriheavy_cuts_test_other.jsonl.gz")

all_files = val_files | test_clean_files | test_other_files

print(f"Validation: {len(val_files)} unique audio files")
print(f"Test Clean: {len(test_clean_files)} unique audio files")
print(f"Test Other: {len(test_other_files)} unique audio files")
print(f"Total unique: {len(all_files)} audio files")

print("\nSample paths:")
for i, path in enumerate(sorted(all_files)[:5]):
    print(f"  {path}")
