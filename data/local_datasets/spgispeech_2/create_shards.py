import json
import os
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm

def write_shard(shard_path, examples):
    """Writes a list of examples to a tar shard."""
    with tarfile.open(shard_path, "w") as tar:
        for key, data in examples:
            # Write JSON metadata
            json_bytes = json.dumps(data["metadata"]).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))
            
            # Write Audio
            # We read the audio file from disk and write it to the tar
            # To avoid reading into memory, we can use the file object directly, 
            # but tarfile.addfile with fileobj needs known size.
            
            try:
                audio_path = data["audio_path"]
                file_size = os.path.getsize(audio_path)
                with open(audio_path, "rb") as f:
                    audio_info = tarfile.TarInfo(name=f"{key}.wav")
                    audio_info.size = file_size
                    tar.addfile(audio_info, fileobj=f)
            except FileNotFoundError:
                print(f"Warning: Audio file not found {audio_path}")
                continue

import io

def process_split(data_dir, output_dir, split_name, json_filename, max_count=1000, limit=None):
    """Processes a single split (train/dev/test)."""
    
    json_path = os.path.join(data_dir, json_filename)
    if not os.path.exists(json_path):
        print(f"Metadata file not found: {json_path}")
        return

    print(f"Loading metadata for {split_name} from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Keys like "160602"
    keys = list(meta["call_id"].keys())
    
    print(f"Found {len(keys)} examples in {split_name}.")
    
    if limit:
         keys = keys[:limit]
         print(f"Limiting to first {limit} examples.")

    examples_buffer = []
    shard_idx = 0
    
    split_out_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    for k in tqdm(keys, desc=f"Processing {split_name}"):
        call_id = str(meta["call_id"][k])
        snippet_id = str(meta["snippet_id"][k])
        
        # Original Audio Path
        # Construction logic from previous script: audio_files/call_id/snippet_id.wav
        audio_path = os.path.join(data_dir, "audio_files", call_id, f"{snippet_id}.wav")
        
        # Metadata record
        record = {
            "call_id": call_id,
            "snippet_id": snippet_id,
            "transcript": meta["raw_transcript"][k],
            "raw_transcript": meta["raw_transcript"][k],
            "speakers": meta["speakers"][k],
            "wav_filesize": meta["wav_filesize"][k],
        }
        
        key = f"{call_id}_{snippet_id}"
        examples_buffer.append((key, {"metadata": record, "audio_path": audio_path}))
        
        if len(examples_buffer) >= max_count:
            shard_name = f"{split_name}-{shard_idx:05d}.tar"
            shard_path = os.path.join(split_out_dir, shard_name)
            write_shard(shard_path, examples_buffer)
            examples_buffer = []
            shard_idx += 1
            
    # Write remaining
    if examples_buffer:
        shard_name = f"{split_name}-{shard_idx:05d}.tar"
        shard_path = os.path.join(split_out_dir, shard_name)
        write_shard(shard_path, examples_buffer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SPGISpeech 2 to WebDataset shards")
    parser.add_argument("--data_dir", default="/mnt/data/SPGISPeech_2", help="Source directory")
    parser.add_argument("--output_dir", default="/mnt/data/SPGISPeech_2/shards", help="Output directory for shards")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per split for testing")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of examples per shard")
    
    args = parser.parse_args()
    
    # Process Test first (smaller)
    process_split(args.data_dir, args.output_dir, "test", "test.json", max_count=args.shard_size, limit=args.limit)
    process_split(args.data_dir, args.output_dir, "validation", "dev.json", max_count=args.shard_size, limit=args.limit)
    process_split(args.data_dir, args.output_dir, "train", "train.json", max_count=args.shard_size, limit=args.limit)
