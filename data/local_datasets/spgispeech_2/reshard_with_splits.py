
import argparse
import io
import json
import os
import tarfile
import logging
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_alignment_path(base_alignment_dir, call_id):
    """
    Locates the alignment file for a given call_id.
    Assumes structure: base_dir / {something} / {call_id}.json
    Or base_dir / {call_id}.json
    
    Based on user observation: alignment_files/0/0.json
    It seems the parent folder is a prefix or hash of the call_id.
    Let's try a few strategies.
    
    Strategy 1: Check if call_id is numeric and folder is suffix/prefix
    Strategy 2: Check direct match
    Strategy 3: Check glob (slow)
    
    Given the example 'alignment_files/0/0.json', maybe the folder is the call_id itself if call_id is '0'? 
    Or folder is first/last digit?
    """
    # Try exact match first (flat structure)
    p = os.path.join(base_alignment_dir, f"{call_id}.json")
    if os.path.exists(p):
        return p
        
    # Try nested by first digit/char?
    # If call_id is "12345", maybe "1/12345.json" or "5/12345.json"?
    # The user example was "0/0.json". "0" starts with "0".
    
    prefix = str(call_id)[0]
    p = os.path.join(base_alignment_dir, prefix, f"{call_id}.json")
    if os.path.exists(p):
        return p
        
    # Try nested by last digit?
    suffix = str(call_id)[-1]
    p = os.path.join(base_alignment_dir, suffix, f"{call_id}.json")
    if os.path.exists(p):
        return p

    return None

def split_audio_and_text(audio_array, sample_rate, alignment_data, max_duration=28.0):
    """
    Splits audio and text into chunks < max_duration.
    
    Args:
        audio_array: numpy array of audio
        sample_rate: sampling rate (int)
        alignment_data: list of dicts with 'start_time', 'end_time', 'word'
        max_duration: max duration in seconds for a chunk
        
    Returns:
        List of dicts: [{'audio': np_array, 'transcript': str}, ...]
    """
    total_duration = len(audio_array) / sample_rate
    chunks = []
    
    # Sort alignment by start time just in case
    words = sorted(alignment_data, key=lambda x: x['start_time'])
    
    current_start_time = 0.0
    current_word_idx = 0
    
    while current_start_time < total_duration:
        target_end_time = current_start_time + max_duration
        
        # If the remainder is small enough, just take it
        if total_duration - current_start_time <= max_duration:
            split_time = total_duration
            next_start_time = total_duration # Finish loop
            
            # Collect words in this range
            chunk_words = [w for w in words[current_word_idx:]]
            # Advance index
            current_word_idx = len(words) 
        else:
            # Find the best split point before target_end_time
            # We want to split at a word boundary that is closest to target_end_time
            # but definitely before it (or slightly after if necessary to avoid cutting a word, but better before)
            
            # Find words that start before the target end time
            candidates = []
            for i in range(current_word_idx, len(words)):
                if words[i]['end_time'] <= target_end_time:
                    candidates.append(i)
                else:
                    break
            
            if not candidates:
                # No words fit? This implies a single word > 30s or empty space.
                # Force split at max_duration
                split_time = target_end_time
                chunk_words = [] # Should handle edge case
                next_start_time = split_time
            else:
                # Pick the last candidate as the break point
                last_word_idx = candidates[-1]
                
                # The split time should be after this word ends, but before the next word starts
                split_time = words[last_word_idx]['end_time']
                
                # Check gap to next word
                if last_word_idx + 1 < len(words):
                    next_word_start = words[last_word_idx + 1]['start_time']
                    # Ideally split in the middle of silence
                    if next_word_start > split_time:
                        split_time = (split_time + next_word_start) / 2.0
                        # Ensure we don't exceed max_duration relative to current_start_time
                        if split_time - current_start_time > max_duration:
                             split_time = words[last_word_idx]['end_time']
                
                next_start_time = split_time
                chunk_words = words[current_word_idx : last_word_idx + 1]
                current_word_idx = last_word_idx + 1

        # Extract Audio
        start_sample = int(current_start_time * sample_rate)
        end_sample = int(split_time * sample_rate)
        
        # Clamp
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_array), end_sample)
        
        if end_sample <= start_sample:
             break

        chunk_audio = audio_array[start_sample:end_sample]
        
        # Extract Text
        chunk_text = " ".join([w['word'] for w in chunk_words])
        
        if len(chunk_audio) > 0:
            chunks.append({
                "audio": chunk_audio,
                "transcript": chunk_text
            })
            
        current_start_time = next_start_time
        
    return chunks

def process_tar_shard(input_path, output_path, alignment_dir):
    """
    Reads a tar shard, processes every sample, and writes to a new tar shard.
    """
    with tarfile.open(input_path, "r|*") as input_tar, tarfile.open(output_path, "w") as output_tar:
        
        # We need to buffer members to find pairs of .json and .wav
        # Since it's a stream, we might need a buffer or rely on order.
        # Usually webdataset shards store .json then .wav or vice versa immediately.
        # To be robust, let's read the whole shard into memory if it fits? 
        # Shards are usually 100MB-1GB. Might be risky.
        # Better: iterate and buffer by key.
        
        buffer = {} # key -> {json: member, wav: member, json_data: ..., wav_data: ...}
        
        for member in input_tar:
            if member.name.endswith(".json"):
                f = input_tar.extractfile(member)
                if f:
                    data = json.load(f)
                    key = member.name.rsplit('.', 1)[0]
                    if key not in buffer: buffer[key] = {}
                    buffer[key]["json_data"] = data
                    
            elif member.name.endswith(".wav"):
                f = input_tar.extractfile(member)
                if f:
                    content = f.read()
                    key = member.name.rsplit('.', 1)[0]
                    if key not in buffer: buffer[key] = {}
                    buffer[key]["wav_bytes"] = content

            # Check if we have a full pair
            # Note: This simple logic assumes keys come somewhat together. 
            # If not, buffer grows.
            # We can flush processed keys.
            
            finished_keys = []
            for key, item in buffer.items():
                if "json_data" in item and "wav_bytes" in item:
                    # PROCESS PAIR
                    process_sample(key, item, output_tar, alignment_dir)
                    finished_keys.append(key)
            
            for k in finished_keys:
                del buffer[k]

def process_sample(key, item, output_tar, alignment_dir):
    meta = item["json_data"]
    wav_bytes = item["wav_bytes"]
    
    call_id = meta.get("call_id")
    snippet_id = meta.get("snippet_id")
    
    # helper for writing to tar
    def write_to_tar(k, valid_wav_bytes, valid_meta):
        # Write JSON
        json_bytes = json.dumps(valid_meta).encode("utf-8")
        ti = tarfile.TarInfo(name=f"{k}.json")
        ti.size = len(json_bytes)
        output_tar.addfile(ti, io.BytesIO(json_bytes))
        
        # Write Wav
        ti = tarfile.TarInfo(name=f"{k}.wav")
        ti.size = len(valid_wav_bytes)
        output_tar.addfile(ti, io.BytesIO(valid_wav_bytes))

    # Check duration
    # We need to decode to check duration and potentially split
    try:
        audio_array, sample_rate = sf.read(io.BytesIO(wav_bytes))
    except Exception as e:
        logging.error(f"Failed to decode audio for {key}: {e}")
        return

    duration = len(audio_array) / sample_rate
    
    if duration <= 30.0:
        # Pass through
        write_to_tar(key, wav_bytes, meta)
        return
        
    # Proceed to split
    # 1. Find alignment
    if not call_id:
        logging.warning(f"No call_id for {key}, cannot look up alignment. Skipping.")
        return
        
    align_path = get_alignment_path(alignment_dir, call_id)
    if not align_path:
        logging.warning(f"Alignment file not found for call_id {call_id} (key {key}). Skipping or keeping original?")
        # If we keep original, it will be filtered out by training. 
        # Better to skip or keep? Let's skip to be clean, or check provided logic.
        # User constraint: "we cannot increase max audio length due to whisper limitation"
        # So keeping it is useless.
        return

    try:
        with open(align_path, "r") as f:
            full_alignment = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read alignment {align_path}: {e}")
        return
        
    # Convert flat dict to sorted list of words
    # keys are string indices "0", "1", ...
    # We need to robustly handle the keys.
    all_words = []
    for k, v in full_alignment.items():
        try:
            # v should be a dict with 'word', 'start_time' etc.
            if isinstance(v, dict):
                v['id'] = int(k)
                all_words.append(v)
        except ValueError:
            pass # Skip non-integer keys if any
            
    # Sort by ID (sequence order) or start_time
    all_words.sort(key=lambda x: x['id'])
    
    # Match transcript to find the subsequence
    # 1. Clean transcript
    target_text = meta.get("transcript", "")
    # Simple tokenization: split by space, remove punctuation
    if not target_text:
        logging.warning(f"No transcript for {key}. Skipping.")
        return

    import string # ensure imported
    
    def tokenize(text):
        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return [w.lower() for w in text.split()]
        
    target_tokens = tokenize(target_text)
    
    if not target_tokens:
        logging.warning(f"No tokens in transcript for {key}. Skipping.")
        return

    # 2. Find subsequence in all_words
    # This is a naive search: O(N*M)
    alignment_tokens = [w['word'].lower().translate(str.maketrans('', '', string.punctuation)) for w in all_words]
    
    start_idx = -1
    
    # We try to find the start sequence
    # Heuristic: Find first token match, then check subsequent tokens
    # Since alignment might have extra words or slight mismatches, exact match is brittle.
    # But usually ASR training data is exact.
    
    for i in range(len(alignment_tokens) - len(target_tokens) + 1):
        if alignment_tokens[i] == target_tokens[0]:
            # potential match
            match = True
            for j in range(1, len(target_tokens)):
                if alignment_tokens[i+j] != target_tokens[j]:
                    match = False
                    break
            if match:
                start_idx = i
                break
                
    if start_idx == -1:
         # Fallback: fuzzy match or try to match start/end via audio duration? 
         # But we assume the alignment provided corresponds to the text.
         # Let's try matching just the first 10 tokens to find start
         logging.warning(f"Could not find exact text match for {key}. Trying partial match...")
         # Try matching first 5 words
         partial_len = min(5, len(target_tokens))
         for i in range(len(alignment_tokens) - partial_len + 1):
             if alignment_tokens[i] == target_tokens[0]:
                 match = True
                 for j in range(1, partial_len):
                     if alignment_tokens[i+j] != target_tokens[j]:
                         match = False
                         break
                 if match:
                     start_idx = i
                     break
                     
         if start_idx == -1:
             logging.warning(f"Failed to match text for {key} in alignment. Skipping.")
             return
             
    # Extract the relevant words
    # We take len(target_tokens) words starting from start_idx
    # But we might want to capture more if the duration implies it?
    # No, we only want words corresponding to the transcript.
    word_data = all_words[start_idx : start_idx + len(target_tokens)]

    
    # 2. Split
    chunks = split_audio_and_text(audio_array, sample_rate, word_data)
    
    # 3. Write chunks
    for i, chunk in enumerate(chunks):
        new_key = f"{key}_part{i:03d}"
        
        # Encode audio back to bytes (WAV)
        out_bio = io.BytesIO()
        sf.write(out_bio, chunk["audio"], sample_rate, format='WAV')
        new_wav_bytes = out_bio.getvalue()
        
        # Update metadata
        new_meta = meta.copy()
        new_meta["transcript"] = chunk["transcript"]
        new_meta["raw_transcript"] = chunk["transcript"]
        new_meta["orig_duration"] = duration
        new_meta["chunk_index"] = i
        
        write_to_tar(new_key, new_wav_bytes, new_meta)
    
    logging.info(f"Split {key} ({duration:.2f}s) into {len(chunks)} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Source shards directory (e.g. .../shards/train)")
    parser.add_argument("--dst_dir", required=True, help="Destination directory")
    parser.add_argument("--alignment_dir", required=True, help="Directory containing alignment json files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of shards to process")
    args = parser.parse_args()
    
    os.makedirs(args.dst_dir, exist_ok=True)
    
    # List all tar files
    src_files = sorted([f for f in os.listdir(args.src_dir) if f.endswith(".tar")])
    if args.limit:
        src_files = src_files[:args.limit]
    
    for fname in tqdm(src_files, desc="Processing Shards"):
        src_path = os.path.join(args.src_dir, fname)
        dst_path = os.path.join(args.dst_dir, fname)
        
        try:
           process_tar_shard(src_path, dst_path, args.alignment_dir)
        except Exception as e:
           logging.error(f"Failed to process shard {fname}: {e}")

