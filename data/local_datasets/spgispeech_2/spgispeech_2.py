
import json
import os
import tarfile
import glob
import datasets

_DESCRIPTION = """\
SPGISpeech 2 Dataset.
https://datasets.kensho.com/datasets/spgispeech2
"""

# Root of the extracted corpus, containing shards/{train,validation,test}/*.tar and
# alignment_files/{call_id}/{snippet_id}.json. Override with SPGISPEECH2_ROOT; the
# builder's own `data_dir` config option still takes precedence when supplied.
SPGISPEECH2_ROOT = os.environ.get("SPGISPEECH2_ROOT", "")

class Spgispeech2(datasets.GeneratorBasedBuilder):
    """SPGISpeech 2 dataset."""

    VERSION = datasets.Version("1.0.1")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "call_id": datasets.Value("string"),
                    "snippet_id": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16000),
                    "transcript": datasets.Value("string"),
                    "raw_transcript": datasets.Value("string"),
                    "speakers": datasets.Sequence(datasets.Value("int32")),
                    "wav_filesize": datasets.Value("float32"),
                }
            ),
            supervised_keys=("audio", "transcript"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = self.config.data_dir
        if not data_dir:
            data_dir = SPGISPEECH2_ROOT

        shards_dir = os.path.join(data_dir, "shards")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "shard_paths": sorted(glob.glob(os.path.join(shards_dir, "train", "*.tar"))),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "shard_paths": sorted(glob.glob(os.path.join(shards_dir, "validation", "*.tar"))),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "shard_paths": sorted(glob.glob(os.path.join(shards_dir, "test", "*.tar"))),
                },
            ),
        ]

    def _generate_examples(self, shard_paths):
        """Yields examples from shards."""
        # Optional: Support for multi-worker loading
        try:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                shard_paths = shard_paths[worker_info.id :: worker_info.num_workers]
        except ImportError:
            pass

        import soundfile as sf
        import io
        import numpy as np
        
        for shard_path in shard_paths:
            with tarfile.open(shard_path, "r|*") as tar:
                meta_buffer = {}
                
                for member in tar:
                    if member.name.endswith(".json"):
                        f = tar.extractfile(member)
                        if f:
                            key = member.name[:-5]
                            meta_buffer[key] = json.load(f)
                            
                    elif member.name.endswith(".wav"):
                        key = member.name[:-4]
                        if key in meta_buffer:
                            metadata = meta_buffer.pop(key)
                            f = tar.extractfile(member)
                            if not f: continue
                            
                            wav_bytes = f.read()
                            
                            # Check duration
                            try:
                                audio_array, sample_rate = sf.read(io.BytesIO(wav_bytes))
                                duration = len(audio_array) / sample_rate
                            except Exception as e:
                                print(f"Error decoding {key}: {e}")
                                continue

                            if duration <= 30.0:
                                yield key, {
                                    "call_id": metadata["call_id"],
                                    "snippet_id": metadata["snippet_id"],
                                    "audio": {"path": member.name, "bytes": wav_bytes},
                                    "transcript": metadata.get("transcript", ""),
                                    "raw_transcript": metadata.get("raw_transcript", ""),
                                    "speakers": metadata.get("speakers", []),
                                    "wav_filesize": metadata.get("wav_filesize", 0.0),
                                }
                            else:
                                # Split long audio
                                call_id = metadata.get("call_id")
                                snippet_id = metadata.get("snippet_id")
                                chunks = self._split_sample(audio_array, sample_rate, metadata, call_id, snippet_id)
                                
                                for i, chunk in enumerate(chunks):
                                    chunk_key = f"{key}_part{i:03d}"
                                    
                                    # Encode chunk to wav bytes
                                    out_bio = io.BytesIO()
                                    sf.write(out_bio, chunk["audio"], sample_rate, format='WAV')
                                    chunk_bytes = out_bio.getvalue()
                                    
                                    yield chunk_key, {
                                        "call_id": metadata["call_id"],
                                        "snippet_id": metadata["snippet_id"],
                                        "audio": {"path": f"{chunk_key}.wav", "bytes": chunk_bytes},
                                        "transcript": chunk["transcript"],
                                        "raw_transcript": chunk["transcript"], # use same for now
                                        "speakers": metadata.get("speakers", []),
                                        "wav_filesize": float(len(chunk_bytes)),
                                    }

    def _split_sample(self, audio_array, sample_rate, metadata, call_id, snippet_id):
        # Locate alignment
        # Structure is alignment_files/{call_id}/{snippet_id}.json under the corpus root.
        align_path = os.path.join(
            SPGISPEECH2_ROOT, "alignment_files", str(call_id), f"{snippet_id}.json"
        )

        if not os.path.exists(align_path):
             # Fallback: maybe just return original or skip?
             # If we return original > 30s it will crash training potentially.
             # Let's skip or try to force split blindly? Blind split is bad for ASR.
             return []
             
        try:
             with open(align_path, "r") as f:
                 alignment_data = json.load(f)
        except:
             return []
             
        # Normalize alignment data to list of words
        # The file format is { "0": {word...}, "1": {word...} }
        words = []
        for k, v in alignment_data.items():
            if isinstance(v, dict):
                words.append(v)
        words.sort(key=lambda x: x['start_time'])
        
        return self._perform_split(audio_array, sample_rate, words)

    def _perform_split(self, audio_array, sample_rate, words, max_duration=28.0):
        total_duration = len(audio_array) / sample_rate
        chunks = []
        current_start_time = 0.0
        current_word_idx = 0
        
        while current_start_time < total_duration:
            target_end_time = current_start_time + max_duration
            
            if total_duration - current_start_time <= max_duration:
                split_time = total_duration
                next_start_time = total_duration
                chunk_words = words[current_word_idx:]
                current_word_idx = len(words)
            else:
                # Find best split point
                candidates = [i for i in range(current_word_idx, len(words)) 
                              if words[i]['end_time'] <= target_end_time]
                
                if not candidates:
                    # Force split
                    split_time = target_end_time
                    chunk_words = []
                    next_start_time = split_time
                else:
                    last_word_idx = candidates[-1]
                    split_time = words[last_word_idx]['end_time']
                    
                    # Try to extend into silence
                    if last_word_idx + 1 < len(words):
                        next_start = words[last_word_idx+1]['start_time']
                        if next_start > split_time:
                            mid_point = (split_time + next_start) / 2
                            if mid_point - current_start_time <= max_duration:
                                split_time = mid_point
                                
                    next_start_time = split_time
                    chunk_words = words[current_word_idx : last_word_idx+1]
                    current_word_idx = last_word_idx + 1
            
            # Extract
            start_sample = int(current_start_time * sample_rate)
            end_sample = int(split_time * sample_rate)
            
            # Clamp
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_array), end_sample)
            
            if end_sample > start_sample:
                chunk_audio = audio_array[start_sample:end_sample]
                chunk_text = " ".join([w['word'] for w in chunk_words])
                
                if len(chunk_audio) > 0 and len(chunk_text.strip()) > 0:
                    chunks.append({
                        "audio": chunk_audio,
                        "transcript": chunk_text
                    })
            
            current_start_time = next_start_time
            if current_word_idx >= len(words) and current_start_time >= total_duration:
                break
                
        return chunks
