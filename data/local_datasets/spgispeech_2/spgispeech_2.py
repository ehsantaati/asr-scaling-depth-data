
import json
import os
import tarfile
import glob
import datasets

_DESCRIPTION = """\
SPGISpeech 2 Dataset.
https://datasets.kensho.com/datasets/spgispeech2
"""

class Spgispeech2(datasets.GeneratorBasedBuilder):
    """SPGISpeech 2 dataset."""

    VERSION = datasets.Version("1.0.0")

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
            data_dir = "/mnt/data/SPGISPeech_2"

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
                # Distribute shards among workers
                # Worker 0 gets 0, 4, 8...
                # Worker 1 gets 1, 5, 9...
                shard_paths = shard_paths[worker_info.id :: worker_info.num_workers]
        except ImportError:
            pass

        for shard_path in shard_paths:
            with tarfile.open(shard_path, "r|*") as tar:
                # Buffer for metadata
                meta_buffer = {}
                
                for member in tar:
                    if member.name.endswith(".json"):
                        f = tar.extractfile(member)
                        if f:
                            key = member.name[:-5] # remove .json
                            meta_buffer[key] = json.load(f)
                            
                    elif member.name.endswith(".wav"):
                        key = member.name[:-4] # remove .wav
                        if key in meta_buffer:
                            metadata = meta_buffer.pop(key)
                            f = tar.extractfile(member)
                            if f:
                                yield key, {
                                    "call_id": metadata["call_id"],
                                    "snippet_id": metadata["snippet_id"],
                                    "audio": {"path": member.name, "bytes": f.read()},
                                    "transcript": metadata["transcript"],
                                    "raw_transcript": metadata["raw_transcript"],
                                    "speakers": metadata["speakers"],
                                    "wav_filesize": metadata["wav_filesize"],
                                }

