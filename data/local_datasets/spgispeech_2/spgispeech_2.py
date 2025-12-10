
import json
import os
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

        tar_path = os.path.join(data_dir, "audios.tar.gz")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "json_path": os.path.join(data_dir, "train.json"),
                    "archive_iterator": dl_manager.iter_archive(tar_path),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "json_path": os.path.join(data_dir, "dev.json"),
                    "archive_iterator": dl_manager.iter_archive(tar_path),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "json_path": os.path.join(data_dir, "test.json"),
                    "archive_iterator": dl_manager.iter_archive(tar_path),
                },
            ),
        ]

    def _generate_examples(self, json_path, archive_iterator):
        """Yields examples."""
        # Load JSON metadata
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        metadata_map = {}
        # Keys in the column-oriented JSON (e.g., "160602")
        keys = list(data["call_id"].keys())
        
        for k in keys:
            call_id = str(data["call_id"][k])
            snippet_id = str(data["snippet_id"][k])
            
            # Construct record
            record = {
                "call_id": call_id,
                "snippet_id": snippet_id,
                "raw_transcript": data["raw_transcript"][k],
                "speakers": data["speakers"][k],
                "wav_filesize": data["wav_filesize"][k],
                "transcript": data["raw_transcript"][k], 
            }
            metadata_map[(call_id, snippet_id)] = record

        # Iterate over tarball
        for path, f in archive_iterator:
            # Path format: ./audio_files/{call_id}/{snippet_id}.wav 
            # or audio_files/{call_id}/{snippet_id}.wav
            
            # Normalize path
            # Example encountered: ./audio_files/2617/0.wav
            
            parts = path.strip("./").split("/")
            if len(parts) >= 3 and parts[-3] == "audio_files" and parts[-1].endswith(".wav"):
                # parts[-2] is call_id, parts[-1] is snippet_id.wav
                call_id = parts[-2]
                snippet_id = os.path.splitext(parts[-1])[0]
                
                key = (call_id, snippet_id)
                if key in metadata_map:
                    record = metadata_map[key]
                    yield f"{call_id}_{snippet_id}", {
                        "call_id": record["call_id"],
                        "snippet_id": record["snippet_id"],
                        "audio": {"path": path, "bytes": f.read()},
                        "transcript": record["transcript"],
                        "raw_transcript": record["raw_transcript"],
                        "speakers": record["speakers"],
                        "wav_filesize": record["wav_filesize"],
                    }

