
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

        audio_path = os.path.join(data_dir, "audio_files")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "json_path": os.path.join(data_dir, "train.json"),
                    "audio_path": audio_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "json_path": os.path.join(data_dir, "dev.json"),
                    "audio_path": audio_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "json_path": os.path.join(data_dir, "test.json"),
                    "audio_path": audio_path,
                },
            ),
        ]

    def _generate_examples(self, json_path, audio_path):
        """Yields examples."""
        # Load JSON metadata
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Keys in the column-oriented JSON (e.g., "160602")
        keys = list(data["call_id"].keys())
        
        for k in keys:
            call_id = str(data["call_id"][k])
            snippet_id = str(data["snippet_id"][k])
            
            wav_path = os.path.join(audio_path, call_id, f"{snippet_id}.wav")
            
            if os.path.exists(wav_path):
                yield f"{call_id}_{snippet_id}", {
                    "call_id": call_id,
                    "snippet_id": snippet_id,
                    "audio": wav_path,
                    "transcript": data["raw_transcript"][k],
                    "raw_transcript": data["raw_transcript"][k],
                    "speakers": data["speakers"][k],
                    "wav_filesize": data["wav_filesize"][k],
                }

