from .. import types

SPGISPEECH_2_CONFIG = types.DatasetConfig(
    name="spgispeech_2",
    path="/mnt/asr-data-scaling/data/local_datasets/spgispeech_2",
    transcript_field="transcript",
    audio_field="audio",
    streaming=False,
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=512724),
        types.DatasetSplitConfig(
            name="validation", num_samples=6195, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=6932),
    ],
)

configs = [
    SPGISPEECH_2_CONFIG,
]
