from .. import types

SPEECHCOLAB_GIGASPEECH_BASE = types.DatasetConfig(
    name="speechcolab-gigaspeech",
    path="speechcolab/gigaspeech",
    streaming=False,
)

SPEECHCOLAB_GIGASPEECH_M_CONFIG = types.DatasetConfig(
    name="speechcolab-gigaspeech-m",
    base="speechcolab-gigaspeech",
    subset="m",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=680072),
        types.DatasetSplitConfig(
            name="validation", num_samples=6750, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(
            name="test", num_samples=25619, split=types.DatasetSplit.TEST
        ),
    ],
    audio_field="audio",
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

configs = [SPEECHCOLAB_GIGASPEECH_BASE, SPEECHCOLAB_GIGASPEECH_M_CONFIG]
