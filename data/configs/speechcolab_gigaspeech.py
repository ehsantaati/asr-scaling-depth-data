from .. import types

SPEECHCOLAB_GIGASPEECH_BASE = types.DatasetConfig(
    name="speechcolab-gigaspeech",
    path="speechcolab/gigaspeech",
    streaming=False,
    # Gated, but a parquet conversion exists for every config (m: 29 train shards),
    # so the loading script is not needed. Using it is actively worse: a load through
    # trust_remote_code=True hung for >4 minutes without producing a sample, while
    # the parquet route resolves per split. Access still requires an accepted gate
    # plus a valid HF_TOKEN.
    trust_remote_code=False,
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
