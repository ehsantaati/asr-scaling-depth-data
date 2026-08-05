"""LibriSpeech configs, used as the out-of-domain (OOD) evaluation sets.

Ported from branch `dev` for the revision campaign (action B7 / R1-5.5, R3-10).
Only the `test` splits are exercised by the campaign; the train/validation
splits are retained so the configs stay usable for other purposes.
"""

from .. import types

# Base config for the openslr/librispeech_asr dataset on HF.
LS_BASE_CONFIG = types.DatasetConfig(
    name="openslr-librispeech-asr",
    path="openslr/librispeech_asr",
    # Cached, not streamed: the two test splits total ~0.64 GiB and are re-read by
    # every run in the campaign. trust_remote_code stays False (the inherited
    # default) so the Hub's parquet export is used and only the requested split is
    # downloaded -- the script loader materialises train.100 and train.360 as well,
    # 30 GB for 0.33 GiB of need.
    streaming=False,
)

# Clean splits
LS_CLEAN_CONFIG = types.DatasetConfig(
    name="openslr-librispeech-asr-clean",
    base="openslr-librispeech-asr",
    subset="clean",
    splits=[
        types.DatasetSplitConfig(
            name="train.100", num_samples=28_539, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="train.360", num_samples=104_014, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_703, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=2_620),
    ],
)

# Other splits
LS_OTHER_CONFIG = types.DatasetConfig(
    name="openslr-librispeech-asr-other",
    base="openslr-librispeech-asr",
    subset="other",
    splits=[
        types.DatasetSplitConfig(
            name="train.500", num_samples=148_688, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_864, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=2_939),
    ],
)


configs = [
    LS_BASE_CONFIG,
    LS_CLEAN_CONFIG,
    LS_OTHER_CONFIG,
]
