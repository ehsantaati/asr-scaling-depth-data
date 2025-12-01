from .. import types

SPGISPEECH_BASE_CONFIG = types.DatasetConfig(
    name="spgispeech",
    path="kensho/spgispeech",
    transcript_field="transcript",
)

SPGISPEECH_S_CONFIG = types.DatasetConfig(
    name="spgispeech-s",
    base="spgispeech",
    path="kensho/spgispeech",
    subset="S",
    transcript_field="transcript",
    audio_field="audio",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=-1),
        types.DatasetSplitConfig(
            name="validation", num_samples=-1, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=-1),
    ],
)

SPGISPEECH_M_CONFIG = types.DatasetConfig(
    name="spgispeech-m",
    base="spgispeech",
    subset="M",
    transcript_field="transcript",
    audio_field="audio",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=-1),
        types.DatasetSplitConfig(
            name="validation", num_samples=-1, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=-1),
    ],
)

SPGISPEECH_L_CONFIG = types.DatasetConfig(
    name="spgispeech-l",
    base="spgispeech",
    subset="L",
    transcript_field="transcript",
    audio_field="audio",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=-1),
        types.DatasetSplitConfig(
            name="validation", num_samples=-1, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=-1),
    ],
)

configs = [
    SPGISPEECH_BASE_CONFIG,
    SPGISPEECH_S_CONFIG,
    SPGISPEECH_M_CONFIG,
    SPGISPEECH_L_CONFIG,
]
