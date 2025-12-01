from .. import types

SPGISPEECH_BASE_CONFIG = types.DatasetConfig(
    name="spgispeech",
    path="kensho/spgispeech",
    transcript_field="transcript",
)

SPGISPEECH_S_CONFIG = types.DatasetConfig(
    name="spgispeech-s",
    base="spgispeech",
    subset="S",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=0),
        types.DatasetSplitConfig(
            name="validation", num_samples=0, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=0),
    ],
)

SPGISPEECH_M_CONFIG = types.DatasetConfig(
    name="spgispeech-m",
    base="spgispeech",
    subset="M",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=0),
        types.DatasetSplitConfig(
            name="validation", num_samples=0, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=0),
    ],
)

SPGISPEECH_L_CONFIG = types.DatasetConfig(
    name="spgispeech-l",
    base="spgispeech",
    subset="L",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=0),
        types.DatasetSplitConfig(
            name="validation", num_samples=0, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=0),
    ],
)

configs = [
    SPGISPEECH_BASE_CONFIG,
    SPGISPEECH_S_CONFIG,
    SPGISPEECH_M_CONFIG,
    SPGISPEECH_L_CONFIG,
]
