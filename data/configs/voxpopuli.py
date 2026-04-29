from .. import types

VOXPOPULI_BASE_CONFIG = types.DatasetConfig(
    name="voxpopuli",
    path="facebook/voxpopuli",
    transcript_field="normalized_text",
    audio_field="audio",
)

VOXPOPULI_EN_CONFIG = types.DatasetConfig(
    name="voxpopuli-en",
    base="voxpopuli",
    subset="en",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=182482, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="validation", num_samples=1753, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(
            name="test", num_samples=1842, split=types.DatasetSplit.TEST
        ),
    ],
)

VOXPOPULI_EN_ACCENTED_CONFIG = types.DatasetConfig(
    name="voxpopuli-en-accented",
    base="voxpopuli",
    subset="en_accented",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=8387, split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    VOXPOPULI_BASE_CONFIG,
    VOXPOPULI_EN_CONFIG,
    VOXPOPULI_EN_ACCENTED_CONFIG,
]
