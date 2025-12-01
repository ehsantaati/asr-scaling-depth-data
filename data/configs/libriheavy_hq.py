from .. import types

LIBRIHEAVY_HQ_BASE_CONFIG = types.DatasetConfig(
    name="libriheavy-hq",
    path="mythicinfinity/Libriheavy-HQ",
    transcript_field="text_transcription",
)

LIBRIHEAVY_HQ_SMALL_CONFIG = types.DatasetConfig(
    name="libriheavy-hq-small",
    base="libriheavy-hq",
    subset="small",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=122_526),
    ],
)

configs = [
    LIBRIHEAVY_HQ_BASE_CONFIG,
    LIBRIHEAVY_HQ_SMALL_CONFIG,
]
