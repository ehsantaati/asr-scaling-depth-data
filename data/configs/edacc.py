from .. import types

EDACC_CONFIG = types.DatasetConfig(
    name="edacc",
    path="edinburghcstr/edacc",
    # transcript_field defaults to "text"
    splits=[
        types.DatasetSplitConfig(
            name="train",
            num_samples=8_865,
            split=types.DatasetSplit.TRAIN,
            source_split="validation[:8865]",
        ),
        types.DatasetSplitConfig(
            name="validation",
            num_samples=985,
            split=types.DatasetSplit.VALIDATION,
            source_split="validation[8865:]",
        ),
        types.DatasetSplitConfig(
            name="test", num_samples=9_290, split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    EDACC_CONFIG,
]
