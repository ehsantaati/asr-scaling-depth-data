from .. import types

EDACC_CONFIG = types.DatasetConfig(
    name="edacc",
    path="edinburghcstr/edacc",
    # transcript_field defaults to "text"
    splits=[
        types.DatasetSplitConfig(
            name="validation", num_samples=9_850, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(
            name="test", num_samples=9_290, split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    EDACC_CONFIG,
]
