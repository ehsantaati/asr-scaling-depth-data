from .. import types

LIBRIHEAVY_CONFIG = types.DatasetConfig(
    name="libriheavy",
    path="pkufool/libriheavy",
    # Text is extracted from supervisions[0]['text'] (handled in GenericDataset)
    splits=[
        types.DatasetSplitConfig(
            name="validation", num_samples=5_350, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(
            name="test", num_samples=56_200, split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    LIBRIHEAVY_CONFIG,
]
