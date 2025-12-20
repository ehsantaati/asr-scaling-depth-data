from .. import types

# Base config for the MLCommons/peoples_speech dataset on HF.
PEOPLES_SPEECH_BASE = types.DatasetConfig(
    name="mlcommons-peoples-speech",
    path="MLCommons/peoples_speech",
    streaming=True,
)

# Train splits
PEOPLES_SPEECH_CLEAN_CONFIG = types.DatasetConfig(
    name="mlcommons-peoples-speech-clean",
    base="mlcommons-peoples-speech",
    subset="clean",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=1_550_000, split=types.DatasetSplit.TRAIN
        ),
    ],
)

PEOPLES_SPEECH_CLEAN_SA_CONFIG = types.DatasetConfig(
    name="mlcommons-peoples-speech-clean-sa",
    base="mlcommons-peoples-speech",
    subset="clean_sa",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=311_000, split=types.DatasetSplit.TRAIN
        ),
    ],
)

PEOPLES_SPEECH_DIRTY_CONFIG = types.DatasetConfig(
    name="mlcommons-peoples-speech-dirty",
    base="mlcommons-peoples-speech",
    subset="dirty",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=5_530_000, split=types.DatasetSplit.TRAIN
        ),
    ],
)

PEOPLES_SPEECH_DIRTY_SA_CONFIG = types.DatasetConfig(
    name="mlcommons-peoples-speech-dirty-sa",
    base="mlcommons-peoples-speech",
    subset="dirty_sa",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=602_000, split=types.DatasetSplit.TRAIN
        ),
    ],
)

PEOPLES_SPEECH_MICROSET_CONFIG = types.DatasetConfig(
    name="mlcommons-peoples-speech-microset",
    base="mlcommons-peoples-speech",
    subset="microset",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=336, split=types.DatasetSplit.TRAIN
        ),
    ],
)

# Validation and Test splits
# Assuming the validation subset contains a 'validation' split and test contains 'test'.
# If they only contain 'train', we would map name="train" -> split=VALIDATION/TEST.
# But usually matching names is safer first guess.
PEOPLES_SPEECH_VALIDATION_CONFIG = types.DatasetConfig(
    name="mlcommons-peoples-speech-validation",
    base="mlcommons-peoples-speech",
    subset="validation",
    splits=[
        types.DatasetSplitConfig(
            name="validation", num_samples=18_600, split=types.DatasetSplit.VALIDATION
        ),
    ],
)

PEOPLES_SPEECH_TEST_CONFIG = types.DatasetConfig(
    name="mlcommons-peoples-speech-test",
    base="mlcommons-peoples-speech",
    subset="test",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=34_900, split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    PEOPLES_SPEECH_BASE,
    PEOPLES_SPEECH_CLEAN_CONFIG,
    PEOPLES_SPEECH_CLEAN_SA_CONFIG,
    PEOPLES_SPEECH_DIRTY_CONFIG,
    PEOPLES_SPEECH_DIRTY_SA_CONFIG,
    PEOPLES_SPEECH_MICROSET_CONFIG,
    PEOPLES_SPEECH_VALIDATION_CONFIG,
    PEOPLES_SPEECH_TEST_CONFIG,
]