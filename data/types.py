import dataclasses
import enum
import json
import logging
from typing import Any, Dict, List, Optional
from simple_parsing import helpers

class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

@dataclasses.dataclass
class DatasetOptions:
    name: str
    weight: float = 1.0

@dataclasses.dataclass
class VoiceDatasetArgs:
    """Global arguments for train/val/test dataset creation."""

    split: DatasetSplit = DatasetSplit.TRAIN
    """Which split of the dataset to use."""
    shuffle: bool = False
    """Whether to shuffle the dataset."""
    shuffle_seed: int = 42
    """Seed for shuffling the dataset."""
    shuffle_buffer_size: int = 1000
    """Buffer size for shuffling the dataset. Only used for streaming datasets."""
    max_audio_duration_secs: float = 30
    """Whether to skip samples with audio longer than this duration. -1 means no filtering"""
    max_input_characters: Optional[int] = 2200
    """Used for direct messages input. Skips samples with input characters longer than this value."""
    max_samples: int = -1
    """max number of samples to use per dataset"""

    def __post_init__(self):
        if isinstance(self.split, str):
            self.split = DatasetSplit(self.split.lower())


@dataclasses.dataclass
class TrainDatasetArgs(VoiceDatasetArgs):
    split: DatasetSplit = DatasetSplit.TRAIN
    shuffle: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.split == DatasetSplit.TRAIN


@dataclasses.dataclass
class ValDatasetArgs(VoiceDatasetArgs):
    split: DatasetSplit = DatasetSplit.VALIDATION
    max_samples: int = 256

    def __post_init__(self):
        super().__post_init__()
        assert self.split == DatasetSplit.VALIDATION
        assert self.shuffle is False


@dataclasses.dataclass
class EvalDatasetArgs(VoiceDatasetArgs):
    split: DatasetSplit = DatasetSplit.TEST
    max_audio_duration_secs: float = -1  # -1 means no filtering

    def __post_init__(self):
        super().__post_init__()
        if self.split != DatasetSplit.TEST:
            logging.warning(
                f"EvalDatasetArgs is being used with non-test split: {self.split}"
            )
        if self.shuffle:
            logging.warning(
                f"EvalDatasetArgs is being used with shuffle=True and shuffle_seed={self.shuffle_seed}"
            )


@dataclasses.dataclass
class DatasetSplitConfig(helpers.Serializable):
    name: str
    """Name of the split."""
    num_samples: int
    """Number of samples in the split"""
    split: Optional[DatasetSplit] = None
    """Type of split, i.e., train, test, or validation."""

    def __post_init__(self):
        """Automatically set split type based on split name"""
        if self.split is None:
            try:
                self.split = DatasetSplit(self.name.lower())
            except ValueError:
                raise ValueError(
                    f"Could not automatically determine split type from split name '{self.name}'. Please explicitly specify split_type for splits that are not named 'train', 'validation', or 'test'."
                )


# Eval config for a single metric, added to the dataset config
@dataclasses.dataclass
class EvalConfig(helpers.Serializable):
    metric: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    extra_kwargs_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    """Mapping of field names to use as extra_kwargs for the sample.
    key is name in extra_kwargs, value is name in dataset row."""


@dataclasses.dataclass
class DatasetConfig(helpers.Serializable):
    # Note that subclasses can override any of these fields, but they currently can't
    # extend structured fields like splits or user_template_args.
    # See _merge_configs below for the current implementation.
    name: str
    """Name of the dataset."""
    base: Optional[str] = None
    """Base dataset config to inherit from."""
    path: Optional[str] = None
    """Directory of the dataset, or huggingface dataset name; must be set for "generic" datasets. If not set, it is automatically inferred for predefined dataset types."""
    subset: Optional[str] = None
    """Name of the dataset, or huggingface dataset config/subset name."""
    splits: Optional[List[DatasetSplitConfig]] = None
    """List of splits to use, e.g. [{"name": "train", "num_samples": 1000}, {"name": "validation", "num_samples": 100}]."""
    transcript_field: Optional[str] = None
    """Name of the field in the dataset that contains the transcript text (default: 'text')."""
    audio_field: Optional[str] = None
    """Field in the dataset that contains the audio, use None if the dataset does not contain audio."""
    audio_path_field: Optional[str] = None
    """Field that contains a filesystem path to the audio file."""
    audio_root: Optional[str] = None
    """Root directory used to resolve relative audio paths."""

    streaming: Optional[bool] = None
    """Whether to load the dataset using streaming mode."""
    features: Optional[Any] = None
    """Optional Hugging Face Features schema to enforce when loading the dataset."""
    eval_config: Optional[EvalConfig] = None
    """Eval config for the dataset."""

    def __post_init__(self):
        """Set defaults only if this is a root config, so that said defaults in a subclass don't act as overrides."""
        DEFAULTS = {
            "splits": [],
            "transcript_field": "text",
            "audio_field": "audio",
            "audio_path_field": None,
            "audio_root": None,

            "streaming": True,
            "features": None,
            "eval_config": None,
        }
        if self.base is None:
            for attr, default_value in DEFAULTS.items():
                if getattr(self, attr) is None:
                    setattr(self, attr, default_value)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
