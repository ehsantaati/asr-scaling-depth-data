import dataclasses
from typing import Any, Dict, List, Optional
from pathlib import Path
import simple_parsing
from data import types

@dataclasses.dataclass
class LoraConfigArgs:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    # Blocks of modules to initialize with separate seed resets to ensure consistency
    init_blocks: Optional[List[List[str]]] = None


@dataclasses.dataclass
class BaseConfig:
    # Model parameters
    model_id: str = "openai/whisper-tiny"
    language: str = "en"
    task: str = "transcribe"
    # Path to a trained checkpoint to load for inference or continuation
    checkpoint_path: Optional[str] = None
    # List of strings to match against parameter names. If None, all parameters are trainable.
    target_modules: Optional[List[str]] = None
    
    # Data parameters
    eval_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    eval_dataset_args: types.EvalDatasetArgs = simple_parsing.field(
        default_factory=types.EvalDatasetArgs
    )

    # Training/Inference parameters
    output_dir: Path = Path("outputs")
    fp16: bool = True
    seed: int = 42
    data_seed: int = 42
    
    # Evaluation
    eval_batch_size: int = 8
    
    # Fast Inference
    use_fast_inference: bool = True
    num_inference_workers: int = 0
    
    # Reproducibility
    strict_reproducibility: bool = False

    def get_eval_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.eval_sets]


@dataclasses.dataclass
class InferenceConfig(BaseConfig):
    pass


@dataclasses.dataclass
class TrainConfig(BaseConfig):
    # LoRA Configuration
    lora_config: Optional[LoraConfigArgs] = None
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    weight_decay: float = 0.0

    # Data parameters
    train_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    val_sets: List[Dict[str, Any]] = simple_parsing.list_field()

    # Run Name (Optional manual override)
    run_name: Optional[str] = None

    train_dataset_args: types.TrainDatasetArgs = simple_parsing.field(
        default_factory=types.TrainDatasetArgs
    )
    val_dataset_args: types.ValDatasetArgs = simple_parsing.field(
        default_factory=types.ValDatasetArgs
    )

    # Scaling parameters
    # List of fractions to train on, e.g. [0.1, 0.5, 1.0]
    data_fractions: List[float] = simple_parsing.list_field(1.0)
    # Number of random subsets to train for each fraction
    num_subsets: int = 1

    # Training parameters
    num_epochs: float = 3.0
    batch_size: int = 4
    dataloader_num_workers: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    warmup_ratio: float = 0.0
    max_steps: int = 0  # if > 0, overrides num_epochs

    # Evaluation / Train
    do_train: bool = True
    do_predict: bool = True
    eval_steps: int = 1000

    def get_train_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.train_sets]

    def get_val_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.val_sets]
