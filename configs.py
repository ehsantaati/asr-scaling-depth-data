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
    
    # Out-of-domain evaluation sets (action B7 / R1-5.5, R3-10). Scored after the
    # in-domain sets with the identical decode spec, inside the training job.
    ood_eval_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    ood_eval_dataset_args: types.EvalDatasetArgs = simple_parsing.field(
        default_factory=types.EvalDatasetArgs
    )

    # Evaluation
    eval_batch_size: int = 8

    # Decoding (the fixed path -- see decoding.py). Stated explicitly rather than
    # inherited from generation_config defaults, because R1-7.1 asks for them.
    decode_num_beams: int = 1
    decode_max_new_tokens: int = 200
    decode_force_language: bool = True
    decode_dtype: str = "fp16"  # fp16 | fp32 | bf16; applies in-job AND standalone

    # Legacy scoring path (inference.run_inference_map), kept so the old-vs-new
    # evaluation delta can be measured. Both paths run at decode_dtype, so the delta
    # isolates the decode path rather than mixing in a precision change.
    score_legacy_path: bool = True

    # Fast Inference
    use_fast_inference: bool = True
    num_inference_workers: int = 0

    # Reproducibility
    strict_reproducibility: bool = False

    def get_eval_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.eval_sets]

    def get_ood_eval_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.ood_eval_sets]


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
    
    # Layer-Specific Dropout (overrides standard dropout for indicated decoder layers)
    layer_specific_dropout: float = 0.0
    layer_specific_attention_dropout: float = 0.0
    layer_specific_activation_dropout: float = 0.0
    layer_specific_dropout_layers: Optional[List[int]] = None

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
    max_steps_fraction: float = 1.0  # Fraction of the calculated_max_steps to use

    # Optimizer / scheduler, stated explicitly instead of inherited silently from
    # TrainingArguments defaults. R1-7.1 asks for exactly these values.
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Data-order seed policy (see the audit, §1.1 / §3.3).
    #   "fixed" -> shuffle_seed = data_seed always (original behaviour: run-to-run
    #              variation comes only from initialization and dropout)
    #   "seed"  -> shuffle_seed = seed always (also changes WHICH samples a
    #              fractional run sees, conflating subset identity with noise)
    #   "auto"  -> seed at fraction == 1.0, data_seed otherwise  [campaign default]
    data_order_seed_mode: str = "auto"

    # Evaluation / Train
    do_train: bool = True
    do_predict: bool = True
    # Absolute step grids, so curves from a 1.1k-step data-limited run and a 42k-step
    # fixed-budget run land on a comparable x-axis (action Z3).
    logging_steps: int = 25
    eval_steps: int = 250
    eval_on_start: bool = True
    log_dataset_metadata: bool = True
    # Number of training samples logged before training starts. Doubles as the
    # non-empty check and as the data-order fingerprint recorded in the manifest.
    # Costs one extra pass over the head of the stream; set to 0 only if that pass
    # is prohibitive (it should not be -- see the note in train.py).
    preview_samples: int = 3

    # Periodic trainable-only checkpoints and the best-val-vs-final comparison
    # (action B8 / R2-6c). Enabled on deep full-FT runs only; checkpoints are
    # deleted in-job once the comparison is written.
    checkpoint_fraction: float = 0.0  # 0 disables; 0.1 => ~10 checkpoints
    run_best_vs_final: bool = False
    best_vs_final_max_samples: int = 2000

    # Weight persistence. "adapter" saves LoRA adapters only; "trainable" saves a
    # trainable-only fp16 state dict (the encoder is frozen, so persisting it is pure
    # waste); "full" restores the old behaviour. Default resolves by method.
    save_mode: str = "auto"  # auto | adapter | trainable | full

    def get_train_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.train_sets]

    def get_val_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.val_sets]
