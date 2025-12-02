import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import simple_parsing
import wandb
import torch
from dotenv import load_dotenv
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)



from data import datasets, registry, types, partitioning


@dataclasses.dataclass
class LoraConfigArgs:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None


@dataclasses.dataclass
class TrainConfig:
    # Model parameters
    model_id: str = "openai/whisper-tiny"
    language: str = "en"
    task: str = "transcribe"
    # List of strings to match against parameter names. If None, all parameters are trainable.
    target_modules: Optional[List[str]] = None
    
    # LoRA Configuration
    lora_config: Optional[LoraConfigArgs] = None

    # Data parameters
    train_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    val_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    eval_sets: List[Dict[str, Any]] = simple_parsing.list_field()

    train_dataset_args: types.TrainDatasetArgs = simple_parsing.field(
        default_factory=types.TrainDatasetArgs
    )
    val_dataset_args: types.ValDatasetArgs = simple_parsing.field(
        default_factory=types.ValDatasetArgs
    )
    eval_dataset_args: types.EvalDatasetArgs = simple_parsing.field(
        default_factory=types.EvalDatasetArgs
    )

    # Scaling parameters
    # List of fractions to train on, e.g. [0.1, 0.5, 1.0]
    data_fractions: List[float] = simple_parsing.list_field(1.0)
    # Number of random subsets to train for each fraction
    num_subsets: int = 1

    # Training parameters
    output_dir: Path = Path("outputs")
    num_epochs: float = 3.0
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 0  # if > 0, overrides num_epochs
    fp16: bool = True
    seed: int = 42

    # Evaluation
    do_eval: bool = True
    eval_steps: int = 1000

    def get_train_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.train_sets]

    def get_val_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.val_sets]

    def get_eval_sets(self) -> List[types.DatasetConfig]:
        return [types.DatasetConfig.from_dict(ds) for ds in self.eval_sets]


def prepare_dataset(
    data_opts: List[types.DatasetConfig],
    data_args: types.VoiceDatasetArgs,
    max_samples: Optional[int] = None,
) -> datasets.SizedIterableDataset:
    data_sets = []
    for ds_config in data_opts:
        # Register if not already registered (or just ensure it's in the map)
        if ds_config.name not in registry.DATASET_MAP:
             registry.register_datasets([ds_config])
        
        ds = registry.create_dataset(ds_config.name, data_args, verbose=True)
        data_sets.append(ds)

    # Interleave if multiple datasets
    if len(data_sets) > 1:
        dataset = datasets.InterleaveDataset(data_sets)
    else:
        dataset = data_sets[0]

    # Apply range limit if specified
    if max_samples is not None:
        dataset = datasets.Range(dataset, max_samples)
    elif data_args.max_samples != -1:
        dataset = datasets.Range(dataset, data_args.max_samples)
        
    return dataset

    return dataset


class WhisperDataproc(datasets.Dataproc):
    def __init__(self, dataset, processor):
        super().__init__(dataset)
        self.processor = processor

    def _process(self, sample):
        # Process audio
        audio = sample.audio
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features[0]
        
        # Process text
        labels = self.processor(text=sample.text).input_ids
        
        return {
            "audio": {"array": input_features},
            "text_input_ids": labels,
            "reference": sample.text,
        }


def set_trainable_parameters(model: torch.nn.Module, target_modules: Optional[List[str]]) -> None:
    """
    Sets the trainability of the model parameters based on the target_modules list.
    If target_modules is None or empty, all parameters are trainable.
    Otherwise, only parameters whose names contain one of the target_modules strings are trainable.
    """
    if not target_modules:
        return

    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if any(target in name for target in target_modules):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            
    logging.info(f"Set trainability based on target_modules: {target_modules}")
    logging.info(f"Trainable params: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")


def run_inference(
    model: torch.nn.Module,
    processor: WhisperProcessor,
    dataset: datasets.SizedIterableDataset,
    device: torch.device,
    batch_size: int = 1,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Runs inference on the dataset and computes WER.
    Returns metrics, predictions, and references.
    """
    logging.info(f"Starting inference with batch_size={batch_size}...")
    model.eval()
    metric = evaluate.load("wer")
    
    predictions = []
    references = []
    
    # Prepare dataset and dataloader
    dataset_proc = WhisperDataproc(dataset, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    
    # Note: num_workers=0 is safer for iterable datasets sometimes, but >0 can help speed.
    # Since we are using an IterableDataset, we can't use shuffle=True (already handled)
    dataloader = torch.utils.data.DataLoader(
        dataset_proc,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0, 
    )
    
    from tqdm import tqdm
    
    for i, batch in enumerate(tqdm(dataloader, desc="Inference")):
        # Move inputs to device and cast to model's dtype
        input_features = batch["input_features"].to(device, dtype=model.dtype)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(input_features)
        
        # Decode
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Get references
        # If collator passed through 'references', use them.
        # Otherwise decode labels (fallback)
        if "references" in batch:
            batch_references = batch["references"]
        else:
            # Fallback: decode labels (replace -100 with pad_token_id)
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            batch_references = processor.batch_decode(labels, skip_special_tokens=True)

        # Normalization (simple lowercasing for now)
        transcriptions = [t.lower() for t in transcriptions]
        batch_references = [r.lower() for r in batch_references]
        
        predictions.extend(transcriptions)
        references.extend(batch_references)
        
        if i < 3:
            logging.info(f"Batch {i} Sample 0:")
            logging.info(f"  Ref: {batch_references[0]}")
            logging.info(f"  Pred: {transcriptions[0]}")

    wer = metric.compute(predictions=predictions, references=references)
    logging.info(f"Final Inference WER: {wer}")
    
    return {"wer": wer}, predictions, references


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["audio"]["array"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["text_input_ids"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (
            labels[:, 0] == self.processor.tokenizer.bos_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        # Pass through references if present
        if "reference" in features[0]:
            batch["references"] = [feature["reference"] for feature in features]
            
        return batch


def main():
    load_dotenv()
    os.environ["WANDB_PROJECT"] = "asr-data-scaling"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Register pre-defined datasets
    import data.configs
    registry.register_datasets(data.configs.ALL_CONFIGS)
    
    # Parse config
    config = simple_parsing.parse(TrainConfig, add_config_path_arg=True)
    
    # Set seed
    transformers.set_seed(config.seed)

    # Load processor
    processor = WhisperProcessor.from_pretrained(
        config.model_id, language=config.language, task=config.task
    )

    # Prepare validation dataset (fixed across experiments)
    val_dataset = prepare_dataset(
        config.get_val_sets(), config.val_dataset_args
    )
    
    val_dataset_proc = WhisperDataproc(val_dataset, processor)

    # Determine total training samples from full dataset to calculate fractions
    # We need to instantiate the full train dataset once to get its length
    full_train_dataset = prepare_dataset(
        config.get_train_sets(), config.train_dataset_args
    )
    total_train_samples = len(full_train_dataset)
    logging.info(f"Total available training samples: {total_train_samples}")

    # Experiment Loop
    for fraction in config.data_fractions:
        # Calculate number of partitions based on fraction
        # e.g. 0.2 -> 5 partitions
        total_partitions = int(1 / fraction)
        if abs(1/fraction - total_partitions) > 1e-6:
             logging.warning(f"Fraction {fraction} does not divide 1.0 cleanly. Using {total_partitions} partitions, which is approx {1/total_partitions:.2f}")
        
        logging.info(f"--- Starting experiments for fraction: {fraction} (Total Partitions: {total_partitions}) ---")
        
        # Limit number of subsets to total available partitions if needed
        num_subsets_to_run = min(config.num_subsets, total_partitions)
        
        for i in range(num_subsets_to_run):
            logging.info(f"Subset {i+1}/{num_subsets_to_run} (Partition Index: {i})")
            
            # 1. Initialize Model
            model = WhisperForConditionalGeneration.from_pretrained(
                config.model_id
            )
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []

            # Untie embeddings and reinitialize output weights
            if model.config.tie_word_embeddings:
                model.config.tie_word_embeddings = False
                # Clone the input embeddings to initialize the output projection
                # Whisper's output projection is typically 'proj_out'
                if hasattr(model, "proj_out"):
                     # The decoder embeddings are usually the source
                     decoder_embed_weight = model.model.decoder.embed_tokens.weight
                     model.proj_out = torch.nn.Linear(decoder_embed_weight.shape[1], decoder_embed_weight.shape[0], bias=False)
                     model.proj_out.weight.data.copy_(decoder_embed_weight.data)
                else:
                    logging.warning("Could not find 'proj_out' to reinitialize after untying embeddings.")
            
            
            # Set trainable parameters or apply LoRA
            if config.lora_config is not None:
                logging.info("Applying LoRA...")
                peft_config = LoraConfig(
                    # task_type=TaskType.SEQ_2_SEQ_LM, # Removed to avoid input_ids assumption
                    inference_mode=False,
                    r=config.lora_config.r,
                    lora_alpha=config.lora_config.lora_alpha,
                    lora_dropout=config.lora_config.lora_dropout,
                    target_modules=config.lora_config.target_modules or config.target_modules
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                set_trainable_parameters(model, config.target_modules)
            
            # 2. Prepare Training Dataset for this specific run
            # We use a FIXED seed for the base dataset to ensure the stream order is consistent across partitions
            # Then we use PartitionedDataset to pick the i-th slice
            train_args = dataclasses.replace(
                config.train_dataset_args, 
                shuffle_seed=config.seed, # Fixed seed for consistent stream
                shuffle=True
            )
            
            # Get the base streaming dataset
            base_train_dataset = prepare_dataset(
                config.get_train_sets(), 
                train_args,
                max_samples=None # Do not limit yet
            )
            
            # Wrap with PartitionedDataset
            train_dataset = partitioning.PartitionedDataset(
                base_train_dataset,
                partition_index=i,
                total_partitions=total_partitions
            )
            
            train_dataset_proc = WhisperDataproc(train_dataset, processor)

            # 3. Setup Trainer
            run_name = f"frac_{fraction}_subset_{i}"
            output_dir = config.output_dir / run_name
            
            training_args = Seq2SeqTrainingArguments(
                run_name=run_name,
                output_dir=str(output_dir),
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.grad_accum_steps,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                num_train_epochs=config.num_epochs,
                fp16=config.fp16,
                logging_steps=10,
                eval_strategy="steps" if config.do_eval else "no",
                eval_steps=config.eval_steps,
                save_strategy="no", # Save only at end to save space
                report_to=["wandb", "tensorboard"],
                remove_unused_columns=False, # Required for custom collator/dataset
                label_names=["labels"], 
            )

            trainer = Seq2SeqTrainer(
                args=training_args,
                model=model,
                train_dataset=train_dataset_proc,
                eval_dataset=val_dataset_proc,
                data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
                processing_class=processor.feature_extractor,
            )

            # 4. Train
            if config.do_eval:
                logging.info("Running initial evaluation...")
                metrics = trainer.evaluate()
                logging.info(f"Initial metrics: {metrics}")

            trainer.train()
            
            # 5. Save Final Model
            trainer.save_model()
            
            # 6. Run Inference
            logging.info("Running post-training inference on EVAL sets...")
            
            # Prepare evaluation dataset
            eval_dataset = prepare_dataset(
                config.get_eval_sets(), config.eval_dataset_args
            )
            
            # Ensure model is on correct device
            device = trainer.args.device
            metrics, predictions, references = run_inference(model, processor, eval_dataset, device)
            
            # Save metrics
            import json
            with open(output_dir / "inference_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Log to WandB and TensorBoard
            if wandb.run is not None:
                # wandb.log({f"inference/{k}": v for k, v in metrics.items()})
                
                # Create a summary table for WER
                # Construct dataset name from config
                eval_dataset_names = [ds["name"] for ds in config.eval_sets]
                eval_dataset_name = "+".join(eval_dataset_names) if eval_dataset_names else "evaluation"
                
                table = wandb.Table(columns=["Dataset", "WER"])
                table.add_data(eval_dataset_name, metrics["wer"])
                wandb.log({"inference/wer_summary": table})
            
            # For TensorBoard, we can use the trainer's callback if available, or just rely on WandB for custom metrics
            # But since the trainer loop is done, we might need manual logging.
            # However, standard training metrics are already logged.
            # Let's try to use the SummaryWriter from the trainer's callback if accessible, or create a new one.
            # A simple way is to use torch.utils.tensorboard.SummaryWriter
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(output_dir / "runs"))
            for k, v in metrics.items():
                tb_writer.add_scalar(f"inference/{k}", v, global_step=trainer.state.global_step)
            tb_writer.close()

            # Cleanup
            del model
            del trainer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
