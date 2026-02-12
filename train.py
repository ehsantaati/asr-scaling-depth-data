#!/usr/bin/env python
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

from utils import prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding, WhisperDataproc
from inference import run_inference_pipeline, run_inference_map, save_inference_results
from data import datasets, registry, types, partitioning
from configs import BaseConfig, TrainConfig, LoraConfigArgs


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
    
    trainable_names = []
    for name, param in model.named_parameters():
        all_params += param.numel()
        if any(target in name for target in target_modules):
            param.requires_grad = True
            trainable_params += param.numel()
            trainable_names.append(name)
        else:
            param.requires_grad = False
            
    logging.info(f"Set trainability based on target_modules: {target_modules}")
    logging.info(f"Trainable parameter names: {trainable_names}")
    logging.info(f"Trainable params: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")



def main():
    load_dotenv()
    os.environ["WANDB_PROJECT"] = "asr-data-scaling"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
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

    if config.strict_reproducibility:
        logging.info("Strict reproducibility mode ENABLED.")
        logging.info("Enforcing deterministic CUDA operations...")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if config.dataloader_num_workers > 0:
            logging.warning(f"Overriding dataloader_num_workers from {config.dataloader_num_workers} to 0 for strict reproducibility.")
            config.dataloader_num_workers = 0

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
            
            # 0. Setup Output Directory & Logging
            # Extract experiment ID from output_dir (e.g. "outputs/002" -> "002")
            if config.run_name:
                exp_id = config.run_name
            else:
                exp_id = config.output_dir.name
            
            run_name = f"{exp_id}_frac_{fraction}_subset_{i}"
            output_dir = config.output_dir / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Add file handler to logging for this specific run
            file_handler = logging.FileHandler(output_dir / "train.log")
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)

            # 1. Initialize Model
            model = WhisperForConditionalGeneration.from_pretrained(
                config.model_id,
                dropout=config.dropout
            )
            # Log dropout
            logging.info(f"Initialized model with dropout={config.dropout}")
            logging.info(f"Model config dropout={model.config.dropout}")
            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = []

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
                trainable_params, all_param = model.get_nb_trainable_parameters()
                trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
                logging.info(f"Trainable parameter names: {trainable_names}")
                logging.info(
                    f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
                )
                
                # Re-initialize specific blocks with seed resets to ensure consistency across configurations
                if config.lora_config.init_blocks:
                    logging.info(f"Re-initializing LoRA modules in {len(config.lora_config.init_blocks)} blocks with seed resets...")
                    for block_idx, block in enumerate(config.lora_config.init_blocks):
                        logging.info(f"Initializing block {block_idx+1}: {block}")
                        # Reset global and library seeds to the base experiment seed
                        transformers.set_seed(config.seed)
                        
                        # Traverse model and re-init matching modules
                        for name, module in model.named_modules():
                            if any(target in name for target in block):
                                if hasattr(module, "reset_lora_parameters"):
                                    module.reset_lora_parameters("default", init_lora_weights=True)
                                    logging.debug(f"Reset LoRA parameters for {name}")

            else:
                set_trainable_parameters(model, config.target_modules)
            
            # 2. Prepare Training Dataset for this specific run
            # We use a FIXED seed for the base dataset to ensure the stream order is consistent across partitions
            # Then we use PartitionedDataset to pick the i-th slice
            train_args = dataclasses.replace(
                config.train_dataset_args, 
                shuffle_seed=config.data_seed, # Fixed seed for consistent stream
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
            
            # Wrapper with max length filtering
            max_label_length = getattr(model.config, "max_length", 448)
            train_dataset_proc = WhisperDataproc(train_dataset, processor, max_label_length=max_label_length)

                        # Calculate max_steps based on num_epochs, batch size, and gradient accumulation
            num_update_steps_per_epoch = len(train_dataset) // (config.batch_size * config.grad_accum_steps)
            if num_update_steps_per_epoch == 0:
                 # If dataset is smaller than batch * grad_accum, we at least have 1 step if possible, or 0.
                 # Warn about this.
                 logging.warning(f"Dataset length ({len(train_dataset)}) is smaller than effective batch size ({config.batch_size * config.grad_accum_steps}).")
            
            calculated_max_steps = int(num_update_steps_per_epoch * config.num_epochs)
            logging.info(f"Calculated max_steps: {calculated_max_steps} (Epochs: {config.num_epochs}, Steps/Epoch: {num_update_steps_per_epoch}, Dataset Len: {len(train_dataset)})")

            # Calculate warmup steps
            if config.warmup_ratio > 0:
                warmup_steps = int(calculated_max_steps * config.warmup_ratio)
                logging.info(f"Using warmup_ratio {config.warmup_ratio}. Calculated warmup_steps: {warmup_steps}")
            else:
                warmup_steps = config.warmup_steps
                logging.info(f"Using fixed warmup_steps: {warmup_steps}")

            # Print 3 random samples for verification
            logging.info("Printing 3 random training samples (Raw)...")
            try:
                for idx, sample in enumerate(train_dataset):
                    if idx >= 3:
                        break
                    logging.info(f"Raw Sample {idx+1}: {sample.text}")
            except Exception as e:
                logging.warning(f"Failed to print raw training samples: {e}")
            
            logging.info("Printing 3 random training samples (Processed)...")
            try:
                for idx, sample in enumerate(train_dataset_proc):
                    if idx >= 3:
                        break
                    # Decode the input_ids to check content
                    if "text_input_ids" in sample:
                        decoded_text = processor.decode(sample["text_input_ids"], skip_special_tokens=True)
                        logging.info(f"Proc Sample {idx+1} (Decoded): {decoded_text}")
                    elif "reference" in sample:
                        logging.info(f"Proc Sample {idx+1} (Ref): {sample['reference']}")
            except Exception as e:
                logging.warning(f"Failed to print processed training samples: {e}")

            # 3. Setup Trainer

            
            training_args = Seq2SeqTrainingArguments(
                run_name=run_name,
                output_dir=str(output_dir),
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.eval_batch_size,
                dataloader_num_workers=config.dataloader_num_workers,
                gradient_accumulation_steps=config.grad_accum_steps,
                learning_rate=config.learning_rate,
                warmup_steps=warmup_steps,
                max_steps=calculated_max_steps,
                num_train_epochs=config.num_epochs,
                fp16=config.fp16,
                logging_steps=25,
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

            train_result = trainer.train()
            training_time = train_result.metrics.get("train_runtime")
            logging.info(f"Training completed in {training_time} seconds.")
            
            # 5. Run Inference
            logging.info("Running post-training inference on EVAL sets...")
            
            # Prepare evaluation dataset
            eval_dataset = prepare_dataset(
                config.get_eval_sets(), config.eval_dataset_args
            )
            
            # Ensure model is on correct device
            device = trainer.args.device
            
            if config.use_fast_inference:
                logging.info("Using FAST inference mode (dataset.map)...")
                metrics, predictions, references = run_inference_map(
                    model,
                    processor,
                    eval_dataset,
                    batch_size=config.eval_batch_size,
                    language=config.language,
                    num_workers=config.num_inference_workers
                )
            else:
                logging.info("Using PIPELINE inference mode...")
                metrics, predictions, references = run_inference_pipeline(
                    model, 
                    processor, 
                    eval_dataset, 
                    device, 
                    batch_size=config.eval_batch_size,
                    language=config.language
                )

            # 6. Merge and Save Final Model
            if config.lora_config is not None:
                logging.info("Merging LoRA adapters into base model before saving...")
                model = model.merge_and_unload()
                trainer.model = model
            
            # Ensure config matches training args
            if hasattr(model, "config"):
                model.config.dropout = config.dropout
                
            trainer.save_model()
            
            # Save processor (tokenizer) alongside model for self-contained checkpoints
            processor.save_pretrained(str(output_dir))
            
            # Save metrics and predictions using the shared function
            save_inference_results(
                output_dir,
                metrics,
                predictions,
                references,
                model_path=output_dir, # In train.py, the model is in the output_dir
                dataset_name=f"{'+'.join([d['name'] for d in config.eval_sets])}_{config.eval_dataset_args.split}",
                training_time=training_time
            )
            
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

            # Finish WandB run to ensure next iteration starts a new one
            if wandb.run is not None:
                wandb.finish()

            # Cleanup
            del model
            del trainer
            
            # Remove file handler to prevent duplicate logs in next iteration
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
