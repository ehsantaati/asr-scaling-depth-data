#!/usr/bin/env python
import dataclasses
import datetime
import json
import logging
import os
import shutil
import sys
import time
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

import callbacks as cb
import decoding
import manifest as mf
from utils import (
    prepare_dataset,
    DataCollatorSpeechSeq2SeqWithPadding,
    WhisperDataproc,
    compute_and_save_dataset_metadata,
    save_trainable_state,
    trainable_state_dict,
)
from inference import run_inference_pipeline, run_inference_map, save_inference_results, evaluate_datasets
from data import datasets, registry, types, partitioning
from configs import BaseConfig, TrainConfig, LoraConfigArgs


def _matches_target(name: str, target: str) -> bool:
    """Name-boundary match, not substring.

    The original `target in name` test meant that a target of
    "model.decoder.layers.1" also selected layers 10-19. The shipped configs happen to
    enumerate every layer so no published run was affected, but any new layer-wise
    configuration (B10's GigaSpeech L3/L4, B9's Whisper-Small remap) would trip it
    silently.
    """
    return name == target or name.startswith(target + ".")


def set_trainable_parameters(
    model: torch.nn.Module,
    target_modules: Optional[List[str]],
    allow_encoder_adaptation: bool = False,
) -> List[str]:
    """Freeze everything except parameters selected by ``target_modules``.

    Returns the resolved trainable parameter names so they can be recorded in the run
    manifest (R1-7.2).

    ``allow_encoder_adaptation`` is a scoped exception for action B5 only, which exists
    to answer R1-5.2 -- whether decoder-side adaptation really captures the dominant
    domain-specific gains -- and cannot be run without adapting encoder layers. It
    defaults to False so every other run in the campaign is still protected by the
    frozen-encoder invariant, and it must be set explicitly in the config, so any run
    that used it says so in its own manifest. Do not flip the default.
    """
    if not target_modules:
        # Previously this returned silently, leaving the whole model -- encoder
        # included -- trainable. Every claim in this study assumes a frozen encoder.
        raise ValueError(
            "target_modules is empty. Refusing to train all parameters implicitly; "
            "set target_modules explicitly (use save_mode/full FT configs as a guide)."
        )

    trainable_params = 0
    all_params = 0

    trainable_names = []
    for name, param in model.named_parameters():
        all_params += param.numel()
        if any(_matches_target(name, target) for target in target_modules):
            param.requires_grad = True
            trainable_params += param.numel()
            trainable_names.append(name)
        else:
            param.requires_grad = False

    leaked = [n for n in trainable_names if ".encoder." in n or n.startswith("model.encoder.")]
    if leaked and not allow_encoder_adaptation:
        raise ValueError(
            f"target_modules selected {len(leaked)} encoder parameters "
            f"(first: {leaked[:3]}). The encoder must remain frozen. If this is the B5 "
            f"encoder experiment, set allow_encoder_adaptation: true in the config."
        )
    if leaked:
        logging.warning(
            "allow_encoder_adaptation is set: %d encoder parameters are trainable "
            "(first: %s). This run is OUTSIDE the frozen-encoder scope that every other "
            "claim in the study assumes, and must be reported separately.",
            len(leaked), leaked[:3],
        )

    logging.info(f"Set trainability based on target_modules: {target_modules}")
    logging.info(f"Trainable parameter names: {trainable_names}")
    logging.info(f"Trainable params: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")
    return trainable_names


def resolve_shuffle_seed(config, fraction: float) -> Tuple[int, str]:
    """Pick the data-order seed according to ``config.data_order_seed_mode``.

    Under "auto" (the campaign default) full-data runs vary their data order with the
    run seed, giving full fine-tuning a genuine data-order noise source; fractional
    runs keep the fixed data_seed so that varying the seed does not also change *which*
    samples the run sees (PartitionedDataset slices the shuffled stream).
    """
    mode = getattr(config, "data_order_seed_mode", "auto")
    if mode == "seed":
        return config.seed, mode
    if mode == "fixed":
        return config.data_seed, mode
    if mode == "auto":
        return (config.seed if fraction >= 1.0 else config.data_seed), mode
    raise ValueError(f"Unknown data_order_seed_mode: {mode!r}")



def _config_path_from_argv() -> Optional[str]:
    """Recover --config_path so results.json records it (it was always null before)."""
    for i, a in enumerate(sys.argv):
        if a == "--config_path" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--config_path="):
            return a.split("=", 1)[1]
    return None


def _best_checkpoint_step(metrics_path: Path, ckpt_steps: List[int]) -> Optional[Tuple[int, float]]:
    """Lowest eval_loss among steps that actually have a checkpoint on disk.

    Selection is on validation *loss*: the Trainer runs eval without
    predict_with_generate, so no in-training WER exists to select on. This is a
    deliberate cost trade-off and is recorded in the manifest.
    """
    if not metrics_path.exists():
        return None
    allowed = set(ckpt_steps)
    best: Optional[Tuple[int, float]] = None
    with open(metrics_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "eval" or "eval_loss" not in rec:
                continue
            step = rec.get("step")
            if step not in allowed:
                continue
            loss = float(rec["eval_loss"])
            if best is None or loss < best[1]:
                best = (step, loss)
    return best


def run_best_vs_final(
    *, config, model, processor, output_dir: Path, ckpt_steps: List[int], final_state
) -> Dict[str, Any]:
    """Score the best-validation checkpoint against the final model (action B8).

    R2-6c: evaluating only the final model can specifically penalise the deep
    full-parameter configurations that the paper argues are "harder to optimize", so
    the ranking has to be shown to survive checkpoint selection. Runs here, in-job,
    because the checkpoints do not outlive this process.
    """
    from safetensors.torch import load_file

    best = _best_checkpoint_step(output_dir / "metrics.jsonl", ckpt_steps)
    if best is None:
        raise RuntimeError("No eval_loss record coincides with a written checkpoint.")
    best_step, best_loss = best

    eval_args = dataclasses.replace(
        config.eval_dataset_args, max_samples=config.best_vs_final_max_samples
    )
    eval_cfg = config.get_eval_sets()[0]
    spec = decoding.spec_from_config(config, path="best_vs_final")

    def _score(tag: str) -> Dict[str, Any]:
        ds = prepare_dataset([eval_cfg], eval_args)
        m, _, _ = decoding.transcribe_and_score(model, processor, ds, spec, desc=tag)
        return m

    ckpt_path = output_dir / "ckpt" / f"step_{best_step:07d}.safetensors"
    logging.info(f"best-vs-final: loading best checkpoint step {best_step} (eval_loss={best_loss:.5f})")
    missing, unexpected = model.load_state_dict(load_file(str(ckpt_path)), strict=False)
    if unexpected:
        logging.warning(f"best-vs-final: {len(unexpected)} unexpected keys in checkpoint")
    best_metrics = _score(f"best@{best_step}")

    logging.info("best-vs-final: restoring final weights")
    model.load_state_dict(final_state, strict=False)
    final_metrics = _score("final")

    result = {
        "eval_set": eval_cfg.name,
        "n_utts": final_metrics["denominator"],
        "selection_metric": "eval_loss",
        "checkpoint_steps": ckpt_steps,
        "best_step": best_step,
        "best_eval_loss": best_loss,
        "wer_best": best_metrics["wer"],
        "wer_final": final_metrics["wer"],
        "delta_wer_final_minus_best": final_metrics["wer"] - best_metrics["wer"],
        "spec": spec.to_dict(),
    }
    logging.info(f"best-vs-final: {result}")
    return result


def main():
    load_dotenv()
    os.environ.setdefault("WANDB_MODE", "offline")
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
        config.get_val_sets(), config.val_dataset_args, raw_dicts=config.val_sets
    )
    
    val_dataset_proc = WhisperDataproc(val_dataset, processor)

    # Determine total training samples from full dataset to calculate fractions
    # We need to instantiate the full train dataset once to get its length
    full_train_dataset = prepare_dataset(
        config.get_train_sets(), config.train_dataset_args, raw_dicts=config.train_sets
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
        
        # Limit number of subsets to total available partitions if needed.
        # subset_index offsets the window so one job can run a single named partition
        # (see configs.TrainConfig.subset_index); at its default of 0 this is the
        # original range(num_subsets).
        first_subset = config.subset_index
        if not 0 <= first_subset < total_partitions:
            raise ValueError(
                f"subset_index {first_subset} out of range for fraction {fraction} "
                f"({total_partitions} partitions available)"
            )
        num_subsets_to_run = min(config.num_subsets, total_partitions - first_subset)

        for i in range(first_subset, first_subset + num_subsets_to_run):
            logging.info(
                f"Subset {i - first_subset + 1}/{num_subsets_to_run} (Partition Index: {i})"
            )

            # Re-seed per run. set_seed was previously called once before the loops, so
            # on the 2nd+ (fraction, subset) iteration the RNG state depended on loop
            # history and a run was not reproducible in isolation.
            transformers.set_seed(config.seed)

            # 0. Setup Output Directory & Logging
            # Extract experiment ID from output_dir (e.g. "outputs/002" -> "002")
            if config.run_name:
                exp_id = config.run_name
            else:
                exp_id = config.output_dir.name

            run_name = f"{exp_id}_frac_{fraction}_subset_{i}"
            output_dir = config.output_dir / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            run_started_at = datetime.datetime.now().astimezone().isoformat()
            run_t0 = time.perf_counter()
            # Written before anything can fail, so a crashed job is still self-describing.
            run_manifest = mf.build_pre_manifest(
                config=config,
                run_name=run_name,
                output_dir=output_dir,
                fraction=fraction,
                partition_index=i,
                total_partitions=total_partitions,
                started_at=run_started_at,
            )
            mf.write_json(output_dir / "manifest_pre.json", run_manifest)

            # Add file handler to logging for this specific run
            file_handler = logging.FileHandler(output_dir / "train.log")
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)

            # 1. Initialize Model
            model = WhisperForConditionalGeneration.from_pretrained(
                config.model_id,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation_dropout=config.activation_dropout
            )
            # Log dropout
            logging.info(f"Initialized model with dropout={config.dropout}, attention_dropout={config.attention_dropout}, activation_dropout={config.activation_dropout}")
            logging.info(f"Model config dropout={model.config.dropout}, attention_dropout={model.config.attention_dropout}, activation_dropout={model.config.activation_dropout}")
            
            # Apply layer-specific dropout if configured
            if config.layer_specific_dropout_layers:
                num_layers_applied = 0
                for layer_idx in config.layer_specific_dropout_layers:
                    if layer_idx < len(model.model.decoder.layers):
                        layer = model.model.decoder.layers[layer_idx]
                        layer.dropout = config.layer_specific_dropout
                        layer.activation_dropout = config.layer_specific_activation_dropout
                        layer.self_attn.dropout = config.layer_specific_attention_dropout
                        layer.encoder_attn.dropout = config.layer_specific_attention_dropout
                        num_layers_applied += 1
                    else:
                        logging.warning(f"Requested layer-specific dropout for layer {layer_idx}, but decoder only has {len(model.model.decoder.layers)} layers.")
                logging.info(f"Applied layer-specific dropout to {num_layers_applied} decoder layers: {config.layer_specific_dropout_layers}")
                logging.info(f"Layer-specific dropout values - dropout: {config.layer_specific_dropout}, attention: {config.layer_specific_attention_dropout}, activation: {config.layer_specific_activation_dropout}")

            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = []

            # Untie embeddings and reinitialize output weights.
            # See manifest.UNTIE_PROCEDURE for the description recorded per run (R1-7.3).
            untied = False
            if model.config.tie_word_embeddings:
                model.config.tie_word_embeddings = False
                # Clone the input embeddings to initialize the output projection
                # Whisper's output projection is typically 'proj_out'
                if hasattr(model, "proj_out"):
                     # The decoder embeddings are usually the source
                     decoder_embed_weight = model.model.decoder.embed_tokens.weight
                     model.proj_out = torch.nn.Linear(decoder_embed_weight.shape[1], decoder_embed_weight.shape[0], bias=False)
                     model.proj_out.weight.data.copy_(decoder_embed_weight.data)
                     # Assert the untie actually took: proj_out must now be a distinct
                     # tensor holding the same values. A silent failure here would make
                     # L0 train nothing at all.
                     assert model.proj_out.weight.data_ptr() != decoder_embed_weight.data_ptr(), \
                         "proj_out is still tied to embed_tokens after untying"
                     assert torch.equal(
                         model.proj_out.weight.data, decoder_embed_weight.data
                     ), "proj_out was not initialised from embed_tokens"
                     untied = True
                     logging.info(
                         f"Untied proj_out from decoder.embed_tokens: "
                         f"{tuple(model.proj_out.weight.shape)} "
                         f"({model.proj_out.weight.numel():,} params now independently trainable)"
                     )
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
                    n_reset_total = 0
                    for block_idx, block in enumerate(config.lora_config.init_blocks):
                        logging.info(f"Initializing block {block_idx+1}: {block}")
                        # Reset global and library seeds to the base experiment seed
                        transformers.set_seed(config.seed)

                        # Traverse model and re-init matching modules
                        n_reset = 0
                        for name, module in model.named_modules():
                            if any(target in name for target in block):
                                if hasattr(module, "reset_lora_parameters"):
                                    module.reset_lora_parameters("default", init_lora_weights=True)
                                    n_reset += 1
                                    logging.debug(f"Reset LoRA parameters for {name}")
                        # A typo in init_blocks used to fail silently, leaving the
                        # cross-config initialisation guarantee quietly unmet.
                        if n_reset == 0:
                            logging.warning(
                                f"init_blocks entry {block!r} matched no LoRA module -- "
                                f"initialisation is NOT synchronised for it."
                            )
                        n_reset_total += n_reset
                    logging.info(f"Re-initialised {n_reset_total} LoRA modules across init_blocks.")

            else:
                set_trainable_parameters(
                    model,
                    config.target_modules,
                    allow_encoder_adaptation=getattr(
                        config, "allow_encoder_adaptation", False),
                )

            resolved_modules = mf.resolve_target_modules(
                model, config.target_modules, is_lora=config.lora_config is not None
            )

            # 2. Prepare Training Dataset for this specific run
            # The stream order is controlled by shuffle_seed; PartitionedDataset then
            # picks the i-th modulo slice. See resolve_shuffle_seed for the policy.
            effective_shuffle_seed, seed_mode = resolve_shuffle_seed(config, fraction)
            logging.info(
                f"Data-order policy '{seed_mode}': shuffle_seed={effective_shuffle_seed} "
                f"(seed={config.seed}, data_seed={config.data_seed}, fraction={fraction})"
            )
            train_args = dataclasses.replace(
                config.train_dataset_args,
                shuffle_seed=effective_shuffle_seed,
                shuffle=True
            )

            # Get the base streaming dataset
            base_train_dataset = prepare_dataset(
                config.get_train_sets(), 
                train_args,
                max_samples=None, # Do not limit yet
                raw_dicts=config.train_sets
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

            if config.log_dataset_metadata:
                compute_and_save_dataset_metadata(train_dataset, output_dir / "dataset_metadata.json")

                        # Calculate max_steps based on num_epochs, batch size, and gradient accumulation
            if config.num_epochs <= 0:
                logging.info("num_epochs <= 0. Calculating steps based on FULL dataset size.")
                num_update_steps_per_epoch = total_train_samples // (config.batch_size * config.grad_accum_steps)
                calculated_max_steps = num_update_steps_per_epoch
                if config.max_steps_fraction < 1.0:
                    calculated_max_steps = int(calculated_max_steps * config.max_steps_fraction)
                    logging.info(f"Applying max_steps_fraction of {config.max_steps_fraction}, new calculated_max_steps: {calculated_max_steps}")
                effective_num_epochs = 1.0  # Use local variable instead of mutating config
            else:
                num_update_steps_per_epoch = len(train_dataset) // (config.batch_size * config.grad_accum_steps)
                effective_num_epochs = config.num_epochs
                calculated_max_steps = int(num_update_steps_per_epoch * effective_num_epochs)

            if num_update_steps_per_epoch == 0:
                 # If dataset is smaller than batch * grad_accum, we at least have 1 step if possible, or 0.
                 # Warn about this.
                 logging.warning(f"Dataset length ({len(train_dataset)}) is smaller than effective batch size ({config.batch_size * config.grad_accum_steps}).")
            
            logging.info(f"Calculated max_steps: {calculated_max_steps} (Epochs: {effective_num_epochs}, Steps/Epoch: {num_update_steps_per_epoch}, Dataset Len: {len(train_dataset)})")

            # Calculate warmup steps
            if config.warmup_ratio > 0:
                warmup_steps = int(calculated_max_steps * config.warmup_ratio)
                logging.info(f"Using warmup_ratio {config.warmup_ratio}. Calculated warmup_steps: {warmup_steps}")
            else:
                warmup_steps = config.warmup_steps
                logging.info(f"Using fixed warmup_steps: {warmup_steps}")

            # Preview the head of the training stream. This doubles as the
            # non-empty check and as the data-order fingerprint that the smoke
            # test's --seed-divergence check compares across runs.
            #
            # ONE pass, deliberately. This previously opened three separate
            # iterators (raw preview, processed preview, and a next(iter(...))
            # empty-check) plus the Trainer's own. On a streaming dataset each
            # iterator restarts the shuffle buffer from scratch -- measured at
            # ~8.5 minutes per startup for VoxPopuli at shuffle_buffer_size=1000
            # -- so the job spent ~34 minutes filling buffers before its first
            # optimizer step. WhisperDataproc carries the raw text through as
            # "reference", so a single pass over train_dataset_proc yields both
            # the raw and the tokenised view.
            first_sample_texts = []
            n_preview = max(0, config.preview_samples)
            logging.info(f"Previewing first {n_preview} training samples (single pass)...")
            try:
                for idx, sample in enumerate(train_dataset_proc):
                    if idx >= n_preview:
                        break
                    raw = sample.get("reference")
                    if raw is not None:
                        first_sample_texts.append(raw)
                        logging.info(f"Sample {idx+1} (raw): {raw}")
                    if "text_input_ids" in sample:
                        decoded = processor.decode(sample["text_input_ids"], skip_special_tokens=True)
                        logging.info(f"Sample {idx+1} (decoded): {decoded}")
            except Exception as e:
                logging.warning(f"Failed to preview training samples: {e}")

            if n_preview > 0 and not first_sample_texts:
                raise ValueError(
                    f"Training dataset '{train_dataset.name}' yielded no samples after "
                    f"filtering! Stopping before training."
                )

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
                num_train_epochs=effective_num_epochs,
                weight_decay=config.weight_decay,
                fp16=config.fp16,
                logging_steps=config.logging_steps,
                do_train=config.do_train,
                do_eval=config.do_train,
                do_predict=config.do_predict,
                eval_strategy="steps" if config.do_train else "no",
                eval_steps=config.eval_steps,
                eval_on_start=config.eval_on_start,
                save_strategy="no",  # checkpointing is custom; see callbacks.py
                # WITHOUT seed=, TrainingArguments.seed stays at its default of 42 and
                # Trainer.__init__ calls set_seed(42), so config.seed only ever affected
                # model/LoRA init -- dropout masks and any post-construction randomness
                # were identical across "different" seeds.
                seed=config.seed,
                data_seed=config.data_seed,
                # Explicit rather than inherited: R1-7.1 asks for these by name.
                optim=config.optim,
                lr_scheduler_type=config.lr_scheduler_type,
                max_grad_norm=config.max_grad_norm,
                adam_beta1=config.adam_beta1,
                adam_beta2=config.adam_beta2,
                adam_epsilon=config.adam_epsilon,
                # HF's own memory deltas, as a cross-check on CostCallback (action B6).
                skip_memory_metrics=False,
                report_to=["tensorboard"],
                remove_unused_columns=False, # Required for custom collator/dataset
                label_names=["labels"],
            )

            metrics_cb = cb.JsonlMetricsCallback(output_dir / "metrics.jsonl")
            cost_cb = cb.CostCallback()
            ckpt_cb = cb.PeriodicTrainableCheckpointCallback(
                ckpt_dir=output_dir / "ckpt",
                save_fn=lambda m, p: save_trainable_state(
                    m, p,
                    allow_encoder_adaptation=getattr(
                        config, "allow_encoder_adaptation", False),
                ),
                fraction=config.checkpoint_fraction,
                max_steps=calculated_max_steps,
                eval_steps=config.eval_steps,
            )
            progress_cb = cb.ProgressHeartbeatCallback(
                output_dir / "progress.json", calculated_max_steps
            )

            trainer = Seq2SeqTrainer(
                args=training_args,
                model=model,
                train_dataset=train_dataset_proc,
                eval_dataset=val_dataset_proc,
                data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
                processing_class=processor.feature_extractor,
                callbacks=[metrics_cb, cost_cb, ckpt_cb, progress_cb],
            )

            # 4. Train
            # (The empty-dataset check lives in the preview pass above; repeating it
            # here would cost another full shuffle-buffer refill.)
            training_time = 0
            train_metrics: Dict[str, Any] = {}
            if config.do_train:
                # The step-0 anchor now comes from eval_on_start, which routes through
                # the callbacks; the old manual trainer.evaluate() logged outside them.
                train_result = trainer.train()
                train_metrics = dict(train_result.metrics)
                training_time = train_metrics.get("train_runtime", 0)
                logging.info(f"Training completed in {training_time} seconds.")

                # 5. Persist weights -- trainable parameters only.
                # The previous code merged LoRA and then wrote the whole ~3 GB model,
                # so adapters were never saved and the frozen encoder was stored with
                # every run.
                save_mode = config.save_mode
                if save_mode == "auto":
                    save_mode = "adapter" if config.lora_config is not None else "trainable"

                weight_info: Dict[str, Any] = {"save_mode": save_mode}
                final_state = trainable_state_dict(model)

                if save_mode == "adapter":
                    # Must happen while the model is still a PeftModel.
                    adapter_dir = output_dir / "adapter"
                    model.save_pretrained(str(adapter_dir))
                    weight_info["path"] = str(adapter_dir)
                    weight_info["bytes"] = sum(
                        p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file()
                    )
                elif save_mode == "trainable":
                    weight_info.update(
                        save_trainable_state(
                            model, output_dir / "final.safetensors",
                            allow_encoder_adaptation=getattr(
                                config, "allow_encoder_adaptation", False),
                        )
                    )
                elif save_mode == "full":
                    trainer.save_model()
                    weight_info["path"] = str(output_dir)
                else:
                    raise ValueError(f"Unknown save_mode: {save_mode!r}")
                logging.info(f"Saved weights: {weight_info}")

                processor.save_pretrained(str(output_dir))

            # Cast once, to the decode dtype, so the in-job numbers and the standalone
            # vanilla baseline are produced with identical numerics.
            decode_dtype = decoding.torch_dtype(config.decode_dtype)
            model = model.to(decode_dtype)

            # 5b. Best-val checkpoint vs final model (action B8 / R2-6c), performed
            # here because the checkpoints are deleted at the end of this job.
            best_vs_final = None
            bvf_t0 = time.perf_counter()
            if config.do_train and config.run_best_vs_final and ckpt_cb.steps:
                try:
                    best_vs_final = run_best_vs_final(
                        config=config,
                        model=model,
                        processor=processor,
                        output_dir=output_dir,
                        ckpt_steps=ckpt_cb.steps,
                        final_state=final_state,
                    )
                    mf.write_json(output_dir / "best_vs_final.json", best_vs_final)
                except Exception as exc:
                    logging.warning(f"best-vs-final comparison failed: {exc}")
                    best_vs_final = {"error": str(exc)}
            elif config.run_best_vs_final:
                logging.warning("run_best_vs_final requested but no checkpoints were written.")
            bvf_wall_s = time.perf_counter() - bvf_t0

            # 5c. Merge LoRA only now that adapters are saved and B8 is done.
            if config.do_train and config.lora_config is not None:
                logging.info("Merging LoRA adapters into base model for evaluation...")
                model = model.merge_and_unload()
                trainer.model = model

            # 6. Run Inference (in-domain + OOD, fixed and legacy paths)
            all_metrics = {}
            infer_t0 = time.perf_counter()
            if config.do_predict:
                logging.info("Running post-training inference on EVAL + OOD sets...")
                all_metrics = evaluate_datasets(
                    config=config,
                    model=model,
                    processor=processor,
                    batch_size=config.eval_batch_size,
                    device=trainer.args.device,
                    output_dir=output_dir,
                    model_path=output_dir,
                    config_path=_config_path_from_argv(),
                    training_time=training_time,
                )
            inference_wall_s = time.perf_counter() - infer_t0

            # 7. Cost metrics (action B6 / R1-6)
            steps_completed = trainer.state.global_step
            effective_bs = config.batch_size * config.grad_accum_steps * max(1, trainer.args.world_size)
            cost = {
                "train_wall_s": round(cost_cb.train_wall_s, 3),
                "train_wall_excl_eval_s": round(cost_cb.train_wall_excl_eval_s, 3),
                "eval_wall_s": round(cost_cb.eval_wall_s, 3),
                "best_vs_final_wall_s": round(bvf_wall_s, 3),
                "inference_wall_s": round(inference_wall_s, 3),
                "run_wall_s": round(time.perf_counter() - run_t0, 3),
                "peak_mem_alloc_bytes": cost_cb.peak_mem_alloc_bytes,
                "peak_mem_reserved_bytes": cost_cb.peak_mem_reserved_bytes,
                "peak_mem_alloc_gib": (
                    round(cost_cb.peak_mem_alloc_bytes / 2**30, 3)
                    if cost_cb.peak_mem_alloc_bytes
                    else None
                ),
                "max_steps": calculated_max_steps,
                "steps_completed": steps_completed,
                # May exceed the nominal epoch count: max_steps is derived from the
                # declared split size, while filtering means fewer samples actually
                # reach the model, so the iterator wraps. Recorded, not hidden.
                "epochs_consumed": trainer.state.epoch,
                "effective_batch_size": effective_bs,
                "samples_seen": steps_completed * effective_bs,
                "hf_train_metrics": train_metrics,
                "trainable_params": resolved_modules.get("n_trainable_parameters"),
                "weights": weight_info if config.do_train else None,
                "checkpoint_steps": ckpt_cb.steps,
                "fingerprint": mf.env_fingerprint(),
            }
            if cost_cb.train_wall_excl_eval_s > 0:
                cost["train_samples_per_second"] = round(
                    steps_completed * effective_bs / cost_cb.train_wall_excl_eval_s, 4
                )
                cost["train_steps_per_second"] = round(
                    steps_completed / cost_cb.train_wall_excl_eval_s, 4
                )
            # total_flos is HF's 6*params*tokens proxy, not a measured FLOP count.
            cost["total_flos_proxy"] = train_metrics.get("total_flos")
            mf.write_json(output_dir / "cost.json", cost)

            # 8. Final manifest. Written before checkpoint cleanup so that a crash
            # during cleanup leaves the run looking incomplete and it gets re-run.
            run_manifest["status"] = "complete"
            run_manifest["run"]["finished_at"] = datetime.datetime.now().astimezone().isoformat()
            run_manifest["adaptation"].update(
                {
                    "target_modules": resolved_modules,
                    "lora": (
                        dataclasses.asdict(config.lora_config)
                        if config.lora_config is not None
                        else None
                    ),
                }
            )
            run_manifest["model"].update(
                {
                    "untied_embeddings": untied,
                    "dtype_train": "fp16-amp" if config.fp16 else "fp32",
                    "dtype_decode": config.decode_dtype,
                    "dropout": config.dropout,
                    "attention_dropout": config.attention_dropout,
                    "activation_dropout": config.activation_dropout,
                }
            )
            run_manifest["optimization"] = {
                "optim": config.optim,
                "lr_scheduler_type": config.lr_scheduler_type,
                "learning_rate": config.learning_rate,
                "warmup_steps": warmup_steps,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "max_grad_norm": config.max_grad_norm,
                "adam_beta1": config.adam_beta1,
                "adam_beta2": config.adam_beta2,
                "adam_epsilon": config.adam_epsilon,
                "fp16": config.fp16,
                "batch_size": config.batch_size,
                "grad_accum_steps": config.grad_accum_steps,
                "effective_batch_size": effective_bs,
            }
            run_manifest["schedule"].update(
                {
                    "num_epochs_requested": config.num_epochs,
                    "max_steps_source": "full_dataset" if config.num_epochs <= 0 else "subset",
                    "total_train_samples_full": total_train_samples,
                    "subset_samples": len(train_dataset),
                    "num_update_steps_per_epoch": num_update_steps_per_epoch,
                    "max_steps": calculated_max_steps,
                    "steps_completed": steps_completed,
                    "epochs_consumed": trainer.state.epoch,
                }
            )
            run_manifest["seeding"] = {
                "seed": config.seed,
                "training_args_seed": trainer.args.seed,
                "data_seed": config.data_seed,
                "data_order_seed_mode": seed_mode,
                "effective_shuffle_seed": effective_shuffle_seed,
                "shuffle_buffer_size": config.train_dataset_args.shuffle_buffer_size,
                "strict_reproducibility": config.strict_reproducibility,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "dataloader_num_workers": config.dataloader_num_workers,
                "first_train_samples": first_sample_texts,
            }
            run_manifest["data"] = {
                "train_sets": config.train_sets,
                "val_sets": config.val_sets,
                "eval_sets": config.eval_sets,
                "ood_eval_sets": config.ood_eval_sets,
                "max_label_length": max_label_length,
                "train_pass_counts": getattr(base_train_dataset, "last_pass_counts", None),
            }
            run_manifest["decoding"] = decoding.spec_from_config(config).to_dict()
            run_manifest["artifacts"] = {
                "metrics_jsonl": "metrics.jsonl",
                "cost_json": "cost.json",
                "tensorboard_dir": "runs",
                "eval_dir": "eval",
                "weights": weight_info if config.do_train else None,
                "best_vs_final_json": "best_vs_final.json" if best_vs_final else None,
            }
            run_manifest["results"] = {
                name: {
                    "wer_fixed": m.get("wer_fixed"),
                    "wer_legacy": m.get("wer_legacy"),
                    "denominator": m.get("denominator"),
                    "denominator_legacy": m.get("denominator_legacy"),
                    "kind": m.get("kind"),
                }
                for name, m in all_metrics.items()
            }
            if best_vs_final:
                run_manifest["results"]["best_vs_final"] = best_vs_final
            mf.write_json(output_dir / "run_manifest.json", run_manifest)

            # 9. Storage: drop intermediate checkpoints now that B8 has consumed them.
            ckpt_dir = output_dir / "ckpt"
            if ckpt_dir.exists():
                freed = sum(p.stat().st_size for p in ckpt_dir.glob("*") if p.is_file())
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                logging.info(f"Removed intermediate checkpoints ({freed / 2**20:.1f} MiB freed).")

            # Cleanup
            metrics_cb.close()
            del model
            del trainer

            # Remove file handler to prevent duplicate logs in next iteration
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
