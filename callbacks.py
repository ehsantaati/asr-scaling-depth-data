"""TrainerCallbacks that make a run self-documenting.

These callbacks record per-run metrics and resource usage:
original checkpoints were deleted, so any diagnostic not captured while the job is
alive is lost. Four concerns, four callbacks:

  JsonlMetricsCallback   loss / grad_norm / learning_rate / eval_loss -> metrics.jsonl
  CostCallback           wall-clock, peak GPU memory, throughput      -> cost.json
  PeriodicTrainableCheckpointCallback   trainable-only checkpoints for B8
  ProgressHeartbeatCallback             progress.json for the job queue

``on_log`` is the only hook where train and eval logs converge, which is why the
metrics writer is a callback rather than inline code.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback


class JsonlMetricsCallback(TrainerCallback):
    """Append every Trainer log event to a JSONL file.

    TensorBoard already receives these, but parsing event files to build figures is
    fragile and TB was the *only* copy in the original runs. Line-buffered append
    means a crashed job still has every record up to the crash.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        record: Dict[str, Any] = {
            "step": state.global_step,
            "epoch": state.epoch,
            "wall_s": round(time.time() - self._t0, 3),
            "kind": "eval" if any(k.startswith("eval_") for k in logs) else "train",
        }
        record.update({k: v for k, v in logs.items()})
        self._f.write(json.dumps(record, default=str) + "\n")

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        self._f.flush()


class CostCallback(TrainerCallback):
    """Measure wall-clock time, peak GPU memory, and throughput.

    HF's ``skip_memory_metrics`` defaults to True and nothing in the original code
    touched ``torch.cuda.max_memory_allocated``, so no cost table could be built.
    Eval time is accumulated separately so training throughput is not diluted by
    the periodic validation passes.
    """

    def __init__(self):
        self.train_wall_s: float = 0.0
        self.eval_wall_s: float = 0.0
        self.peak_mem_alloc_bytes: Optional[int] = None
        self.peak_mem_reserved_bytes: Optional[int] = None
        self._t0: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # eval_runtime is measured by the Trainer itself; reuse it rather than
        # re-timing and risking a different definition.
        if logs and "eval_runtime" in logs:
            self.eval_wall_s += float(logs["eval_runtime"])

    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_mem_alloc_bytes = int(torch.cuda.max_memory_allocated())
            self.peak_mem_reserved_bytes = int(torch.cuda.max_memory_reserved())
        if self._t0 is not None:
            self.train_wall_s = time.perf_counter() - self._t0

    @property
    def train_wall_excl_eval_s(self) -> float:
        return max(0.0, self.train_wall_s - self.eval_wall_s)


class PeriodicTrainableCheckpointCallback(TrainerCallback):
    """Write trainable-only checkpoints at regular fractions of the run.

    HF's ``save_strategy="steps"`` is unusable here: for Whisper-Medium full FT it
    writes model *plus* AdamW state, ~9-10 GB per checkpoint. Saving only the
    trainable parameters in fp16 costs ~308 MB (L6 full FT) or ~27 MB (LoRA r=64).

    The interval is snapped to a multiple of ``eval_steps`` so every checkpoint has a
    co-located ``eval_loss`` in metrics.jsonl — otherwise best-checkpoint selection
    would have to interpolate.
    """

    def __init__(self, ckpt_dir: Path, save_fn, fraction: float, max_steps: int, eval_steps: int):
        self.ckpt_dir = Path(ckpt_dir)
        self.save_fn = save_fn
        self.steps: List[int] = []
        self.interval = 0
        if fraction and fraction > 0 and max_steps > 0:
            raw = max(1, int(round(max_steps * fraction)))
            if eval_steps > 0:
                snapped = int(round(raw / eval_steps)) * eval_steps
                self.interval = max(eval_steps, snapped)
            else:
                self.interval = raw
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            logging.info(
                f"Periodic trainable checkpoints every {self.interval} steps "
                f"(fraction={fraction}, max_steps={max_steps}, eval_steps={eval_steps})"
            )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.interval <= 0 or not state.is_world_process_zero:
            return
        step = state.global_step
        if step > 0 and step % self.interval == 0:
            path = self.ckpt_dir / f"step_{step:07d}.safetensors"
            try:
                self.save_fn(model, path)
                self.steps.append(step)
            except Exception as exc:  # never let checkpointing kill a run
                logging.warning(f"Failed to write checkpoint at step {step}: {exc}")


class ProgressHeartbeatCallback(TrainerCallback):
    """Write progress.json so the queue can show real progress and detect stalls."""

    def __init__(self, path: Path, max_steps: int):
        self.path = Path(path)
        self.max_steps = max_steps
        self._t0 = time.time()

    def _write(self, state, phase: str):
        elapsed = time.time() - self._t0
        step = state.global_step
        eta = None
        if step > 0 and self.max_steps > 0:
            eta = round(elapsed / step * max(0, self.max_steps - step), 1)
        payload = {
            "phase": phase,
            "step": step,
            "max_steps": self.max_steps,
            "epoch": state.epoch,
            "elapsed_s": round(elapsed, 1),
            "eta_s": eta,
            "updated_at": time.time(),
        }
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f)
            tmp.replace(self.path)
        except Exception:
            pass

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(state, "train")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            self._write(state, "train")

    def on_train_end(self, args, state, control, **kwargs):
        self._write(state, "post_training")
