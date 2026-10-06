"""Per-run reproducibility manifest.

Answers R1 §7 (7.1-7.4) in machine-readable form, and feeds writing task W3. A
manifest is written twice per run:

  * ``manifest_pre.json`` right after the run directory is created, so a job that
    crashes mid-training is still self-describing;
  * ``run_manifest.json`` at the end, with resolved counts, results and a
    ``status`` field. Completed artifacts are authoritative for executed
    schedules and reproducibility checks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2

# Literal description of the input/output embedding untying performed in train.py.
# Record this procedure so independent runs can reproduce initialization.
UNTIE_PROCEDURE = (
    "Whisper ties proj_out to model.decoder.embed_tokens. Before adaptation we set "
    "model.config.tie_word_embeddings=False and replace model.proj_out with a fresh "
    "torch.nn.Linear(d_model, vocab_size, bias=False) whose weight is initialised by "
    "copying model.model.decoder.embed_tokens.weight. The model is therefore "
    "numerically identical to the pretrained checkpoint at step 0, but proj_out.weight "
    "is an independent parameter (51865 x 1024 = 53.1M params) that can be adapted "
    "separately; this is the entire trainable budget of the L0 configuration."
)

PARTITION_RULE = (
    "Samples are drawn from the split shuffled with shuffle_seed, then "
    "PartitionedDataset keeps those whose 0-based stream index i satisfies "
    "i % total_partitions == partition_index, with total_partitions = round(1/fraction)."
)


def _run_git(args: List[str], cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_provenance(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Commit, branch and dirty-state of the working tree that produced the run."""
    root = repo_root or str(Path(__file__).resolve().parent)
    diff = _run_git(["diff", "HEAD"], root)
    return {
        "commit": _run_git(["rev-parse", "HEAD"], root),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(diff),
        # Hash rather than the diff itself: identifies an uncommitted state without
        # bloating the manifest.
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def env_fingerprint() -> Dict[str, Any]:
    """Hardware and library identity. B6 claims 'identical hardware'; this proves it."""
    import torch

    gpus = []
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": props.total_memory,
                }
            )
    except Exception:
        pass

    driver = None
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip().splitlines()[0]
    except Exception:
        pass

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "nvidia_driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {
            p: _pkg_version(p)
            for p in (
                "transformers",
                "peft",
                "datasets",
                "accelerate",
                "safetensors",
                "evaluate",
                "jiwer",
                "numpy",
                "librosa",
                "soundfile",
                "truecase",
            )
        },
    }


def config_to_dict(config: Any) -> Dict[str, Any]:
    """asdict() with Paths and enums coerced to strings so it round-trips through JSON."""

    def _coerce(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(v) for v in obj]
        if hasattr(obj, "value") and hasattr(obj, "name"):  # enum
            return obj.value
        return obj

    return _coerce(dataclasses.asdict(config))


def config_sha256(config: Any) -> str:
    """Hash of the *resolved* config, so YAML formatting changes don't split groups."""
    payload = json.dumps(config_to_dict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_target_modules(model, requested: Optional[List[str]], is_lora: bool) -> Dict[str, Any]:
    """Record which model modules the requested target list matched.

    PEFT matches by exact name or dotted suffix; set_trainable_parameters matches by
    name-boundary prefix. Those are different rules applied to the *same* YAML list,
    so 'requested' alone is not enough to reproduce a run.
    """
    resolved: Dict[str, Any] = {"requested": list(requested or [])}

    if is_lora:
        lora_modules = []
        try:
            from peft.tuners.lora.layer import LoraLayer

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    lora_modules.append(name)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logging.warning(f"Could not enumerate LoRA layers: {exc}")
        resolved["lora_layers"] = sorted(lora_modules)
        resolved["n_lora_layers"] = len(lora_modules)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    resolved["trainable_parameter_names"] = sorted(trainable)
    resolved["n_trainable_parameters"] = len(trainable)
    resolved["encoder_parameters_trainable"] = sorted(
        n for n in trainable if ".encoder." in n or n.startswith("model.encoder.")
    )
    return resolved


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    # Atomic within the filesystem: a reader never observes a half-written manifest.
    os.replace(tmp, path)


def build_pre_manifest(
    *,
    config: Any,
    run_name: str,
    output_dir: Path,
    fraction: float,
    partition_index: int,
    total_partitions: int,
    started_at: str,
) -> Dict[str, Any]:
    is_lora = getattr(config, "lora_config", None) is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "run": {
            "job_id": os.environ.get("QUEUE_JOB_ID"),
            "run_name": run_name,
            "output_dir": str(output_dir),
            "started_at": started_at,
            "finished_at": None,
        },
        "provenance": {
            "git": git_provenance(),
            "config_sha256": config_sha256(config),
            "config_resolved": config_to_dict(config),
            "argv": sys.argv,
        },
        "environment": env_fingerprint(),
        "model": {
            "model_id": config.model_id,
            "language": config.language,
            "task": config.task,
            "untie_procedure": UNTIE_PROCEDURE,
        },
        "adaptation": {"method": "lora" if is_lora else "full"},
        "schedule": {
            "regime": "data_limited" if config.num_epochs > 0 else "fixed_budget",
            "fraction": fraction,
            "partition_index": partition_index,
            "total_partitions": total_partitions,
            "partition_rule": PARTITION_RULE,
        },
    }
