#!/usr/bin/env python
"""Download and cache the corpora a campaign needs, before any job starts.

Why this exists: with streaming, HF fetches shards over the network on every run
and keeps nothing. Measured on VoxPopuli it delivered ~2 MB/s against a link that
does ~25 MB/s, so a single full-data pass would take ~25 h -- repeated for every
seed. Cached, VoxPopuli-en train is a ~62 min one-off and every later run reads
local disk.

Doing it here rather than inside the first training job means the download is
visible, resumable, and not competing with a GPU that is sitting idle waiting for it.

    python scripts/prefetch_datasets.py --sets voxpopuli-en openslr-librispeech-asr-clean
    python scripts/prefetch_datasets.py --config configs/rev/smoke/smoke_l4_lora.yaml
    python scripts/prefetch_datasets.py --sets voxpopuli-en --splits train
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets as hf_datasets  # noqa: E402

import data.configs  # noqa: E402
from data import registry, types  # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def prefetch(name: str, splits) -> None:
    cfg = registry.DATASET_MAP.get(name)
    if cfg is None:
        raise SystemExit(f"Unknown dataset: {name}")

    # Walk base configs the same way registry.create_dataset does, so we resolve the
    # same path/subset/splits the training job will use.
    chain, temp = [], name
    while temp:
        c = registry.DATASET_MAP[temp]
        chain.insert(0, c)
        temp = c.base
    merged = registry._merge_configs(chain)

    wanted = [s for s in (merged.splits or []) if splits is None or s.name in splits]
    if not wanted:
        logging.warning(f"{name}: no matching splits (have: "
                        f"{[s.name for s in (merged.splits or [])]})")
        return

    for split in wanted:
        src = split.source_split or split.name
        logging.info(f"--- {name} [{src}] ({split.num_samples} rows declared) ---")
        t0 = time.perf_counter()
        ds = hf_datasets.load_dataset(
            merged.path,
            merged.subset,
            split=src,
            streaming=False,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
            trust_remote_code=True,
        )
        dt = time.perf_counter() - t0
        nbytes = ds.dataset_size or 0
        logging.info(
            f"    cached {len(ds)} rows, {human(nbytes)} on disk in {dt/60:.1f} min"
            + (f" ({human(nbytes/dt)}/s)" if dt > 0 and nbytes else "")
        )
        if split.num_samples and len(ds) < split.num_samples:
            logging.warning(
                f"    declared num_samples={split.num_samples} but the split has "
                f"{len(ds)}; max_steps is computed from the declared value."
            )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None, help="dataset names to prefetch")
    ap.add_argument("--config", default=None,
                    help="YAML config; prefetch every train/val/eval/ood set it names")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="restrict to these split names (default: all declared)")
    args = ap.parse_args()

    registry.register_datasets(data.configs.ALL_CONFIGS)

    names = list(args.sets or [])
    if args.config:
        # Read the YAML directly rather than through simple_parsing: `load` does not
        # exist in simple_parsing 0.1.7, and all we need are the dataset names.
        import yaml

        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
        for key in ("train_sets", "val_sets", "eval_sets", "ood_eval_sets"):
            for d in raw.get(key) or []:
                n = d.get("name") if isinstance(d, dict) else d
                if n and n not in names:
                    names.append(n)
    if not names:
        raise SystemExit("Nothing to do: pass --sets and/or --config")

    logging.info(f"Prefetching: {', '.join(names)}")
    for n in names:
        prefetch(n, set(args.splits) if args.splits else None)
    logging.info("Done.")


if __name__ == "__main__":
    main()
