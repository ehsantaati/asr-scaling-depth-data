import os
from pathlib import Path

from .. import types

# Path to the local HF loading script (data/local_datasets/spgispeech_2/).
# Resolved relative to this file so the repo is portable; the previous hardcoded
# /mnt/asr-data-scaling/... path does not exist on the current host. Override with
# SPGISPEECH2_LOADER if the loader lives elsewhere. The corpus itself (shards/,
# alignment_files/) is located separately by SPGISPEECH2_ROOT — see
# data/local_datasets/spgispeech_2/spgispeech_2.py.
_LOADER_DIR = str(Path(__file__).resolve().parents[1] / "local_datasets" / "spgispeech_2")
SPGISPEECH_2_LOADER = os.environ.get("SPGISPEECH2_LOADER", _LOADER_DIR)

SPGISPEECH_2_CONFIG = types.DatasetConfig(
    name="spgispeech_2",
    path=SPGISPEECH_2_LOADER,
    transcript_field="transcript",
    audio_field="audio",
    streaming=False,
    # Local loading script; there is no parquet export to fall back to.
    trust_remote_code=True,
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=512724),
        types.DatasetSplitConfig(
            name="validation", num_samples=6195, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=6932),
    ],
)

configs = [
    SPGISPEECH_2_CONFIG,
]
