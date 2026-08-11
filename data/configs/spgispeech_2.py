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
        # 156753 is the number of source *snippets*, not of training examples. The
        # loader splits every snippet over 30 s at word boundaries into chunks of at
        # most 28 s, so the train split actually yields ~512724 chunks -- 3367 h of
        # audio at ~23.6 s per chunk. Validation (6195) and test (6932) need no such
        # split and match exactly.
        #
        # The snippet count is kept on purpose, exactly as GigaSpeech keeps 680072.
        # train.py derives max_steps from the declared count when num_epochs <= 0 and
        # then hardcodes effective_num_epochs = 1.0, so every original SPGISpeech run
        # trained 156753 // 16 = 9797 steps = 156752 chunks = 30.6% of the corpus, or
        # ~1029 h, while logging "epoch 1.0". The archived TensorBoard confirms it:
        # full-data runs stop at step 9797 with train/epoch = 0.9999, and the
        # data-scaling runs at 4898 / 1959 / 979 -- exactly 50%, 20% and 10% of that.
        # This is the origin of Table 3's ~1026 effective hours (action W5, R1-7.5).
        #
        # Raising it to 512724 would train 32045 steps, 3.27x the original budget, and
        # the rerun would no longer be comparable to the numbers this revision exists
        # to re-measure. The budget stays; what changes is that we now state it.
        # run_manifest.json records rows_read alongside declared_total, so every run
        # carries the gap in its own artifacts.
        types.DatasetSplitConfig(name="train", num_samples=156753),
        types.DatasetSplitConfig(
            name="validation", num_samples=6195, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(name="test", num_samples=6932),
    ],
)

configs = [
    SPGISPEECH_2_CONFIG,
]
