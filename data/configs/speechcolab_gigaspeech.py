from .. import types

SPEECHCOLAB_GIGASPEECH_BASE = types.DatasetConfig(
    name="speechcolab-gigaspeech",
    path="speechcolab/gigaspeech",
    streaming=False,
    # Gated, but a parquet conversion exists for every config (m: 29 train shards),
    # so the loading script is not needed. Using it is actively worse: a load through
    # trust_remote_code=True hung for >4 minutes without producing a sample, while
    # the parquet route resolves per split. Access still requires an accepted gate
    # plus a valid HF_TOKEN.
    trust_remote_code=False,
)

SPEECHCOLAB_GIGASPEECH_M_CONFIG = types.DatasetConfig(
    name="speechcolab-gigaspeech-m",
    base="speechcolab-gigaspeech",
    subset="m",
    splits=[
        # 680072 is NOT the size of this split. The HF "m" parquet conversion holds
        # 910140 train segments / 999.95 h -- the campaign's own measurement, kept in
        # outputs/gigaspeech_dur.json, says so. Validation (6750) and test (25619) do
        # match exactly; only train is short.
        #
        # It is left wrong on purpose. train.py derives max_steps from the declared
        # count when num_epochs <= 0 (`total_train_samples // (batch * grad_accum)`)
        # and then hardcodes effective_num_epochs = 1.0, so every original full-data
        # GigaSpeech run trained 42504 steps = 680064 segments = 74.7% of M (~747 h of
        # 999.9 h) while logging "epoch 1.0". All 74 archived one-epoch runs show
        # last_step 42504 and epoch 1.0000; not one reaches 56883. The data-scaling
        # fractions inherit the same base, so "10%" is 68007 segments = 7.5% of M.
        #
        # Correcting it to 910140 would change the training budget by a third and make
        # the rerun incomparable to the numbers the revision exists to re-measure. So
        # the budget stays; what changes is that we now state it. run_manifest.json
        # records rows_read alongside declared_total, so every GigaSpeech run carries
        # the gap in its own artifacts (W3/W8, and the reviewer's "subset" question).
        types.DatasetSplitConfig(name="train", num_samples=680072),
        types.DatasetSplitConfig(
            name="validation", num_samples=6750, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(
            name="test", num_samples=25619, split=types.DatasetSplit.TEST
        ),
    ],
    audio_field="audio",
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

configs = [SPEECHCOLAB_GIGASPEECH_BASE, SPEECHCOLAB_GIGASPEECH_M_CONFIG]
