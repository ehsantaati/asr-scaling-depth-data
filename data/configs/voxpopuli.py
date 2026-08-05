from .. import types

VOXPOPULI_BASE_CONFIG = types.DatasetConfig(
    name="voxpopuli",
    path="facebook/voxpopuli",
    transcript_field="normalized_text",
    audio_field="audio",
    # Download and cache rather than stream. DatasetConfig defaults streaming to
    # True, and streaming VoxPopuli was measured at ~2 samples/s while still pulling
    # whole shards over the network -- so the bandwidth is spent either way, but
    # nothing is kept. Across a 3-5 seed multi-seed cell that is the same download
    # repeated per seed, and a single full-data B1 pass would take ~25 h of pure I/O.
    # Cached, the corpus is fetched once and every later run reads local disk.
    # GigaSpeech and SPGISpeech already do this.
    streaming=False,
)

VOXPOPULI_EN_CONFIG = types.DatasetConfig(
    name="voxpopuli-en",
    base="voxpopuli",
    subset="en",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=182482, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="validation", num_samples=1753, split=types.DatasetSplit.VALIDATION
        ),
        types.DatasetSplitConfig(
            name="test", num_samples=1842, split=types.DatasetSplit.TEST
        ),
    ],
)

VOXPOPULI_EN_ACCENTED_CONFIG = types.DatasetConfig(
    name="voxpopuli-en-accented",
    base="voxpopuli",
    subset="en_accented",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=8387, split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    VOXPOPULI_BASE_CONFIG,
    VOXPOPULI_EN_CONFIG,
    VOXPOPULI_EN_ACCENTED_CONFIG,
]
