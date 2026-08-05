"""Dataset config registry.

Only modules that exist in this tree may be imported here. Commit 979a2cb removed
the edacc / libriheavy / libriheavy_hq / spgispeech configs but left them in this
import list, which made `import data.configs` — and therefore both train.py and
inference.py — fail with ImportError. Keep this list in sync with the directory.
"""

from . import (
    openslr_librispeech_asr,
    peoples_speech,
    speechcolab_gigaspeech,
    spgispeech_2,
    voxpopuli,
)

ALL_CONFIGS = (
    openslr_librispeech_asr.configs
    + peoples_speech.configs
    + speechcolab_gigaspeech.configs
    + spgispeech_2.configs
    + voxpopuli.configs
)
