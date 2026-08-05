# Reproducibility artifact for the revision campaign (R1-7.4): the image pins the
# same interpreter and the same locked dependency set the reported runs used.
#
# Note the campaign itself is driven from the host virtualenv (scripts/*.sh call
# .venv/bin/python). This image exists so a third party can reproduce a run without
# reconstructing that environment by hand.
ARG CUDA_VERSION=12.8.0
ARG BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04"

FROM ${BASE_IMAGE}

ENV \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    # Was PYTHONHASHSEED=random. Set-and-fixed here: this project reports seed
    # variance as a result, so no source of run-to-run randomness may be implicit.
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.4.1 \
    CUDA_VERSION=${CUDA_VERSION} \
    POETRY_HOME=/opt/poetry \
    PROJECT_ROOT=/app \
    # The venv MUST live outside PROJECT_ROOT. docker-compose bind-mounts the repo
    # over /app, and the host repo has its own .venv -- with the venv at
    # /app/.venv the mount silently shadows the image's environment with the
    # host's, so the container would run whatever happens to be installed outside
    # it. That defeats the entire point of shipping an image.
    VENV_PATH=/opt/venv \
    POETRY_VIRTUALENVS_CREATE=false \
    # Model/dataset cache lives on a bind mount so restarts do not re-download
    # hundreds of GB; see docker/docker-compose.yml.
    HF_HOME=/cache/huggingface \
    # Corpus root, overridden by the compose mount. Read by
    # data/local_datasets/spgispeech_2/spgispeech_2.py.
    SPGISPEECH2_ROOT=/data/SPGISPeech_2 \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=offline

WORKDIR ${PROJECT_ROOT}

# Ubuntu 24.04 ships CPython 3.12, which is what pyproject.toml requires and what the
# campaign ran on. The previous deadsnakes/python3.11 pin was inherited from another
# project (its stated reason, a mecab-ko conflict, is not a dependency here).
#
# Runtime libraries that the pipeline genuinely needs and the old image lacked:
#   libsndfile1  - soundfile, used by data/data_sample.py and the SPGISpeech loader
#   ffmpeg       - decoding GigaSpeech opus / VoxPopuli audio via librosa
#   jq, tmux     - required by scripts/queue_*.sh (util-linux supplies flock)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential git \
        ffmpeg libsndfile1 \
        jq tmux util-linux procps \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry in its own venv so it cannot perturb the project environment.
RUN python3.12 -m venv $POETRY_HOME \
    && $POETRY_HOME/bin/pip install --no-cache-dir poetry==$POETRY_VERSION

# Project venv at /opt/venv. POETRY_VIRTUALENVS_CREATE=false makes poetry install
# into the active environment, which VIRTUAL_ENV + PATH point at.
RUN python3.12 -m venv $VENV_PATH
ENV VIRTUAL_ENV="${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${POETRY_HOME}/bin"

# Dependency layer, cached independently of the source tree.
COPY pyproject.toml poetry.lock $PROJECT_ROOT/
# `poetry sync` (not `install --sync`, deprecated in Poetry 2.x) so the image can
# never contain a package the lock does not list. --no-root: package-mode = false.
RUN poetry sync --no-interaction --no-ansi --without dev --no-root \
    && rm -rf ~/.cache/pypoetry

# Source last, so editing code does not invalidate the dependency layer.
# .dockerignore keeps .venv/, outputs/, archive/ and the corpora out of the context.
COPY . $PROJECT_ROOT/

# Fail fast at build time rather than an hour into a run.
RUN python -c "import torch, transformers, peft, datasets, soundfile, librosa; \
    import data.configs; \
    print('deps OK |', torch.__version__, transformers.__version__, \
          '| datasets registered:', len(data.configs.ALL_CONFIGS))"

CMD ["bash"]
