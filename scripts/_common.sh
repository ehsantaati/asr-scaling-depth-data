# Shared helpers for the runner scripts. Source, do not execute.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the Python interpreter, container-first.
#
# Order matters. docker-compose bind-mounts the repo over /app, so the *host*
# .venv is visible inside the container at /app/.venv -- built for the host, with
# host absolute paths baked into its scripts. Picking it up inside the container
# would silently run the wrong environment (this is exactly how an earlier
# verification run reported host package versions while claiming to test the image).
#
#   1. explicit $PY               -- caller knows best
#   2. $VIRTUAL_ENV               -- the image sets this to /opt/venv, outside the mount
#   3. <repo>/.venv               -- host development environment
#   4. python3 on PATH            -- last resort
resolve_py() {
    if [[ -n "${PY:-}" ]]; then
        printf '%s' "$PY"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        printf '%s' "${VIRTUAL_ENV}/bin/python"
    elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        printf '%s' "${REPO_ROOT}/.venv/bin/python"
    else
        command -v python3
    fi
}

require_py() {
    local py; py="$(resolve_py)"
    if [[ -z "$py" || ! -x "$py" ]]; then
        echo "Error: no usable Python interpreter (set PY=/path/to/python)" >&2
        return 1
    fi
    printf '%s' "$py"
}
