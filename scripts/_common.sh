# Shared helpers for the single-run scripts. Source, do not execute.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the Python interpreter for a standalone run.
# Explicit overrides and active environments take precedence; otherwise prefer
# Poetry's project environment, then a repository .venv, then python3 on PATH.
resolve_py() {
    if [[ -n "${PY:-}" ]]; then
        printf '%s' "$PY"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        printf '%s' "${VIRTUAL_ENV}/bin/python"
    elif command -v poetry >/dev/null 2>&1; then
        local poetry_py
        poetry_py="$(cd "$REPO_ROOT" && poetry env info --executable 2>/dev/null || true)"
        if [[ -x "$poetry_py" ]]; then
            printf '%s' "$poetry_py"
        elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
            printf '%s' "${REPO_ROOT}/.venv/bin/python"
        else
            command -v python3
        fi
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

require_dependencies() {
    local py="$1"
    if ! "$py" -c 'import simple_parsing, torch, yaml' >/dev/null 2>&1; then
        echo "Error: project dependencies are not installed for $py." >&2
        echo "Run: poetry install --without dev (or use an activated compatible environment)." >&2
        return 1
    fi
}
