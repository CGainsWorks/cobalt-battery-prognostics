#!/usr/bin/env bash
# Set up a working environment from a clean checkout.
#
#   bash scripts/setup.sh
#
# Python 3.12 or newer. The pinned dependency set needs it: numpy 2.5 and
# scipy 1.18 both dropped 3.11.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo "no '$PYTHON' on PATH. Set PYTHON=/path/to/python and try again." >&2
        exit 1
    fi
fi

version="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$version" in
    3.12|3.13|3.14) ;;
    *)
        echo "Python $version is too old: this needs 3.12 or newer." >&2
        echo "Set PYTHON=/path/to/python3.12 and try again." >&2
        exit 1
        ;;
esac

echo "creating $VENV with Python $version"
"$PYTHON" -m venv "$VENV"

if [ -x "$VENV/bin/python" ]; then
    VENV_PYTHON="$VENV/bin/python"       # Linux and macOS
else
    VENV_PYTHON="$VENV/Scripts/python"   # Windows
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements-dev.txt
"$VENV_PYTHON" -m pip install -e .

cat <<'DONE'

Done. Next:

    bash scripts/run_demo.sh     everything, about four minutes
    bash scripts/run_smoke.sh    the same pipeline small, about two minutes

Both will use the virtual environment automatically.
DONE
