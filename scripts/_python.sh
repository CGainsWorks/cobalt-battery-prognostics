# Pick an interpreter. Sourced by the other scripts, not run directly.
#
# Prefers the repository's own virtual environment, so a clean checkout followed
# by scripts/setup.sh and scripts/run_demo.sh works without anybody having to
# remember to activate anything.

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -n "${PYTHON:-}" ]; then
    :
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

export PYTHON
