#!/usr/bin/env python
"""Check the committed generated files for two platform-specific mistakes.

    python scripts/check_generated_files.py

Both of these are invisible on the machine that makes them and break on every
other one, which is why they are checked rather than remembered.

**Carriage returns.** `Path.write_text` follows the platform, so a file written
on Windows and the same file written on Linux differ by a byte per line. For
files that are committed and then compared against a regenerated copy, which is
what `make check` does, that is a permanent spurious diff. Everything goes
through `cobalt.textio` instead.

**Absolute paths.** A `C:\\Users\\...` or `/home/runner/...` string in a
committed results file breaks every other consumer of it, silently and much
later. Nothing written here should contain one.

Lives in a file rather than inline in the workflow so it can be run locally,
linted with everything else, and edited without a shell heredoc eating the
escapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Everything the pipeline writes and the repository commits.
PATTERNS = (
    "results/**/*.json",
    "results/**/*.csv",
    "analysis/**/*.md",
    "artifacts/**/*.json",
)

#: Path prefixes that must never appear in a committed file.
ABSOLUTE_PATH_MARKERS = ("C:" + chr(92), "/home/", "/Users/", "/root/")

CARRIAGE_RETURN = chr(13).encode()


def generated_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in PATTERNS:
        found.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(found))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = generated_files(root)
    if not files:
        print("no committed generated files found, nothing to check")
        return 0

    carriage_returns: list[str] = []
    absolute_paths: list[str] = []

    for path in files:
        name = str(path.relative_to(root))
        raw = path.read_bytes()
        if CARRIAGE_RETURN in raw:
            carriage_returns.append(name)
        text = raw.decode("utf-8", errors="replace")
        for marker in ABSOLUTE_PATH_MARKERS:
            if marker in text:
                absolute_paths.append(f"{name} contains {marker!r}")

    if carriage_returns:
        print("carriage returns in committed generated files:", file=sys.stderr)
        for name in carriage_returns:
            print(f"  {name}", file=sys.stderr)
        print("  fix: write through cobalt.textio, then regenerate", file=sys.stderr)

    if absolute_paths:
        print("absolute paths in committed generated files:", file=sys.stderr)
        for problem in absolute_paths:
            print(f"  {problem}", file=sys.stderr)

    if carriage_returns or absolute_paths:
        return 1

    print(f"{len(files)} generated files: LF endings, no absolute paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
