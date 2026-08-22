"""Writing text files that look the same on every platform.

`Path.write_text` translates newlines to the platform default, so a file written
on Windows gets carriage returns and the same file written on Linux does not.
For generated files that are committed and then checked against a regenerated
copy, that difference is a permanent spurious diff, and the usual patch for it
is a `.gitattributes` rule that hides the problem at the repository boundary
rather than fixing it at the source.

Everything here writes LF explicitly. The committed results then match what a
regeneration produces on any machine, and `make check` compares content rather
than line endings.

`tests/test_packaging.py` asserts no committed generated file contains a
carriage return, which is what stops this drifting back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

#: Line ending for every file this repository generates. Spelled via chr rather
#: than an escape so it survives being edited through a shell heredoc.
LF = chr(10)


def write_text(path: str | Path, text: str) -> Path:
    """Write text with LF line endings regardless of platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalised = text.replace(chr(13) + LF, LF).replace(chr(13), LF)
    path.write_text(normalised, encoding="utf-8", newline=LF)
    return path


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Write pretty JSON with a trailing newline and LF endings."""
    return write_text(path, json.dumps(payload, indent=indent) + LF)


def write_csv(frame: pd.DataFrame, path: str | Path, float_format: str = "%.6g") -> Path:
    """Write a CSV with LF endings.

    pandas honours ``lineterminator`` only when it opens the file itself, so the
    path is handed over rather than an already-open handle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format=float_format, lineterminator=LF)
    return path


def has_carriage_return(path: str | Path) -> bool:
    """True if the file contains a CR. Reads bytes, so encoding cannot mask it."""
    return chr(13).encode() in Path(path).read_bytes()
