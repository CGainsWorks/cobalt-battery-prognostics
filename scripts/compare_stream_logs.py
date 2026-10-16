#!/usr/bin/env python
"""Check a CAN stream log against the committed one.

    python scripts/compare_stream_logs.py EXPECTED ACTUAL

Used after the vcan end to end run. The committed log was made on the
in-process virtual bus; the run being checked crossed the kernel's SocketCAN
stack between two processes. The frames are the same bytes either way, so the
cycles and predictions must be identical, and the link must report no lost
frames, no checksum failures and no incomplete cycles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    expected = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    actual = json.loads(Path(argv[1]).read_text(encoding="utf-8"))

    problems: list[str] = []
    link = actual.get("link", {})
    for counter in ("checksum_errors", "frames_lost_estimate", "incomplete_cycles_dropped"):
        if link.get(counter, -1) != 0:
            problems.append(f"link reported {counter} = {link.get(counter)}")
    if actual["cycles_streamed"] != expected["cycles_streamed"]:
        problems.append(
            f"streamed {actual['cycles_streamed']} cycles, expected {expected['cycles_streamed']}"
        )
    if actual["predictions"] != expected["predictions"]:
        problems.append("predictions differ from the committed virtual-bus log")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print(
        f"{argv[1]} matches {argv[0]}: {actual['cycles_streamed']} cycles, "
        f"{link['frames_decoded']} frames, no loss, identical predictions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
