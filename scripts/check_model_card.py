#!/usr/bin/env python
"""Check a model card leads with its limitation and quotes its worst protocol.

    python scripts/check_model_card.py artifacts/ci/model_card.json

A model card that quotes the flattering number is worse than no model card at
all: it launders the optimism into the one file somebody downstream will
actually read and believe. A serialised model outlives the conversation in which
somebody explained that its training data was simulated, so the card has to
carry that itself, in the first field.

Three assertions, all of which a careless edit to `cobalt.deploy` could break
without any test noticing:

1. ``data_source`` is the first key, and says the data was simulated.
2. Every model quotes ``batch_independent``, the hardest protocol available,
   rather than the one that looks best.
3. ``out_of_scope`` is populated.

Run from the workflow and usable by hand on any card the pipeline has written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: The protocol a card must quote. Anything easier is a flattering number.
REQUIRED_PROTOCOL = "batch_independent"


def check(card: dict) -> list[str]:
    problems: list[str] = []

    keys = list(card)
    if not keys or keys[0] != "data_source":
        problems.append(f"data_source must be the first field, got {keys[:3]}")
    elif "SIMULATED" not in card["data_source"]:
        problems.append(f"data_source does not say SIMULATED: {card['data_source']!r}")

    if not card.get("out_of_scope"):
        problems.append("out_of_scope is empty, so the card claims no limits at all")

    models = card.get("models")
    if not isinstance(models, dict) or not models:
        problems.append("the card lists no models")
        return problems

    for task, entry in models.items():
        reported = entry.get("reported")
        if not isinstance(reported, dict):
            problems.append(f"{task} reports no cross-validated metrics")
            continue
        protocol = reported.get("protocol")
        if protocol != REQUIRED_PROTOCOL:
            problems.append(
                f"{task} quotes {protocol!r} rather than {REQUIRED_PROTOCOL!r}, "
                f"which is the hardest protocol it has"
            )
        else:
            print(
                f"ok: {task} quotes {protocol}, "
                f"rmse={reported.get('rmse_cycles')} cycles, "
                f"picp={reported.get('picp')}"
            )

    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <model_card.json>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"no model card at {path}", file=sys.stderr)
        return 1

    problems = check(json.loads(path.read_text(encoding="utf-8")))
    if problems:
        print(f"\n{path} is not honest about itself:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"{path} leads with its limitation and quotes its worst protocol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
