"""Verify the README's tables against the results that produced them.

A results table in a README is a claim about what a script printed, and it stops
being true the first time somebody reruns the pipeline and forgets to paste the
new numbers in. That is not a hypothetical: it is the normal life cycle of every
results table in every repository.

So the tables are generated, never typed, and this check regenerates them from
``summary.json`` and diffs. Continuous integration runs it on every push, which
makes a stale number a failing build.

    python -m cobalt.check_readme
    python -m cobalt.check_readme --config config/ci.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cobalt.config import load_resolved, repository_root
from cobalt.evaluate.report import render_tables


def extract_block(text: str, name: str) -> str:
    begin = f"<!-- BEGIN:{name} -->"
    end = f"<!-- END:{name} -->"
    if begin not in text or end not in text:
        raise ValueError(f"README has no {begin} ... {end} block")
    body = text.split(begin, 1)[1].split(end, 1)[0]
    return body.strip("\n")


def check(readme_path: Path, summary_path: Path) -> list[str]:
    """Return a list of human-readable problems. Empty means the README is current."""
    if not summary_path.exists():
        return [
            f"no results at {summary_path}. Run `cobalt evaluate --config config/default.yaml` "
            f"or `make demo` first."
        ]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readme = readme_path.read_text(encoding="utf-8")
    expected = render_tables(summary)

    problems: list[str] = []
    for name, table in expected.items():
        try:
            found = extract_block(readme, name)
        except ValueError as error:
            problems.append(str(error))
            continue
        if found != table.strip("\n"):
            problems.append(
                f"block {name!r} is stale.\n"
                f"  README says:\n{_indent(found)}\n"
                f"  results say:\n{_indent(table)}"
            )
    return problems


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--readme", default=None, help="defaults to the repository README")
    args = parser.parse_args(argv)

    root = repository_root()
    config = load_resolved(args.config)
    readme_path = Path(args.readme) if args.readme else root / "README.md"
    summary_path = Path(config.paths.results_dir) / "summary.json"

    problems = check(readme_path, summary_path)
    if problems:
        print("README does not match the committed results:\n", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}\n", file=sys.stderr)
        print(
            "Fix with: cobalt evaluate --config config/default.yaml --update-readme",
            file=sys.stderr,
        )
        return 1

    print(f"README matches {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
