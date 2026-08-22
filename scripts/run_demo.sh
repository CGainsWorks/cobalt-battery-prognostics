#!/usr/bin/env bash
# Everything, end to end, from nothing.
#
#   bash scripts/run_demo.sh
#
# Simulates the 320-cell fleet, builds both feature tables, evaluates every model
# under every protocol, prices both decisions, draws the figures, fits the
# deployed models, streams a pack through them over the hardware interface and
# rewrites the README's tables from the results.
#
# About three minutes on a laptop, almost all of it in the simulator. No download
# and no GPU. Deterministic given the seed in config/default.yaml.

set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/_python.sh
. scripts/_python.sh

CONFIG="${CONFIG:-config/default.yaml}"

echo "==> using $PYTHON with $CONFIG"

echo "==> simulating the fleet"
"$PYTHON" -m cobalt.cli generate --config "$CONFIG"

echo "==> evaluating, pricing, plotting and fitting"
"$PYTHON" -m cobalt.cli evaluate --config "$CONFIG" --update-readme

echo "==> streaming a simulated pack through the deployed model"
"$PYTHON" -m cobalt.cli stream --config "$CONFIG"

echo "==> checking the README against what was just produced"
"$PYTHON" -m cobalt.check_readme --config "$CONFIG"

cat <<'DONE'

==> done

  analysis/default/EVALUATION.md   the full report
  analysis/default/*.png           the figures
  results/default/summary.json     every number, machine readable
  artifacts/default/               the fitted models and their model card

Every number in README.md was regenerated from summary.json. Nothing in it was
typed by hand and `make check` will fail if that stops being true.

Remember what the numbers are: simulated cells, no hardware. docs/LIMITATIONS.md
DONE
