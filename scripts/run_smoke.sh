#!/usr/bin/env bash
# The whole pipeline on a fleet small enough for a continuous integration runner.
#
#   bash scripts/run_smoke.sh
#
# THE ACCURACY FIGURES THIS PRINTS ARE A WIRING CHECK, NOT A RESULT.
#
# Eight batches of six cells is far too few to estimate a batch-independent
# error. The fold standard deviations will be enormous and the numbers will move
# if you change the seed. What this proves is that simulation, feature
# extraction, every split protocol, every model, conformal calibration, both cost
# models, the figures, the model card and the hardware interface all connect.
#
# Everything it writes is stamped with the `ci` config name so it cannot be
# mistaken for the results in the README, which come from config/default.yaml.

set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/_python.sh
. scripts/_python.sh

CONFIG="${CONFIG:-config/ci.yaml}"

warn() {
    echo ""
    echo "  !! smoke run: the accuracy figures below are a wiring check, not a result"
    echo ""
}

warn
echo "==> simulating a small fleet"
"$PYTHON" -m cobalt.cli generate --config "$CONFIG" --force

warn
echo "==> evaluating, pricing, plotting and fitting"
# Deliberately no --update-readme. The README's numbers come from the default
# config, and letting a smoke run overwrite them would be the exact failure the
# README check exists to catch.
"$PYTHON" -m cobalt.cli evaluate --config "$CONFIG"

warn
echo "==> streaming a simulated pack through the deployed model"
"$PYTHON" -m cobalt.cli stream --config "$CONFIG" --stride 20 --max-cycles 600

warn
cat <<'DONE'
==> smoke run finished

Every stage connected. The numbers in analysis/ci/ and results/ci/ mean nothing
beyond that: the fleet is too small for any of them to be an estimate of
anything. The real ones are in analysis/default/ and are produced by
scripts/run_demo.sh.
DONE
