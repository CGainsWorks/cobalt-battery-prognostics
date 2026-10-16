# Status

As of 2026-10-16.

## Complete

- **Simulated fleet and evaluation.** A physics-informed cell simulator, a
  320-cell fleet across 20 batches, early-life (Severson features) and online
  remaining-life models, three split protocols with leakage assertions, split
  conformal intervals with grouped calibration, the newsvendor and Bayes grading
  cost layer, the FastAPI service, the CLI and the container. Every README table
  is generated from `results/default/summary.json` and checked in CI.
- **CAN battery link (simulated at both ends).** `src/cobalt/hal/cobalt_bms.dbc`
  defines three live frames (pack status, four series cell voltages,
  temperatures) and five cycle summary frames carrying every `CycleTelemetry`
  field, each with a rolling counter and checksum. `SimulatedBmsNode` publishes
  them from the existing cell simulator on any python-can bus; `CanBms` decodes
  them with cantools (python-can 4.6.1, cantools 45.0.0), checks checksums and
  counters, counts lost frames and incomplete cycles and feeds the same feature
  function and model the service uses. It replaces the old `SerialBms` stub.
- **CAN tested end to end.** `tests/test_can.py` (16 tests) runs the node and
  the receiver on the in-process virtual bus, asserts every field survives
  within half a bit of its DBC scaling, injects corruption and frame loss, and
  runs the deployed model over the bus. CI additionally runs the node and the
  receiver as two processes on a Linux `vcan0` interface through SocketCAN and
  requires zero loss and predictions identical to the committed log
  (`artifacts/default/stream_demo_can.json`).
- **Real cells.** `cobalt real-data` downloads the Severson et al. (2019)
  dataset from the publisher (8.3 GB, never committed), extracts what the
  protocol needs following the authors' own loading code (124 cells reach end
  of life; labels reproduce the publisher's `cycle_life` for all of them) and
  runs the unchanged evaluation protocol. Results are in
  `results/severson/summary.json` and `analysis/severson/EVALUATION.md`, the
  README table is generated from them and the dataset card is
  `docs/datasets/severson-2019.md`. A CI job re-runs the evaluation from a
  cached extract (downloading on a cache miss) and fails if it drifts from the
  committed summary; if the publisher is unreachable it says so and skips.
- **Tests.** 242 in total, passing on Python 3.12 and 3.13 and inside the
  container image.

## In progress

- Nothing.

## Not started

- Conformalised quantile regression in place of the Gaussian-in-log-space
  assumption used to turn an interval into a full predictive distribution.
- A second public dataset (NASA PCoE or CALCE) to test whether the real-cell
  findings hold beyond one LFP cell type.
- Integrating a per-cycle summary from live current and voltage on the
  receiving side, which a production BMS that sends no summary frames would
  need.

## Limitations

- **No hardware.** No cycler, thermal chamber, cells, CAN adapter or BMS was
  used. Every number outside `results/severson/` and `analysis/severson/` is
  simulated. The CAN link has only ever talked to the simulated node, on the
  python-can virtual bus and on Linux vcan; the DBC is designed here, not a
  vendor's and the checksum is a simple ones-complement sum rather than AUTOSAR
  E2E. See `docs/HARDWARE.md`.
- **Quantisation moves predictions slightly.** Over the 1,200-cycle CAN demo,
  38 of 47 remaining-life estimates match the direct path exactly and the other
  nine move by at most 1.2%, because the gradient-boosted model has thresholds.
- **Real-cell results are one laboratory dataset.** One LFP/graphite cell type at
  30 degC and 4C discharge, three test campaigns rather than manufacturing lots,
  so grouped calibration fits on a single campaign and the early-life intervals
  are fragile. The cost layer is not run on real data.
- **Dataset licence unconfirmed.** The publisher's licence text could not be
  retrieved; only aggregate metrics are committed and the extract lives in the
  Actions cache. See the dataset card.
- **Glitch filter.** 101 of 104,925 real cycle records are dropped as physically
  impossible by rules this repository chose; the counts per rule are recorded.
