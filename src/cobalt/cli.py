"""Command line entry point.

    cobalt generate   simulate a fleet and write it to disk
    cobalt evaluate   features, every protocol, every model, costs, figures, models
    cobalt stream     run a simulated pack through the deployed model, cycle by cycle
    cobalt report     rebuild figures and tables from an existing summary.json
    cobalt serve      run the service

`make demo` is `generate` then `evaluate` then `stream`. Everything takes
``--config`` and everything is deterministic given the seed in it.

Argparse rather than a CLI framework: five commands do not justify a dependency,
and the one thing that would justify it, rich help output, is not worth the
supply chain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cobalt.config import Config, load_resolved, repository_root
from cobalt.data.schema import FleetDataset, write_fleet
from cobalt.deploy import load_deployed, train_and_save
from cobalt.evaluate import report as report_module
from cobalt.evaluate.protocol import build_feature_tables, run_task
from cobalt.features.online import window_features
from cobalt.hal.simulated import SimulatedBms
from cobalt.sim.fleet import simulate_fleet

DEFAULT_CONFIG = "config/default.yaml"


def _log(message: str) -> None:
    print(f"[cobalt] {message}", flush=True)


def _load(config_path: str) -> Config:
    return load_resolved(config_path)


def command_generate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    target = Path(config.paths.data_dir)

    if target.exists() and not args.force:
        try:
            dataset = FleetDataset.load(target)
        except FileNotFoundError:
            pass
        else:
            _log(
                f"{target} already holds {len(dataset.cells)} cells, leaving it alone "
                f"(pass --force to regenerate)"
            )
            return 0

    _log(
        f"simulating {config.fleet.n_batches} batches of {config.fleet.cells_per_batch} "
        f"cells, seed {config.seed}"
    )
    started = time.perf_counter()
    fleet = simulate_fleet(config.fleet, seed=config.seed)
    elapsed = time.perf_counter() - started

    lives = [cell.cycle_life for cell in fleet.cells if cell.cycle_life is not None]
    censored = sum(1 for cell in fleet.cells if cell.censored)
    total_cycles = sum(len(cell.records) for cell in fleet.cells)
    _log(f"simulated {len(fleet.cells)} cells, {total_cycles:,} cell-cycles in {elapsed:.1f}s")
    if lives:
        _log(
            f"cycle life min {min(lives)} median {int(np.median(lives))} max {max(lives)}, "
            f"{censored} censored at the {config.fleet.max_cycles}-cycle ceiling"
        )
    if censored > len(fleet.cells) // 2:
        _log(
            "WARNING: more than half the fleet never reached end of life. Raise "
            "fleet.max_cycles or drive the cells harder, or the labels are mostly fiction."
        )

    written = write_fleet(fleet, target)
    for name, path in written.items():
        _log(f"wrote {name}: {path}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    dataset = FleetDataset.load(Path(config.paths.data_dir))
    _log(f"loaded {len(dataset.cells)} cells, {len(dataset.cycles):,} cell-cycles")

    started = time.perf_counter()
    tables = build_feature_tables(dataset, config)
    early_frame = tables["early_life"]["frame"]
    online_frame = tables["online_soh"]["frame"]
    _log(
        f"features: {len(early_frame)} early-life rows "
        f"(excluded {tables['early_life']['excluded']}), "
        f"{len(online_frame)} online windows"
    )
    if early_frame.empty or online_frame.empty:
        _log("ERROR: a feature table came out empty, nothing to evaluate")
        return 1

    results = {}
    for task, frame in (("early_life", early_frame), ("online_soh", online_frame)):
        _log(f"evaluating {task}")
        results[task] = run_task(task, frame, config)

    summary = report_module.build_summary(
        config,
        dataset,
        results,
        excluded={task: tables[task]["excluded"] for task in tables},
    )

    results_dir = Path(config.paths.results_dir)
    analysis_dir = Path(config.paths.analysis_dir)
    artefacts_dir = Path(config.paths.artifacts_dir)
    for directory in (results_dir, analysis_dir, artefacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    report_module.write_summary(summary, results_dir / "summary.json")
    report_module.write_evaluation_markdown(summary, analysis_dir / "EVALUATION.md")
    _log(f"wrote {results_dir / 'summary.json'} and {analysis_dir / 'EVALUATION.md'}")

    pooled = {
        task: {
            protocol: payload["_pooled"] for protocol, payload in results[task]["protocols"].items()
        }
        for task in results
    }
    report_module.write_prediction_tables(
        pooled["early_life"],
        config.evaluation.deployed_early_life_model,
        results_dir,
        "early_life",
    )
    report_module.write_prediction_tables(
        pooled["online_soh"],
        config.evaluation.deployed_online_model,
        results_dir,
        "online_soh",
    )

    if not args.no_figures:
        _log("drawing figures")
        report_module.figure_capacity_fade(dataset, analysis_dir / "capacity_fade.png")
        report_module.figure_delta_q(dataset, analysis_dir / "delta_q.png")
        report_module.figure_protocol_gap(summary, analysis_dir / "protocol_gap.png")
        report_module.figure_parity(
            pooled["early_life"],
            config.evaluation.deployed_early_life_model,
            analysis_dir / "parity_early_life.png",
            "cycle life",
        )
        report_module.figure_parity(
            pooled["online_soh"],
            config.evaluation.deployed_online_model,
            analysis_dir / "parity_online.png",
            "remaining life",
        )
        report_module.figure_cost(summary, analysis_dir / "cost.png")

    _log("fitting the deployed models on everything")
    written = train_and_save(config, early_frame, online_frame, summary, artefacts_dir)
    for name, path in written.items():
        _log(f"wrote {name}: {path}")

    if args.update_readme:
        readme = repository_root() / "README.md"
        changed = report_module.update_readme(readme, summary)
        _log(f"README {'updated' if changed else 'already matched the results'}")

    _log(f"done in {time.perf_counter() - started:.1f}s")
    return 0


def command_report(args: argparse.Namespace) -> int:
    config = _load(args.config)
    summary_path = Path(config.paths.results_dir) / "summary.json"
    if not summary_path.exists():
        _log(f"ERROR: no summary at {summary_path}, run `cobalt evaluate` first")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis_dir = Path(config.paths.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    report_module.write_evaluation_markdown(summary, analysis_dir / "EVALUATION.md")
    report_module.figure_protocol_gap(summary, analysis_dir / "protocol_gap.png")
    report_module.figure_cost(summary, analysis_dir / "cost.png")

    if args.update_readme:
        changed = report_module.update_readme(repository_root() / "README.md", summary)
        _log(f"README {'updated' if changed else 'already matched the results'}")

    _log(f"rebuilt the report in {analysis_dir}")
    return 0


def command_stream(args: argparse.Namespace) -> int:
    """Run a simulated pack through the deployed model over the hardware interface.

    This is the end-to-end check that the serving path works: telemetry in over
    `cobalt.hal`, features built by the same function training used, a
    prediction and an interval out. It is the shape the real thing would take,
    with the one class that touches hardware replaced by a simulator.
    """
    config = _load(args.config)
    _, online_model = load_deployed(Path(config.paths.artifacts_dir))

    window = config.online.window_cycles
    source = SimulatedBms(pack_id=args.pack_id, seed=args.seed, max_cycles=args.max_cycles)

    history: list[dict[str, float]] = []
    predictions: list[dict[str, float]] = []
    started = time.perf_counter()

    with source:
        while True:
            telemetry = source.read_cycle()
            if telemetry is None:
                break
            record = telemetry.as_dict()
            record.pop("pack_id")
            history.append(record)

            if len(history) >= window and telemetry.cycle % args.stride == 0:
                frame = pd.DataFrame(history[-window:])
                features = window_features(frame, rated_capacity_ah=2.20)
                out = online_model.predict(pd.DataFrame([features]))
                predictions.append(
                    {
                        "cycle": telemetry.cycle,
                        "state_of_health": round(features["soh_now"], 4),
                        "book_after_cycles": round(float(out["lower"][0]), 1),
                        "remaining_estimate": round(float(out["point"][0]), 1),
                        "remaining_upper": round(float(out["upper"][0]), 1),
                    }
                )

    if not predictions:
        _log(
            f"ERROR: the pack produced {len(history)} cycles, fewer than the "
            f"{window}-cycle window the model needs"
        )
        return 1

    elapsed = time.perf_counter() - started
    for row in predictions[:: max(1, len(predictions) // 8)]:
        _log(
            f"cycle {row['cycle']:>5}  SoH {row['state_of_health']:.3f}  "
            f"book after {row['book_after_cycles']:>7.1f} cycles  "
            f"(estimate {row['remaining_estimate']:.0f}, "
            f"upper {row['remaining_upper']:.0f})"
        )

    payload = {
        "pack_id": args.pack_id,
        "source": "cobalt.hal.simulated.SimulatedBms",
        "note": "simulated pack, no hardware involved. See docs/HARDWARE.md.",
        "cycles_streamed": len(history),
        "predictions": predictions,
        "wall_clock_s": round(elapsed, 2),
        "mean_prediction_interval_cycles": round(
            float(np.mean([p["remaining_upper"] - p["book_after_cycles"] for p in predictions])),
            1,
        ),
    }
    output = Path(config.paths.artifacts_dir) / "stream_demo.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _log(f"streamed {len(history)} cycles in {elapsed:.1f}s, wrote {output}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    config_path = str((repository_root() / args.config).resolve())
    os.environ.setdefault("COBALT_CONFIG", config_path)
    _log(f"serving on http://{args.host}:{args.port} with {config_path}")
    uvicorn.run("cobalt.service.api:app", host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cobalt",
        description="Battery prognostics on a simulated fleet. No hardware required.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default=DEFAULT_CONFIG, help="path to a config file")

    generate = subparsers.add_parser("generate", help="simulate a fleet and write it to disk")
    add_config(generate)
    generate.add_argument(
        "--force", action="store_true", help="regenerate even if the data is already there"
    )
    generate.set_defaults(handler=command_generate)

    evaluate = subparsers.add_parser(
        "evaluate", help="run every model under every protocol, price it, and write the report"
    )
    add_config(evaluate)
    evaluate.add_argument("--no-figures", action="store_true", help="skip the plots")
    evaluate.add_argument(
        "--update-readme",
        action="store_true",
        help="rewrite the README's result tables from this run",
    )
    evaluate.set_defaults(handler=command_evaluate)

    report = subparsers.add_parser("report", help="rebuild the report from an existing summary")
    add_config(report)
    report.add_argument("--update-readme", action="store_true")
    report.set_defaults(handler=command_report)

    stream = subparsers.add_parser(
        "stream", help="run a simulated pack through the deployed model over the HAL"
    )
    add_config(stream)
    stream.add_argument("--pack-id", default="SIM-0001")
    stream.add_argument("--seed", type=int, default=7)
    stream.add_argument("--stride", type=int, default=25, help="cycles between predictions")
    stream.add_argument("--max-cycles", type=int, default=1200)
    stream.set_defaults(handler=command_stream)

    serve = subparsers.add_parser("serve", help="run the fleet service")
    add_config(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=command_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except FileNotFoundError as error:
        print(f"[cobalt] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
