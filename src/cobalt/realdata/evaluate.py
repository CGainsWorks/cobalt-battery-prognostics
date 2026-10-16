"""The leakage-free evaluation protocol, run on real cells.

Same feature builders, same split protocols, same leakage assertions, same
models and the same split conformal calibration as the simulated pipeline:
`cobalt.evaluate.protocol.run_task` is called unchanged. Only three things
differ and all three are forced by the data rather than chosen:

* the rated capacity is the cells' 1.1 Ah rather than the simulator's 2.2 Ah;
* there are three batches, so batch-independent evaluation is leave one batch
  out with three folds and the cell-independent protocol uses three folds too;
* there is no business layer. The cost models are priced in assumptions about a
  commercial pack and this is a laboratory dataset, so pricing it would put a
  real-data label on illustrative money.

What a Severson batch is matters for reading the batch-independent number. It
is a **test campaign**: cells cycled together, months apart from the next
campaign, under a different set of charging policies. It is not established
that the batches are different manufacturing lots. So holding one out tests
generalisation to a new campaign and a new policy set, which is the nearest
thing this dataset has to the new-lot question the simulator asks and not the
same thing.

Every number written by this module comes from the public Severson et al.
(2019) cells. None is simulated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from cobalt.config import Config, repository_root
from cobalt.data.schema import FleetDataset
from cobalt.evaluate import report as report_module
from cobalt.evaluate.protocol import build_feature_tables, run_task
from cobalt.models.conformal import MIN_FIT_GROUPS
from cobalt.realdata.severson import NOMINAL_CAPACITY_AH
from cobalt.textio import LF, write_json, write_text

README_BLOCK = "severson"
DATASET_CARD = "docs/datasets/severson-2019.md"

_TASK_LABELS = {
    "early_life": "Early life (cycle life from cycles 2 to 100)",
    "online_soh": "Online RUL",
}
_KEPT_METRICS = ("rmse_cycles", "mae_cycles", "mape", "alpha_lambda", "spearman", "bias_cycles")


def _round(value: Any, digits: int = 6) -> Any:
    """Round to significant figures so a summary does not churn on the last bit."""
    if isinstance(value, float):
        if not math.isfinite(value) or value == 0.0:
            return value
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def _calibration_groups(n_train_groups: int, fraction: float) -> tuple[int, int]:
    """Mirror of ConformalRegressor's grouped split: (held for calibration, left to fit)."""
    held = max(1, int(round(fraction * n_train_groups)))
    return held, n_train_groups - held


def build_summary(config: Config, dataset: FleetDataset, results: dict, excluded: dict) -> dict:
    tasks: dict[str, Any] = {}
    notes: list[str] = []
    n_batches = int(dataset.cells["batch_id"].nunique())
    for task_name, result in results.items():
        protocols = {}
        for protocol, payload in result["protocols"].items():
            models = {
                name: {
                    key: value
                    for key, value in metrics.items()
                    if key in ("n", "n_folds") or key.removesuffix("_std") in _KEPT_METRICS
                }
                for name, metrics in payload["models"].items()
            }
            protocols[protocol] = {"models": models, "intervals": payload["intervals"]}
        tasks[task_name] = {
            "n_rows": result["n_rows"],
            "n_cells": result["n_cells"],
            "n_batches": result["n_batches"],
            "deployed_model": result["deployed_model"],
            "alpha": result["alpha"],
            "excluded": excluded[task_name],
            "protocols": protocols,
        }

    train_batches = n_batches - math.ceil(n_batches / config.evaluation.n_folds)
    held, left = _calibration_groups(train_batches, config.evaluation.calibration_fraction)
    if left < MIN_FIT_GROUPS:
        notes.append(
            f"Batch-independent grouped calibration holds out {held} of the {train_batches} "
            f"training batches and fits on {left}. The library warns below {MIN_FIT_GROUPS}, "
            "so with three batches the grouped-against-random calibration comparison is "
            "starved and its coverage figures are reported, not relied on."
        )
    notes.append(
        "A Severson batch is a test campaign with its own set of charging policies, not a "
        "known manufacturing lot."
    )

    return _round(
        {
            "config": config.name,
            "data_source": "real",
            "dataset": {
                "name": "Severson et al. (2019), Toyota Research Institute",
                "card": DATASET_CARD,
                "n_cells": int(len(dataset.cells)),
                "n_batches": n_batches,
                "cells_per_batch": dataset.manifest.get("cells_per_batch"),
                "censored_cells": len(dataset.manifest.get("censored_cells", [])),
                "cell_cycles": int(len(dataset.cycles)),
                "cycle_life_min": dataset.manifest.get("cycle_life_min"),
                "cycle_life_median": dataset.manifest.get("cycle_life_median"),
                "cycle_life_max": dataset.manifest.get("cycle_life_max"),
                "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
            },
            "evaluation": {
                "n_folds": config.evaluation.n_folds,
                "alpha": config.evaluation.alpha,
                "calibration_fraction": config.evaluation.calibration_fraction,
                "online_window_cycles": config.online.window_cycles,
                "online_stride": config.online.stride,
                "seed": config.seed,
            },
            "tasks": tasks,
            "notes": notes,
        }
    )


# -- tables ----------------------------------------------------------------------


def _row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _table(header: list[str], rows: list[list[str]]) -> str:
    return LF.join([_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)])


def table_headline(summary: dict) -> str:
    """The README block: RMSE and MAPE per model under the two honest protocols."""
    rows = []
    for task_name in ("early_life", "online_soh"):
        task = summary["tasks"][task_name]
        protocols = [p for p in ("cell_independent", "batch_independent") if p in task["protocols"]]
        for model in task["protocols"][protocols[0]]["models"]:
            row = [
                "Early life" if task_name == "early_life" else "Online RUL",
                report_module._label(model, report_module.MODEL_LABELS),
            ]
            for protocol in protocols:
                metrics = task["protocols"][protocol]["models"][model]
                row.append(
                    f"{metrics['rmse_cycles']:.0f} ± {metrics['rmse_cycles_std']:.0f} "
                    f"({metrics['mape']:.1%})"
                )
            rows.append(row)
    header = ["Task", "Model", "Cell-independent", "Batch-independent"]
    return _table(header, rows)


def table_coverage(summary: dict) -> str:
    rows = []
    for task_name in ("early_life", "online_soh"):
        task = summary["tasks"][task_name]
        for protocol in report_module.PROTOCOL_ORDER:
            if protocol not in task["protocols"]:
                continue
            intervals = task["protocols"][protocol]["intervals"]
            rows.append(
                [
                    "Early life" if task_name == "early_life" else "Online RUL",
                    report_module.PROTOCOL_LABELS[protocol],
                    f"{1 - task['alpha']:.0%}",
                    f"{intervals['matched_calibration']['picp']:.3f}",
                    f"{intervals['random_calibration']['picp']:.3f}",
                    f"{intervals['matched_calibration']['median_width_cycles']:.0f}",
                ]
            )
    header = [
        "Task",
        "Protocol",
        "Asked for",
        "Grouped calibration",
        "Random calibration",
        "Median width (cycles, grouped)",
    ]
    return _table(header, rows)


def table_task(summary: dict, task_name: str) -> str:
    task = summary["tasks"][task_name]
    protocols = [p for p in report_module.PROTOCOL_ORDER if p in task["protocols"]]
    header = ["Model"] + [report_module.PROTOCOL_LABELS[p] for p in protocols]
    rows = []
    for model in task["protocols"][protocols[0]]["models"]:
        row = [report_module._label(model, report_module.MODEL_LABELS)]
        for protocol in protocols:
            m = task["protocols"][protocol]["models"][model]
            row.append(
                f"RMSE {m['rmse_cycles']:.0f} ± {m['rmse_cycles_std']:.0f}, "
                f"MAPE {m['mape']:.1%}, rho {m['spearman']:.2f}"
            )
        rows.append(row)
    return _table(header, rows)


def write_markdown(summary: dict, path: Path) -> None:
    data = summary["dataset"]
    lines = [
        "# Evaluation on real cells: Severson et al. (2019)",
        "",
        "Generated by `cobalt real-data evaluate`. Every number here comes from the public",
        f"Severson et al. dataset, not from the simulator. Dataset card: `{data['card']}`.",
        "",
        f"{data['n_cells']} cells in {data['n_batches']} batches "
        f"({', '.join(f'{k}: {v}' for k, v in data['cells_per_batch'].items())}), "
        f"{data['censored_cells']} censored, {data['cell_cycles']:,} cell-cycles. "
        f"Cycle life {data['cycle_life_min']} / {data['cycle_life_median']} / "
        f"{data['cycle_life_max']} (min / median / max).",
        "",
        "Errors are root mean square error in cycles, mean ± standard deviation across",
        "folds, then mean absolute percentage error and Spearman rank correlation.",
        "",
    ]
    for task_name in ("early_life", "online_soh"):
        task = summary["tasks"][task_name]
        lines += [
            f"## {_TASK_LABELS[task_name]}",
            "",
            f"{task['n_rows']} rows from {task['n_cells']} cells. Excluded: "
            + ", ".join(f"{k} {v}" for k, v in task["excluded"].items())
            + f". Deployed model: `{task['deployed_model']}`.",
            "",
            table_task(summary, task_name),
            "",
        ]
    lines += ["## Conformal interval coverage", "", table_coverage(summary), "", "## Notes", ""]
    lines += [f"- {note}" for note in summary["notes"]]
    write_text(path, LF.join(lines) + LF)


def evaluate(
    config: Config, output_dir: Path | None = None, figures: bool = True, log=print
) -> dict:
    data_dir = Path(config.paths.data_dir)
    dataset = FleetDataset.load(data_dir)
    log(f"loaded {len(dataset.cells)} real cells, {len(dataset.cycles):,} cell-cycles")

    tables = build_feature_tables(dataset, config)
    results = {}
    for task in ("early_life", "online_soh"):
        frame = tables[task]["frame"]
        log(f"evaluating {task} on {len(frame)} rows")
        results[task] = run_task(task, frame, config, rated_capacity_ah=NOMINAL_CAPACITY_AH)

    summary = build_summary(
        config, dataset, results, {task: tables[task]["excluded"] for task in tables}
    )
    results_dir = Path(output_dir) if output_dir else Path(config.paths.results_dir)
    write_json(results_dir / "summary.json", summary)
    log(f"wrote {results_dir / 'summary.json'}")

    if output_dir is None:
        analysis_dir = Path(config.paths.analysis_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, analysis_dir / "EVALUATION.md")
        if figures:
            for task, label in (("early_life", "cycle life"), ("online_soh", "remaining life")):
                pooled = {
                    p: payload["_pooled"] for p, payload in results[task]["protocols"].items()
                }
                report_module.figure_parity(
                    pooled,
                    results[task]["deployed_model"],
                    analysis_dir / f"parity_{task}.png",
                    label,
                )
        log(f"wrote {analysis_dir / 'EVALUATION.md'}")
    return summary


def update_readme(summary: dict, readme: Path | None = None) -> bool:
    readme = readme or repository_root() / "README.md"
    original = readme.read_text(encoding="utf-8")
    updated = report_module.apply_blocks(original, {README_BLOCK: table_headline(summary)})
    if updated != original:
        write_text(readme, updated)
        return True
    return False


def compare(
    expected: dict, actual: dict, rtol: float = 2e-3, atol: float = 1e-6, path: str = ""
) -> list[str]:
    """Differences between two summaries beyond a tolerance.

    Numeric results move by parts per million between platforms and library
    builds, so a byte comparison would fail for reasons that are not drift.
    """
    problems: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in actual:
                problems.append(f"{where}: present in only one summary")
            else:
                problems += compare(expected[key], actual[key], rtol, atol, where)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            problems.append(f"{path}: length {len(expected)} against {len(actual)}")
        else:
            for i, (a, b) in enumerate(zip(expected, actual)):
                problems += compare(a, b, rtol, atol, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=rtol, abs_tol=atol):
            problems.append(f"{path}: {expected} against {actual}")
    elif expected != actual:
        problems.append(f"{path}: {expected!r} against {actual!r}")
    return problems


def compare_files(expected: Path, actual: Path, rtol: float = 2e-3) -> list[str]:
    return compare(
        json.loads(Path(expected).read_text(encoding="utf-8")),
        json.loads(Path(actual).read_text(encoding="utf-8")),
        rtol=rtol,
    )
