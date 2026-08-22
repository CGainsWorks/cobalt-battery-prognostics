"""Turning results into figures, tables and the numbers in the README.

One rule holds this together: **no number is typed into the README by hand**.
Every table between a pair of ``BEGIN``/``END`` markers is rendered from
``summary.json`` by the functions here and `cobalt.check_readme` regenerates
them and fails if what is in the file has drifted from what the last run
actually produced. Continuous integration runs that check, so a stale table is a
red build rather than a quiet lie.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cobalt.business.cost import (  # noqa: E402
    ReplacementCostModel,
    compare_policies,
    qualification_time_saving,
)
from cobalt.business.grading import (  # noqa: E402
    GradingPolicy,
    constant_grade_baseline,
    evaluate_grading,
    grade_from_prediction,
    minimum_expected_cost_grade,
)
from cobalt.config import Config  # noqa: E402
from cobalt.data.schema import FleetDataset  # noqa: E402
from cobalt.evaluate.protocol import OutOfFoldPredictions  # noqa: E402
from cobalt.models.conformal import implied_sigma  # noqa: E402
from cobalt.textio import LF, write_csv, write_json, write_text  # noqa: E402

#: Order protocols appear in every table, easiest first so the reader watches the
#: numbers get worse rather than having to hunt for the honest one.
PROTOCOL_ORDER = ("random_cycle", "cell_independent", "batch_independent")

PROTOCOL_LABELS = {
    "random_cycle": "Random cycle (leaky)",
    "cell_independent": "Cell-independent",
    "batch_independent": "Batch-independent",
}

MODEL_LABELS = {
    "fleet_mean": "Fleet mean (floor)",
    "capacity_extrapolation": "Capacity extrapolation",
    "slope_extrapolation": "Slope extrapolation",
    "ridge": "Ridge",
    "elastic_net": "Elastic net",
    "gradient_boosting": "Gradient boosting",
}


def _label(name: str, table: dict[str, str]) -> str:
    return table.get(name, name.replace("_", " ").capitalize())


def _ordered(protocols: dict[str, Any]) -> list[str]:
    return [name for name in PROTOCOL_ORDER if name in protocols]


# -- business analysis ---------------------------------------------------


def price_early_life(
    pooled: dict[str, dict[str, OutOfFoldPredictions]],
    policy: GradingPolicy,
    deployed: str,
    alpha: float,
) -> dict[str, Any]:
    """Cost per cell of grading a lot, by protocol and by decision rule.

    Four rules, because the interesting result is that they do not rank the
    way intuition says. Grading on the conformal lower bound is more cautious
    than grading on the point estimate and more cautious is not cheaper: it
    trades expensive overgrading for cheap undergrading and then buys far too
    much of it. Only the expected-cost rule uses both the prediction and its
    uncertainty the way the cost matrix actually asks.
    """
    out: dict[str, Any] = {
        "grade_boundaries": {
            "A": f">= {policy.grade_a_min_cycles} cycles",
            "B": f"{policy.grade_b_min_cycles} to {policy.grade_a_min_cycles - 1} cycles",
            "C": f"< {policy.grade_b_min_cycles} cycles",
        }
    }

    for protocol, models in pooled.items():
        conformal = models.get(f"{deployed}+conformal")
        point = models.get(deployed)
        if point is None or conformal is None:
            continue

        truth = point.natural_truth
        half_width = float(np.mean(conformal.log_prediction - conformal.log_lower))
        sigma = implied_sigma(half_width, alpha)

        rules = {
            "point_estimate": evaluate_grading(
                truth, grade_from_prediction(point.log_prediction, policy), policy
            ),
            "conformal_lower_bound": evaluate_grading(
                truth,
                grade_from_prediction(
                    conformal.log_prediction, policy, log_lower_bound=conformal.log_lower
                ),
                policy,
            ),
            "minimum_expected_cost": evaluate_grading(
                truth,
                minimum_expected_cost_grade(conformal.log_prediction, sigma, policy),
                policy,
            ),
            "grade_everything_b": constant_grade_baseline(truth, policy),
        }
        out[protocol] = {name: result.as_dict() for name, result in rules.items()}
        out[protocol]["implied_sigma_log10"] = round(sigma, 4)

    return out


def price_online(
    pooled: dict[str, dict[str, OutOfFoldPredictions]],
    model: ReplacementCostModel,
    deployed: str,
    alpha: float,
) -> dict[str, Any]:
    """Cost per pack of scheduling a replacement, by protocol and by policy."""
    out: dict[str, Any] = {
        "cost_per_cycle_early": model.cost_per_cycle_early,
        "cost_per_cycle_late": model.cost_per_cycle_late,
        "optimal_quantile": round(model.optimal_quantile, 4),
    }

    for protocol, models in pooled.items():
        conformal = models.get(f"{deployed}+conformal")
        if conformal is None:
            continue
        results = compare_policies(
            log_truth=conformal.log_truth,
            log_prediction=conformal.log_prediction,
            model=model,
            log_lower_bound=conformal.log_lower,
            training_log_truth=conformal.log_truth,
            alpha=alpha,
        )
        out[protocol] = {result.policy: result.as_dict() for result in results}

    return out


def fleet_statistics(dataset: FleetDataset) -> dict[str, Any]:
    """Descriptive numbers for the fleet the run was done on."""
    cells = dataset.cells
    uncensored = cells[~cells["censored"].astype(bool)]
    lives = uncensored["cycle_life"].to_numpy(dtype=float)
    return {
        "n_cells": int(len(cells)),
        "n_batches": int(cells["batch_id"].nunique()),
        "censored_cells": int(cells["censored"].astype(bool).sum()),
        "median_cycle_life": int(np.median(lives)) if lives.size else 0,
        "min_cycle_life": int(np.min(lives)) if lives.size else 0,
        "max_cycle_life": int(np.max(lives)) if lives.size else 0,
        "cycle_life_spread": round(float(np.max(lives) / max(np.min(lives), 1)), 2)
        if lives.size
        else 0.0,
        "total_cycles_simulated": int(len(dataset.cycles)),
    }


def build_summary(
    config: Config,
    dataset: FleetDataset,
    results: dict[str, dict[str, Any]],
    excluded: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Assemble everything worth keeping into one JSON-serialisable object."""
    pooled = {
        task: {protocol: payload["_pooled"] for protocol, payload in result["protocols"].items()}
        for task, result in results.items()
    }

    fleet = fleet_statistics(dataset)
    summary: dict[str, Any] = {
        "config_name": config.name,
        "seed": config.seed,
        "data_source": "simulated",
        "fleet": fleet,
        "tasks": {},
        "business": {
            "grading": price_early_life(
                pooled["early_life"],
                config.grading,
                config.evaluation.deployed_early_life_model,
                config.evaluation.alpha,
            ),
            "replacement": price_online(
                pooled["online_soh"],
                config.replacement,
                config.evaluation.deployed_online_model,
                config.evaluation.alpha,
            ),
            "qualification": qualification_time_saving(
                median_cycle_life=float(fleet["median_cycle_life"]),
                observation_cycles=100,
                hours_per_cycle=config.qualification.hours_per_cycle,
                channels=config.qualification.channels,
            ),
        },
    }

    for task, result in results.items():
        protocols = {
            protocol: {
                "models": payload["models"],
                "intervals": payload["intervals"],
            }
            for protocol, payload in result["protocols"].items()
        }
        summary["tasks"][task] = {
            "n_rows": result["n_rows"],
            "n_cells": result["n_cells"],
            "n_batches": result["n_batches"],
            "deployed_model": result["deployed_model"],
            "alpha": result["alpha"],
            "excluded": excluded.get(task, {}),
            "protocols": protocols,
        }

    return summary


# -- tables --------------------------------------------------------------


def _row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [_row(header), _row(["---"] * len(header))]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def table_fleet(summary: dict[str, Any]) -> str:
    fleet = summary["fleet"]
    rows = [
        ["Cells simulated", f"{fleet['n_cells']}"],
        ["Manufacturing batches", f"{fleet['n_batches']}"],
        ["Cell-cycles simulated", f"{fleet['total_cycles_simulated']:,}"],
        [
            "Cycle life, min / median / max",
            f"{fleet['min_cycle_life']} / {fleet['median_cycle_life']} / {fleet['max_cycle_life']}",
        ],
        ["Longest life over shortest", f"{fleet['cycle_life_spread']}x"],
        ["Right-censored at the ceiling", f"{fleet['censored_cells']}"],
    ]
    return _table(["Fleet", "Value"], rows)


def _model_rows(task: dict[str, Any], metric: str, fmt: str) -> list[list[str]]:
    protocols = _ordered(task["protocols"])
    model_names = list(task["protocols"][protocols[0]]["models"])
    rows = []
    for model_name in model_names:
        row = [_label(model_name, MODEL_LABELS)]
        for protocol in protocols:
            value = task["protocols"][protocol]["models"][model_name][metric]
            row.append(format(value, fmt))
        rows.append(row)
    return rows


def table_task(summary: dict[str, Any], task_name: str, metric: str, fmt: str) -> str:
    task = summary["tasks"][task_name]
    protocols = _ordered(task["protocols"])
    header = ["Model"] + [PROTOCOL_LABELS[p] for p in protocols]
    return _table(header, _model_rows(task, metric, fmt))


def table_coverage(summary: dict[str, Any]) -> str:
    rows = []
    for task_name in ("early_life", "online_soh"):
        task = summary["tasks"][task_name]
        nominal = 1.0 - task["alpha"]
        for protocol in _ordered(task["protocols"]):
            intervals = task["protocols"][protocol]["intervals"]
            matched = intervals["matched_calibration"]
            random_cal = intervals["random_calibration"]
            rows.append(
                [
                    "Early life" if task_name == "early_life" else "Online RUL",
                    PROTOCOL_LABELS[protocol],
                    f"{nominal:.0%}",
                    f"{matched['picp']:.3f}",
                    f"{random_cal['picp']:.3f}",
                    f"{random_cal['picp'] - matched['picp']:+.3f}",
                ]
            )
    header = [
        "Task",
        "Protocol",
        "Asked for",
        "Grouped calibration",
        "Random calibration",
        "Difference",
    ]
    return _table(header, rows)


def table_grading(summary: dict[str, Any]) -> str:
    grading = summary["business"]["grading"]
    protocols = [p for p in PROTOCOL_ORDER if p in grading]
    rows = []
    for rule in (
        "grade_everything_b",
        "point_estimate",
        "conformal_lower_bound",
        "minimum_expected_cost",
    ):
        row = [
            _label(
                rule,
                {
                    "grade_everything_b": "Grade everything B (no model)",
                    "point_estimate": "Model, point estimate",
                    "conformal_lower_bound": "Model, 90% lower bound",
                    "minimum_expected_cost": "Model, minimum expected cost",
                },
            )
        ]
        for protocol in protocols:
            row.append(f"£{grading[protocol][rule]['cost_per_cell']:.2f}")
        rows.append(row)
    header = ["Grading rule"] + [PROTOCOL_LABELS[p] for p in protocols]
    return _table(header, rows)


def table_replacement(summary: dict[str, Any]) -> str:
    replacement = summary["business"]["replacement"]
    protocols = [p for p in PROTOCOL_ORDER if p in replacement]
    rows = []
    for policy in (
        "oracle",
        "fleet_quantile",
        "point_estimate",
        "conformal_lower_bound",
        "newsvendor_quantile",
    ):
        row = [
            _label(
                policy,
                {
                    "oracle": "Oracle (unreachable floor)",
                    "fleet_quantile": "Fleet quantile (no features)",
                    "point_estimate": "Book on the point estimate",
                    "conformal_lower_bound": "Book on the 90% lower bound",
                    "newsvendor_quantile": "Book on the newsvendor quantile",
                },
            )
        ]
        for protocol in protocols:
            entry = replacement[protocol].get(policy)
            row.append(f"£{entry['cost_per_pack']:.2f}" if entry else "n/a")
        rows.append(row)
    header = ["Scheduling policy"] + [PROTOCOL_LABELS[p] for p in protocols]
    return _table(header, rows)


def render_tables(summary: dict[str, Any]) -> dict[str, str]:
    """Every README block, keyed by its marker name."""
    return {
        "fleet": table_fleet(summary),
        "early_life": table_task(summary, "early_life", "rmse_cycles", ".0f"),
        "early_life_alpha": table_task(summary, "early_life", "alpha_lambda", ".3f"),
        "online": table_task(summary, "online_soh", "rmse_cycles", ".0f"),
        "online_alpha": table_task(summary, "online_soh", "alpha_lambda", ".3f"),
        "coverage": table_coverage(summary),
        "grading": table_grading(summary),
        "replacement": table_replacement(summary),
    }


def apply_blocks(text: str, tables: dict[str, str]) -> str:
    """Replace each marked block in ``text`` with its rendered table."""
    for name, table in tables.items():
        begin = f"<!-- BEGIN:{name} -->"
        end = f"<!-- END:{name} -->"
        if begin not in text or end not in text:
            raise ValueError(f"no {begin} ... {end} block found")
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{table}\n{end}{tail}"
    return text


def update_readme(readme_path: Path, summary: dict[str, Any]) -> bool:
    """Rewrite the README's tables. Returns whether anything changed."""
    readme_path = Path(readme_path)
    original = readme_path.read_text(encoding="utf-8")
    updated = apply_blocks(original, render_tables(summary))
    if updated != original:
        write_text(readme_path, updated)
        return True
    return False


# -- figures -------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def figure_capacity_fade(dataset: FleetDataset, path: Path, max_cells: int = 60) -> None:
    """Fade curves, coloured by batch. The knee and the batch structure at once."""
    _style()
    figure, axes = plt.subplots(figsize=(7.2, 4.0))

    batches = sorted(dataset.cells["batch_id"].unique())
    colours = plt.cm.viridis(np.linspace(0, 0.92, max(len(batches), 1)))
    colour_of = dict(zip(batches, colours))

    shown = 0
    for _, cell in dataset.cells.iterrows():
        if shown >= max_cells:
            break
        cycles = dataset.cycles_for(str(cell["cell_id"]))
        rated = float(cell["nameplate_capacity_ah"])
        axes.plot(
            cycles["cycle"],
            cycles["discharge_capacity_ah"] / rated,
            color=colour_of[str(cell["batch_id"])],
            linewidth=0.8,
            alpha=0.8,
        )
        shown += 1

    axes.axhline(0.80, color="crimson", linestyle="--", linewidth=1.1)
    axes.text(
        axes.get_xlim()[1] * 0.985,
        0.806,
        "end of life, 80% of nameplate",
        color="crimson",
        fontsize=8,
        ha="right",
    )
    axes.set_xlabel("Cycle")
    axes.set_ylabel("State of health (capacity over nameplate)")
    axes.set_title(
        f"Capacity fade, {shown} simulated cells coloured by manufacturing batch",
        fontsize=10,
    )
    figure.savefig(path)
    plt.close(figure)


def figure_delta_q(dataset: FleetDataset, path: Path, max_cells: int = 40) -> None:
    """The early-life signal: dQ(V) curves coloured by the life they went on to have."""
    from cobalt.features.early_life import delta_q

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    uncensored = dataset.cells[~dataset.cells["censored"].astype(bool)]
    if uncensored.empty:
        plt.close(figure)
        raise ValueError("every cell was censored, so there is nothing to plot")

    lives = uncensored["cycle_life"].to_numpy(dtype=float)
    low, high = float(np.min(lives)), float(np.max(lives))
    normalise = plt.Normalize(low, high)
    colourmap = plt.cm.plasma

    variances, plotted_lives = [], []
    for count, (_, cell) in enumerate(uncensored.iterrows()):
        difference = delta_q(dataset, str(cell["cell_id"]))
        if difference is None:
            continue
        life = float(cell["cycle_life"])
        variances.append(float(np.var(difference)))
        plotted_lives.append(life)
        if count < max_cells:
            axes[0].plot(
                dataset.voltage_grid[: len(difference)],
                difference,
                color=colourmap(normalise(life)),
                linewidth=0.9,
                alpha=0.85,
            )

    axes[0].set_xlabel("Terminal voltage (V)")
    axes[0].set_ylabel(r"$\Delta Q_{100-10}$ (Ah)")
    axes[0].set_title("Discharge curve difference, cycle 100 against cycle 10", fontsize=9)
    figure.colorbar(
        plt.cm.ScalarMappable(norm=normalise, cmap=colourmap),
        ax=axes[0],
        label="cycle life",
    )

    axes[1].scatter(
        np.log10(np.maximum(variances, 1e-12)),
        np.log10(plotted_lives),
        s=14,
        c=plotted_lives,
        cmap=colourmap,
        norm=normalise,
        edgecolor="none",
    )
    axes[1].set_xlabel(r"$\log_{10}\ \mathrm{var}(\Delta Q)$")
    axes[1].set_ylabel(r"$\log_{10}$ cycle life")
    if len(variances) > 2:
        correlation = float(
            np.corrcoef(np.log10(np.maximum(variances, 1e-12)), np.log10(plotted_lives))[0, 1]
        )
        axes[1].set_title(f"One feature against the label, r = {correlation:.2f}", fontsize=9)
    figure.savefig(path)
    plt.close(figure)


def figure_protocol_gap(summary: dict[str, Any], path: Path) -> None:
    """Per-model error under each protocol, both tasks. The headline picture."""
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for panel, task_name in zip(axes, ("early_life", "online_soh")):
        task = summary["tasks"][task_name]
        protocols = _ordered(task["protocols"])
        model_names = list(task["protocols"][protocols[0]]["models"])
        width = 0.8 / len(protocols)
        positions = np.arange(len(model_names))

        for index, protocol in enumerate(protocols):
            values = [
                task["protocols"][protocol]["models"][name]["rmse_cycles"] for name in model_names
            ]
            errors = [
                task["protocols"][protocol]["models"][name].get("rmse_cycles_std", 0.0)
                for name in model_names
            ]
            panel.bar(
                positions + index * width - 0.4 + width / 2,
                values,
                width=width,
                yerr=errors,
                capsize=2.5,
                label=PROTOCOL_LABELS[protocol],
            )

        panel.set_xticks(positions)
        panel.set_xticklabels(
            [_label(name, MODEL_LABELS) for name in model_names], rotation=20, ha="right"
        )
        panel.set_ylabel("RMSE (cycles)")
        panel.set_title(
            "Early life: cycles to end of life"
            if task_name == "early_life"
            else "Online: remaining useful life",
            fontsize=10,
        )
        panel.legend(fontsize=8)

    figure.suptitle("The same models, scored three ways", fontsize=11)
    figure.savefig(path)
    plt.close(figure)


def figure_parity(
    pooled: dict[str, dict[str, OutOfFoldPredictions]],
    deployed: str,
    path: Path,
    task_label: str,
) -> None:
    """Predicted against true, one panel per protocol, with the interval shown."""
    _style()
    protocols = [p for p in PROTOCOL_ORDER if p in pooled]
    figure, axes = plt.subplots(
        1, len(protocols), figsize=(4.4 * len(protocols), 4.0), squeeze=False
    )

    for panel, protocol in zip(axes[0], protocols):
        predictions = pooled[protocol].get(f"{deployed}+conformal")
        if predictions is None:
            continue
        truth = np.power(10.0, predictions.log_truth)
        estimate = np.power(10.0, predictions.log_prediction)

        if predictions.log_lower is not None:
            lower = np.power(10.0, predictions.log_lower)
            upper = np.power(10.0, predictions.log_upper)
            panel.vlines(truth, lower, upper, color="tab:blue", alpha=0.15, linewidth=0.8)

        panel.scatter(truth, estimate, s=10, alpha=0.6, edgecolor="none")
        limits = [min(truth.min(), estimate.min()) * 0.85, max(truth.max(), estimate.max()) * 1.1]
        panel.plot(limits, limits, color="black", linewidth=0.9, linestyle="--")
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(limits)
        panel.set_ylim(limits)
        panel.set_xlabel(f"True {task_label} (cycles)")
        panel.set_ylabel(f"Predicted {task_label} (cycles)")
        panel.set_title(PROTOCOL_LABELS[protocol], fontsize=10)

    figure.suptitle(f"{deployed} with 90% conformal intervals, out of fold", fontsize=11)
    figure.savefig(path)
    plt.close(figure)


def figure_cost(summary: dict[str, Any], path: Path) -> None:
    """What each decision rule costs, by protocol. The business panel."""
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    grading = summary["business"]["grading"]
    grading_protocols = [p for p in PROTOCOL_ORDER if p in grading]
    rules = (
        "grade_everything_b",
        "point_estimate",
        "conformal_lower_bound",
        "minimum_expected_cost",
    )
    rule_labels = ("no model", "point estimate", "90% lower bound", "min expected cost")
    positions = np.arange(len(grading_protocols))
    width = 0.8 / len(rules)
    for index, (rule, rule_label) in enumerate(zip(rules, rule_labels)):
        values = [grading[p][rule]["cost_per_cell"] for p in grading_protocols]
        axes[0].bar(
            positions + index * width - 0.4 + width / 2, values, width=width, label=rule_label
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([PROTOCOL_LABELS[p] for p in grading_protocols], rotation=12)
    axes[0].set_ylabel("Cost per cell (£)")
    axes[0].set_title("Lot acceptance: cost of a misgraded cell", fontsize=10)
    axes[0].legend(fontsize=8)

    replacement = summary["business"]["replacement"]
    replacement_protocols = [p for p in PROTOCOL_ORDER if p in replacement]
    policies = (
        "fleet_quantile",
        "point_estimate",
        "conformal_lower_bound",
        "newsvendor_quantile",
    )
    policy_labels = ("no features", "point estimate", "90% lower bound", "newsvendor")
    positions = np.arange(len(replacement_protocols))
    width = 0.8 / len(policies)
    for index, (policy, policy_label) in enumerate(zip(policies, policy_labels)):
        values = [
            replacement[p].get(policy, {}).get("cost_per_pack", 0.0) for p in replacement_protocols
        ]
        axes[1].bar(
            positions + index * width - 0.4 + width / 2, values, width=width, label=policy_label
        )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([PROTOCOL_LABELS[p] for p in replacement_protocols], rotation=12)
    axes[1].set_ylabel("Cost per pack (£)")
    axes[1].set_title("Maintenance scheduling: cost of a booking", fontsize=10)
    axes[1].legend(fontsize=8)

    figure.suptitle("Prediction error, priced", fontsize=11)
    figure.savefig(path)
    plt.close(figure)


# -- the markdown report -------------------------------------------------


def write_evaluation_markdown(summary: dict[str, Any], path: Path) -> None:
    """The long-form report that sits beside the figures."""
    tables = render_tables(summary)
    fleet = summary["fleet"]
    qualification = summary["business"]["qualification"]

    lines = [
        f"# Evaluation: `{summary['config_name']}`",
        "",
        "Generated by `cobalt evaluate`. Do not edit by hand, it is overwritten.",
        "",
        f"Data source: **{summary['data_source']}**. There is no laboratory behind",
        "these numbers. `docs/LIMITATIONS.md` says what that does and does not permit.",
        "",
        "## Figures",
        "",
        "![Capacity fade by batch](capacity_fade.png)",
        "",
        "![Discharge curve differences and the early-life signal](delta_q.png)",
        "",
        "![Every model under every protocol](protocol_gap.png)",
        "",
        "![Early life: predicted against true, with intervals](parity_early_life.png)",
        "",
        "![Online: predicted against true, with intervals](parity_online.png)",
        "",
        "![What each decision rule costs](cost.png)",
        "",
        "## The fleet",
        "",
        tables["fleet"],
        "",
        "## Early life: predicting cycle life from the first 100 cycles",
        "",
        f"Rows: {summary['tasks']['early_life']['n_rows']} cells. "
        f"Excluded: {summary['tasks']['early_life']['excluded']}.",
        "",
        "RMSE in cycles, mean over folds:",
        "",
        tables["early_life"],
        "",
        "Share of cells predicted within 20% of their true life:",
        "",
        tables["early_life_alpha"],
        "",
        "## Online: predicting remaining useful life from a 50-cycle window",
        "",
        f"Rows: {summary['tasks']['online_soh']['n_rows']} windows over "
        f"{summary['tasks']['online_soh']['n_cells']} cells.",
        "",
        tables["online"],
        "",
        tables["online_alpha"],
        "",
        "## Interval coverage",
        "",
        "Conformal intervals calibrated on held-out groups, against the same",
        "intervals calibrated on random rows. Negative difference means the",
        "random calibration promised more coverage than it delivered.",
        "",
        tables["coverage"],
        "",
        "## What the decisions cost",
        "",
        "Lot acceptance, pounds per cell:",
        "",
        tables["grading"],
        "",
        "Maintenance scheduling, pounds per pack:",
        "",
        tables["replacement"],
        "",
        "## Test time",
        "",
        f"Cycling a cell to end of life at {fleet['median_cycle_life']} median cycles takes",
        f"{qualification['full_test_days']} days of a channel. Stopping at 100 cycles takes",
        f"{qualification['early_test_days']} days, a saving of "
        f"{qualification['fraction_saved']:.1%}.",
        "",
    ]
    write_text(path, LF.join(lines) + LF)


def write_summary(summary: dict[str, Any], path: Path) -> None:
    write_json(path, summary)


def write_prediction_tables(
    pooled: dict[str, dict[str, OutOfFoldPredictions]],
    deployed: str,
    directory: Path,
    prefix: str,
) -> None:
    """Out-of-fold predictions, one CSV per protocol, for anybody who wants to check."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for protocol, models in pooled.items():
        predictions = models.get(f"{deployed}+conformal")
        if predictions is None:
            continue
        frame: pd.DataFrame = predictions.as_frame()
        write_csv(frame, directory / f"{prefix}_{protocol}.csv")
