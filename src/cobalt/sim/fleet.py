"""Fleets of cells, organised into batches.

A fleet is not a bag of independent cells and pretending otherwise is the
single most common way a battery prognostics result gets overstated. Cells
arrive in manufacturing lots that share an electrode coating run and they get
deployed to sites that share a climate and a duty cycle. Two cells from the
same batch are far more alike than two cells drawn at random.

So this generator builds correlation in deliberately, at two levels:

* **Batch parameters.** Each batch draws its own multipliers on the SEI rate,
  the plating rate, the knee onset and the delivered capacity. Cells then jitter
  around their batch, with a much smaller spread than the between-batch spread.
* **Batch duty cycle.** Each batch draws a site: a mean ambient temperature and
  a mean charge C-rate. A lot shipped to a hot depot on fast charge ages
  differently from the same lot shipped somewhere temperate.

The consequence is that holding out random cells leaks: for every test cell
there are siblings in the training set built on the same coating run and run at
the same site. Holding out whole batches does not leak and it is what happens
when a new lot arrives. `docs/EVALUATION_PROTOCOL.md` has the full argument and
`cobalt.data.splits` implements both so the gap can be measured rather than
asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cobalt.sim.cell import Cell, CellParameters, CycleRecord, DutyCycle


@dataclass(frozen=True)
class FleetConfig:
    """Everything that varies between one simulated fleet and another."""

    n_batches: int = 12
    cells_per_batch: int = 20
    #: Stop cycling a cell once it crosses this fraction of nameplate capacity.
    end_of_life_soh: float = 0.80
    #: Hard ceiling so a very gently used cell cannot run the simulation forever.
    max_cycles: int = 2400
    #: Keep cycling this many cycles past end of life, so the dataset contains the
    #: post-threshold behaviour a fleet operator actually sees before replacement.
    cycles_past_eol: int = 40

    #: Between-batch spread, as a lognormal sigma on each rate multiplier.
    batch_sigma_sei: float = 0.20
    batch_sigma_plate: float = 0.26
    batch_sigma_knee: float = 0.11
    batch_sigma_capacity: float = 0.013

    #: Within-batch spread. Deliberately several times smaller than the above.
    cell_sigma_sei: float = 0.055
    cell_sigma_plate: float = 0.070
    cell_sigma_knee: float = 0.030
    cell_sigma_capacity: float = 0.006
    cell_sigma_resistance: float = 0.045

    #: Site conditions, drawn per batch then jittered per cell.
    site_temperature_mean_c: float = 27.0
    site_temperature_sigma_c: float = 7.5
    cell_temperature_sigma_c: float = 1.6
    site_charge_c_rate_choices: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    cell_charge_c_rate_sigma: float = 0.10
    discharge_c_rate: float = 1.0
    depth_of_discharge_mean: float = 0.94
    depth_of_discharge_sigma: float = 0.05
    rest_hours_mean: float = 3.0
    rest_hours_sigma: float = 1.2

    #: Cycles at which the full discharge curve is retained for delta-Q features.
    curve_cycles: tuple[int, ...] = (2, 10, 100)
    #: Voltage grid the delta-Q features are evaluated on, high to low.
    voltage_grid_v: tuple[float, float, int] = (3.82, 3.34, 60)


@dataclass
class CellHistory:
    """One simulated cell: its identity, how it was built and used and its log."""

    cell_id: str
    batch_id: str
    parameters: CellParameters
    duty: DutyCycle
    records: list[CycleRecord] = field(default_factory=list)
    #: Discharged capacity on the shared voltage grid, keyed by cycle index.
    voltage_curves: dict[int, np.ndarray] = field(default_factory=dict)
    #: Cycles until state of health first fell below the end-of-life threshold.
    #: ``None`` when the cell hit ``max_cycles`` first, which is right-censoring
    #: and has to be handled rather than quietly dropped.
    cycle_life: int | None = None

    @property
    def censored(self) -> bool:
        return self.cycle_life is None


@dataclass
class Fleet:
    """A simulated fleet plus the voltage grid its curves were sampled on."""

    cells: list[CellHistory]
    voltage_grid: np.ndarray
    config: FleetConfig
    seed: int

    @property
    def batch_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for cell in self.cells:
            seen.setdefault(cell.batch_id, None)
        return list(seen)

    def uncensored(self) -> list[CellHistory]:
        return [cell for cell in self.cells if not cell.censored]


def _lognormal(rng: np.random.Generator, sigma: float) -> float:
    """A multiplier with median 1 and the given log-space spread."""
    if sigma <= 0.0:
        return 1.0
    return float(rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))


def _build_batch(
    rng: np.random.Generator,
    config: FleetConfig,
    base: CellParameters,
    batch_index: int,
) -> tuple[list[CellParameters], list[DutyCycle], str]:
    """Draw one batch: shared lot characteristics, then per-cell jitter."""
    batch_id = f"B{batch_index:02d}"

    lot_sei = _lognormal(rng, config.batch_sigma_sei)
    lot_plate = _lognormal(rng, config.batch_sigma_plate)
    lot_knee = _lognormal(rng, config.batch_sigma_knee)
    lot_capacity = _lognormal(rng, config.batch_sigma_capacity)

    site_temperature = float(
        rng.normal(config.site_temperature_mean_c, config.site_temperature_sigma_c)
    )
    site_charge_c_rate = float(rng.choice(np.asarray(config.site_charge_c_rate_choices)))

    parameters: list[CellParameters] = []
    duties: list[DutyCycle] = []
    for _ in range(config.cells_per_batch):
        cell_parameters = base.scaled(
            k_sei=lot_sei * _lognormal(rng, config.cell_sigma_sei),
            k_plate=lot_plate * _lognormal(rng, config.cell_sigma_plate),
            knee_onset=lot_knee * _lognormal(rng, config.cell_sigma_knee),
            initial_capacity_scale=lot_capacity * _lognormal(rng, config.cell_sigma_capacity),
            initial_resistance_ohm=_lognormal(rng, config.cell_sigma_resistance),
        )
        duty = DutyCycle(
            charge_c_rate=max(
                0.2,
                site_charge_c_rate * _lognormal(rng, config.cell_charge_c_rate_sigma),
            ),
            discharge_c_rate=config.discharge_c_rate,
            ambient_temperature_c=float(
                rng.normal(site_temperature, config.cell_temperature_sigma_c)
            ),
            depth_of_discharge=float(
                np.clip(
                    rng.normal(config.depth_of_discharge_mean, config.depth_of_discharge_sigma),
                    0.55,
                    1.0,
                )
            ),
            rest_hours_per_cycle=float(
                max(0.1, rng.normal(config.rest_hours_mean, config.rest_hours_sigma))
            ),
        )
        parameters.append(cell_parameters)
        duties.append(duty)

    return parameters, duties, batch_id


def simulate_cell(
    cell_id: str,
    batch_id: str,
    parameters: CellParameters,
    duty: DutyCycle,
    config: FleetConfig,
    voltage_grid: np.ndarray,
    rng: np.random.Generator,
) -> CellHistory:
    """Cycle one cell to end of life, or to the ceiling and keep the log."""
    cell = Cell(parameters, duty, rng=rng)
    history = CellHistory(cell_id=cell_id, batch_id=batch_id, parameters=parameters, duty=duty)

    curve_cycles = set(config.curve_cycles)
    stop_at: int | None = None

    while cell.cycle < config.max_cycles:
        record = cell.step()
        history.records.append(record)

        if record.cycle in curve_cycles:
            history.voltage_curves[record.cycle] = cell.capacity_at_voltage(voltage_grid)

        if history.cycle_life is None and cell.state_of_health < config.end_of_life_soh:
            history.cycle_life = record.cycle
            stop_at = record.cycle + config.cycles_past_eol

        if stop_at is not None and record.cycle >= stop_at:
            break

    return history


def simulate_fleet(config: FleetConfig, seed: int, base: CellParameters | None = None) -> Fleet:
    """Simulate a whole fleet. Deterministic given the seed.

    Each cell gets its own generator, spawned from the fleet seed, so adding a
    cell to the end of a batch does not change any cell before it. That property
    is what lets the CI configuration be a genuine subset of the full run rather
    than a different fleet with the same name.
    """
    base = base if base is not None else CellParameters()
    lo, hi, count = config.voltage_grid_v
    voltage_grid = np.linspace(lo, hi, count)

    root = np.random.default_rng(seed)
    structure_rng = np.random.default_rng(root.integers(0, 2**63 - 1))
    cell_seeds = np.random.SeedSequence(seed).spawn(config.n_batches * config.cells_per_batch)

    cells: list[CellHistory] = []
    index = 0
    for batch_index in range(config.n_batches):
        parameters, duties, batch_id = _build_batch(structure_rng, config, base, batch_index)
        for within_batch, (cell_parameters, duty) in enumerate(zip(parameters, duties)):
            history = simulate_cell(
                cell_id=f"{batch_id}-C{within_batch:02d}",
                batch_id=batch_id,
                parameters=cell_parameters,
                duty=duty,
                config=config,
                voltage_grid=voltage_grid,
                rng=np.random.default_rng(cell_seeds[index]),
            )
            cells.append(history)
            index += 1

    return Fleet(cells=cells, voltage_grid=voltage_grid, config=config, seed=seed)
