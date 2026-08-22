"""A single lithium-ion cell that ages.

This is the substrate the whole repository runs on, so it is worth being precise
about what it is and what it is not.

It is a lumped empirical degradation model with the four mechanisms that
dominate the published ageing literature for graphite/NMC-class cells:

1. **SEI growth.** An anode film that thickens with the square root of time and
   consumes cyclable lithium. Accelerated by temperature through an Arrhenius
   term, so a cell parked hot degrades even with no cycling at all.
2. **Lithium plating and loss of active material.** Driven by throughput rather
   than time, worsened by high charge C-rate and by *low* temperature, which is
   why the temperature optimum for a cell is a compromise rather than "colder is
   better".
3. **The knee.** Once the SEI film passes a threshold the plating rate turns up
   sharply, which is what produces the characteristic elbow in a capacity fade
   curve rather than a gentle glide to end of life.
4. **Resistance rise**, from both mechanisms, which shortens the usable
   discharge because the terminal voltage hits the cut-off sooner.

It is **not** fitted to a real cell. No parameter here was regressed against
laboratory data, because there was no laboratory. The parameter values produce
fade curves, knee behaviour and Arrhenius sensitivity of roughly the right shape
and magnitude and that is the whole claim. See `docs/LIMITATIONS.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator

#: Universal gas constant, J/(mol K).
GAS_CONSTANT = 8.314462618

#: Open-circuit voltage anchors for a graphite/NMC-class cell, as
#: (state of charge, volts). Monotonic, so the Pchip interpolant through them is
#: monotonic too and can be inverted without care.
_OCV_ANCHORS_SOC = (0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00)
_OCV_ANCHORS_V = (3.20, 3.45, 3.545, 3.63, 3.725, 3.855, 4.005, 4.095, 4.20)

_OCV = PchipInterpolator(np.asarray(_OCV_ANCHORS_SOC), np.asarray(_OCV_ANCHORS_V))
_SOC_AT_OCV = PchipInterpolator(np.asarray(_OCV_ANCHORS_V), np.asarray(_OCV_ANCHORS_SOC))


def open_circuit_voltage(state_of_charge: float | np.ndarray) -> np.ndarray:
    """Cell OCV in volts at a given state of charge in [0, 1]."""
    return _OCV(np.clip(state_of_charge, 0.0, 1.0))


def state_of_charge_at_voltage(voltage: float | np.ndarray) -> np.ndarray:
    """Inverse of :func:`open_circuit_voltage`, clipped to the modelled range."""
    lo, hi = _OCV_ANCHORS_V[0], _OCV_ANCHORS_V[-1]
    return np.clip(_SOC_AT_OCV(np.clip(voltage, lo, hi)), 0.0, 1.0)


@dataclass(frozen=True)
class CellParameters:
    """Per-cell degradation parameters.

    Every rate constant is a multiplier that manufacturing spread perturbs, so a
    batch is described by shifting these and a cell by jittering around its
    batch. That structure is the point: it is what makes a batch-independent
    split genuinely harder than a random one rather than a formality.
    """

    nameplate_capacity_ah: float = 2.20
    #: Manufacturing spread on delivered capacity. 1.02 means the cell leaves the
    #: line at 102% of nameplate, which is normal and is why state of health is
    #: quoted against nameplate rather than against first measurement.
    initial_capacity_scale: float = 1.00
    initial_resistance_ohm: float = 0.0300

    #: Capacity fraction lost to SEI per sqrt(equivalent day at reference temperature).
    k_sei: float = 4.6e-3
    #: Capacity fraction lost to plating per equivalent full cycle at reference conditions.
    k_plate: float = 2.15e-5

    #: SEI capacity loss at which the plating rate begins to turn up.
    knee_onset: float = 0.052
    #: How abruptly it turns up. Larger is sharper.
    knee_sharpness: float = 110.0
    #: Multiplier on the plating rate once the knee is fully engaged.
    knee_gain: float = 6.0

    #: Fractional resistance rise per sqrt(equivalent day).
    k_r_sei: float = 0.085
    #: Fractional resistance rise per equivalent full cycle.
    k_r_cycle: float = 7.5e-5

    #: Arrhenius activation energy for SEI growth, J/mol.
    activation_energy_j_per_mol: float = 34_000.0
    reference_temperature_k: float = 298.15
    #: Plating acceleration per kelvin below the reference temperature.
    plating_cold_coefficient: float = 0.045
    #: Exponent on charge C-rate in the plating term.
    c_rate_exponent: float = 1.45
    reference_c_rate: float = 1.0

    #: Cell-to-ambient thermal resistance, K/W. Sets the self-heating rise.
    thermal_resistance_k_per_w: float = 9.0

    def scaled(self, **multipliers: float) -> CellParameters:
        """Return a copy with the named fields multiplied by the given factors."""
        updates = {name: getattr(self, name) * factor for name, factor in multipliers.items()}
        return replace(self, **updates)


@dataclass(frozen=True)
class DutyCycle:
    """How a cell is actually used, which matters as much as how it was built."""

    charge_c_rate: float = 1.0
    discharge_c_rate: float = 1.0
    ambient_temperature_c: float = 25.0
    depth_of_discharge: float = 1.0
    #: Idle time between cycles. Calendar ageing happens here.
    rest_hours_per_cycle: float = 1.0
    #: Discharge terminates at this terminal voltage or at empty, whichever comes first.
    voltage_cutoff_v: float = 3.00


@dataclass
class CycleRecord:
    """What a battery management system would log for one cycle.

    Everything on here is observable on a real pack. The two `true_` fields are
    the noise-free ground truth and exist only so the tests and the evaluation
    can separate model error from measurement error. They never reach a feature.
    """

    cycle: int
    discharge_capacity_ah: float
    charge_capacity_ah: float
    internal_resistance_ohm: float
    mean_temperature_c: float
    max_temperature_c: float
    charge_c_rate: float
    discharge_c_rate: float
    charge_time_min: float
    discharge_energy_wh: float
    cumulative_ah_throughput: float
    true_capacity_ah: float
    true_resistance_ohm: float


class Cell:
    """A cell that can be stepped one cycle at a time.

    The state is two accumulators, both monotonic: effective aged time (days at
    the reference temperature) and effective throughput (full cycles at the
    reference C-rate and temperature). Every observable is a function of those
    two plus the cell parameters, so the model is path-independent in a way a
    real cell is not. That simplification is recorded in `docs/LIMITATIONS.md`.
    """

    def __init__(
        self,
        parameters: CellParameters,
        duty: DutyCycle,
        rng: np.random.Generator | None = None,
        capacity_noise_ah: float = 0.0035,
        resistance_noise_frac: float = 0.020,
        temperature_noise_c: float = 0.45,
    ) -> None:
        self.parameters = parameters
        self.duty = duty
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._capacity_noise_ah = capacity_noise_ah
        self._resistance_noise_frac = resistance_noise_frac
        self._temperature_noise_c = temperature_noise_c

        self.cycle = 0
        self._effective_days = 0.0
        self._effective_cycles = 0.0
        self._plating_loss = 0.0
        self._throughput_ah = 0.0

    # -- derived quantities ------------------------------------------------

    @property
    def rated_capacity_ah(self) -> float:
        """Nameplate capacity. State of health is quoted against this."""
        return self.parameters.nameplate_capacity_ah

    @property
    def _sei_loss(self) -> float:
        return self.parameters.k_sei * math.sqrt(self._effective_days)

    @property
    def _knee_multiplier(self) -> float:
        p = self.parameters
        x = p.knee_sharpness * (self._sei_loss - p.knee_onset)
        # Guard the exponential: knee_sharpness is large and x runs to -inf early
        # in life, where math.exp underflows to zero perfectly happily but the
        # positive branch would overflow late in a long simulation.
        sigmoid = 1.0 / (1.0 + math.exp(-x)) if x > -700.0 else 0.0
        return 1.0 + p.knee_gain * sigmoid

    @property
    def true_capacity_ah(self) -> float:
        """Noise-free usable capacity before the voltage cut-off is applied."""
        p = self.parameters
        remaining = 1.0 - self._sei_loss - self._plating_loss
        return max(0.0, p.nameplate_capacity_ah * p.initial_capacity_scale * remaining)

    @property
    def true_resistance_ohm(self) -> float:
        p = self.parameters
        rise = (
            p.k_r_sei * math.sqrt(self._effective_days)
            + p.k_r_cycle * self._effective_cycles * self._knee_multiplier
        )
        return p.initial_resistance_ohm * (1.0 + rise)

    @property
    def state_of_health(self) -> float:
        """Deliverable capacity over nameplate, the number a warranty is written against."""
        return self.deliverable_capacity_ah() / self.rated_capacity_ah

    # -- electrical behaviour ----------------------------------------------

    def cell_temperature_c(self) -> float:
        """Ambient plus ohmic self-heating at the discharge current."""
        current = self.duty.discharge_c_rate * self.rated_capacity_ah
        heat_w = current * current * self.true_resistance_ohm
        return self.duty.ambient_temperature_c + self.parameters.thermal_resistance_k_per_w * heat_w

    def cutoff_state_of_charge(self) -> float:
        """State of charge at which the terminal voltage reaches the cut-off.

        Resistance rise pushes this up, so an aged cell stops delivering before
        it is empty. This is the coupling that makes capacity fade and power fade
        two views of one process rather than two independent numbers.
        """
        current = self.duty.discharge_c_rate * self.rated_capacity_ah
        required_ocv = self.duty.voltage_cutoff_v + current * self.true_resistance_ohm
        return float(state_of_charge_at_voltage(required_ocv))

    def deliverable_capacity_ah(self) -> float:
        """Capacity actually extractable down to the cut-off, noise free."""
        usable_fraction = max(0.0, 1.0 - self.cutoff_state_of_charge())
        return self.true_capacity_ah * usable_fraction

    def discharge_curve(self, points: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Terminal voltage against capacity discharged, for one constant-current discharge.

        Returns ``(capacity_ah, voltage_v)`` with capacity ascending. This is the
        raw material for the delta-Q(V) features, which are the ones that carry
        the early-life signal.
        """
        current = self.duty.discharge_c_rate * self.rated_capacity_ah
        resistance = self.true_resistance_ohm
        cutoff_soc = self.cutoff_state_of_charge()

        soc = np.linspace(1.0, cutoff_soc, points)
        voltage = open_circuit_voltage(soc) - current * resistance
        capacity = self.true_capacity_ah * (1.0 - soc)
        return capacity, voltage

    def capacity_at_voltage(self, voltage_grid: np.ndarray) -> np.ndarray:
        """Capacity discharged by the time the terminal voltage reaches each grid point.

        ``NaN`` where the grid point lies outside the voltage window this cycle
        actually reached, so a caller that compares two cycles is forced to
        notice rather than silently extrapolate.
        """
        capacity, voltage = self.discharge_curve()
        # The curve is descending in voltage; np.interp needs ascending x.
        ascending_v = voltage[::-1]
        ascending_q = capacity[::-1]
        out = np.interp(voltage_grid, ascending_v, ascending_q, left=np.nan, right=np.nan)
        outside = (voltage_grid < ascending_v[0]) | (voltage_grid > ascending_v[-1])
        out[outside] = np.nan
        return out

    # -- ageing -------------------------------------------------------------

    def step(self) -> CycleRecord:
        """Age the cell by one full charge, rest and discharge and log it."""
        p = self.parameters
        duty = self.duty

        temperature_c = self.cell_temperature_c()
        temperature_k = temperature_c + 273.15

        # Wall-clock hours the cycle occupies. Depth of discharge sets how much
        # charge moves, the C-rates set how fast and the rest is calendar time.
        active_hours = duty.depth_of_discharge * (
            1.0 / duty.charge_c_rate + 1.0 / duty.discharge_c_rate
        )
        cycle_hours = active_hours + duty.rest_hours_per_cycle

        arrhenius = math.exp(
            -p.activation_energy_j_per_mol
            / GAS_CONSTANT
            * (1.0 / temperature_k - 1.0 / p.reference_temperature_k)
        )
        self._effective_days += (cycle_hours / 24.0) * arrhenius

        cold_delta_k = max(0.0, p.reference_temperature_k - temperature_k)
        plating_temperature_factor = math.exp(p.plating_cold_coefficient * cold_delta_k)
        rate_factor = (duty.charge_c_rate / p.reference_c_rate) ** p.c_rate_exponent
        equivalent_cycles = duty.depth_of_discharge * rate_factor * plating_temperature_factor

        # The knee multiplies the *rate*, so it has to be applied inside the
        # accumulation rather than to the total. Applying it to the total is a
        # tempting one-liner and it retroactively re-ages the cell's whole past.
        self._plating_loss += p.k_plate * equivalent_cycles * self._knee_multiplier
        self._effective_cycles += equivalent_cycles

        delivered = self.deliverable_capacity_ah()
        self._throughput_ah += delivered
        self.cycle += 1

        rng = self._rng
        measured_capacity = delivered + rng.normal(0.0, self._capacity_noise_ah)
        measured_resistance = self.true_resistance_ohm * (
            1.0 + rng.normal(0.0, self._resistance_noise_frac)
        )
        measured_temperature = temperature_c + rng.normal(0.0, self._temperature_noise_c)

        # Coulombic efficiency below one: charge in exceeds discharge out and the
        # gap widens as the parasitic side reactions grow.
        efficiency = 0.998 - 0.020 * (self._sei_loss + self._plating_loss)
        charge_capacity = measured_capacity / max(efficiency, 0.90)
        charge_time_min = 60.0 * charge_capacity / (duty.charge_c_rate * self.rated_capacity_ah)

        mean_voltage = float(np.mean(self.discharge_curve(points=32)[1]))

        return CycleRecord(
            cycle=self.cycle,
            discharge_capacity_ah=max(0.0, measured_capacity),
            charge_capacity_ah=max(0.0, charge_capacity),
            internal_resistance_ohm=max(1e-6, measured_resistance),
            mean_temperature_c=measured_temperature,
            max_temperature_c=measured_temperature + 1.8,
            charge_c_rate=duty.charge_c_rate,
            discharge_c_rate=duty.discharge_c_rate,
            charge_time_min=charge_time_min,
            discharge_energy_wh=max(0.0, measured_capacity) * mean_voltage,
            cumulative_ah_throughput=self._throughput_ah,
            true_capacity_ah=delivered,
            true_resistance_ohm=self.true_resistance_ohm,
        )
