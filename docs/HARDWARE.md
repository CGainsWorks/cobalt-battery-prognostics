# The rig this needs

`cobalt.hal.serial_bms.SerialBms` raises `NoHardwareError` from every method
that would touch a device. This page is what it would take to make it work and
what would have to be true of the data for the results in this repository to
mean anything about real cells.

It is deliberately specific. A limitation with a shape is a plan; a limitation
without one is an excuse.

---

## 1. Cells

**Minimum: three manufacturing lots, thirty cells each.**

The lot count is not negotiable and it is the constraint most likely to be
overlooked. This repository's central argument is that holding out whole
manufacturing lots is a different and harder question than holding out cells.
With one lot that protocol cannot be run at all. With two, a cross-validated
version of it does not exist. Three is the bare minimum for a single held-out
lot with two remaining to fit on and even then
`cobalt.models.conformal.MIN_FIT_GROUPS` says grouped calibration will be
starved.

The configuration that produced the results here uses twenty lots. Five or six
real ones would support a credible version of the same analysis; three would
support a much weaker one and should be reported as such.

Lots should differ in the way real lots differ: different electrode coating
runs, ideally different production dates. Thirty cells per lot is enough to
estimate within-lot spread against between-lot spread, which is the quantity the
whole protocol argument rests on.

## 2. Cycler

**Eight to sixty-four channels**, for example a Neware CT-4008 or a Maccor 4300.

Channel count sets how many cells age in parallel and cells must age in
parallel because a single channel takes months per cell. Ninety cells at a
median thousand cycles, at roughly 1.6 hours a cycle, is about 6,000 channel-days
of testing. On eight channels that is two years. On sixty-four it is three
months.

This arithmetic is the business case for the early-life model and it is the one
part of the case that does not depend on the model being accurate. It is
computed in `cobalt.business.cost.qualification_time_saving`.

Required capabilities:

- Constant-current constant-voltage charge, programmable C-rate.
- Constant-current discharge to a voltage cut-off.
- Per-cycle logging of capacity, energy and the full voltage-against-capacity
  trace at a resolution fine enough to compute `dQ(V)`. Coarse logging makes the
  headline early-life feature unusable, so this is worth checking before
  starting rather than after.
- Scheduled reference performance tests: a slow full discharge every twenty-five
  cycles at a fixed low C-rate and fixed temperature. This is what makes
  `discharge_capacity_ah` a measurement rather than an estimate and it is what
  the early-life features assume they have.

## 3. Thermal

**One chamber per temperature point, three points minimum**, for example 15 °C,
25 °C and 45 °C.

The Arrhenius term is the single largest driver in the degradation model and it
cannot be characterised at one temperature. The plating term needs the cold
point specifically: it is accelerated at *low* temperature, which is why the
temperature optimum for a cell is a compromise rather than "colder is better"
and a rig that only tests hot will miss it.

Chamber stability of ±2 °C is adequate. Cell surface thermocouples are needed as
well as chamber air temperature, because the self-heating term at 4C is tens of
kelvin and the model's `thermal_resistance_k_per_w` is currently a guess.

## 4. The link

**CAN via `python-can` and a DBC file, or Modbus RTU over RS-485.**

`SerialBms.read_cycle` is where frames become a `CycleTelemetry`. Everything
above that line in this codebase already works, because it is written against
the interface rather than against the simulator.

The fields that have to come off the link are exactly the ones on
`CycleTelemetry` and all of them are on a production battery-management-system
bus already: discharge and charge capacity, internal resistance estimate, mean
and maximum temperature, both C-rates, charge time, discharge energy and
cumulative throughput. Nothing on that record requires a laboratory instrument,
which was the point of choosing it.

## 5. Acceptance tests for the link

Before any of the analysis is worth running, the link should pass these. They
are cheap and each one corresponds to a real failure that is silent otherwise.

1. **Round trip.** Cycle one cell for twenty-five cycles. The cycler's own log
   and the ingested telemetry must agree on capacity to within 0.5%. Disagreement
   here is usually a scaling factor in the DBC.
2. **Monotonicity.** Over a hundred cycles at fixed conditions, the smoothed
   capacity must be non-increasing and the smoothed resistance non-decreasing.
   Violations indicate a reference test that is not actually at fixed
   temperature.
3. **Cut-off consistency.** The measured discharge capacity must fall when the
   discharge C-rate is raised, by an amount consistent with the measured
   resistance. This is the coupling `Cell.cutoff_state_of_charge` models and if
   it is absent in the data the resistance channel is not measuring what it
   claims.
4. **Frame loss.** Run the link for twenty-four hours and count gaps. The
   telemetry store deduplicates and re-sorts on ingest but a systematically
   missing cycle index will silently shorten every window.
5. **Clock.** Cycle indices from the pack and from the cycler must not drift
   apart. The online features are indexed by cycle, not by time and a drifting
   index corrupts every slope in the feature set.

---

## What would not change

The features, the split protocols, the models, the conformal calibration, the
cost models, the service and the whole test suite. All of them consume
`CycleTelemetry` and none of them import `cobalt.sim`.

The work is one class and a DBC file. Keeping it that small was the reason for
putting the seam at `cobalt.hal` and writing everything above it against an
interface, rather than against a simulator that happened to be convenient.
