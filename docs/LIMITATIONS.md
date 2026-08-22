# Limitations

What this repository is evidence for, and what it is not. Written plainly,
because the interesting parts of the work are worth less if the boundary around
them is vague.

---

## The headline limitation

**No physical cell was tested. There was no cycler, no thermal chamber and no
cells.** Every figure, table and model in this repository was produced from the
simulator in `cobalt.sim`.

That is stated in the README's first paragraph, in the first field of the model
card, in the OpenAPI description of the service, and in the docstring of the
module that would talk to real hardware. It is not a caveat added at the end.

---

## What follows from that

### No accuracy number here transfers to a real cell

The simulator is a lumped empirical model. It is not fitted to laboratory data,
because there was no laboratory. Its parameters were chosen to produce fade
curves, knee behaviour and Arrhenius sensitivity of roughly the right shape and
magnitude, and that is the entire claim being made for it.

Real cells are noisier, less consistent and less well-behaved in ways that are
specific and known:

- **Path dependence.** The model's state is two monotonic accumulators,
  effective aged time and effective throughput, so its history collapses to two
  numbers. A real cell remembers the *order* things happened in: a cold fast
  charge early in life leaves plated lithium that changes everything afterwards.
- **No sudden failures.** Separator defects, tab welds, vent events and internal
  short circuits are absent. The simulated fleet has a smooth cycle-life
  distribution with no tail of early deaths, and that tail is where the money is
  in a real fleet.
- **Idealised measurement.** Sensor noise is independent and Gaussian. Real
  battery-management-system telemetry has drift, quantisation, dropped frames,
  temperature-dependent current-sensor bias and occasional nonsense.
- **One chemistry, one form factor, one nominal capacity.**

### One symptom of this is measurable inside the repository

The elastic net, fitted on this data, puts most of its weight on
`log_abs_fade_slope` and `charge_time_drift`, and gives the `dQ(V)` variance a
small coefficient. In the published real-cell result the `dQ(V)` variance
dominates.

That difference is not a contradiction of the paper, it is a measurement of the
simulator. A capacity fade slope taken over a hundred simulated cycles is a
clean, low-noise estimate; on real hardware the same slope is buried in
measurement noise and cell-to-cell scatter, which is exactly the condition that
makes `dQ(V)` worth the trouble of computing. **A simulator cannot reproduce the
motivation for a feature it has designed away.**

This is worth stating because it is the most concrete evidence in the repository
about the direction of its own bias, and it points the same way as everything
else: the simulated problem is easier than the real one.

### The methodology is the transferable part

Three results here are properties of the evaluation and the decision theory
rather than of the chemistry, and they hold on any dataset with group structure:

1. A random split over overlapping windows leaks, and the leak is large.
2. Conformal intervals calibrated on random rows undercover on a held-out group,
   and calibrating on held-out groups fixes it — provided enough groups remain
   in the fit.
3. Under asymmetric costs the decision is a quantile of the predictive
   distribution, not a point estimate, and the naive "use the interval bound"
   heuristic overshoots.

Those would be worth writing up from real cells too, and the code that produces
them does not care where the telemetry came from.

---

## Modelling choices that would need revisiting on real data

- **Conformal intervals become a predictive distribution by assuming Gaussian
  residuals in log space.** `implied_sigma` does the conversion and says so.
  This is wrong in the tails, which is where the expensive decisions live. A
  conformalised quantile regressor would remove the assumption entirely and is
  the obvious next piece of work.
- **The interval is a constant width per fold.** Split conformal gives one
  half-width for every prediction, so a cell the model is genuinely unsure about
  gets the same interval as one it is confident about. Conformalised quantile
  regression fixes this too.
- **Right-censored cells are dropped, not modelled.** Six of 320 cells in the
  default fleet never reached end of life inside the cycle ceiling, and they are
  excluded with a count. On real data censoring is much more common and dropping
  it biases the labels short. Survival analysis is the correct tool and is not
  used here.
- **Cycle life is defined as first crossing of 80% of nameplate.** A single
  noisy measurement can trigger it. A real qualification would require the
  threshold to be crossed on a reference performance test, or on several
  consecutive cycles.
- **The knee is a sigmoid on the plating rate.** It produces the right shape.
  Whether it produces the right *timing* under an unusual duty cycle is
  untested, because testing it needs cells.

---

## Things the numbers are not

- **The cost figures are illustrative.** £2.40 per cycle of discarded life,
  £14.00 per cycle run past end of life, and the grading cost matrix, are sized
  for a small commercial pack and live in `config/default.yaml`. They are
  configuration, not findings. They set the shape of the answer, and changing
  them changes which decision rule wins.
- **The £637 per pack is not a saving anybody can bank.** It is the cost of one
  scheduling policy against another under one set of assumed prices on one
  simulated fleet. The transferable part is that the ordering of the policies
  follows from the cost ratio and not from the model.
- **The service is not production software.** The telemetry store is an
  in-memory dictionary behind a lock. There is no authentication, no persistence
  and no multi-process story. Those are real decisions with real operational
  consequences and making them here on no evidence would be guessing.
- **Nothing here is qualified for a safety-related decision**, and the model card
  says so under `out_of_scope`.

---

## What would change this

The rig is specified in [`HARDWARE.md`](HARDWARE.md), including why three
coating lots is the minimum that makes this repository's central argument
testable at all: with fewer, a batch-independent protocol cannot be run.

Nothing else in the codebase would change. The features, the split protocols,
the models, the conformal calibration, the cost models and the service all
consume `CycleTelemetry` and never touch the simulator. That was the point of
putting the seam where it is.
