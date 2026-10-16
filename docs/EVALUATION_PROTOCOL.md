# Evaluation protocol

The rules every number in this repository is produced under and the reasoning
behind each one. If a result here is ever quoted somewhere else, this is the
page that says what it means.

---

## 1. The three protocols

### `random_cycle` (leaky, reported to show the size of the leak)

Shuffle every cell-cycle row and cut. Available for the online task only.

Cycle 604 of a cell can be in training while cycle 629 of the same cell is in
test. With a fifty-cycle window and a twenty-five-cycle stride, those two rows
share half their samples, come from the same cell, the same coating run and the
same depot and have labels twenty-five cycles apart. The model is not
generalising, it is interpolating between neighbours it has already seen.

This protocol is in the repository for one reason: so the size of the leak is
visible rather than asserted. It is labelled "(leaky)" in every table it appears
in and it is never the number quoted anywhere.

### `cell_independent`

Hold out whole cells. No cycle of a test cell appears in training.

Honest about cells and still optimistic about lots. A held-out cell has siblings
in training built on the same electrode coating run, filled from the same
electrolyte batch, shipped to the same site and run on the same charge policy.
The model can learn "cells from lot B07 last about 1,600 cycles" and score well
on a new cell from B07 without having learned anything about ageing.

This is the right number if the question is "a cell in my existing fleet has no
history, what do I expect from it".

### `batch_independent` (the number this repository leads with)

Hold out whole manufacturing batches.

This is the arrival of a new lot at a new site, which is the situation a fleet
operator is actually in and the only protocol under which the model has to have
learned something about degradation rather than about bookkeeping.

It is also the protocol with the largest fold-to-fold variance, because with
five lots held out per fold, which five you get matters. That variance is
reported as a standard deviation next to every mean and it is part of the
result, not noise to be smoothed away.

---

## 2. Why batches are not an abstraction

The generator in `cobalt.sim.fleet` builds correlation in at two levels and it
does so deliberately.

**Lot characteristics.** Each batch draws multipliers on the SEI rate, the
plating rate, the knee onset and the delivered capacity, from lognormals with
between-batch sigmas of 0.20, 0.26, 0.11 and 0.013. Cells then jitter around
their batch with sigmas of 0.055, 0.070, 0.030 and 0.006, roughly a quarter of
the between-batch spread.

**Deployment site.** Each batch also draws a mean ambient temperature and a
charge C-rate and cells jitter around those. A lot shipped to a hot depot on
4C fast charge ages nothing like the same lot shipped somewhere temperate on
0.5C.

The consequence is that cells within a lot are several times more alike than
cells across lots, which is the case in reality and is the reason a
batch-independent split is a harder question. `tests/test_fleet_and_schema.py`
asserts the variance ratio holds, so if a future change to the generator
flattens it the whole argument fails loudly rather than quietly.

---

## 3. Leakage rules

Every one of these is enforced in code, not by convention.

1. **Folds partition the groups.** Every row is in exactly one test fold, so the
   pooled out-of-fold predictions are a complete leave-one-group-out prediction
   for the whole dataset. `evaluate_model` raises if any row was never tested.
2. **Every split is checked against its own claim.** `assert_no_leakage` runs
   before any model is fitted on any fold and re-derives the grouping from the
   frame rather than trusting the split's label.
3. **A split with overlapping train and test indices cannot be constructed.**
   `Split.__post_init__` rejects it.
4. **Features are selected by an explicit allow-list**, not by dropping label
   columns. Adding a diagnostic column to a feature table cannot silently feed
   it to a model.
5. **Ground-truth columns never reach a feature.** The simulator emits noise-free
   `true_capacity_ah` and `true_resistance_ohm` for the tests and the evaluation
   to use. They are not in `OBSERVABLE_CYCLE_COLUMNS` and not in either
   `FEATURE_NAMES` tuple.
6. **Model refits happen inside the fold.** A fresh model is built per fold from
   the registry. The deployed model is fitted separately, on everything, in
   `cobalt.deploy` and its quoted numbers come from cross-validation rather than
   from that fit.
7. **Conformal calibration happens inside the training fold**, on data the base
   model was not fitted on and grouped by the same column the protocol splits
   on.

---

## 4. Calibrating the intervals

Split conformal prediction assumes exchangeability between the calibration set
and the test set. That assumption is exactly what a batch-independent protocol
violates if you calibrate the ordinary way.

Calibrate on rows drawn at random from the training lots and the residuals
measure how well the model does on lots it already knows. Test on a new lot and
the residuals are larger, so the interval is too narrow and the coverage it
advertises is not the coverage it delivers. On the online task the effect is a
90% interval covering 58%.

So `ConformalRegressor` takes a `groups` argument and holds out whole groups for
calibration. `cobalt.evaluate.protocol` runs the deployed model both ways at
every protocol and reports both coverages, which is what makes the cost of the
mistake a number instead of a warning.

The finite-sample correction matters here too. The conformal quantile is taken
at `ceil((n+1)(1-alpha))/n` rather than at `1-alpha`. Without the `(n+1)` the
interval undercovers on small calibration sets, which is precisely the regime a
held-out-lot calibration lands in.

### Where grouped calibration stops paying

Grouped calibration is not free: it removes whole groups from the fit, not just
rows. When there are few groups to begin with, the model loses more from the
missing between-group variety than the calibration gains from being out of
group and a grouped-calibrated model can cover *worse* than a randomly
calibrated one.

That is not a counter-example to the argument above, it is a different effect
with the same sign. `MIN_FIT_GROUPS = 4` in `cobalt.models.conformal` marks the
line and a configuration that crosses it sets `group_warning` rather than
reporting the result silently. Both shipped configs clear it: `default.yaml`
leaves eleven lots in the fit, `ci.yaml` leaves four.

This was found by a test on a four-batch fixture, not by reasoning. It is
recorded here because the effect is easy to run into and easy to misread as
evidence against grouping.

---

## 5. What is reported and why four families of metric

Cycle life spans roughly an order of magnitude across a fleet, so everything is
fitted in `log10` cycles. A squared error in linear cycles would let the
long-lived cells set the whole objective. Everything is reported in both spaces,
because a maintenance planner does not think in decades.

- **Error size**: RMSE and MAE, in log space and in cycles.
- **Relative error**: MAPE and the alpha-lambda accuracy the prognostics
  literature uses: the share of predictions inside a ±20% band around the truth.
- **Ranking**: Spearman correlation. Triage is a sorting problem and a model can
  rank a fleet correctly while being biased on every single cell.
- **Interval quality**: coverage against the width it cost. Coverage without
  width is free.

They disagree often enough to be worth keeping all four. The clearest case in
this repository is `slope_extrapolation` on the online task, which has a *worse*
RMSE than predicting the fleet mean (802 against 666) and a much better
alpha-lambda (0.317 against 0.155). It is right about most packs and
catastrophically wrong about the ones early in life, where the fade slope is not
yet distinguishable from zero. A single headline metric would hide that
completely, in either direction depending on which one was chosen.

---

## 6. What the model card is allowed to say

`cobalt.deploy.build_model_card` quotes the hardest protocol available for the
task, never the best one. Its first field states that the training data was
simulated. `tests/test_service.py` asserts both, because a model card that
quotes the flattering number is worse than no model card: it launders the
optimism into somewhere it will be believed.

## 7. The same protocol on real cells

`cobalt real-data evaluate` runs `cobalt.evaluate.protocol.run_task` unchanged
on the public Severson et al. (2019) cells: the same splits, the same leakage
assertions, the same models and the same conformal calibration. Two settings
differ because the data forces them. The rated capacity is the cells' 1.1 Ah
and there are three test campaigns, so both grouped protocols use three folds
and batch-independent becomes leave one campaign out. A Severson batch is a
test campaign with its own charging policies rather than a known manufacturing
lot, which [`datasets/severson-2019.md`](datasets/severson-2019.md) explains.
The results are in [`../analysis/severson/EVALUATION.md`](../analysis/severson/EVALUATION.md).
