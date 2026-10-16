# Dataset card: Severson et al. (2019) cycle-life data

The only real data this repository uses. Everything else is simulated.

## Source

| | |
| --- | --- |
| Publisher | Toyota Research Institute, with MIT and Stanford |
| Landing page | <https://data.matr.io/1/> |
| Paper | K. A. Severson, P. M. Attia et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383 to 391 (2019), [doi:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8) |
| Authors' loading code | <https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation> (`LoadData.m`, `Load Data.ipynb`) |
| Fetched | 16 October 2026 |

Three MATLAB v7.3 files (HDF5 containers), one per test campaign:

| File | Bytes | Download |
| --- | --- | --- |
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3,025,320,241 | `https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download` |
| `2017-06-30_batchdata_updated_struct_errorcorrect.mat` | 2,007,331,155 | `https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download` |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3,236,690,412 | `https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download` |

The sizes are checked on extraction. The publisher's ETags are multipart
upload tags rather than content hashes, so there is no published checksum to
verify against.

## Licence

**Not confirmed from the publisher.** The data platform at data.matr.io is a
single-page app that renders its licence text from site configuration, and
that text could not be retrieved when the data was fetched. Neither the paper
nor the authors' repository states a data licence. A published review,
*Lithium-ion battery data and where to find it* (Energy and AI, 2021), reports
it as Open Data Commons ODbL v1.0 with DbCL v1.0; a Kaggle mirror labels it
MIT. Neither is the publisher's own statement.

Because the terms are unconfirmed, this repository is conservative:

- **No raw or derived data is committed.** The 8.3 GB of `.mat` files and the
  few-megabyte extract both live under `data/`, which is git-ignored and
  docker-ignored.
- **Only aggregate results are committed:** fold-level error metrics, interval
  coverage, cell counts and parity plots, in `results/severson/` and
  `analysis/severson/`.
- **CI keeps the extract in the GitHub Actions cache**, which is private to the
  repository's workflows and expires and otherwise downloads from the
  publisher.
- **The paper is cited** wherever the data is used.

If the publisher's terms turn out to forbid any of that, the cache and the
committed metrics are what would have to go.

## What the cells are

From the paper and the files: commercial A123 APR18650M1A cylindrical cells,
lithium iron phosphate cathode and graphite anode, 1.1 Ah nominal. Cycled in a
temperature chamber at 30 degC, fast-charged under one- or two-step policies
(70 distinct policy strings in the files, first-step rates from 1C to 8C) and
discharged at 4C. End of life is 80% of nominal, 0.88 Ah.

A "batch" here is a **test campaign**: cells started together, months apart
from the next campaign, under a different set of charging policies. Nothing in
the data says the campaigns used different manufacturing lots, so the
batch-independent protocol on this data tests a new campaign and a new set of
policies, not a new lot.

## What is read

`cobalt.realdata.severson` opens the files with h5py, follows the MATLAB
object references and reads only:

| Field in the files | Becomes | Notes |
| --- | --- | --- |
| `summary.QDischarge` | `discharge_capacity_ah` | per cycle |
| `summary.QCharge` | `charge_capacity_ah` | |
| `summary.IR` | `internal_resistance_ohm` | |
| `summary.Tavg`, `summary.Tmax` | `mean_temperature_c`, `max_temperature_c` | |
| `summary.chargetime` | `charge_time_min` | |
| `cycles.Qdlin` on `Vdlin` | `discharge_energy_wh` and the delta-Q curves | energy is integrated between 3.5 V and 2.0 V, the window the publisher interpolated over, so it is not the full discharge energy |
| `policy_readable` | `charge_c_rate` | the first fast-charge step |
| (constant) | `discharge_c_rate` = 4.0 | every cell was discharged at 4C |
| (derived) | `cumulative_ah_throughput` | running sum of discharge capacity over the records kept |

The discharge curves (`Qdlin`) are kept only at cycles 2, 10 and 100, which is
all the early-life features use.

## Preprocessing, every decision

Each one follows the authors' own loading code unless marked otherwise.

1. **Continuations joined.** Five batch 1 cells continued into the second
   campaign on new channels (`b2c7`, `b2c8`, `b2c9`, `b2c15`, `b2c16`). Their
   records are appended to `b1c0` to `b1c4`, which keep their batch 1 identity.
2. **Noisy channels removed.** `b3c37`, `b3c2`, `b3c23`, `b3c32`, `b3c42` and
   `b3c43`, as the authors' notebook does.
3. **Cycle life** is the first cycle whose discharge capacity is below 0.88 Ah.
   A test that stopped with its last capacity at or below 0.885 Ah is labelled
   one cycle past its last record; one that stopped above that never reached
   end of life and is marked censored. This is `LoadData.m`'s rule and it
   reproduces the publisher's `cycle_life` field for all 124 finished cells,
   continuations included. The five censored cells (`b1c8`, `b1c10`, `b1c12`,
   `b1c13`, `b1c22`) are the five the authors remove for not finishing.
4. **Empty records dropped.** The first record of every test run is all zeros.
   46 such records are dropped.
5. **Glitch records dropped (this repository's decision, not the authors').**
   101 of 104,925 records are physically impossible. The rules fire 112 times
   between them, since a record can trip more than one: a charge or discharge
   capacity above 1.3 Ah from a 1.1 Ah cell (29), a resistance of zero (22), a
   charge time more than three times or less than a third of the cell's own
   median (28; the glitches are hundreds of minutes against about ten) and a
   discharge energy more than 50% from the cell's median (33, including
   negative energies). The bounds are wide on purpose. A first
   attempt with a 50% bound on charge time removed 2,450 records, because real
   charge time roughly doubles approaching end of life; that would have
   deleted genuine ageing signal, so the bound was widened. The counts per rule
   are in the extract's `manifest.json` and the rules are
   `cobalt.realdata.severson.implausible_records`. Labels are computed before
   this step, from the raw capacities, so it cannot move a cycle life.
6. **One cell has a logger gap.** `b1c18` has only 73 records in its first
   hundred cycles, so the early-life task excludes it and counts it as
   `gaps_in_window`. The online task still uses it.

Result: 129 cells, 124 with a cycle life (41, 43 and 40 per campaign), 104,824
cell-cycles, cycle lives from 148 to 2,237 with a median of 736. These are
written by `extract` to `manifest.json` and to the committed
`results/severson/summary.json`.

## Known limitations

- One cell type, one chamber temperature, one discharge rate. It says nothing
  about NMC, about cold or hot operation or about variable duty.
- Laboratory cycling. No pack in service gets a constant 4C discharge.
- Three campaigns is the bare minimum for a batch-independent protocol. Grouped
  conformal calibration fits on one campaign and calibrates on the other,
  which the library flags as starved.
- The discharge energy is a windowed proxy, as noted above.
- The gradient-boosting figures in the committed summary vary slightly by
  processor (histogram sums in vector code). The CI comparison allows 3% on
  their means and ignores their fold spreads; every other figure must match to
  0.2%.
