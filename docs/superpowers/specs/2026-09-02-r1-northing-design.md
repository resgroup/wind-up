# Design — R1: northing errors (shared fix)

**Date:** 2026-09-02
**Status:** approved design
**Issue:** `docs/v1/issues_campaigns.md` § R1
**Extends:** `2026-08-28-robustness-failure-modes-design.md` (R-series ground rules),
`2026-09-01-c2-campaign-context-seam-design.md` (the seam decisions R1 fills in)
**Branch of work:** `v1-R1`, developed off `v1`

## Problem

A turbine's direction reference carries **step changes** in its north calibration —
a recalibration, a sensor swap, a controller replacement. wind-up must recover a known
uplift regardless.

Two things stand in the way today.

**The estimator is too slow to use.** `src/wind_up_v0/optimize_northing.py` (762 lines)
hill-climbs over a hand-rolled move set — shift a changepoint forward, shift it back,
add *n* changepoints via `ruptures.BottomUp` with a custom circular L1 cost — rescoring
the whole turbine on every move, with a step size that decays and re-inflates through
a `1/(DECAY_FRACTION ** (pi * (tries_left + 1))) % 10` schedule. It is slow enough that
it is switched off in practice: both `examples/` set `optimize_northing_corrections=False`
and use pre-computed tables, and the benchmarking layer reads a vendored
`optimized_northing_corrections.yaml` (the result of a prior run of `optimize_northing.py`).

**No v1 method can see a northing error.** `benchmarking/baselines/power_model/features.py`
builds features from reference active power, availability and ERA5 — no turbine direction
signal of any kind. A step injected into `YawAngleMean` is invisible to it, so the
R-series' phase 1 ("the fault bites") is unreachable until the feature exists.

## Scope

All four parts land together, in this order:

1. a fast, `ruptures`-free northing estimator in `src/wind_up/northing.py`;
2. a shared northing step in the campaign runner, upstream of every method, plus the
   `power_model` reference-direction feature that makes a northing error visible;
3. a fault injector and the tiny fixture that proves *bites* then *fixed*;
4. `src/wind_up_v0/optimize_northing.py` reduced to a thin adapter over the new core,
   with its tests ported and `ruptures` dropped from `pyproject.toml`.

`naive_ratio` and `toggle_specialist` use no direction signal and are out of scope.

## Key decisions

1. **Exact dynamic programming, not a numerical optimiser.** Locating step changes is
   combinatorial; a continuous optimiser has nothing to descend, which is the fragility
   the current hill-climb exhibits. Given the segments, each offset is closed-form (a
   circular median). So the estimator contains no optimiser at all.

2. **Aggregate before searching.** The residual is piecewise-constant plus noise, so a
   daily circular median loses nothing a changepoint search needs and turns 105k rows
   (2 years at 10 minutes) into ~730 points. This is what makes the search cheap; local
   refinement recovers sub-day changepoint timing afterwards.

3. **The core is frame-agnostic and device-neutral.** It takes an index, a direction
   array, a reference-direction array and a caller-supplied `usable` mask. Turbine-specific
   logic (generating above 5% of rated, not in downtime) lives in a helper, not the core,
   so masts and LiDARs are a mask away rather than a rewrite.

   **`usable` is also how wake steering is handled.** A steering turbine is deliberately
   yawed off the wind, so a steered period looks exactly like a northing offset that appears
   and disappears on the steering schedule. Excluding those rows via the mask is the whole
   fix, and it needs no change to the core — which is a second reason to keep mask
   construction outside it. C5 supplies the steered-period mask; R1 only has to not
   foreclose it.

4. **Knobs are in physical units.** `min_step_deg` (the smallest step worth reporting)
   and changepoints-per-year, not sample counts or pruning constants. See the prior-art
   review below for why this is worth insisting on.

5. **The returned table is always absolute** — offsets relative to the **raw** field,
   never "further corrections to an already-corrected field". This is what makes a supplied
   table and an estimated one directly comparable, and repeated runs composable. See
   *Designed for, not implemented*.

6. **Two passes are preserved.** Pass 1 norths each turbine to reanalysis wind direction;
   the northed yaws give a farm direction; pass 2 norths to that. Pass 2 is far less noisy,
   but pass 1 is what anchors the farm in absolute terms — without it a farm that is
   uniformly 180° wrong looks perfectly self-consistent.

7. **The fault is a measurement corruption, not an upgrade.** It changes direction only,
   never power, so ground truth is untouched by construction.

8. **Success is invariance**, per the R-series ground rules: the target is
   `power_model`-under-fault ≈ `power_model`-clean, not a race against v0.

9. **How small a step may be attributed depends on the reference.** Reanalysis is a modelled
   direction carrying its own drift, and a shift in it is indistinguishable from every turbine
   shifting at once — so against reanalysis only steps above `REANALYSIS_MIN_STEP_DEG` (10°)
   are attributable, whatever tier the caller chose (`against_reanalysis(effort)`). A farm
   consensus shares that common-mode error, so against one a residual step really is that
   turbine's and the caller's `min_step_deg` applies. Where the farm reference has fallen back
   to reanalysis (fewer than three turbines) the second pass stays conservative too. This is
   what v0 was already expressing as `best_score_margin=0.5` on its reanalysis pass. Not
   foreseen in design — forced by the measurement in *Evidence* below.

## Evidence: reference-side drift is real and had to be designed for

Measured on the Homer July-2023 fixture while porting the v0 tests. Against reanalysis, both
turbines appeared to step on **2023-07-12 within 40 minutes of each other** -- T01 by 3.8
degrees, T02 by 5.0. Two independent sensor recalibrations on the same afternoon is not
credible, so the shared signal was isolated by comparing the turbines with **each other**
instead of with reanalysis:

| comparison | before 07-12 | after 07-12 | step |
|---|---|---|---|
| T01 - reanalysis | 55.80 | 52.00 | **-3.80** |
| T02 - reanalysis | -118.40 | -123.30 | **-4.90** |
| T01 - T02 (no reanalysis) | 174.00 | 175.00 | **+1.00** |

About 4 degrees of the apparent step is common-mode: it is in the reanalysis reference, not in
the turbines, whose relative alignment barely moved. An estimator that attributes it to the
turbines produces two spurious changepoints and shifts the northing by ~1.7 degrees.

`REANALYSIS_MIN_STEP_DEG = 10` is sized from this: about twice the measured common-mode drift,
and far below a real sensor swap (the ported test injects 30-degree steps, and a north
recalibration is a large move by nature). It is a floor on the threshold, not a new tier, so a
caller asking for `thorough` still gets 1-degree resolution against the farm consensus.

The measurement agrees with the physics: **reanalysis is not accurate to better than ~10° as a
representation of the wind direction at a specific turbine's hub height.** It is a coarse-grid
modelled field, not a measurement at the rotor. So attributing a sub-10° step to a turbine on
reanalysis evidence alone is asking the reference for precision it does not have, whatever any
one fixture shows.

This is the same failure HOGER's >50% pairwise-consensus vote addresses, reached from the other
direction -- which is why pairwise consensus stays the recorded next step rather than a
speculative one.

## Evidence: site veer, and why a threshold alone cannot fix it

Comparing the first implementation's north tables against the vendored Hill of Towie table (the
old optimizer's output, and the only "old" answer we need — it need not be re-run) exposed a
second, larger problem than the reanalysis drift above.

**84 changepoints in 2017–18 against the old table's 7.** T06 alone got 12, sitting exactly on
its `6/year × 2` budget ceiling. Its offsets oscillated between ≈ −6° and ≈ −13° and **netted
0.38° across all 12 steps** — ending where they began. A recalibration is permanent; this was an
excursion being approximated by a square wave.

The cause is **site veer**: the wind direction genuinely differs from turbine to turbine across a
site, varying with bulk direction, atmospheric stability and wind speed. So a turbine's residual
against the farm median has a level that depends on *which directions the wind blew from*, and a
shift in the direction mix moves that level with nothing at the turbine having changed. The
spurious-jump count tracks the veer amplitude exactly, with the cut falling on
`balanced`'s 3° threshold:

| turbine | monthly-residual range | spurious jumps |
|---|---|---|
| T06 | 4.7° | 12 |
| T09 | 3.5° | 10 |
| T15 | 3.4° | 8 |
| T07 | 2.0° | **0** |
| T11 | 1.3° | **0** |

Raising `min_step_deg` was measured and rejected as the fix. It works, but only by trading away
real detections — at 10° the worst northing error jumps to 7.9° (a genuine step between 7° and
10° goes unfound), and veer amplitude is turbine-specific, so one global threshold is either too
loose for T06 or too tight for T11. **The tiers were left at the user's original 5 / 3 / 1°** and
the cause attacked instead, by two mechanisms.

### 1. Veer normalisation (`veer_normalised`)

Subtract each direction sector's own whole-record median from the residual before searching. A
genuine north offset shifts every sector alike and survives; a change in the direction mix cannot
move the level at all. 30° sectors by default (20° measured no better).

Two details make it correct rather than circular:

- **Detection only.** Segment offsets are estimated from the *raw* residual, so the correction
  stays absolute and the 1°-accuracy goal is untouched.
- **De-step first.** Measuring sector levels on a record that contains large steps lets the steps
  leak into the veer signature, and uneven direction sampling between segments then distorts the
  very steps being looked for — this broke the ported Homer changepoint test outright. So a first
  pass detects on the raw residual purely to remove the step structure, the veer signature is
  measured on that de-stepped residual, and the real search runs on the normalised one.

### 2. Ironing out self-cancelling excursions (`_prune_transient_steps`)

After detection, drop changepoints that do not *persistently* move the level: for each one,
compare the duration-weighted level of everything before it with everything after. Veer wanders
away and back, so either side sits at the same place; a recalibration leaves the level moved.

**With an amplitude gate, which is essential.** A first cut without one pruned by persistence
alone and sent the worst northing error to 17°, because real recalibrations *do* sometimes
reverse: the vendored table has T16 stepping 98°, 9°, 7°, 89° for a net of only +11.4°. So a step
above `max_transient_step_deg` (10°) is never ironed out — its size is the evidence it happened.
Below that, the same table shows T11 (four steps of 2–9°, net +0.3°) and T10 (8.2°/9.6°, net
+1.5°), which look exactly like the veer the filter is meant to remove.

It is a threshold rule, not an oracle: an oscillation biased enough that its halves sit at
genuinely different levels keeps the changepoints carrying that difference (a unit test pins
this, so the limit is documented rather than discovered later).

### Measured effect

Farm-scale, Hill of Towie, 21 turbines, 2017–18, at the unchanged `balanced` 3° threshold:

| configuration | jumps | turbines | worst err |
|---|---|---|---|
| first implementation | 84 | 19 | 3.665 |
| + veer normalisation | 51 | 11 | 3.471 |
| **+ excursion pruning (shipped)** | **8** | **4** | 5.125 |
| *old v0 (vendored table)* | *7* | *3* | *5.121* |

**Mean error is deliberately not in that table.** It is *best* on the 84-jump row, so it rewards
over-detection and cannot discriminate — more free parameters always fit better. Jump count
against the real rate (~0.3/turbine/year, from the vendored table's 49 entries over 8 years and
21 turbines) and worst-case error are the honest measures.

The decisive result is not the totals but **which** turbines. The shipped estimator finds
`{T01: 2, T05: 2, T13: 1, T16: 3}`; the vendored v0 table's 2017–18 changepoints are
`{T01: 2, T05: 2, T16: 3}`. It **independently rediscovers v0's changepoints exactly** — same
turbines, same counts — and adds one on T13, whose worst-case error improves 3.85° → 3.67°.

Honest attribution: **the excursion pruning does most of the work**; direction binning contributes
5–10%, and is kept because it is physically right and cheap, not because it carries the result.

On the fixture the effect is starker still: the clean arms now discover **0** changepoints and the
faulted arms exactly **1** — the injected fault and nothing else, where the first implementation
found 7–11 spurious ones per run.

## Prior art

**HOGER** (Homogenization Of GEneral Regressions), Engie + CENER, merged into FLASC as
`flasc/data_processing/northing_offset_change_hoger.py`
([PR #240](https://github.com/NatLabRockies/flasc/pull/240)), is the closest published work.

| | HOGER | R1 |
|---|---|---|
| reference | pairwise turbine-vs-every-other differences (`wrap_180`) | two-pass: reanalysis → farm-median yaw |
| detection | `DecisionTreeRegressor` on time → difference; splits are knots | exact DP on daily circular medians |
| optimality | greedy (CART), `max_depth=4` caps knots at 15 | globally optimal for a given K |
| consensus | keeps a jump only if it appears in >50% of that turbine's pairwise comparisons | robust circular median across the farm |
| tuning | `min_samples_split=1000`, `min_samples_leaf=500`, `ccp_alpha=0.09` | `min_step_deg`, changepoints/year, `min_segment` |
| absolute anchor | none — homogenises only | reanalysis pass |

Three conclusions:

- **HOGER is purely differential.** It makes turbines agree with each other but cannot
  detect that they all agree on the wrong north. A farm uniformly 180° out is invisible
  to it. That is the argument for keeping the reanalysis pass, and it is why the
  180°-wrong-farm unit test below is a first-class acceptance test rather than an edge case.
- **Physical knobs beat pruning constants.** `ccp_alpha=0.09` is not a quantity an analyst
  can reason about. `min_step_deg=3` is.
- **The pairwise-consensus trick beats a farm median when N is small.** One jumping
  turbine contaminates a 4-turbine median, and the R1 fixture is exactly 4 turbines. Recorded
  as a named future option; the reanalysis pass is the anchor in the meantime.

Background, not code: [Bromm et al., WES 2018](https://wes.copernicus.org/articles/3/395/2018/)
on detecting alignment changes from SCADA; [SkySpecs on north offset](https://skyspecs.com/blog/addressing-north-offset-in-wind-turbines-scada-data/).
OpenOA has no northing module. Nothing found combines globally-optimal circular
segmentation with an absolute anchor.

## The estimator core — `src/wind_up/northing.py`

### Public surface

```python
@dataclass(frozen=True)
class NorthingEffort:
    changepoints_per_year: float
    min_step_deg: float
    refine: bool
    grid: pd.Timedelta = pd.Timedelta(days=1)
    min_segment: pd.Timedelta = pd.Timedelta(days=7)

FAST / BALANCED / THOROUGH: NorthingEffort

def estimate_north_table(
    index: pd.DatetimeIndex,
    direction_deg: npt.NDArray[np.float64],
    *,
    reference_deg: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    effort: NorthingEffort | Literal["fast", "balanced", "thorough"] = "balanced",
) -> pd.DataFrame:            # columns: timestamp, north_offset

def apply_north_table(
    index: pd.DatetimeIndex,
    direction_deg: npt.NDArray[np.float64],
    *,
    north_table: pd.DataFrame,
) -> npt.NDArray[np.float64]  # (direction + offset) % 360

def yaw_usable(
    *, power: NDArray, downtime_s: NDArray, reference_deg: NDArray,
    rated_power: float, timebase_s: int,
) -> npt.NDArray[np.bool_]
```

`estimate_north_table` works on any direction field. `yaw_usable` is the turbine mask
(the existing `add_ok_yaw_col` rule: reference present, power above 5% of rated, downtime
below a quarter of the timebase). Masts and LiDARs need a wind-speed-based mask instead;
documented as not yet wired up.

`apply_north_table` is array-in/array-out so **one table can be applied to several fields
of the same device** — derive the correction from yaw position, apply it to yaw position
*and* a measured wind-direction channel.

### Algorithm

1. **Residual.** `d = circ_diff(direction_deg, reference_deg)` where `usable`, NaN elsewhere.
2. **Aggregate** `d` to `effort.grid` bins by per-bin circular median, carrying each bin's
   count as a weight. ~730 points for two years.
3. **Prefix sums** of `w·sin(d)`, `w·cos(d)`, `w`. Any segment's weighted resultant-length
   cost is then O(1):
   `C(i,j) = W(i,j) − hypot(Σ w·sin, Σ w·cos)` — the loss minimised by the circular mean.
4. **Exact DP.** `best[k][j] = min_i best[k−1][i] + C(i,j)`, subject to `min_segment`,
   with `K ≤ ceil(changepoints_per_year × years_of_data)`. Choose K by penalised total,
   the per-changepoint penalty derived from `min_step_deg` (a step smaller than that is
   not worth a changepoint). Vectorised per k; milliseconds at m ≈ 730.
5. **Refine** (when `effort.refine`): re-scan each changepoint at native resolution within
   ±1 grid bin, same cost function.
6. **Offsets.** Per segment, `−circ_median(d)` over its native usable rows — robust, and
   only K+1 of them.
7. **Iron out excursions.** Drop changepoints that do not persistently move the level and whose
   own step is under `max_transient_step_deg` — site veer wandering away and back.
8. **Prune small steps.** Drop any remaining changepoint whose step is under `min_step_deg`, which
   is what makes that knob mean what it says.

Steps 1–5 run **twice**: once on the raw residual to locate the step structure, then again on the
veer-normalised residual (the veer signature measured on the de-stepped record). Offsets always
come from the raw residual, so the correction is absolute.

Grid stays at one day for every tier: local refinement already recovers sub-day timing, so
a finer grid would quadruple the DP for nothing. `min_segment` prevents pathological
micro-splits.

### Effort tiers

There is **one setting**, not a menu — `NorthingSettings`, exposed as `DEFAULT_NORTHING`:

| field | value | why |
|---|---|---|
| `changepoints_per_year` | 12 | a rate, so a longer record gets a larger budget |
| `min_changepoints` | 3 | a floor, so a short record can still hold several corrections |
| `min_step_deg` | 3° | the smallest step reported |
| `refine` | `True` | measured free (3.6s vs 3.7s on a farm-year) |
| `veer_sector_deg` | 30° | see *Evidence: site veer* |
| `max_transient_step_deg` | 10° | above this a step is never ironed out as wander |
| `grid` / `min_segment` | 1 day / 7 days | search resolution and shortest gap |

**The effort tiers were built, measured and removed.** The knob was introduced to trade speed for
quality, and the trade turned out not to exist: across a 21-turbine, 2-year farm the whole
spread from the cheapest to the most thorough setting was **3.1 s to 5.2 s**, out of a ~40 s run
dominated by data handling rather than the search. Worse, the cheap tier was *lower quality for
no real saving* — its `1/year` budget missed a genuine changepoint and left a 7.9° worst-case
error against 5.1° for the default. A dial whose cheap end is worse and no faster is not a dial.

`NorthingSettings` survives as an expert override, not a menu; the tier names and the string API
are gone. Nothing pretends to be a speed control.

`max_changepoints = ceil(changepoints_per_year × years_of_data)`, so the budget scales with
record length rather than being a fixed count. `fast` is the R1 workhorse: one large
injected step is exactly the K≤1-per-year, no-refinement case.

### Two-pass driver

```python
def north_farm(
    index: pd.DatetimeIndex,
    *,
    direction_deg: Mapping[str, npt.NDArray[np.float64]],   # device -> direction, on ``index``
    usable: Mapping[str, npt.NDArray[np.bool_]],            # device -> mask, on ``index``
    reanalysis_deg: npt.NDArray[np.float64],                # on ``index``
    effort: NorthingEffort | str = "balanced",
    min_devices_for_farm_reference: int = 3,
) -> dict[str, pd.DataFrame]                                # device -> absolute north table
```

Every device shares one `index`, which is what lets the farm reference be computed by
position. Pass 1 norths each device to `reanalysis_deg`. The northed directions give a farm
direction (circular median across devices, at least `min_devices_for_farm_reference` present
at a timestamp, else NaN). Pass 2 norths each device to that. Returns one absolute north
table per device.

## Seam 1 — the shared step

Runs in `CampaignRunner`, which holds the `CampaignSpec` (per the C2 decision that the
runner, not a method, owns this). It writes `columns.northed(role)` — `northed_YawAngleMean`
for `nacelle_position` — **alongside the untouched original**, so existing plots and
diagnostics keep meaning what they say. There is **no `northing_applied` flag**: the
column's presence is the state.

The step takes a list of direction roles to correct (default `["nacelle_position"]`),
derives one table per turbine from yaw position, and writes `northed_<col>` for each role.

### `north_offsets`: supplied or not

`CampaignSpec.north_offsets` becomes `list[...] | None`, defaulting to `None`. Two states:

| value | meaning |
|---|---|
| `None` (default) | **auto-calculate.** The analyst supplied nothing; wind-up norths from the data. The usual case. |
| a list (possibly empty) | **apply exactly this, discover nothing.** `[]` is simply the case with no corrections to apply, so it needs no separate rule. |

wind-up does **no checking** of a supplied table in R1 — it applies it and moves on. Checking
a prior and reporting confirmation-or-amendments is real future usage (see *Designed for, not
implemented*), but building it now would mean designing a disagreement threshold and a report
with no caller to validate them against. This mirrors what v0 already does with
`optimize_northing_corrections` versus `northing_corrections_utc`.

Either way the step writes `northed_<col>`, so downstream consumers find the column
regardless of which branch ran.

**Consequence for the benchmark:** a campaign that supplies a table never exercises
discovery. The placebo currently loads the real Hill of Towie YAML, so it would
apply-as-supplied and silently stop testing the thing R1 builds. Campaigns meant to exercise
discovery — the R1 fixture above, and C3/C5 — pass `None` explicitly, a per-campaign choice
made visibly rather than a property of the type.

C3/C5 drop their bespoke northing wiring in favour of this step.

## Seam 2 — `power_model` sees direction

`build_reference_features` gains each reference's northed direction, as `sin`/`cos`
companions (LightGBM cannot see that 359° ≈ 1°). Guarded: the method raises, naming the
missing column, if the shared step has not run. `check_reference_only` already blocks the
test turbine's own direction, which is the design-note §3 rule — northing does not make a
post-treatment signal safe.

**Northed replaces raw, never both.** Direction features come only from northed columns; a
raw direction offered in `extra_cols` when a northed counterpart exists is dropped, with a
log line.

**It is opt-in, `direction_feature=False` by default — an open decision, not the end state.**
The shared northing step runs in `CampaignRunner`, so the *campaign* path has a northed column
but the *study* path (`build_replicates` → `score_one`, which drives both frozen benchmarks)
does not. Turning the feature on by default would make `power_model` raise on every study
driver. So the flag ships off, campaign method factories turn it on, and two things remain to
decide:

1. whether the study path should also north (which means the step moving somewhere both paths
   share, rather than living only in the runner), and
2. flipping the default and regenerating `study_power_model_compare_baseline.json`.

Until (1) lands, R1's bites/fixed evidence comes from the campaign path only. That is enough
for the fixture, but it means the frozen benchmark does **not** yet move — contrary to what
this design assumed.

## Seam 3 — v0 adapter

`auto_northing_corrections(wf_df, *, cfg, plot_cfg)` keeps its signature and its two-pass
shape. It loops turbines, builds the four core arguments from `RAW_YAWDIR_COL`,
`REANALYSIS_WD_COL` and `WINDFARM_YAWDIR_COL`, and calls the core. v0's own
supplied-versus-discovered switch is unchanged: `cfg.northing_corrections_utc` is applied by
`apply_northing_corrections` as it is today, and `auto_northing_corrections` is what runs
when the analyst asked for discovery.

Deleted: `CostCircularL1`, `_northing_score`, the move generator, the hill-climb,
`_calc_max_changepoints_to_add`, and the `ruptures` import. `ruptures` then leaves
`pyproject.toml` (dependency and `mypy` override).

`northing.py`'s `apply_northing_corrections`, `add_wf_yawdir` and `check_wtg_northing` are
unchanged, as are the northing plots.

This introduces a `wind_up_v0` → `wind_up` import, so the releasable v1 package does not
depend on the legacy one, and W2 promotes the module with no second move.

**`circular_math` moves too.** The core needs `circ_diff`, `circ_median` and
`rolling_circ_median_approx`, which today live in `src/wind_up_v0/circular_math.py` — v1
importing them from v0 would be exactly the dependency direction this decision avoids. So
the module moves to `src/wind_up/circular_math.py` and `src/wind_up_v0/circular_math.py`
becomes a re-export, leaving the seven v0 importers and four test modules untouched.

**Blast radius is small and was verified:** `auto_northing_corrections` is reached only when
`optimize_northing_corrections=True`; both `examples/` set it `False` and use pre-computed
tables, and `hot_context` reads the vendored YAML. No frozen example or benchmark number
moves from this swap.

**v0 stays verified end-to-end** by re-running the SMARTEOLE and WeDoWind examples. Where a
northing table is supplied the results must be **identical** (that path does not touch the
estimator at all). Where auto-northing runs they need only be **similar** — a different
optimiser finding a slightly different table is the expected outcome, not a regression. Note
that both examples ship with `optimize_northing_corrections=False`, so the auto-northing arm
has to be run with the flag deliberately flipped; it is not exercised by default.

## The fault and the fixture

### Fault

`NorthingStep(turbine, at, offset_deg)` adds `offset_deg` to a turbine's reported
`nacelle_position` from `at`. It changes no power, so `true_uplift` is untouched by
construction.

This earns a `faults: list = []` field on `SyntheticCampaign` — private ground truth, like
`upgrades` — applied after upgrade injection and to `synthetic_df` only. `CampaignSpec`
never sees it, and the fixture leaves `north_offsets=None` so the step must discover the
step change rather than be told about it: an analyst does not know it happened. The protocol
stays minimal so R2–R4 inherit it: `__call__(synthetic_df, *, columns) -> pd.DataFrame` plus
a `description` for run metadata.

### Calibrating the fault

Damage is not monotonic in offset size. Two levers matter more than magnitude:

- **Timing.** Worst when the step coincides with the changeover in prepost, or falls in the
  exact middle of a toggle campaign — that is when the corruption aligns with the contrast
  the method is measuring.
- **Where the offset lands.** What matters is how much the power-ratio-versus-direction
  shape changes, so a **30° offset can be more damaging than 180°** if it moves a crucial
  wake onto a well-populated direction sector.

So calibration sweeps timing and offset rather than winding magnitude up until something
breaks, and the chosen fault is justified by which sector it moves the wake into.

### Fault target

Both v0 and `power_model` key on the **reference** turbine's direction — `main_analysis.py`
sets `ref_wd_col = "ref_YawAngleMean"`, which feeds detrending, the waking scenarios, the
`ref_wd_filter` and the pp binning (`test_wd_col` appears only in a pre/post sanity check);
and `power_model` is barred from the test turbine's own direction by §3. So there is one
fault target — a reference — and one row per mode in the bites table.

### Fixture

`benchmarking/campaigns/northing_fixture.py`: **T06** plus its three nearest stable
neighbours, over a 12-month 2017 baseline into 2018.

- T06 is the measured best fixture turbine (`power_model` mean |err| 0.34%, swing 0.72pp
  across the placebo window sweep) and 12mo→2018 is the best-ranked window.
- **T05 is excluded** despite being T06's natural best reference: it carries real northing
  steps in 2017–18, so injecting on top of them would muddy attribution.
- Injected uplift is `ws_dependent_cp` (+10% Cp below 5 m/s fading to 0 by 12 m/s) — the
  AeroUp shape — so truth is non-zero and we measure error, not placebo drift.
- Declared in **both modes**: prepost changing over 2018-01-01, toggle in 50-minute blocks.

### Natural-case probe (up front)

Before any injection: run v0 on T06 prepost 2017→2018 with and without northing correction,
using T05 as reference, to size the naturally occurring instance of this failure mode. This
is a sighting shot that calibrates how large an injected step needs to be to be realistic.

## Acceptance

### Per mode, per method (`power_model`, `v0`) — a 2×2, not a pair

| | northing off | northing on |
|---|---|---|
| **clean** | reference error | must be no worse — *no harm* |
| **faulted** | must be significantly worse — **bites** | must return to ≈ clean — **fixed** |

The *no harm* cell is the one most easily skipped and the one that would sink C3 if it were
wrong. Where the fault does not bite in toggle (cancellation across on/off blocks), that is
**recorded explicitly** as "no mitigation needed there" — determined empirically, never
assumed. Fault magnitude is calibrated until it bites, per the R-series ground rules.

Concrete thresholds, so the table has pass/fail rather than adjectives, with `e` the signed
error against the fixture's known truth:

- **bites**: `|e(faulted, off)| − |e(clean, off)| ≥ 1.0 pp`. T06's `power_model` swing across
  the placebo window sweep was 0.72 pp, so a 1 pp degradation is outside its natural
  window-to-window scatter and cannot be luck.
- **fixed**: `|e(faulted, on)| − |e(clean, off)| ≤ 0.25 pp`, i.e. the fault's residual damage
  is within a third of that natural scatter.
- **no harm**: `|e(clean, on)| − |e(clean, off)| ≤ 0.25 pp`.

These are the acceptance thresholds; if the natural-case probe or the clean re-baseline
(which changes when `power_model` gains the direction feature) shows T06's scatter is
materially different from 0.72 pp, the thresholds are re-derived from the measured scatter
and the change recorded — they are not loosened to make a run pass.

### Fixture results (measured)

T06 + T15/T10/T08, 12 months of 2017 baseline into 2018, `ws_dependent_cp` uplift injected, a
40° `NorthingStep` on T15 at the changeover (prepost) / mid-campaign (toggle). Errors in
percentage points of energy ratio:

| mode | method | clean/raw | faulted/raw | clean/northed | faulted/northed | bites | fixed | no harm |
|---|---|---|---|---|---|---|---|---|
| prepost | `power_model` | 0.451 | 1.782 | 0.399 | 0.534 | **+1.331 ✓** | **+0.083 ✓** | **−0.052 ✓** |
| prepost | `naive_ratio` | 5.892 | 5.892 | 5.892 | 5.892 | 0.000 | — | — |
| toggle | `power_model` | 0.114 | 0.089 | 0.040 | 0.072 | −0.025 ✗ | +(−0.042) ✓ | −0.075 ✓ |
| toggle | `naive_ratio` | 0.014 | 0.014 | 0.014 | 0.014 | 0.000 | — | — |
| toggle | `toggle_specialist` | 0.014 | 0.014 | 0.014 | 0.014 | 0.000 | — | — |

**Prepost: the fault bites and the shared step fixes it.** 1.331 pp of damage against the
1.0 pp threshold, closed to 0.083 pp against the 0.25 pp threshold. The threshold was derived
from T06's 0.72 pp placebo swing *before* this was run, and the clean error came out at 0.451 pp
— consistent, so the bar was not set to fit the answer.

**Toggle: the fault does not bite** (−0.025 pp), so no mitigation is needed there. This is the
cancellation the R-series design anticipated: with on/off blocks interleaved at 50 minutes, a
corruption present in both halves of the contrast largely cancels. **Recorded empirically, as
the ground rules require — not assumed.**

**`naive_ratio` and `toggle_specialist` are unmoved to the last digit** in all four arms, which
confirms the scoping decision: they read no direction signal, so northing is neither a risk nor
a benefit to them.

Two further observations:

- **The direction feature is doing real work.** In the power model's gain ranking the six
  `northed_wtc_NacelPos_mean_{sin,cos} @ {T08,T10,T15}` features come in immediately after the
  three reference active-power columns and the T10 power minimum — ahead of every ERA5 column.
  Without them the fault could not bite at all, which is why Seam 2 is a prerequisite rather
  than an enhancement.
- **Northing helps on clean data too** — the *no harm* cell is negative in both modes (0.451 →
  0.399 prepost, 0.114 → 0.040 toggle). The step discovers 7–11 changepoints across the four
  turbines even in the clean arm, where the vendored table (an old-optimizer product) says there
  are none. Given the farm-scale result, the likely reading is that these are real corrections
  the hill-climb missed rather than false positives — but it is inferred from the error moving
  the right way, not directly confirmed, and the small-N farm consensus stays a recorded risk.

### v0 swap — three pieces of evidence

1. **Ported tests.** `tests/test_optimize_northing.py`'s three `wind_direction_offset` cases
   and its injected-changepoint second half pass at the same or tighter tolerances (currently
   `abs=1.0` / `abs=1.5` degrees).
2. **The 180°-wrong farm.** A new unit test where every turbine is uniformly 180° out,
   proving the reanalysis pass is load-bearing and that pass 2 alone is blind to a
   common-mode offset. This is the case HOGER cannot address.
3. **Farm-scale real-data comparison.** Re-derive Hill of Towie's northing with both
   implementations and compare turbine-by-turbine, with runtime measured for both.

### Measured results

**Homer, July 2023, 2 turbines** (the ported v0 test): the new estimator reproduces the old one
**exactly** — identical median yaw and identical max northing error on all three
`wind_direction_offset` cases. Runtime is a wash at this size (0.6s vs 0.7s): the old
optimizer's cost is in the *search*, which barely runs on one month of two turbines.

**Hill of Towie, 21 turbines, 2017–2018** (2,207,520 rows, both passes):

| | old | new |
|---|---|---|
| runtime | 389.9 s | **39.3 s** (9.9x faster) |
| mean max northing error | 2.336° | **2.057°** |
| worst max northing error | 5.121° | **3.665°** |

Quality is v0's own metric (`check_wtg_northing`: max 20-day rolling circular-median error
against the wind-farm yaw direction), so neither implementation is scored on its own objective.
The new estimator is better or equal on 14 of 21 turbines and never worse by more than 0.25°;
the two largest gains are **T06 5.12 → 2.07** and **T15 4.01 → 2.72**, both turbines where it
finds a real changepoint the hill-climb missed. T06 being the biggest win matters directly —
it is the R1 fixture turbine.

Agreement is tight: median |new − old| ≤ 0.45° on every turbine. The turbines with a larger p95
(T06 5.26°, T09 3.46°, T15 3.20°) are precisely those where the new estimator found an extra
changepoint, which is also where its error metric improves most.

So "same or better performance" holds on both axes, and "MUCH faster" is **9.9x at farm scale**
— a number, and one that grows with record length and turbine count, since the old search cost
scales far worse than the DP's.

### Test strategy

The core is pure and array-based, so its tests are fast and synthetic. They land in
`tests/wind_up/test_northing.py` (alongside the existing `tests/wind_up/test_farm.py`);
`tests/test_northing.py` and `tests/test_optimize_northing.py` stay where they are, testing
v0's unchanged helpers and the adapter respectively.

- known steps at known times, recovered to within tolerance;
- wrap-around at 0/360 in both the raw and the northed signal;
- the all-180° case;
- noise floor: a step just below `min_step_deg` is not reported, one just above is;
- degenerate input (empty, all-NaN, all-unusable, a single segment) returns a valid one-row table;
- effort tiers: `fast` finds one large step; `thorough` finds small ones `fast` misses.

The fixture runs are drivers, not unit tests. The pytest layer gets a tiny-frame end-to-end
proving the shared step wires through `CampaignRunner`, and that `power_model` raises a
named error when the northed column is absent.

**At least one test runs on real data** — Hill of Towie, already available via git-lfs.
Synthetic tests pin the algorithm's contract but cannot expose what real SCADA does to it,
so a purely synthetic suite leaves a gap exactly where this issue lives.

## Designed for, not implemented

**The incremental re-run.** An analyst supplies a prior north table and asks wind-up to
check it again — either from scratch or with the prior already applied — and wants back
either "confirmed" or a list of amendments. This is normal usage on a live campaign: re-run
monthly as data arrives, and the new data may contain a north jump nobody knew about.

**Decision 5 (tables are always absolute) is the whole of what R1 does for this**, and it is
enough: a supplied table and a freshly estimated one are directly comparable, so "confirmed
or amended" is a subtraction over two absolute tables. Nothing else needs to exist yet.

Left for later: a mode that estimates *and* compares rather than choosing between them; a
`prior_mode` selecting whether supplied changepoints are pinned or re-optimised; and the
report that states "confirmed" or lists amendments.

Deliberately **not** built now: `estimate_north_table` takes no `prior` argument. Seeding the
search from a supplied table has no caller under the supplied-or-discovered rule above, and a
parameter that sits unused across issues drifts — the same argument the C2 design made for
keeping unread fields off `CampaignContext`. It is a small addition when a caller exists.

The success condition for all of this is that it stays rarely used — a norther fast and
accurate enough that supplying a table stops being worth the analyst's time.

**Pairwise consensus.** HOGER's ">50% of pairwise comparisons" vote as an alternative to
the farm-median reference, for farms with few turbines.

**Masts and LiDARs.** The core already accepts any direction field; what is missing is a
wind-speed-based `usable` helper and the plumbing to declare non-turbine devices.

## Risks

- **Making the fault bite `power_model` at all.** The direction feature is new, so how much
  the model leans on it is unknown until measured. If it leans lightly, the injected step may
  need to be large to bite, which strains realism. Mitigated by the natural-case probe, which
  sizes a real occurrence first. If the feature turns out to carry little weight, ERA5 wind
  direction can be withheld from `power_model` as a deliberate intervention, forcing it onto
  the turbine direction signal and putting the northing step under real pressure.
- **Small-N farm reference.** Four turbines with one jumping makes the farm-median direction
  noisier than at HoT's 21. The reanalysis pass anchors it, and pairwise consensus is the
  recorded fallback.
- **`min_step_deg` → penalty conversion.** The mapping from a degrees threshold to a
  resultant-length penalty needs calibrating against the noise floor rather than derived
  once on paper; the noise-floor test is what pins it.
- **v0 parity on real data.** "Same or better" is judged on the HoT farm-scale comparison.
  A turbine where the new table differs materially needs explaining, not averaging away.
