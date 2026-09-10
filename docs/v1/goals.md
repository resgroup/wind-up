# wind-up v1 — goals

## Context

wind-up v0 is a beta tool (published on PyPI as `res-wind-up`) that measures the
energy-yield **uplift** of a turbine upgrade — `(upgraded MWh) / (baseline MWh) − 1`
for matched conditions — using a **binned power-curve, test-vs-reference** method
with uncertainty estimation (including a block bootstrap). It is used in production
at RES.

v1 is a major upgrade. Its purpose is to make wind-up measure uplift **more
accurately**, from **shorter campaigns**, and with **more insight into how an
upgrade's performance depends on conditions** — while keeping the tool practical
and easy to use.

## North-star vision

> Turn wind-up from a single-method tool into a **platform for measuring
> turbine-upgrade uplift**, in which alternative methods are pluggable and
> objectively benchmarked, and in which more of the final report is produced by
> the tool rather than by hand.

## Goals

### G1 — Pluggable, objectively benchmarked methodology *(the central thrust)*
The choice of "best" uplift method is currently unknown and should not be
constrained by premature architectural assumptions. v1 makes the uplift method a
**pluggable component** and provides a **public synthetic-data evaluation harness**
with known ground truth, so candidate methods can be compared objectively against
the v0 baseline. New methods are merged only once they demonstrably beat the
baseline.

### G2 — Better results from shorter campaigns
Reduce the data (campaign duration) needed to reach a given accuracy/precision.
Short-campaign robustness is a primary evaluation axis, not an afterthought.

### G3 — Conditional / heterogeneous uplift information
Report not just a single uplift number but **how uplift varies with conditions** —
wakes vs free-stream, day vs night, wind direction, atmospheric stability — so
upgrades can be understood and targeted. (This is a natural output of
treatment-effect / ML methods conditioned on more than wind speed.)

### G4 — Pipeline as independent, composable steps
Make each stage of the analysis runnable on its own:
1. **Pre-processing** — source data (SCADA, ERA5, optionally mast/LiDAR), filter,
   feature engineering.
2. **Measure campaign uplift + uncertainty** per turbine.
3. **Long-term extrapolation** of uplift + uncertainty.
4. **Aggregate** results across turbines.

### G5 — Matured I/O and configuration
Make wind-up easier to configure and use (cleaner inputs/outputs, clearer config).

### G6 — More of the report auto-generated
Today wind-up produces most of the per-pair analysis plots (the report appendix),
but the executive summary, results tables, farm-layout figure, exclusion-period
table, combined/by-pair uplift charts, and reference-suitability table are made by
hand. v1 should generate **more of this report content** to cut manual
post-processing. *(Lower priority than G1–G4; high practical value.)*

## What v1 is NOT (initial scope guards)

- **Not** an uncertainty-model overhaul *first*. The first methodology effort
  targets **P50 accuracy and precision only**. The P95 / uncertainty model
  (block bootstrap, conformal OOD, density-ratio long-term weighting) is
  **deferred** until the best P50 method is identified, because P95 depends on the
  chosen point method.
- **Not** a big-bang rewrite. Foundational refactors (G4, G5) are done only as far
  as needed to unblock the methodology work — enough to avoid baking in
  assumptions, no more.

## Success criteria

1. A candidate method recovers a **known injected uplift** on synthetic datasets
   more accurately and/or precisely than the v0 baseline (P50).
2. A method reaches a target accuracy/precision from a **shorter campaign** than
   the v0 baseline requires.
3. The uplift method is selectable via config, with the v0 method preserved.
4. (Stretch / later) The conditional-uplift story (G3) and auto-generated report
   content (G6) are available from the tool.

## Source material

- Internal v1 planning notes — the v1 plan and goals.
- `wind-up-ml-uplift-design-note.md` — the ML / treatment-effect (R-learner)
  methodology candidate and staged plan.
- Example v0 uplift reports (Hill of Towie AeroUp/Pitch, Tallentire, Earlseat,
  wake-steering) — basis for the report-generation gap analysis (G6).
- Key open data: Hill of Towie SCADA (Zenodo 20204946) and
  `resgroup/hill-of-towie-open-source-analysis`.
- Other open data: SMARTEOLE (see existing wind-up example notebook), Kelmarsh (https://zenodo.org/records/16807551), Penmanshiel (https://zenodo.org/records/16807304), WeDoWind (pitch-angle and vortex-generator examples in wind-up).
