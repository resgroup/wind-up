# wind-up v1 — roadmap

This file decomposes v1 into workstreams (epics) and lays out the phasing. See
[goals.md](goals.md) for the goals these serve, and [issues.md](issues.md) for the
concrete first issues.

## Workstreams (epics)

### WS1 — Evaluation harness *(public; critical path)*
The objective yardstick for everything else. Two parts:
- **Synthetic upgrade-dataset generator** — take real SCADA from stable,
  no-upgrade periods (Hill of Towie and other open wind farms) and inject a
  **known** uplift to create realistic datasets with a ground-truth answer.
  Support a range of injected uplift *profiles*: constant Cp change,
  wind-speed-dependent Cp change (region-2-only, tailing to 0 at rated — the AeroUp
  shape), condition-dependent change (e.g. wake-direction-dependent — the
  wake-steering shape; or stability-dependent, as TuneUp may be), rated-power
  change, etc.
- **P50 scoring + short-campaign robustness sweep** — given a method's estimate
  and the known injected truth, compute **accuracy** (bias) and **precision**
  (spread) metrics; sweep campaign length to measure how each degrades with less
  data (serves G2).

Designed to stand alone so external collaborators can compete on it (WeDoWind /
Kaggle-style), mirroring the Hill of Towie power-prediction challenge.

### WS2 — Methods
The competing uplift estimators, all judged on WS1.
- **v0 binned power-curve method as the baseline** — the bar to beat.
- **Candidate: cross-fit R-learner (P50)** — treatment-effect framing from the
  design note; treatment-invariant reference-only features (never the test
  turbine's own SCADA wind speed); LightGBM nuisances.
- **Later candidates** — DSWE / `funGP` as a cross-check, Astolfi
  multivariate-linear residual method, others.
- Multi-dimensional binning (i.e. the AWC validation methodology by S. Kanev,
  TNO 2020 R11300, Aug 2020) — wind speed × wind direction binning with adaptive
  bin sizing. See [references.md](references.md).

### WS3 — Minimal foundations *(only what unblocks WS2)*
- **Assessment data contract** — a standardized per test-reference, pre/post
  conditioned dataset (with treatment-invariant features) that ANY method
  consumes. This is the interface methods plug into.
- **Pluggable method selector** — a thin `assessment_method` config field that
  picks the estimator, reusing existing inputs / test-ref pairing / result
  objects. Kept deliberately minimal; not a big refactor (see goals.md scope
  guards).

### WS4 — Uncertainty / P95 *(deferred)*
Begins only after a P50 winner is identified, because the uncertainty model
depends on the chosen point method. From the design note: block bootstrap
(autocorrelation-aware), conformalized quantile regression for OOD filtering, and
density-ratio weighting for long-term (ERA5) extrapolation under covariate shift.

### WS5 — Reporting & I/O maturation *(parallel-later)*
- **Report content generation (G6)** — auto-produce the report pieces currently
  made by hand: executive-summary numbers, per-turbine results table, combined and
  by-pair uplift charts, farm-layout figure with test/reference highlighting,
  exclusion-period table, reference-suitability table.
- **I/O & config maturation (G5)** — cleaner inputs/outputs and configuration.
- **Pipeline step independence (G4)** — run preprocess / measure / long-term /
  aggregate as standalone steps.

## Phasing

| Phase | Focus | Workstreams |
|-------|-------|-------------|
| **1 — now** | Harness + baseline + first new method, on **P50 only** | WS1, WS2 (baseline + R-learner), WS3 (minimal contract) |
| **2** | More methods; conditional/heterogeneous uplift (G3); short-campaign study (G2) | WS2, WS1 |
| **3** | Uncertainty / P95 model | WS4 |
| **4 (overlap)** | Reporting (G6), I/O maturation (G5), step independence (G4) | WS5 |

## Dependency notes

- WS1 (harness) gates WS2 — no objective comparison without it.
- WS3 (data contract) is the seam between WS1 and WS2: methods consume the
  contract; the harness feeds synthetic data through it.
- WS4 strictly follows a P50 winner from Phase 1–2.
- WS5 is independent and can proceed whenever there is appetite, since it builds
  on the existing v0 outputs.

## Management

- All v1 PRs target the `v1` branch.
- First issues are drafted in [issues.md](issues.md) and created on GitHub
  (`resgroup/wind-up`) once wording is settled — suggested labels `v1`,
  `WS1-harness`, `WS2-methods`, `WS3-foundations`, and a `v1` milestone.
- Implementation of each issue should go through the normal design → plan → build
  flow in its own session.
