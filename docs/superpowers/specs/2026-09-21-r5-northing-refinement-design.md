# R5 — Northing refinement: small-N devices, and absolute accuracy from wake nadirs

**Status:** design, 2026-09-21. Lands in PR 144 (`northing-improvement` → `v1`).
**Issue:** `docs/v1/issues_campaigns.md` §R5 (Part A: small-N devices; Part B: wake-nadir absolute anchor).

## Context

PR 144 already simplified `north_farm` into two passes:

- **Pass 1** — a constant bulk anchor: one offset per device nulling its whole-record direction to
  reanalysis, no changepoints. It exists to build the consensus and to hold the farm in absolute
  terms; it never attributes changepoints, because reanalysis is short-term unreliable and a
  pass-1 changepoint would leak weather excursions into the very consensus pass 2 trusts.
- **Pass 2** — each device is northed against the circular-median consensus of its nearest
  neighbours (WGS84 geodesic, `nearest_neighbours`), where all changepoint detection happens.
  `north_farm` currently **refuses below `min_devices_for_farm_reference=3`**.

`against_reanalysis()` and `REANALYSIS_MIN_STEP_DEG=10` still exist in `northing.py` but are now
**unused** — they are exactly what pass 3 revives. `Layout` (`src/wind_up/layout.py`) already
bundles coordinates, rotor diameters, geodesic distance/bearing and IEC disturbed sectors, all
validated. `WakePair` / `derive_wake_steering_pairs` (`benchmarking/synthetic/geometry.py`) already
turn geometry into directed pairs with nadir bearings.

R5 turns this into the four-pass pipeline below.

## Goal

North devices a farm consensus cannot reach (Part A), and improve the **absolute** accuracy of the
answer — not just its internal consistency — using wake nadirs as a physical anchor (Part B).

## The four-pass pipeline

`north_farm` becomes four passes, with a **whole-farm switch** between 2 and 3:

1. **Pass 1 — ERA5 constant anchor.** Unchanged. Always computed. Builds the consensus and is the
   absolute base / safety fallback.
2. **Pass 2 — neighbour consensus + changepoints.** Unchanged mechanism. Runs **when the farm is
   large enough** — `len(devices) >= min_devices_for_farm_reference` and a consensus is formable.
3. **Pass 3 — ERA5 + changepoints** *(new)*. Runs **when pass 2 did not** (small farm). Reanalysis
   reference with changepoint detection, via `against_reanalysis(settings)`, at a
   `REANALYSIS_MIN_STEP_DEG` lowered by evidence. Supersedes pass 1 for small farms (pass 1 is its
   no-changepoint special case).
4. **Pass 4 — wake-nadir absolute nudge** *(new)*. Runs when a `Layout` and per-device `power` are
   supplied and geometry gives at least one usable pair within the wake cutoff. Produces one
   **absolute correction per turbine**, added on top of that turbine's changepoint table. Never
   changes changepoints or which rows downstream analysis treats as valid.

**The 3-device floor stops being a hard `raise`.** A 1- or 2-device farm now runs pass 1 + pass 3
(+ pass 4 where geometry allows) — this is Part A. `min_devices_for_farm_reference` is retained as
the **documented boundary between two supported regimes** (consensus vs reanalysis), not an error.

## Interface changes

`north_farm`'s signature changes to carry geometry and power (both chosen deliberately in
brainstorming):

- `coordinates: Mapping[str, tuple[float, float]] | None` → **`layout: Layout | None`**. `Layout`
  already carries coordinates, rotor diameters and geodesic geometry. `layout=None` — written
  explicitly, as `coordinates=None` is today — keeps the whole-farm-consensus path and disables
  pass 4. Neighbour selection reuses `layout.distance_m` rather than recomputing.
- **`power: Mapping[str, ndarray] | None`** — per-device power on the shared `index`, for pass 4's
  deficit-versus-direction curve. `None` disables pass 4.

Device names in `direction_deg` must resolve to `Layout` rows; validated with a clear error.
Reference-only turbines in the layout that are not being northed are ignored.

## Pass 3 design (Part A)

Reanalysis reference + changepoints, veer-normalisation on (it also subtracts reanalysis's *static*
direction-dependent bias per sector). The floor is lowered **by evidence, not assertion**.

Why a lower floor is safer than it first appears: `_worst_transient` prunes on **persistence** (how
much a changepoint moves the *long-run* level), not raw step size, so a weather wobble that returns
has persistence ≈ 0 and is dropped whatever the floor. Lowering `REANALYSIS_MIN_STEP_DEG` from 10°
mainly exposes edge / short-segment artefacts, already guarded by the support-scaled required step
(`_required_step`), and genuinely ambiguous small persistent steps.

- Set `REANALYSIS_MIN_STEP_DEG` from the **21-turbine one-at-a-time experiment** (below), scored
  against the published table and, later, the full-farm golden table.
- **One lever held in reserve:** decoupling the reported-step floor from the persistence threshold
  (a separate `min_persistence_deg` for the reanalysis regime). Added **only if** evidence shows the
  floor alone cannot separate small real steps from artefacts. Not built speculatively.

## Pass 4 design (Part B)

A new module (`src/wind_up/wake_nadir.py`), invoked as pass 4 inside `north_farm`.

**Per-turbine correction, not one global shift.** After pass 2 has put a large farm in one consensus
frame the per-turbine values largely agree (≈ a global rotation), but the per-turbine form is what
works for pass-3 small farms and lets a back-row turbine with no downstream neighbour be filled by
its neighbours. It stays **one absolute number per turbine** (not per segment), so "offset only,
changepoints untouched" holds.

**Geometric nadir.** For a directed pair (X upstream, Y downstream) within the wake cutoff
(**10 rotor diameters**, from `layout`), the geometric nadir bearing β is the wind-from direction at
which X's wake lands on Y — the geodesic bearing Y→X (reusing `WakePair` logic on `layout`).

**Measured (apparent) nadir and its uncertainty.** For turbine X as the direction reference, bin
Y's power against X's **northed** wind direction in a sector around β; locate the deficit minimum by
a robust quadratic fit near the dip. The fit yields both the centre and a **quick standard error**
from the dip's curvature (sharpness), depth, and in-sector row count: deep, sharp, well-sampled dips
get small σ; shallow or thin ones get large σ. Apparent − geometric = X's residual δ.

**Aggregate per turbine.** Combine a turbine's pairs by **inverse-variance-weighted circular mean**
(σ⁻²), with light outlier down-weighting, so the most certain wake centres dominate. Store δ_X and
its combined σ_X.

**Spatial inheritance.** A turbine with no resolvable δ (no downstream neighbour within the cutoff,
or all σ too large) **inherits the circular median of δ from up to its 4 nearest turbines that did
resolve** (`layout` distances). A turbine with neither its own nor any resolvable neighbour gets 0.

**Apply.** δ_X is added to every offset in X's table — a single absolute shift per turbine on top of
the changepoint structure. Pass 4 outputs offsets only; it never alters row validity, so a turbine's
wake-affected rows are left for downstream analysis to treat as it sees fit.

**Graceful degradation** falls out: when every pair is shallow/thin the weights are small, δ stays
near zero with wide σ (logged); with no geometry at all pass 4 is skipped.

## Validation and the ERA5 / published-table caveat

ERA5 is a poor proxy for hub-height wind at the specific site, so the wake-nadir anchor can
legitimately move the answer **by up to ~10°** from the ERA5-anchored / published answer. That move
is the point of Part B, not a regression. Therefore:

- The **published HoT table** (`optimized_northing_corrections.yaml`) validates **changepoint
  structure** — same turbines, dates, counts, relative steps — i.e. passes 1–3. It does **not**
  bound the pass-4 absolute shift.
- **Pass 4 correctness** is proven on **synthetic** ground truth: inject a known absolute offset θ
  (rotate all directions) on top of a `WakeSteering` dataset and assert pass 4 recovers θ.
- On real HoT, pass 4's per-turbine shifts are checked for **plausibility** (spatially smooth across
  neighbours; magnitude within a sane bound), not for agreement with the published table.

**Golden tables.** The full pipeline (1→2→4) on a complete farm is the best available absolute
answer. Record each complete farm's golden table as a small fixture (new file — flagged for the user
to `git add`; do not stage). Small-N / subset challenges are then scored against the **golden +
published** tables.

## Harness plumbing

`north_scada` already has per-turbine power (it builds `usable` from it) and coordinates. It will
build a `Layout` from site metadata (lat/lon + rotor diameter) and pass `layout` + `power` into
`north_farm`. The campaign runner and study replicates inherit it unchanged. Coordinates-only
callers (no diameters) still work — `Layout` fills missing diameters and pass 4 is simply weaker.

## Testing plan

- **Part A:** grow `TestSingleTurbineAgainstReanalysis` into all-21-turbines-scored-against-the-
  published-table at the chosen floor; synthetic challenge cases — small steps, steps near the
  record edge, steps during an outage.
- **Part B:** synthetic injected-absolute-offset recovery (pass 4 recovers θ); on HoT, changepoints
  unchanged by pass 4 and per-turbine shifts plausible/spatially smooth; graceful degradation with
  too few pairs.
- **Subsets** of HoT (and Kelmarsh / Penmanshiel where data exists) down to **1 turbine**, using
  **contiguous** groups so a wake pair exists for pass 4; subset answers scored against the golden
  table.
- Existing real-data and rotation-invariance suites stay green (call sites migrated to `layout=` /
  `power=`).

## Sequencing (staged, reviewable commits within PR 144)

1. **Layout + power signature; whole-farm switch; floor removal.** Migrate `north_farm` and all call
   sites/tests to `layout` + `power`; pass 2/pass 3 switch on `min_devices_for_farm_reference`;
   small farms no longer raise.
2. **Pass 4 on full farms → golden tables.** Implement `wake_nadir.py` (uncertainty-weighted
   per-turbine nadir + spatial inheritance) and wire it as pass 4; run the full pipeline on complete
   HoT (and other sites where available); validate structure against the published table; record
   golden tables.
3. **Pass 3 + evidence-driven floor.** Implement pass 3; run the one-at-a-time and contiguous-subset
   experiments; set `REANALYSIS_MIN_STEP_DEG` (and, only if needed, the persistence lever) from the
   results.
4. **Harness plumbing + final regression tests**, including the synthetic pass-4 recovery test.

## Decisions locked (from brainstorming)

- Pass 2/3 boundary: **whole-farm switch** on `min_devices_for_farm_reference` (not a new knob).
- Pass 4 seam: **inside `north_farm`** as a true fourth pass.
- Geometry input: **accept a `Layout`** (replacing the coordinates mapping).
- Pass 4 shape: **per-turbine** absolute correction, uncertainty-weighted, with spatial inheritance
  (median of up to 4 nearest resolved). One number per turbine, not per segment.
- Pass 4 magnitude: **may reach ~10°**; validated on synthetic ground truth, not against the
  published table.
- Pass 3 floor: **evidence-first**, persistence lever in reserve.

## Out of scope

- Uncertainty / P95 modelling of the northing correction itself (Phase 1 is P50-only).
- Per-segment (time-varying) pass-4 shifts.
- R6 (curtailment) and C5 (direction-offset treatment), which consume R5 but are separate issues.
