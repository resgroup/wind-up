# Design — campaign design: choosing test turbines that comply with the wind-up methodology

**Date:** 2026-09-11
**Status:** Implemented 2026-09-11; departures listed at the end
**Motivated by:** the W1a/W3 dry runs, whose randomly drawn campaigns clustered test turbines
(commit f373fe5 patched `placebo_instance` with a quick spacing rule; this replaces it)
**Informed by:** a private TuneUp campaign-design prototype (greedy priority walk, IEC front-row
classification, 3 references per test turbine). Ideas only; nothing from it is copied.
**Methodology:** `docs/wind-up uplift validation methodology v3.pdf`, step 2 (and step 3)
**Branch of work:** the user's call (brainstormed on `v1-w1a`)

## Problem

Choosing which turbines to upgrade for a validation campaign is currently done by hand or, in our
synthetic campaigns, by an unconstrained random draw. Either can cluster test turbines, leaving
some with no nearby reference to compare against. The quick fix of 2026-09-10 in
`benchmarking/campaigns/placebo.py` (`_spaced_draw`) is local to the placebo, uses a flat-earth
distance approximation, and has no notion of representativeness.

The methodology (v3, step 2) says: *"A set of reference turbines is chosen for each test turbine.
At least three reference turbines are used where possible."* It favours references whose power
correlates most strongly with the test turbine (*"Normally these are also the closest
turbines"*), with similar mean wind speed, with no known issues or changes, and ideally un-waked
in the predominant direction, *"especially by the test turbine itself"*. Step 3 adds a
representativeness requirement for curtailed turbines.

Separately, test turbines should occupy a representative share of turbine positions: if 25% of
the farm is tested, about 25% of the test turbines should be front-row turbines.

## Goal

A generic, public campaign-design utility in `src/wind_up` that:

1. picks the **most test turbines the site supports** while complying with the rules below;
2. among designs of that size, honours a user **test priority** order (TuneUp uses priority to
   make sure the spread of expected uplifts gets tested);
3. fills in the priority for any turbine the user did not rank, in a **seeded random order** — so
   a user with no priority method passes nothing, and our synthetic campaigns vary by seed;
4. reports compliance per test turbine (a **reference compliance report**) and checks any
   test-turbine set, however it was chosen;
5. ships with accurate geodesy used everywhere in the repo, replacing v0's hand-cached helpers,
   the approximate helpers in `benchmarking`, and the `utm` dependency.

`power_model` is unaffected: it keeps using every available non-test turbine as a candidate
reference. The per-test references are a compliance record, not a method input.

## Non-goals

- General stratification. Front row is the only group with a share rule. Internally a group is
  a named set of turbines so another can be added later without redesign.
- Choosing references by power correlation or mean wind speed (methodology items i-ii). Distance
  is the proxy, as the methodology's own wording suggests.
- Weighting front-row classification by the site's wind rose, a KML export, a CLI.
- Uncertainty or campaign-length planning.

## Terms

All sets below are drawn from the **farm under design** unless stated.

| Term | Meaning |
|---|---|
| layout | every turbine that matters geometrically, including neighbouring farms and unnamed turbines |
| farm under design | the turbines of one named `wind_farm` in the layout |
| blocker | any layout turbine, for front-row purposes (neighbours, unnamed and excluded included) |
| excluded | neither test nor reference (offline, another campaign, known issues); still a blocker |
| reference-only | never a test turbine; may be a reference |
| available | farm under design minus excluded |
| reference-eligible | available (reference-only included) |
| candidates | available minus reference-only, in resolved priority order |
| front row | a turbine with a contiguous IEC-clear arc of at least 90° (see Layout) |
| pool of i, `P(i)` | the `k` nearest reference-eligible turbines to i (excluding i), within the distance limit; ties broken by name |
| nearest neighbour of i, `nn(i)` | the nearest reference-eligible turbine to i |

## The rules

A set of test turbines complies when:

1. **Roles.** Every test turbine is a candidate (a named farm turbine, not excluded, not
   reference-only).
2. **Pool.** Each test turbine has at least 3 non-test turbines in its pool. Pool size
   `reference_pool_size` defaults to 4; `3` recovers the strict "the 3 nearest are never test
   turbines" rule. The pool only contains turbines within `max_reference_distance_d` (default 20)
   of the test turbine's own rotor diameters, so a turbine with fewer than 3 reference-eligible
   turbines inside the limit can never be tested.
3. **Nearest neighbour.** Each test turbine's nearest neighbour is not a test turbine — so the
   closest turbine is always a reference.
4. **Fair front-row share.** With `A` available turbines of which `F` are front row, and `n` test
   turbines, the front-row test count is within 0.5 of `n × F / A` — the nearest whole number,
   either one on an exact tie. This bounds overshoot and undershoot alike. Reference-only
   turbines count in `A` and `F`; excluded turbines do not, so a half-offline site shifts the
   target instead of distorting the test/reference balance.

A turbine may be a reference for any number of test turbines.

Each test turbine's **references** are the 3 nearest non-test turbines in its pool, nearest
first. Under rule 3 the nearest neighbour is always the first.

## Interface

New modules, all in `src/wind_up/`, none importing `benchmarking`:

| Module | Holds |
|---|---|
| `geodesy.py` | `distance_and_bearing`, `geodesic_matrices`, `local_east_north` |
| `layout.py` | layout validation, `iec_disturbed_sector_deg` (moved from v0), the upwind helper, `front_row` |
| `campaign_design.py` | `design_campaign`, `check_design`, `CampaignDesign`, `ComplianceReport`, `write_design`, `write_compliance` |
| `campaign_design_plots.py` | the two maps |

### Layout frame

One row per turbine. Column names are matched case-insensitively.

| Column | Required | Notes |
|---|---|---|
| `name` | no | unique where present; farm-under-design turbines must be named; unnamed turbines are blockers only |
| `latitude`, `longitude` | yes | WGS84 degrees |
| `rotor_diameter_m` | no | missing values take the largest known diameter (worst case: widest disturbed sector); the fills are reported; no known diameter at all is an error |
| `wind_farm` | no | the farm's name, or empty when unknown; colours the map |

This is a superset of the campaign declaration's `turbines.csv` sidecar.

### `design_campaign`

```python
design_campaign(
    layout, *,
    wind_farm=None,                # defaults to the only wind_farm present; error if ambiguous
    test_priority=(),              # highest first
    reference_only=(),
    excluded=(),
    n_test=None,                   # None = the most the site supports
    seed=0,                        # orders the unlisted tail
    reference_pool_size=4,
    max_reference_distance_d=20.0,
    front_row_min_clear_deg=90.0,
) -> CampaignDesign
```

Raises on malformed input with a message saying what to fix: an unnamed farm turbine, a name in
more than one of `test_priority` / `reference_only` / `excluded`, a role name that is not a farm
turbine, a duplicate name, no rotor diameters, an ambiguous `wind_farm`. Also raises when no
compliant design exists, or when `n_test` exceeds the maximum (the message states the maximum).

`CampaignDesign` carries: the farm name; the test turbines in commit order; each test turbine's 3
references; the front-row set; `max_test_turbines` (always computed, so a smaller `n_test` can be
compared with the ceiling); the resolved priority order, with which part came from
`test_priority` and which from the seeded tail; per candidate, whether it was committed, skipped
(with a reason) or not needed; the filled rotor diameters; the settings used; and the
`ComplianceReport` from running `check_design` on itself. A helper returns the declaration roles:
`upgraded` = the test turbines, `references` = every available non-test farm turbine,
`excluded` = the excluded turbines.

### `check_design`

```python
check_design(
    layout, *,
    test_turbines,
    wind_farm=None,
    reference_only=(),
    excluded=(),
    reference_pool_size=4,
    max_reference_distance_d=20.0,
    front_row_min_clear_deg=90.0,
) -> ComplianceReport
```

Raises only on malformed input. Non-compliance is reported, never raised, so a failing hand-made
design can still be inspected.

`ComplianceReport` carries:

- `compliant: bool` and `problems: list[str]`, one plain-language line per violation (e.g.
  "T05 has only 2 non-test turbines among its 4 nearest (T02 and T04 are test turbines)");
- `table`, one row per test turbine: `test_turbine`, `front_row`, then for each of
  `reference_1..3` the name, `distance_m`, `distance_d`, `front_row` and neighbour rank (1 = its
  nearest reference-eligible turbine), then `compliant`. A failing turbine lists what it has and
  leaves the rest blank;
- `summary`: farm, available and reference-eligible counts; the test count; available front-row
  count; front-row test count against the fair share; the pool size and distance limit; the
  filled rotor diameters.

## Algorithm

**References are fixed by geometry.** Pools and nearest neighbours come from the geodesic distance
matrix once, before any selection.

**The model** is a feasibility MILP over one binary `x_i` per candidate (non-candidates are 0),
solved with `scipy.optimize.milp` (HiGHS; scipy is already a core dependency):

- count: `Σ x_i = n`;
- pool, for each candidate i with `|P(i)| ≥ 3`: `3·x_i + Σ_{j ∈ P(i)} x_j ≤ |P(i)|`;
  candidates with `|P(i)| < 3` are fixed at 0;
- nearest neighbour, for each candidate i whose `nn(i)` is a candidate: `x_i + x_nn(i) ≤ 1`;
- fair share: `n·F/A − 0.5 ≤ Σ_{i ∈ front row} x_i ≤ n·F/A + 0.5`, posed in integers as
  `2nF − A ≤ 2A·Σ_{i ∈ front row} x_i ≤ 2nF + A` so the tie case is exact (the checker uses the
  same integer form);
- committed turbines fixed at 1.

**Step 1: the count.** If `n_test` is given, one feasibility solve confirms it. Otherwise one MILP
maximises `Σ x_i` subject to the pool and nearest-neighbour rules only, giving an upper bound `U`;
then `n = U, U−1, …` are tried with the share rule until one is feasible. The scan is linear
because rounding makes feasibility non-monotone in `n`. The maximum is computed in both cases, for
`max_test_turbines`.

**Step 2: the walk.** Resolve the priority order: `test_priority`, then every unlisted candidate in
an order drawn from `numpy.random.default_rng(seed)`. Walk it. For each candidate, ask the solver
whether a compliant design of size `n` exists containing the committed turbines plus this one. If
yes, commit it; if no, skip it with the first reason that applies:

1. a direct conflict with a committed turbine Tk, found by checking the rules on the committed set
   plus this candidate alone: "reference of Tk (its nearest neighbour)", "would leave Tk with
   fewer than 3 references", or "Tk is its nearest neighbour";
2. "the front-row share is full" or "the rest-of-farm share is full";
3. "no compliant design of n includes it alongside the higher-priority picks".

Stop at `n` commits; the rest are "not needed". Each commit keeps a feasible completion alive, so
the walk always finishes. The result is the first compliant design of size `n` in priority order:
the count is maximised first, and priority decides which turbines fill it.

Candidates failing rule 2 before the walk (fewer than 3 reference-eligible turbines within the
limit) carry that as their reason.

**Cost.** About `U + |candidates|` small solves; well under a second for 60 turbines.

## Layout geometry

- **Upwind.** Turbine j is upwind of i at wind direction θ when the bearing from i to j is within
  half the IEC 61400-12-1 Annex A disturbed sector of θ (strict `<`, as v0), the sector computed
  from the separation in rotor diameters of the blocker j (as v0).
- **Front row.** Sweep θ = 0…359° in 1° steps; a turbine is front row when its longest contiguous
  run of directions with no upwind blocker, wrapping through 0°/360°, is at least
  `front_row_min_clear_deg` (default 90°). Computed for every layout turbine against every other
  layout turbine.

## Geodesy

Always the WGS84 ellipsoidal geodesic (Karney's algorithm via `geographiclib`, kept after
reviewing pyproj, karney, geodistpy and hand-rolled options). No haversine, equirectangular or
UTM-grid shortcut anywhere in the repo.

- `distance_and_bearing(origin, destination) -> (distance_m, bearing_deg)`: one `Inverse` call;
  `origin`/`destination` are `(latitude, longitude)`; bearing clockwise from true north in
  `[0, 360)`.
- `geodesic_matrices(latitudes, longitudes) -> (distance_m, bearing_deg)`: N×N arrays, entry
  `[i, j]` from i to j, one `Inverse` per ordered pair; distance 0 and bearing NaN on the
  diagonal. About 12 ms for 21 turbines, about 1 s for 200.
- `local_east_north(latitudes, longitudes) -> (east_m, north_m)`: geodesic distance `d` and
  bearing `β` from the site centroid give `east = d·sin β`, `north = d·cos β` (azimuthal
  equidistant on WGS84, about 2 × 10⁻⁶ distortion at 20 km); then shifted so the minimum easting
  and minimum northing are 0, so the maxima are the farm's physical extent. North is true north.

**Consolidation.**

- `wind_up_v0/waking_state.py`: `calc_distance`, `calc_bearing` and `get_distance_and_bearing`
  delegate to `distance_and_bearing`; the module-global dict cache is replaced with
  `functools.cache` on the v0 wrapper (v0's per-direction sweep needs a cache; v1 uses the
  matrices). `iec_disturbed_sector_deg` moves to `wind_up.layout` and v0 re-exports it, as
  `wind_up_v0/circular_math.py` already does. v0 outputs are expected bit-identical.
- `wind_up_v0/plots/misc_plots.py` `bubble_plot`: `local_east_north` replaces `utm.from_latlon`,
  and the plot keeps subtracting the median, so its look is unchanged. On Hill of Towie (near UTM
  zone 30's central meridian) it barely moves; elsewhere it turns by the grid convergence, a
  correction to true north. `utm` leaves `pyproject.toml` and the mypy overrides.
- `benchmarking/synthetic/geometry.py`: the haversine `distance_m` and spherical `bearing_deg` are
  deleted; `derive_wake_steering_pairs` and `benchmarking/synthetic/upgrades.py` use
  `wind_up.geodesy` and `wind_up.layout`. WakeSteering nadir bearings move by up to about 0.06°
  and a pair at the 7 D limit could flip; affected pins are re-pinned and each move is named in
  the commit.
- `benchmarking/campaigns/placebo.py`: `_spaced_draw`, `_nearest`, `_separation` and
  `_METRES_PER_DEGREE` are deleted.

## Outputs

`write_design(design, out_dir)` writes:

- `compliance.csv`: the report's per-test table;
- `turbines.csv`: one row per farm turbine: role (test / reference in use / reference-eligible
  unused / reference-only / excluded), front row, resolved priority rank and whether it came from
  `test_priority` or the seeded tail, and the skip reason or "not needed". This answers "why was
  T12 not picked";
- `roles.yaml`: a `turbines:` block in the campaign declaration's own keys (`upgraded`,
  `references`, `excluded`), ready to paste;
- `design_map.png`: east/north metres from `local_east_north`, equal aspect;
- `design_map_latlon.png`: longitude/latitude degrees, axis aspect `1 / cos(mean latitude)`.

Both maps: test turbines red, references in use blue, other available turbines grey,
reference-only a hollow blue marker, excluded an ×, front row a heavy ring; other layout turbines
small and coloured by `wind_farm`; a thin line from each test turbine to each of its references;
names where present; the title gives the test count and the front-row share against its target.

`write_compliance(report, out_dir)` writes `compliance.csv` for a hand-checked set.

## Integration

- **`placebo_instance`** keeps its signature (`mode, *, seed, coords, turbines=None`) and builds a
  layout from `coords` (new `HOT_ROTOR_DIAMETER_M = 82` next to `HOT_RATED_POWER_KW`,
  `wind_farm="Hill of Towie"`), then calls
  `design_campaign(layout, reference_only=("T17",), seed=seed)` with `n_test=None`, so every
  instance is maximal and the seed varies which turbines. `PLACEBO_INSTANCE_MIN_UPGRADED`,
  `PLACEBO_INSTANCE_MAX_UPGRADED_FRACTION`, `PLACEBO_INSTANCE_NEAREST_REFS`,
  `MIN_SCREENABLE_REFERENCES` and `_max_upgraded` go; only `test_placebo.py` reads them, and its
  size assertions become compliance assertions. Its synthetic `LINE_COORDS` fixtures (and
  `test_handover.py`'s call) may need spacing that respects the 20 D limit. The treatment-start
  draw is unchanged.
- **Campaign declaration loader** (`benchmarking/campaigns/loader.py`) skips `turbines.csv` rows
  with no name, so the design's layout file can serve as the sidecar. Extra columns are already
  ignored.
- **Docs:** a short `docs/designing-a-campaign.md`: inputs and roles, the four rules, how the count
  and priority interact, reading the outputs, a worked Hill of Towie example.
- **Already done in the brainstorm session:** the accurate-geodesy rule added to `CLAUDE.md`
  (uncommitted).

## Evidence from the brainstorm

Hill of Towie (21 turbines, 82 m rotors, nearest spacing 4.1–4.9 D), measured 2026-09-11 with a
throwaway MILP, no front-row share constraint:

| Rule | Most test turbines | Nearest-neighbour test pairs | Furthest 3rd reference |
|---|---|---|---|
| 3 nearest never test (pool 3) | 7 | 0 | 3rd nearest |
| order-dependent greedy walk with locked references (2000 random orders) | 6–9 | 0 | 7th nearest |
| pool 4, no nearest-neighbour rule | 10 | 2 | 4th nearest |
| **pool 4 + nearest-neighbour rule (chosen)** | **10** | **0** | 4th nearest |
| pool 5 / 6 + nearest-neighbour rule | 10 / 12 | 0 | 5th / 6th nearest |

The approximations being replaced, against the geodesic on the same layout: equirectangular
distance up to 0.24% off, haversine distance 0.35%, spherical bearing 0.06°. The 3-nearest sets
happened to agree.

## Decisions (each agreed in the brainstorm)

- Per-test references are a compliance record; `power_model` keeps the whole pool.
- Front row is the only share group; the share is two-sided (no overshoot, no undershoot).
- References are chosen by distance only; no front-row preference for references.
- Greedy walk with an exact solver look-ahead, over a pure optimizer (weights cannot express
  strict priority) or backtracking; the compliance checker is separate from the selector.
- Order-free pool rule (pool 4, nearest neighbour always a reference) instead of the
  order-dependent "later picks may sit beside earlier ones" rule.
- References may be shared. Distance limit 20 D, as a keyword.
- Share denominator excludes excluded turbines. Unlisted farm turbines are appended to the
  priority in seeded random order. A `reference_only` role exists.
- Layout: `wind_farm` name column (not a flag), optional names, optional rotor diameters filled
  with the largest known.
- Names: `test_priority`, `reference_only`, `excluded`.
- `n_test` optional; default is the most the site supports.
- 90° front-row threshold, kept as a keyword.
- Keep `geographiclib`, fix the call pattern; drop `utm`; east/north origin at the minimum
  easting and northing; a latitude/longitude map as well.
- The placebo uses the maximum count.

## Testing

**Geodesy** (`tests/wind_up/test_geodesy.py`): the expected values of v0's `test_get_bearing`,
`test_get_distance` and `test_get_distance_and_bearing` ported with inline coordinates (the v0
tests stay and must pass untouched); `geodesic_matrices` equals the scalar function for every
pair; `local_east_north` has minimum easting and northing exactly 0, puts a point 1 km due north at
(0, 1000) to the millimetre, and matches pairwise geodesic distances to 10⁻⁵ relative on Hill of
Towie.

**Layout** (`tests/wind_up/test_layout.py`): v0's parametrised `test_get_iec_upwind_turbines`
cases ported to the upwind helper; disturbed-sector values; longest clear arc (all clear, all
blocked, a run wrapping 0°/360°, exactly 90° against 89°); three turbines in a line (ends front
row, middle not); validation errors; diameter fill.

**Design and checker** (`tests/wind_up/test_campaign_design.py`):

- each rule on a tiny hand-derivable layout: pool 3 and 4, nearest neighbour, the distance
  limit, shared references, fair share blocking overshoot and undershoot, the tie case;
- brute force over every subset on layouts of up to 12 turbines: the solver's maximum is the true
  maximum, and the walk returns the first compliant set in priority order;
- adjacent #1 and #2 priorities: #1 is tested, #2 is its reference, with that reason;
- priority honoured; the seeded tail reproducible and seed-dependent; "not needed" distinguished
  from "skipped"; an infeasible `n_test` error states the maximum;
- `design_campaign` output passes `check_design` on Hill of Towie across 20 seeds;
- the W1a-era clustered Hill of Towie draw (T02/T04/T05 triangle, T13/T14 pair) is reported
  non-compliant with the expected problem lines;
- writers and plots: smoke tests; `roles.yaml` loads through the declaration loader.

**Integration:** `tests/benchmarking/campaigns/test_placebo.py` checks instances comply, are
maximal, never test T17, are reproducible per seed and vary across seeds; WakeSteering/synthetic
pins re-pinned where the geodesic switch moves them; the v0 suite passes; `utm` is no longer
importable in the environment. All under `poe test-fast`; nothing new is `slow`.

## Phases

1. **Geodesy and layout**: `geodesy.py`, `layout.py`, v0 delegation, `utm` removal, the
   benchmarking helpers switched, re-pins. Pinned first because it is the only phase that can
   move existing numbers.
2. **Design and checker**: `design_campaign`, `check_design`, the result types.
3. **Outputs**: writers, both maps, `roles.yaml`, the loader skipping unnamed rows.
4. **Integration**: `placebo_instance`, `docs/designing-a-campaign.md`.

## After implementation (2026-09-11)

Implemented in commits 614c4e8, ffd08d7, 7b81bc0 and 792c64b on `v1-w1a`. Departures from the
design above, each agreed during the session or forced by what the code found:

- **Nearness tie-break.** After the listed `test_priority`, the unlisted candidates are walked
  under the least reference-distance limit that still allows the design's size
  (`CampaignDesign.reference_limit_d`). It sits after the listed priority because a strict
  minimax applied first usually leaves one design, so neither priority nor seed would matter.
- **The placebo designs one below the maximum** (`PLACEBO_INSTANCE_BELOW_MAX = 1`), with every
  candidate listed in a shuffled priority: at the maximum Hill of Towie has only 2 compliant
  designs, one below it 60.
- **`summary.yaml`** is written alongside the other outputs (counts, share, maximum, limit,
  problems).
- **The maps zoom to the farm under design** and size to its shape; neighbours outside the view
  are clipped.
- **"No compliant design" names its cause:** too few available turbines, none within the
  distance limit, or the rules together.

Evidence is recorded as CF16 in `docs/v1/findings_campaigns.md`.
