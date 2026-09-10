# Design — W1a: the composed `wind-up` method (compose and declare)

**Date:** 2026-09-08
**Status:** Approved in brainstorm; awaiting spec review
**Issue:** `docs/v1/issues_campaigns.md` § W1a, § W3
**Extends:** `2026-08-28-c1-campaign-runner-placebo-design.md` (the campaign runner and the
placebo), `2026-09-01-c2-campaign-context-seam-design.md` (the campaign context),
`2026-09-02-r1-northing-design.md` (the shared northing step),
`2026-09-04-r3-invalid-references-design.md` (the reference-validity screen),
`2026-08-28-v1-productization-release-design.md` (W2, which this issue feeds)
**Branch of work:** `v1-w1a`, developed off `v1`

## Problem

v1 has a winning estimator and two robustness fixes, but no single thing called `wind-up`.
What exists is a benchmark: `carried_forward_methods` builds three methods side by side,
`CampaignRunner` norths the farm and scores them all against injected truth, and every entry
point is a Python driver written per campaign.

Two consequences.

**There is no product-shaped way in.** An analyst cannot describe a campaign and get an answer;
they must write a driver. v0 at least had `WindUpConfig.from_yaml`, and v1 has regressed on
that ergonomic.

**There is no truth-free path at all.** `CampaignRunner` takes a `SyntheticDataset`, which
requires `original_df`; `Replicate` requires the same; `score_one` requires a `truth` argument.
Every code path that runs a method today is a benchmark path that knows the answer. Real data
has no `original_df`, so real data cannot currently be run.

W1a delivers the *interface*: a named, self-configuring, truth-free way to declare and run a
campaign. W1b settles whether the composition is *right*. W3 cannot start without W1a, which
is why W1a is hoisted ahead of C3–C6.

## Scope

Everything lands in `benchmarking/`. `src/wind_up` is not touched.

Measured before deciding: a runnable composed method in `src/` would drag ~5,700 lines behind
it — `power_model` (~3,060), the seam types `method.py`/`context.py`/`toggle.py`/`conditions.py`
(529), the whole `diagnostics/` package (~1,050), `era5_sync` + `filtering` (~250), the shared
northing step (288), and `ColumnSchema` out of `synthetic/schema.py`. Deciding product-versus-
harness for each of those *is* W2's promotion step, and W1a's own done-when says the composition
keeps moving as C3–C6 land, so the judgement would be made now and remade after C6. The
intermediate option — declaration and CLI in `src/wind_up`, method left in `benchmarking` — is
worse than either, because `src/wind_up` would then import `benchmarking` to reach the method.

## Decision 1 — `wind-up` is campaign-level, not a `Method`

At the method level `wind-up` would be a no-op relabel. The pieces W1a is asked to compose are
already on by default or already applied upstream:

| piece | where it already is |
|---|---|
| `power_model` | `carried_forward_methods`, with the accepted defaults |
| R3 reference-validity screen | *inside* `PowerModelMethod` (`reference_screen=True` by default) |
| R1 northing | `CampaignRunner._visible_dataset()`, farm-wide, before any method is built |

A `WindUpMethod(MethodInput) -> MethodOutput` in `benchmarking/baselines` would therefore
produce numbers bit-identical to the `power_model` the campaigns already run: two leaderboard
rows, one estimator.

The composition has content one level up. Northing is farm-wide and runs once, so it cannot sit
inside a per-turbine method without misrepresenting its scope; the farm aggregation, the guards
and the report are already campaign-level. `wind-up` is therefore the campaign-level entity:
the shared northing step, plus one `power_model` built from the accepted defaults, plus the
truth-free report, behind one name and one YAML declaration.

The multi-method `carried_forward_methods` path is untouched — the benchmark comparisons keep
running three methods. `wind-up` is the product-shaped way in, not a replacement for it.

W1b then validates *that configuration's* farm numbers against truth across C1–C6 and R1–R4,
and settles `toggle_specialist` — which is a real question at campaign level (does the toggle
arm swap the estimator?) in a way it is not at method level.

## Decision 2 — the truth-free core, and the benchmark as a layer over it

```
                        CampaignSpec + scada_df
                                  |
                    +-------------+-------------+
                    |  shared northing step (R1) |   farm-wide, runs once
                    +-------------+-------------+
                                  |
                         estimate_campaign()          truth-free core
                    per turbine: context -> MethodInput
                    -> method.estimate -> TurbineUplift
                            -> farm_uplift()
                                  |
                    +-------------+-------------+
                    |                           |
              write_report()              CampaignRunner
              (analyst-facing,            (benchmark: adds truth,
               no truth columns)           scores, truth-vs-estimate plots)
```

`CampaignRunner` keeps its current public behaviour and becomes a thin truth-adding layer over
the core. Its `_visible_dataset` / `_visible_mask` are re-cut to operate on frames rather than
on a `SyntheticDataset`, so the core can call them and the runner can call them twice (once for
`synthetic_df`, once for `original_df`).

Faking truth on the analyst path — `original_df = synthetic_df` — is rejected: it would put a
column of zeros labelled `truth` in the report a W3 analyst reads, which is worse than absent.

**The truth-free report must carry reference-turbine self-uplift, not just test-turbine and farm
uplift.** Every real Hill of Towie report to date gives this equal billing with the headline
number, not appendix treatment: each candidate reference is estimated as if it were a test
turbine, and a healthy campaign reads near 0% — the analyst's evidence that the references held
stable and that no undetected fault or creeping degradation leaked into one of them over the
campaign. It needs no truth column: it is the method judging its own references, the same check
R3's screen already runs internally. `PowerModelMethod` computes it for free
(`report_reference_uplifts=True` by default, `MethodOutput.reference_uplifts`); nothing in
`campaigns/report.py` reads it today. `write_report()` must include it as a
`reference_stability` table alongside `per_turbine`/`farm_uplift` — without it, an analyst has no
way to judge whether R3's automatic screen caught everything a reference-turbine fault could have
caused, which is the whole reason the real reports carry it.

This also settles W3's isolation in the strong sense. Once the product path is truth-free, the
answer key cannot reach the analyst's output directory, because the code that writes it has no
access to it.

## Decision 3 — the declaration

One YAML file, campaign facts only. Method configuration stays on the method with the accepted
defaults — the deliberate break from v0's `WindUpConfig`, which mixes both into one file.

```yaml
name: placebo_prepost

data:
  scada: scada.parquet          # source-native column names
  schema: hill_of_towie         # a named ColumnSchema
  turbines: turbines.csv        # name, latitude, longitude

turbines:
  upgraded:   [T06, T07, T11, T12, T16, T19]
  references: [T01, T02, T03, T04, T05, T08, T09, T10, T13, T14, T15, T17, T18, T20, T21]
  excluded:   []
  rated_power_kw: 2050

timing:
  mode: prepost
  changeover: 2018-01-01T00:00:00Z

analysis_period:
  start: 2017-01-01T00:00:00Z
  end:   2019-01-01T00:00:00Z    # exclusive

northing:
  discover: true
```

**`name`** names the run's output subdirectory, so a declaration is self-identifying rather
than depending on where it was invoked from.

**`data.schema`** is a key into a small name-to-`ColumnSchema` registry in the loader
(`hill_of_towie` -> `HOT_COLUMNS` is the only entry W1a needs). An unknown name raises, listing
the known ones; inline schema definition is not supported here.

**Turbine geometry goes to a sidecar CSV**, which is how site metadata already arrives
(`load_hot_metadata` returns `Name` / `Latitude` / `Longitude`). This is what keeps the YAML
short for a 21-turbine farm, which W2's acceptance test requires of the three real Hill of
Towie campaigns.

**Reanalysis is not declared.** `wind_up_v0.era5.get_era5_hourly_df(lat, lon, start_date,
end_date, cache_dir)` already fetches and caches, keyed by a hash of its arguments, honouring
`WIND_UP_CACHE_DIR` then `~/.cache/wind_up`. `wind-up` self-serves from the farm centroid of
`turbines.csv` and the analysis period, and the analyst never names a file. The fetch window is
**rounded out to whole calendar years**: the cache key includes the dates, so exact analysis
windows would re-download for every campaign on the same site. (This generalises the trick
`hot_context` hard-codes as a fixed wide `2000-01-01`..`2026-05-01` range.) Cost: the optional
`era5` dependency group and network on first run.

**Timestamps.** Every time field is a datetime. `analysis_period` is a `start`/`end` mapping
rather than a positional pair, so the exclusive end is stated. Timezone rule: **tz-aware is
honoured and converted to UTC; naive is treated as UTC**, and the resolved UTC values are
echoed into the run output, so a mistake is visible rather than silent — the same call already
made on northing (visible beats infallible). Implementation trap: PyYAML's implicit timestamp
resolver silently converts an offset such as `+01:00` into a naive UTC datetime, so the loader
takes the raw scalar and coerces through `pd.Timestamp` itself rather than trusting what YAML
hands back.

**`northing.discover`** maps onto the existing `north_offsets` tri-state: `true` is `None`
(discover from the data); `false` with a `table:` applies exactly that table; `false` with no
table is the empty list — apply nothing, discover nothing.

**`timing` is a tagged block** with `mode` as the discriminator, so a third mode is additive
rather than a schema break.

## Decision 4 — what W1a deliberately does not build

Per-turbine changeovers and date-ranged exclusions are **C8**, which is ordered before W1b and
is explicitly the issue that generalises the campaign declaration. `CampaignSpec` today is flat:
`timing_for()` returns the same timing for every turbine and `usable_mask()` is whole-turbine
on/off, with nothing overriding either. C8 widens both together; adding half of it here means
C8 rewrites that section of the schema.

AeroUp's commissioning ramp — an upgrade applied over a week or more, where the pre/post
separation is not a single instant — needs **no new axis**. A window that is neither clean-pre
nor clean-post is a window excluded from both, i.e. a per-turbine time-ranged exclusion, which
is what C8 adds. `timing_for(turbine)` is already per-turbine at the signature level, so C8
fills it in without a schema break. What W1a owes the future is only the tagged `timing` block.

**Distance-bounded automatic reference selection is not needed, and not built.** The real TuneUp
campaign was manually split into three zones (North/South/East), each with its own hand-picked
reference subset, because keeping references spatially local mattered to that method. That
zoning was a real cost to produce by hand and is exactly the kind of thing `wind-up` should do
for itself if it ever needs to — but the evidence so far says it doesn't: `power_model`'s
LightGBM-based regression handles all 21 Hill of Towie turbines as one undifferentiated candidate
pool without degradation, so the flat `turbines.references` list needs no zoning today. If
evidence later says otherwise, the fix is an automatic distance cutoff computed from
`turbines.csv`'s lat/longs (references within N rotor diameters of the test turbine, or the
nearest K), not a return to hand-drawn zones — but that is future work, built against evidence of
need, not built ahead of it here.

## Phasing

Three phases, each independently green, so the risky refactor is finished and pinned before
anything is built on top of it.

1. **The truth-free core.** Pin `test_runner.py` / `test_placebo_end_to_end.py`, re-cut
   `_visible_dataset` to frame level, extract `estimate_campaign()`, split `report.py`, and put
   `CampaignRunner` back on top. No new user-facing surface. Existing tests pass untouched.
2. **Declaration and entry point.** `loader.py`, `composed.py`, `__main__.py`. Ends with the
   placebo runnable from a YAML file in both modes.
3. **Handover and the dry run.** `handover.py`, the randomised placebo instance, the brief, and
   one W3 dry run driven end to end with its transcript kept.

## Components

| file | what it does | size |
|---|---|---|
| `campaigns/loader.py` | YAML → `CampaignSpec`; timestamp coercion and the tz rule, `turbines.csv`, named `ColumnSchema`, ERA5 centroid + whole-year window | ~150 |
| `campaigns/run.py` | truth-free `estimate_campaign()` + `CampaignReport`; frame-level northing/clip factored out of `CampaignRunner._visible_dataset` | ~150 |
| `campaigns/composed.py` | the composed `wind-up`: default method config, `run_declaration(path) -> CampaignReport` | ~100 |
| `campaigns/__main__.py` | `python -m benchmarking.campaigns run campaign.yaml --out DIR`; `--out` defaults to `WIND_UP_BENCHMARKING_OUTPUT_DIR`/`name`, matching the existing drivers | ~50 |
| `campaigns/report.py` | split: truth-free analyst report (incl. reference-turbine self-uplift table), benchmark overlay on top | edit |
| `campaigns/runner.py` | refactored onto the core; behaviour unchanged | edit |
| `campaigns/handover.py` | writes the W3 analyst directory | ~100 |

`composed.py` is a placeholder name — `wind_up.py` inside `benchmarking/campaigns/` reads badly
next to the real `wind_up` package. Its module docstring says so, so the next reader knows the
name is unsettled rather than considered.

`runner.py`'s refactor is W1a's only behaviour-break risk, so `test_runner.py` and
`test_placebo_end_to_end.py` are pinned before anything moves.

## The W3 dry run on the placebo

```
/tmp/w3-runs/2026-09-08-placebo-prepost/     outside the checkout
  analyst/                                    all the analyst ever sees
    brief.md            narrative, as an owner would write it
    campaign.yaml       template: commented, values blank
    data/scada.parquet  synthetic_df ONLY
    data/turbines.csv   name, latitude, longitude
    docs/               the documentation under test
    out/                appears after their run
  key/ground_truth.json injected upgrades, faults, true farm uplift, seed
```

**The instance is randomised.** `placebo_campaign()` already takes `upgraded=` / `turbines=`;
a seeded pick plus a shifted campaign start is added. Otherwise a populated YAML matching the
checked-in `PLACEBO_UPGRADED` (`T07 T11 T12 T06 T16 T19`) and `2018-01-01` defaults identifies
the campaign without the analyst reading a line of method code.

**The parquet is `synthetic_df` only.** `original_df` and `run_metadata` — which carries
`test_wtgs` and the injected upgrades — never leave the repo.

**Isolation is audited, not asserted.** A fresh agent has a filesystem and can reach the
checkout whatever it is told. Since the transcript is already W3's deliverable, it is read for
checkout access and runs that took it are discarded. Note that an installed `benchmarking`
package leaks only the *menu* of upgrade and fault classes, which W3 gives the analyst
deliberately ("the menu is given, not hidden"); it does not leak which was used.

**The placebo is a smoke test, not a score.** Its (a)/(b) answer — no upgrade, no faults — is
guessable without doing any work, so it cannot score the analyst's conclusion. What it does
test is W1a's own column of W3's table (*could they produce a run at all*) and whether the
outputs let a reader distinguish zero from noise without over-reading it. The first **scoring**
dry run is C3 (AeroUp, prepost); the first **toggle** dry run is C4. The toggle YAML shape is
still built and unit-tested here — it is simply not the subject of the first run, since the
toggle placebo is six months of `power_model` and a smoke test does not need it.

## Testing

Tests first throughout. `test_runner.py` and `test_placebo_end_to_end.py` are pinned before the
refactor moves anything.

- `test_loader.py` — both modes; the northing tri-state; naive→UTC and aware→converted; the
  PyYAML offset trap; error paths (unknown schema name, missing turbine in the CSV, references
  overlapping upgraded, end before start).
- `test_run.py` — the truth-free core on a small synthetic frame; the report's reference-turbine
  self-uplift table (`MethodOutput.reference_uplifts`) is present and carries no truth column.
- `test_composed.py` — the `wind-up` configuration equals the accepted `power_model` defaults,
  so a default drift in either is caught.
- `test_handover.py` — the handover directory's contents.

Two assertions carry the isolation guarantee and are written as such:

1. the analyst report has **no truth columns**;
2. the handover parquet has **no `original_df` and no `run_metadata`**.

The real end-to-end placebo run stays `slow`-marked.

## Done when

- `wind-up` runs self-configured from a YAML-declared `CampaignSpec` on the C1/C2 campaigns, in
  both modes.
- A campaign runs on data with no ground truth, and its report carries no truth columns.
- **One W3 dry run has been driven end-to-end against the placebo**, with the transcript kept
  and the gaps it found recorded for W2. This replaces W1a's current unfalsifiable *"and W3 can
  be attempted against it"*.
- `poe all-fast` green; the existing campaign behaviour is unchanged (`test_runner.py`,
  `test_placebo_end_to_end.py` pass untouched).

## Edits to `docs/v1/issues_campaigns.md`

- **W1a scope** — add the truth-free core (currently unmentioned, and it is the bulk of the
  work); record that `wind-up` is campaign-level and why; record the declaration decisions
  (ERA5 self-served from the centroid, tagged `timing` block, naive-means-UTC-and-echo); state
  explicitly that per-turbine timing and date-ranged exclusions are C8, and that AeroUp's ramp
  is a C8 exclusion rather than a new axis; record that the truth-free report carries a
  reference-turbine self-uplift table (`MethodOutput.reference_uplifts`) alongside per-turbine
  and farm uplift, since every real report to date treats it as equal-billing content and R3's
  screen alone does not give the analyst that visibility; record that distance-bounded automatic
  reference selection is deferred (not needed while `power_model` handles the whole candidate
  pool undifferentiated) rather than built ahead of evidence.
- **W1a done-when** — replace *"and W3 can be attempted against it"* with the concrete clause
  above.
- **W3** — add the isolation mechanics (truth-free path, randomised instance, transcript
  audit), the smoke-test-not-a-score caveat, and the C3/C4 first-real-run ordering.

## Risks

- **The `runner.py` refactor changes benchmark numbers.** Mitigated by pinning the existing
  tests first and requiring them to pass untouched.
- **ERA5 self-service needs network and the `era5` extra.** A first run in a fresh analyst
  environment downloads; a W3 run that fails on this is a documentation finding, not a method
  finding, and should be recorded as such.
- **The analyst directory is only as isolated as the audit.** Accepted: the transcript audit is
  the control, and a compromised run is discarded rather than scored.
