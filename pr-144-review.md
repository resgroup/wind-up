# PR 144 review — status (2026-09-23)

PR: https://github.com/resgroup/wind-up/pull/144 ("v1 Northing improvement", `northing-improvement` → `v1`).
Local branch is **2 commits ahead of origin, not pushed**:

- `25288d9` Address PR 144 review: explicit layouts, one pass-4 run, tidy wake-nadir
- `715919f` Test the default layout-driven northing path on real Hill of Towie data (**fails, on purpose**)

## 1. Why CI was red (resolved)

`tests/test_optimize_northing.py::test_auto_northing_corrections_with_changepoints` (all 3 rotations, +22.9 deg off).
Root cause: commit `7c4d394` (pass 3) made `against_reanalysis()` also raise `min_segment` to 30 days.
v0's adapter uses `against_reanalysis` for its 2-turbine Homer farm, and the test injects two steps
10 days apart in a 31-day record, so no split is possible any more. Not a threading bug, and not a
bug in the new code: it's the new floor meeting a one-month synthetic case.

Decision (Alex): drop the test; no special v0 logic; v0 will be deprecated. Dynamically shortening
the 30-day floor for short records was considered and rejected: the floor is about how slowly
reanalysis error drifts, which doesn't depend on record length.
Done in `25288d9`.

## 2. Review items and what happened

| # | Item | Status |
|---|------|--------|
| 1 | v0 test regression | Test dropped |
| 2 | PR description out of date (describes `coordinates`/`neighbours`/`nearest_neighbours`; omits pass 3, pass 4, golden tables, YAML wrap) | **Open**: rewrite once the default-path fix (section 3) lands |
| 3 | Undeclared v0 behaviour changes (`_required_step` fix, YAML wrap) | Won't fix: v0 is being deprecated |
| 4 | Harness `coords=None` defaults silently chose whole-farm with no pass 4 | Fixed: `NorthingInputs(era5_wd, layout)` bundle in `benchmarking/harness/replicates.py`. Five HoT drivers were really affected and now pass `hot_layout(context.metadata_df)`. `study_toggle_specialist_uncertainty.py` restored to v1 (its coords were never used; it never norths) |
| 5 | No real-data test of the default v1 path | Partly done: `TestKnownChangepointsWithTheLayout` in `tests/wind_up/test_northing_real_data.py`. **It fails**, see section 3. Pass 4 on real data is still untested, see section 4 |
| 6 | `DEFAULT_ROTOR_DIAMETER_M = 82` quietly assumed | Fixed: constant and `Layout.from_coordinates` deleted; `from_frame` again refuses a layout with no diameters. `north_scada(layout=...)`; `CampaignSpec.layout` replaces `coords` (`spec.coords` is now a derived property); the turbines CSV a campaign loads needs `rotor_diameter_m`; `GreenbyteFarm.rotor_diameter_m` (Kelmarsh 92, Penmanshiel 82); helpers `layout_from_coords`, `layout_coords`, `hot_layout` |
| 7 | `north_scada` ran `north_farm` twice to draw the bubble plot | Fixed: public `add_wake_nadir(tables, ...) -> (tables, corrections)`; `north_farm` uses it internally; test asserts one run |
| 8 | Public `nearest_neighbours` unused | Dropped; ranking tests moved onto `_neighbours_from_layout` |
| 9 | `_Nadir.volume` unused | Dropped (the `_Nadir` NamedTuple is gone; pairs are plain floats) |
| 10 | Log messages name the removed `min_devices_for_farm_reference` | Fixed (`MIN_DEVICES_FOR_FARM_REFERENCE`) |
| 11 | One NaN wind speed dropped the whole wind-speed signal for a pair | Fixed: per-row mask with per-bin population check; test added |
| 12 | Small-farm branch computes pass 1 then discards it | Won't fix: negligible cost |

Lint and mypy are clean. Fast suite: everything passes except the three new real-data layout tests
and `test_uplift_plots::test_all_three_plots_are_written`, which fails on Windows only because of the
path separator (it predates this work; passes on Linux CI).

## 3. Blocker resolved (2026-09-24): pass 2 repeats until it converges

The layout path handed turbines their neighbours' steps (T15 with T05/T16, T17 with T19, T11 in the
June 2020 outage; on 5 years also T14, T03/T04, then T05/T07 with T01/T02's 180 deg flip).
`north_farm` now repeats pass 2 until a round moves no changepoint and no offset > 0.5 deg
(`_MAX_CONSENSUS_ROUNDS = 10`, helper `_converged_pass_two`, check `_table_change`). Larger k alone
did not fix it. The 5-year HoT record converges at round 7. See **CF21** in
`docs/v1/findings_campaigns.md`. Cost: several times one round's pass 2 on a slow-settling farm.

The study re-run (one-round code, before the change) reproduced CF20 exactly (floor sweep
199/197/176/106/76/34 spurious, tuned 13/28 recall with 1 spurious). CF19 was a synthetic scratch
experiment, not reproducible from the repo.

## 4. Tests rewritten on a new fixture

- `tests/test_data/hot/northing/northing_farm_inputs.parquet` (15 MB, git-lfs): the study's own HoT
  inputs for 2017-2020 (yaw, power, ws, yaw_usable mask, ERA5), integer-scaled. Built by
  `benchmarking/baselines/make_northing_farm_fixture.py`. The old yaw-only
  `northing_inputs.parquet` is removed (`git rm`).
- `tests/wind_up/hot_northing.py`: shared loader and helpers.
- `test_northing_real_data.py`: the default path (layout plus pass 4) on 2017-18 and 2019-20 finds
  exactly the published changepoints, is quiet in the outages, and matches offsets across the
  window boundary. Also the study's degradation cases, held to recorded worst error x1.5 + 1 deg;
  the costly ones are slow-marked. Plus the whole-farm fallback and pass 3.
- `test_northing_rotation.py`: same tests on the new fixture, layout path without pass 4.

## 5. State (2026-09-24, end of session)

Committed locally, not pushed (branch is 7 commits ahead of origin):
45d4dfa converging pass 2 + CF21 + golden tables; 4d125d1 real-data tests on the new fixture;
0d345c4 Layout refuses impossible layouts + HoT coordinates for campaign fixtures (HOT_COORDINATES,
hot_coords); 3d79a5a Windows path test fix; bd96188 study results in CF21.
PR description rewritten on GitHub.

Study re-run (converged, all three farms incl. Kelmarsh/Penmanshiel from F:\kelmarsh_and_penmanshiel):
no crashes, CF20 reproduced exactly.

## 6. Still to do

1. Push; CI runs the slow suite for the first time on the new tests.
2. Diagnose the 6-turbine HoT subset getting worse with converged pass 2 (2.0 -> 6.9 deg, CF21).
3. Optionally: Penmanshiel N=10 (9.2 deg) and 7-day (9.4 deg) cases, first run, no baseline.
