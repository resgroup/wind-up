# wind-up v1.0.0 release checklist

Draft, started 2026-11-17. This replaces the "W2 after W1b after C3-C6" ordering in
[issues_campaigns.md](issues_campaigns.md) as the route to a release: v1.0.0 ships when the
requirements below are met, and method development (power model, R-learner, C4-C6, the
remaining R issues) continues after it in minor releases.

## Release principle

**v1.0.0 freezes the interface, not the methods.**

- The interface is: the import paths, the method seam (`Method`, `MethodInput`, `MethodOutput`,
  `CampaignContext`), the campaign declaration (YAML), the column schema, the CLI and the names
  and columns of the core output files. Breaking any of these needs v2.
- A method's numbers may change in a minor release, with a changelog entry saying what moved
  and why. A method is labelled **stable** or **provisional**; a provisional method's options
  may also change in a minor release.
- New methods, new optional declaration keys, new optional `MethodOutput` fields and new output
  files are additive and go in minor releases.

Proposed labels at v1.0.0: `ToggleEnergyRatioMethod` stable; the v0 binned method stable
(frozen); `PowerModelMethod` and the composed `wind-up` run provisional; R-learner not shipped
until it is merged.

## Decisions needed before the work is sized

| # | decision | options | proposal |
|---|---|---|---|
| D1 | What "reproduce any v0 analysis" means | (a) `wind_up_v0` ships in the wheel unchanged and its examples still run; (b) as (a), plus the v0 method runs from a v1 declaration through the seam (`V0BinnedMethod` promoted); (c) the v1 method itself covers every v0 use, including long-term AEP and P95 | (b). (c) makes the release wait on the methods again |
| D2 | Where the methods live | (a) `wind_up.methods.*` with the seam in `wind_up.method`; (b) flat `wind_up.toggle_specialist` etc. | (a) |
| D3 | What happens to `import wind_up` for v0 users | It becomes the v1 package; v0 code is `wind_up_v0`. Already done on `v1` (W0) but it is the one break every existing user hits | Keep, with a migration note at the top of the README and the release notes |
| D4 | Does the composed `wind-up` run ship at 1.0.0 or stay in `benchmarking/` until W1b | ship provisional / hold back | Ship provisional: the declaration is the interface that most needs real users |
| D5 | Does the synthetic generator ship | (a) stays in the repo, not in the wheel; (b) `wind_up.synthetic` | (a) for 1.0.0; G1's "public harness" is met by the repo |
| D6 | Pre-releases | alpha, beta and release candidate, see "Pre-release stages" | Yes; `benchmarking` is never a top-level package in any release, pre-releases included (decided 2026-11-17) |

## Interface principle: facts, question, method (decided 2026-11-17)

An analysis is described in three layers, and the interface keeps them apart.

1. **Facts.** The analyst declares what is true of the wind farm and the campaign in
   method-agnostic terms: layout, rated power, which column means what, instruments that are
   not turbines (a mast or LiDAR: where it is, what it measures), which turbines changed and
   when, the toggle signal and its cycle, the controller's settling time after a toggle change,
   exclusion periods, northing corrections, a wake steering lookup table.
2. **Question.** What is being measured over which records: the effect, the subject turbines,
   and the subset of records (a direction sector, the records another question did not use).
   One campaign and one set of facts can carry several questions.
3. **Method and its options.** Each method answers a question from the facts and has options
   that are its own (bin width, block length, pairing gap, years of detrend data).

Each method decides whether it has enough information:

- a required fact is missing: the method raises `MethodConfigurationError` before doing any
  work, naming the fact;
- a declared fact is one the method cannot use (a LiDAR reference for a method that compares
  power): it runs and records an entry in the analysis warnings register.

Reference choice splits the same way: "this turbine was not controlled" is a fact, "this
reference drifts" is a judgement the reference screen makes and the analyst may override.

- [ ] The declaration and `CampaignContext` carry every fact in the list above; today they lack
      instruments that are not turbines, the settling time, the toggle cycle and datum, and the
      steering lookup table.
- [ ] A type for the question, between the declaration and the method input. Today a record
      subset can only be expressed by editing the frame or by a method's own exclusion column.
- [ ] `ToggleEnergyRatioMethod` takes `toggle_period`, `toggle_datum`, `rated_power_kw` and
      `columns` as its own options; all four are facts and move to the input. Do this before
      `1.0.0a1`, since it changes the constructor an alpha user calls.
- [ ] The same pass over `binned_power_curve` and `counterfactual_power` when they move.
- [ ] `binned_power_curve` takes v0's own options as its method options, so any v0 analysis can
      be expressed; it currently exposes five and hard-codes the rest.
- [ ] Every method documents the facts it requires and the facts it ignores.

The Hill of Towie Dynamic Yaw campaign is the worked case for this
(`scripts/wfc_analysis_2026` in `resgroup/hill-of-towie-open-source-analysis`): collective
control first, then wake steering, each with `binned_power_curve` (reproducing the published
result) and `toggle_energy_ratio`. Each place the interface gets in the way is added here.

## Worked case: Hill of Towie Dynamic Yaw, collective control (2026-11-18)

Reference runs of the original scripts with released `res-wind-up` 0.7.0 on Zenodo record
22662930: wake steering +1.32 %, sigma 0.54 % (as published); collective control +0.53 %,
sigma 0.46 % (published +0.56 %, 0.47 %; accepted as close enough). The collective control
inputs were then run through the v1 seam.

| Method | Farm uplift, 13 turbines | Per-turbine sigma | Hours on per turbine | Run time |
|---|---|---|---|---|
| v0, local run of the original script (references T01, T13, T18 and a LiDAR) | +0.53 %, sigma 0.46 % | 0.6 to 1.1 % | 180 to 440 | 2 h |
| `toggle_energy_ratio` (references T01, T13, T18) | +0.13 %, no farm sigma | 0.4 to 0.9 % | 710 to 1080 | 2 min |

The two agree within their uncertainty. `toggle_energy_ratio` ran with no change to `src`: an
explicit toggle frame as the timing, the declared exclusions as `valid_for_uplift`, and the
steering records through `ColumnSchema.exclude_row`.

What the interface could not express, each found by doing it:

- [ ] **The analysed campaign is shorter than the toggle signal.** The toggle frame has on and
      off records from 2025-11-05 but the analysis starts 2026-01-07. The script cut the data
      and the frame by hand; the campaign window is a fact and belongs on the input.
- [ ] **Settling time has nowhere to go.** The script blanked the first record after each change
      in the toggle frame. It is a fact of the campaign (already listed above); a method then
      applies it or says it cannot.
- [ ] **Toggle cycle and datum were stated on the method** although the toggle frame implies
      them (already listed above). When both are given they should be checked against each other.
- [ ] **A declared reference the method cannot use is dropped silently.** The LiDAR has no power,
      so the script left it out. This is the "declared fact the method cannot use" case: the
      method should receive it and put a line in the warnings register.
- [ ] **A farm result has no sigma.** `farm_uplift` returns the energy-weighted uplift and the
      spread across turbines. The turbines share references, so their errors are correlated and
      the farm sigma cannot be built from the per-turbine sigmas by the caller.
- [ ] **`MethodOutput` does not carry the energy behind the estimate.** `farm_uplift` needs each
      turbine's actual energy and record count; the script rebuilt them from `labeled_rows`.
- [ ] **`ColumnSchema` demands roles the method never reads.** `wind_speed`, `wind_speed_sd` and
      `gen_rpm` are required to construct it; `toggle_energy_ratio` reads power and availability.
      Each method already checks the roles it needs, so the rest can be optional.
- [ ] **One call per subject.** Thirteen `estimate` calls, each given the whole frame. Fine for
      this method; v0 analyses all subjects in one run (see below).

A prototype of `binned_power_curve` outside `src` (v0's run behind `MethodInput` and
`CampaignContext`) needed these passed around the seam:

- [ ] turbine type data: rotor diameter, rated power, hub height, cut-in and cut-out;
- [ ] turbine positions and the timestamp convention (`metadata_df`);
- [ ] reanalysis data;
- [ ] non-turbine reference data (the LiDAR) and its name as a reference;
- [ ] the campaign window and the settling time, as above;
- [ ] a northing table, since v0 applies its own (v1 would hand it northed directions);
- [ ] several subjects in one run, and a result per subject-reference pair, which
      `MethodOutput` has no field for.

The declared exclusions did pass through the seam, as `valid_for_uplift` over the whole record,
but not with the meaning v0 gives them. Two runs of the prototype against the original's 64
subject-reference results:

| How the exclusions reached v0 | Pairs reproduced | Largest uplift difference |
|---|---|---|
| v0's own exclusion periods | 64 of 64, to rounding | 0 |
| `valid_for_uplift`, rows dropped before v0 | 55 of 64 | 0.51 pp (T12 against T13, 375 h on against 459) |

- [ ] **`valid_for_uplift` flattens two different facts.** v0 blanks an excluded record for the
      uplift but still reads its raw power to decide whether that turbine is waking others. A
      record dropped before the method leaves the wake state unknown, and the records of
      turbines downwind of it are lost. An excluded record whose data is right (a turbine on the
      wrong side of the test, a curtailment) still describes the wind farm; one whose data is
      wrong does not. A method needs to tell them apart, so the seam should carry both: the
      records as measured, and why each is not valid for uplift.
- [ ] **`binned_power_curve` reproduces v0 exactly once the items above are on the input.** The
      reproduction of the published collective control result is then a test that can be kept.

## Worked case: Hill of Towie Dynamic Yaw, wake steering (2026-11-18)

The question is different in kind from collective control. There is one question per steer (a
steering turbine and the turbine it steers away from), asked only of the records in the
direction sector where the controller steers, and the answer wanted is the net uplift of the
two turbines together. The published figure is the mean over the steers with at least 100
hours on.

| Steer (sector) | v0, net of the pair | `toggle_energy_ratio`, net of the pair |
|---|---|---|
| T02 to T05 (211 to 241) | +0.8 %, sigma 1.1 %, 105 h on | +2.4 %, 150 h on |
| T03 to T07 (192 to 224) | +2.1 %, sigma 1.2 %, 104 h on | +1.3 %, 152 h on |
| T11 to T14 (228 to 255) | +1.0 %, sigma 0.5 %, 169 h on | +2.1 %, 189 h on |
| T05 to T04 (131 to 161) | +1.2 %, sigma 2.4 %, 44 h on | +1.9 %, 104 h on |
| Mean of the steers with at least 100 h on | +1.32 %, sigma 0.54 % (3 steers) | +1.9 % (4 steers; +1.9 % on v0's 3) |

Both methods put the gain on the downstream turbine and read the steering turbine near zero.
`toggle_energy_ratio` gives per-turbine sigmas of 0.7 to 2.2 % on these steers and no sigma for
a pair or for the mean; treating the pair as fully correlated and the steers as independent
gives about 0.8 % on the mean. Its run took 2 minutes for all nine steers; v0 took 5 h 20 min
for 36 runs. The `binned_power_curve` prototype, given the sector and v0's own exclusions,
reproduces the original for the one steer checked (T11 to T14 against T13).

What the interface could not express:

- [ ] **The record subset of the question has nowhere to go.** The sector was passed to
      `toggle_energy_ratio` as an exclusion column written onto the data, and to v0 as a method
      option. It belongs to the question (the second layer), beside the subjects.
- [ ] **A subset by direction needs a direction, and the seam does not carry one.** v0 takes the
      sector on each reference's own northed direction, so the records selected differ by
      reference. For `toggle_energy_ratio` the script built a farm direction from the references'
      northed nacelle positions. Direction must not come from a subject: the upgrade changes it.
      The declaration should say which direction signal a subset is taken on.
- [ ] **The subject of a question can be a group of turbines.** The net uplift of a pair, and
      its sigma, were built outside both methods: v0's `calc_net_uplift` for one, `farm_uplift`
      (no sigma) for the other. Same gap as the farm sigma above.
- [ ] **Several questions, then one combined answer.** The mean over steers, the 100 hour floor
      and the combined sigma are all in the script. v1 needs a view on whether combining
      questions is the tool's job or the analyst's; either way each answer must carry what the
      combination needs (hours on and off, energy, sigma).
- [ ] **A question with no records returns NaN and no warning.** The T14 to T12 sector (2 to 6
      degrees) had no usable records; `toggle_energy_ratio` returned NaN silently. It should be a
      line in the warnings register.
- [ ] **A long-term correction is a method option that the question can rule out.** The original
      turns v0's long-term distribution off for steering, since a sector result cannot be
      extrapolated to a year by the method.
- [ ] **Secondary measures confirmed.** The original reports the change in yaw activity beside
      the uplift. Yaw activity per hour came straight from `labeled_rows`
      (steering turbines +9 to +32 %; v0's pairs +12 to +47 %), so that split works, provided
      `labeled_rows` keeps the input's other columns.

## Method names (decided 2026-11-17)

A method is named for its mechanism and scope, not for a version, a year or a judgement. The
class is `<Name>Method`, the module `wind_up.methods.<name>`, and the name string written to
output files (and later read from the declaration's `method:` block) is the same `<name>`.

| name | class | was | status |
|---|---|---|---|
| `toggle_energy_ratio` | `ToggleEnergyRatioMethod` | `toggle_specialist`, `ToggleSpecialistMethod` | renamed with the move into `src/` |
| `binned_power_curve` | `BinnedPowerCurveMethod` | `v0_binned`, `V0BinnedMethod` | rename when it moves into `src/` |
| `counterfactual_power` | `CounterfactualPowerMethod` | `power_model`, `PowerModelMethod` | rename when it moves into `src/`; the frozen benchmarks are keyed by the old name |

`naive_ratio` stays in `benchmarking/` as a yardstick and is not shipped. Documents written
before the renames (`issues*.md`, `findings*.md`) keep the old names.

## Pre-release stages

pip installs a pre-release only when asked (`pip install res-wind-up==1.0.0a1` or `--pre`), so
0.7.x users are unaffected until 1.0.0.

| stage | published when | what may still change |
|---|---|---|
| `1.0.0a1` | PRs 169 and 170 are merged to `v1`, the toggle energy ratio method and what it imports are in `src/wind_up`, and the wheel contains no `benchmarking` package | anything, including import paths and the seam |
| `1.0.0aN` | each further move into `src/` (power model, campaign run, CLI) | anything |
| `1.0.0b1` | R1-R3 met: every method importable, the interface reviewed, the method registry in place | the interface, in response to feedback |
| `1.0.0rc1` | everything below met except the merge to `main` and the tag | bug fixes only |

## Requirements

### R1. Every method is importable from the installed package

Today `src/wind_up` exports only `farm_uplift`; the seam and every method live in
`benchmarking/`, which is packaged "temporarily" for exactly this reason.

- [ ] `from wind_up.methods import ToggleEnergyRatioMethod` (and the power model, the v0 binned
      method, the naive ratio) works after `pip install res-wind-up` with no repo checkout.
- [ ] `ToggleEnergyRatioMethod` imports and runs with the base dependencies only. LightGBM, pvlib
      and scikit-learn stay behind the `ml` extra and are imported only by the power model; a
      missing extra raises a message naming the extra.
- [ ] Nothing under `src/` imports `benchmarking`; add a test that keeps it true.
- [ ] First move (2026-11-17, for `1.0.0a1`): `wind_up.methods.toggle_energy_ratio` with the seam
      (`wind_up.method`, `wind_up.campaign_context`), `wind_up.schema`, `wind_up.toggle`,
      `wind_up.conditions`, `wind_up.operating_state`, `wind_up.record_selection`, `wind_up.waking`,
      `wind_up.block_bootstrap`, `wind_up.filtering` and `wind_up.diagnostics`.
- [ ] `benchmarking*` is removed from `[tool.setuptools.packages.find]`; a built wheel is
      inspected to contain `wind_up` and `wind_up_v0` only.
- [ ] A smoke test in CI installs the built wheel into a clean environment and imports the
      public names.

### R2. The interface is solid enough to live with through 1.x

Everything listed under "the interface" gets one deliberate naming and shape review before it
is frozen. Known items:

- [ ] The seam depends on the synthetic generator: `MethodInput.upgrade_timing` and
      `CampaignContext.timing` are typed with `benchmarking.synthetic.ToggleSchedule`, and
      methods take `benchmarking.synthetic.schema.ColumnSchema`. Both types move to `src/` and
      the generator imports them from there.
- [ ] `MethodInput` carries the same fact twice (`upgrade_timing`/`turbine_col` and the
      context, reconciled in `__post_init__`). Decide whether the public form is context-only
      with a convenience constructor.
- [ ] `MethodOutput` names: `p50_overall`, `p50_by_condition`, `sigma_overall`. Confirm them or
      rename now. Decide where P95 and a long-term/AEP result will go when they arrive, so they
      land as optional fields and not as a reshaped output.
- [ ] `MethodOutput` fields that are frames (`reference_uplifts`, `screen_passes`,
      `selection_accounting`, `labeled_rows`) have their columns documented in one place; the
      documented columns are part of the interface, extra columns are not.
- [ ] Conditions are a closed tuple (`CONDITIONS`) and an unknown name raises. A new method must
      be able to report on its own axis without editing a central list.
- [ ] The toggle timing accepts a `ToggleSchedule` or a bare DataFrame with `toggle_on` /
      `toggle_off`. Give the explicit form a named type.
- [ ] One documented error contract: `MethodConfigurationError` fails the campaign, a
      `ValueError` marks one turbine unmeasured. Consider a named exception for the second.
- [ ] The declaration carries a `version:` key and rejects unknown keys with a message naming
      the nearest valid key, so a later schema change can be detected and migrated.
- [ ] Output files: list which files and columns are stable (`farm_uplift.csv`,
      `per_turbine.csv`, `conditional.csv`, `reference_stability.csv`, `analysis_warnings.csv`)
      and say everything else is diagnostic and may change.
- [ ] The public names are exported from `wind_up/__init__.py` or documented subpackages;
      anything else is private by convention (leading underscore or not in `__all__`).

### R3. Extensible for the methods still in development

- [ ] A method is selected and configured without editing wind-up: a registry keyed by name
      (built-ins registered on import; a third-party method by entry point or by passing the
      object), and a `method:` block in the declaration with `name` and that method's own options.
- [ ] Method options are validated by the method (its dataclass), not by the declaration
      loader, so a new method adds options without a loader change.
- [ ] Method-specific outputs have a home that does not need a `MethodOutput` change: a method
      writes its own files under its own folder of the run, and `uncertainty_diagnostics`-style
      pass-through frames stay uninterpreted.
- [ ] The R-learner branch (PR 172) is rebased onto the moved layout as the test of this: it
      should land touching only its own module, its registration and its tests.
- [ ] A short "writing a method" page with a minimal working method.

### R4. v0 analyses can be reproduced (scope per D1)

- [ ] `wind_up_v0` ships in the wheel and its existing tests pass unchanged.
- [ ] The v0 examples (SMARTEOLE notebook, WeDoWind, Kelmarsh) run against `wind_up_v0` with
      only the import changed, and the README says so.
- [ ] `wind_up_v0/constants.py` no longer depends on the repo's root folders (`config/`,
      `input_data/`, `cache/`); those folders are deleted.
- [ ] If D1 = (b): the v0 binned method is selectable in a declaration (`method: v0_binned`) and
      one real analysis, Hill of Towie AeroUp T13, is run both ways with the differences
      explained (northing, reference choice).
- [ ] A gap table in the docs: what v0 reports that the v1 run does not yet (long-term AEP
      extrapolation, P95, per-pair detrend plots, combined results) and what to use meanwhile.

### R5. User feedback has been considered

From [discussion 101](https://github.com/resgroup/wind-up/discussions/101):

- [ ] **An explicit, machine-checked input contract.** One documented statement of required
      columns, units, sign conventions, timestamp rules (timezone, interval, which end is
      stamped) and valid ranges, validated before any analysis with messages that name the
      column and the rule. `ColumnSchema`, `TimestampConvention` and the full turbine x
      timestamp grid cover part of this; units and ranges are not checked.
- [ ] **Data description separate from method options.** Columns, rated power and the reference
      frame sit in the data part of the declaration; method options sit in the `method:` block
      (R3).
- [ ] **Validated on open datasets, and said so.** Hill of Towie, Kelmarsh and Penmanshiel
      loaders exist; decide which ship (as `wind_up.datasets` or as examples) and state in the
      docs which datasets the release was validated on.
- [ ] **Tests run without Git LFS.** The Hill of Towie, SMARTEOLE and WeDoWind test data are LFS
      files. Confirm a clone without LFS runs `poe test-fast` with those tests skipped, not
      failed, and say so in the contributing notes.
- [ ] Reply on the discussion with what was adopted and what was not.

From [analyst_feedback.md](analyst_feedback.md), triage every entry to `done`, `1.0.0`, or
`after 1.0.0`. Proposed for 1.0.0, because they change the interface or are cheap:

- [ ] AF1: the reference-relative estimate and its standard error as columns of
      `farm_uplift.csv` (a stable output file; add the columns before freezing it).
- [ ] `rated_power_kw` documented and validated against the data.
- [ ] `northing_corrections.yaml` gets a header, units and sign convention.
- [ ] `timing.mode` given a mapping reports that it must be one of two strings.
- [ ] Document the reference screen's rule and what "reads near 0 %" means.

### R6. Benchmarking is tidied: what is kept moves to `src/`

| stays in `benchmarking/` (repo only) | moves to `src/wind_up/` |
|---|---|
| `synthetic/` (generator, faults, upgrades, ground truth, sources) | `ColumnSchema`, `ToggleSchedule` (from `synthetic/`) |
| `harness/` scoring, metrics, leaderboard, replicates, calibration, campaign sweep, plots | `harness/` method seam, context, conditions, toggle, record selection, operating state, reanalysis, northing, waking, wind rose, label checks |
| `baselines/` study drivers, fixtures, `prepost_matrix/`, `old/`, overnight helpers | `baselines/` toggle energy ratio, power model, v0 binned, block bootstrap, filtering |
| `campaigns/` placebo, rollout, the `*_fixture` and probe modules, handover | `campaigns/` declaration, loader, plans, preparation, data screen, run, composed, report, plots, CLI |
| | `diagnostics/` (the toggle energy ratio method and the composed run both write through it) |

- [ ] Move in dependency order (types, seam, shared steps, methods, campaign run, CLI), one PR
      each, tests moving with the code under `tests/wind_up/`.
- [ ] `src/` rules apply to moved code: behaviour-only docstrings, no markdown emphasis, no
      finding numbers, keyword-only arguments, full test coverage. the toggle energy ratio method's and
      the seam's docstrings need this pass.
- [ ] `python -m benchmarking.campaigns run` becomes a `wind-up` console script
      (`wind-up run campaign.yaml`, `wind-up design ...`).
- [ ] Delete `benchmarking/baselines/old/` and study drivers whose findings are recorded.
- [ ] `benchmarking/` imports the product from `wind_up`; the frozen benchmarks are re-run after
      the move and are bit-identical.

### R7. Documentation and examples

- [ ] `README.md` rewritten for v1: install, a five-minute toggle example and prepost example,
      the stable/provisional table, the v0 migration note.
- [ ] `docs/methodology.md` from `docs/v1/method.md`, with the validity domain and the analysis
      warnings; the v3 PDF is marked as describing the v0 method.
- [ ] A toggle energy ratio page: inputs, the two legs, the outputs, a worked example on open data.
- [ ] At least one v1 example per mode that runs from a fresh install on open data.
- [ ] Tracked docs no longer cite untracked paths (`docs/superpowers/`, `CLAUDE.md`).
- [ ] A `CHANGELOG.md` starting at 1.0.0 with the stability policy above.

### R8. Merge and release mechanics

- [ ] Land or close the open stack onto `v1`: PRs 169, 170; 172 after R3. Close or rebase the
      stale PRs against `main` (24, 56) and the dependabot bump.
- [ ] `v1` merges to `main` through PR 100 (372 files); `main` is tagged `v0.7.x` first and a
      `v0` maintenance branch is left for fixes.
- [ ] `pyproject.toml`: version 1.0.0, classifier `Production/Stable`, a `[project.scripts]`
      entry, extras reviewed (`era5`, `ml`, `examples`), `toml`/`seaborn`/`tabulate` kept only
      if still imported, Python 3.14 considered.
- [ ] CI green on 3.10-3.13 including the wheel smoke test; `poe all` (with slow tests) passes
      once on the release commit.
- [ ] The publish workflow's header says to merge to `main` first; pre-releases are GitHub
      Releases marked as pre-release with `v1` as the target.
- [ ] `1.0.0rc1` installed by one outside project and one fresh environment before `1.0.0`.
- [ ] Release notes: what v1 is, the migration from 0.x, what is stable and what is provisional.

## Not required for v1.0.0

These continue after the release and do not block it:

- C3's remaining items (accepting a prepost baseline, the reference sigma, the step 15 bins
  near cut-in), C4-C6, R5-R9, W1b's verdict on the composed method.
- The R-learner headline and any other new method.
- Long-term AEP extrapolation (method step 17) and a P95 model in the v1 run.
- G6 report generation beyond what the composed run already writes.
- AF2, AF3, AF5, AF6, AF7 and the unranked asks not listed under R5.
