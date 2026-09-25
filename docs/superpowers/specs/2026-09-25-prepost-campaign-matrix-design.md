# Prepost campaign matrix — design

Date: 2026-09-25. Branch: `v1-C3`. Status: agreed in brainstorming, awaiting written-spec review.
Untracked (specs stay untracked).

## Purpose

Before fixing F2 (reference turbines reading positive under prepost), gather evidence across many
C3-like campaigns so the fix is not tuned to the single real HoT T13 anecdote. The evidence is a
committed prepost benchmark — the prepost counterpart of `study_power_model_compare_baseline.json`
— that runs **shipped `wind-up` with every default** (reference screen on, northing discovered from
scratch) over the real HoT T13 campaign and many synthetic trial-then-rollout campaigns on Hill of
Towie, Penmanshiel and Kelmarsh. It measures two things:

1. **Reference readings reach 0% reliably** (truth 0) — the headline concern.
2. **Test-turbine uplift does not degrade** (truth known on synthetic campaigns), including whether
   errors are symmetric in the sign of the uplift.

Declaring such campaigns must be easy and realistic, so the campaign declaration matures first.

## Decomposition

One spec, three sub-projects, each with its own implementation plan, built and verified in order:

1. **Campaign timeline and period selection** — the timeline part of C8, plus automatic
   per-test-turbine analysis spans and nearest-K power references.
2. **Synthetic rollout generator** — realistic trial-then-rollout works schedules.
3. **Prepost benchmark matrix** — the HPC-friendly study driver, metrics and committed baseline.

Out of scope: C8's vocabulary sweep ("upgrade" → "change", retiring "treated"), change naming in
titles, and the harness "window" disambiguation. They stay on C8 as a follow-up. The F2 fix itself
(the self-weighting reversal, `2026-09-17-reversal-shrinkage-design.md`) is judged against this
benchmark afterwards; it is not part of this work. A laptop-sized subset run is a later addition.

---

## Sub-project 1: Campaign timeline and period selection

### What the analyst declares

The analyst states which turbines to analyse and supplies data. The only dates they must give are
**works windows**. Everything else is optional.

```yaml
name: hot_aeroup_t13
data:
  scada: data/scada.parquet
  schema: hill_of_towie
  turbines: data/turbines.csv
  works: data/works.csv            # optional; per-turbine works windows
timing:
  mode: prepost
turbines:
  upgraded: [T13]                  # the turbines to analyse
  rated_power_kw: 2300.0
exclusions:                        # optional
  - {turbine: ALL, start: 2022-09-03T00:00:00Z, end: 2023-02-12T00:00:00Z}
# analysis_period: optional; omitted means wind-up chooses a span per test turbine
```

- **Works table** (`works.csv`): the Zenodo shape — `Turbine`, `First date of AeroUp works`,
  `Last date of AeroUp works`. The loader reads a `Turbine` column plus the first column whose name
  starts `First date` and the first whose name starts `Last date`. Dates are whole days: the window covers `[first 00:00, last + 1 day 00:00)`
  UTC. Any turbine may appear, analysed or not. A turbine may appear more than once (several
  windows).
- **An analysed turbine's changeover** is the end of its works window. Every analysed turbine must
  have exactly one works window. `timing.changeover` stays accepted as the degenerate case: one
  shared changeover, no works table, as today.
- **Exclusions** are `(turbine | ALL, start, end)`, end exclusive.
- **`analysis_period`** becomes optional. Declared at campaign level, or per analysed turbine
  (`analysis_period: {T13: {start, end}}`), it fixes the span and bypasses selection.
- **`references`** keeps its meaning: a restriction of the candidate list. When a span is declared,
  a declared reference is used as a power reference **even if its works window overlaps the span**,
  with a warning in the report. This is how an analyst forces compromised references in.
- Adding the Greenbyte schema to the loader's `SCHEMAS` so Penmanshiel and Kelmarsh campaigns are
  declarable.

### What `CampaignSpec` carries

Public facts only, as now:

- `works: dict[str, list[tuple[Timestamp, Timestamp]]]` — works windows per turbine.
- `exclusions: list[tuple[str | None, Timestamp, Timestamp]]` — `None` means farm-wide.
- `analysis_period` becomes `tuple | dict[str, tuple] | None`.
- `timing_for(turbine)` returns that turbine's changeover (its works end), or the shared changeover.
- `usable_mask(turbine, index)` masks the turbine's own works windows and its exclusions, and
  farm-wide exclusions. `run.visible_mask` already applies it. What masking means per role is below.

`SyntheticCampaign` gains the same `works` and `exclusions`. Injection starts at each upgraded
turbine's own works end; `generate_dataset` takes a per-turbine timing map instead of one timestamp
(the single timestamp remains the degenerate case).

### What masking means per role

| Masked rows of | Effect |
|---|---|
| The test turbine (its works, its exclusions) | Rows dropped from fit and prediction. |
| A power reference (its exclusions) | Power and direction NaN over the period; see the power_model switch below. |
| Any turbine (farm-wide exclusion) | Rows dropped for everyone. |
| A turbine's works window | Masked; the turbine is ineligible as a power reference for any span the window overlaps. |

### Period selection (`src/wind_up/analysis_period.py`)

A pure, public function of the layout, the works table, the exclusions, each turbine's data extent,
and settings. For each analysed turbine `t` with works window `[a_t, b_t)` it returns an
`AnalysisPlan`: the span `[s, e)`, the power references, the waking-only turbines, the pre and post
lengths, whether the pool rule is met, and a plain-language reason for every turbine dropped.

- **Eligible power reference** for span `[s, e)`: a turbine other than `t` with data in the span
  and no works window intersecting `[s, e)`. A turbine changed entirely before `s` contributes
  post-change data; entirely after `e`, pre-change data. Either is fine. Other analysed turbines
  are eligible on the same terms.
- **Pool rule** (reused from `campaign_design`, same defaults): among the 4 nearest turbines to
  `t`, at least 3 are eligible and within 20 D, and the nearest is eligible.
- **Power references**: the nearest `K` eligible turbines within 20 D (`K` default 4). Every other
  turbine is waking-only for this test turbine (the existing `wake_only` channel).
- **Lengths** are usable time: calendar time on each side minus the test turbine's masked rows.
- **Candidate edges** — the search is exact over these: `s` from the data start, `a_t − pre_cap`,
  and each other turbine's works end before `a_t`; `e` from the data end, `b_t + post_cap`, and each
  other turbine's works start after `b_t`. Candidates are clipped to the data extent.
- **Order of preference** (lexicographic):
  1. the pool rule is met, with both sides at least `min_side` (3 months);
  2. maximise `min(pre, post)`, capped at 12 months;
  3. maximise `post`, capped at 12 months;
  4. maximise `pre`, capped at `pre_cap` (24 months);
  5. the tightest worst power-reference distance.
- **Degrading gracefully**: if no candidate meets step 1, drop the pool-rule requirement, pick by
  steps 2–5, set `pool_rule_met = False`, and record why (in the manner of `why_no_design`). If no
  candidate gives even `min_side` on both sides, raise with the reason.
- **Override**: a declared `analysis_period` fixes `[s, e)`; power references are still the nearest
  `K` eligible, plus any declared reference forced in as above.
- Settings (`pre_cap`, short-side cap, `min_side`, `K`, pool-rule parameters) are keyword
  arguments with the defaults above.

### Wiring into the shipped run

- The runner resolves an `AnalysisPlan` per analysed turbine and passes its span, power references
  and waking-only turbines to the method through `CampaignContext`. The run window for that turbine
  is the plan's span.
- **Reference readings** (`reference_stability`): each candidate reference is read across the test
  turbine's changeover, over the test turbine's span, with its own nearest `K` eligible power
  references; the test turbine is waking-only for it.
- **The screen** screens the power references. An ejected reference becomes waking-only and the pool
  is refilled from the next-nearest eligible turbine within 20 D, then re-screened, so it stays at
  `K`. If refilling runs out, the pool shrinks and the report says the pool rule broke.
- The report writes each plan: span, pre and post lengths, power references with distances,
  waking-only turbines with the reason each is not a power reference, and `pool_rule_met`.

### power_model change: waking channels for references with exclusions

Today only `power_free` and `wake_only` turbines get the `waking` / `normal_operation` booleans, in
place of their power. New switch on `PowerModelMethod`:
`exclusion_channels: Literal["booleans", "nan"]`.

- `"nan"` — today's behaviour: every column of the reference is NaN over its exclusions.
- `"booleans"` — a power reference with any exclusion also carries its waking and normal-operation
  booleans over the whole record, derived from its raw data; only its power and direction are NaN
  over the exclusions.

The default is provisional (`"booleans"`) until the benchmark A/B decides it (see sub-project 3).

### The real HoT T13 declaration

Declared from the Zenodo works table and the curtailment exclusion in the tracked
`tests/test_data/hot/HoT_AeroUp_T13.yaml`, with no custom build script. It replaces
`c3-prework/build.py`'s hand-cut gap, truncated post and blanked months.

### Tests (TDD)

- Loader: works table parsing, per-turbine `analysis_period`, forced references, exclusions,
  Greenbyte schema; the flat changeover still loads unchanged.
- `usable_mask` per role; `generate_dataset` staggered injection (truth starts at each turbine's own
  works end).
- Period selector on small hand-built layouts: every preference step decides at least one test;
  the long-span/near-reference trade-off; the graceful fallback and its reason; the override; `K`.
- Real T13 behaves sensibly: the chosen span keeps the nearest neighbours as power references and
  ends before their works start. The exact expected span is recorded after the first run and
  reviewed with the user, not asserted beforehand.
- power_model switch: both settings, column presence, and that NaNs appear only over exclusions.
- Existing campaign tests and recorded benchmarks: unchanged for a flat declaration.

---

## Sub-project 2: Synthetic rollout generator (`benchmarking/campaigns/rollout.py`)

### The works schedule

Settings: number of teams, working days per turbine (a range), rollout start, turbine order, seed.

- **Working days**: weekdays, excluding UK bank holidays (England and Wales rules, Easter from
  `dateutil.easter`) and a Christmas shutdown from 24 December to 2 January inclusive.
- Each team works one turbine at a time and takes the next turbine in the order as soon as it
  finishes, so at most `n_teams` turbines are in works at once. Works start and end on working
  days.
- Working days per turbine are drawn uniformly from 5–12, inspired by the shorter HoT windows.
- Output: a works table in the Zenodo shape, so synthetic and real campaigns share the declaration
  path.

### A trial-then-rollout campaign

From a seed and a site:

1. **Trial turbines** from `design_campaign`: the count is drawn from 1 to max−1 (never below 1),
   the rule the placebo uses. The design's test turbines are the analysed turbines.
2. **Trial works**: teams drawn from {1, 2}; the start date is drawn so that 12–24 months of source
   data lie before it and at least 12 months after the last trial works end.
3. **Optional full rollout**: every other turbine is worked, starting 6–9 months after the last
   trial works end (a uniform draw), with its own team draw.
4. **Injected uplift**: the AeroUp shape — a region-2 Cp gain tailing to 0 by rated —
   `WindSpeedCpChange(ws_points=(4, 6, 8, 12, 16), deltas=(0.0, 0.07, 0.07, 0.045, 0.0))`, times a
   multiplier of +1, 0 or −1. Applied to trial turbines and, when present, full-rollout turbines,
   from each turbine's own works end.
5. **Post length**: the data is cut at the last trial works end plus `L` months, for `L` in 1, 2, 3,
   6, 9 and 12. The period selector works within whatever data remains.

Sources and periods: HoT 2016–2020 (no real AeroUp in the record), Penmanshiel and Kelmarsh over
the span their sources provide. Kelmarsh's 6 turbines may allow only 1 trial turbine every time;
that is accepted.

### Tests

The calendar (a known bank holiday and the shutdown are skipped); team concurrency never exceeds
`n_teams`; works start and end on working days; the pre-data bounds hold; the rollout lag bounds
hold; the same seed gives the same campaign; the works table round-trips through the loader.

---

## Sub-project 3: Prepost benchmark matrix (`benchmarking/baselines/study_prepost_campaign_matrix.py`)

### The matrix

- **Synthetic campaigns**: 8 seeds per site (HoT, Penmanshiel, Kelmarsh). Seeds alternate between
  with and without a full rollout (4 each).
- **Real**: HoT T13, with the post cut at 2021-09-30 plus `L` months.
- **Axes**: multiplier ∈ {+1, 0, −1} (synthetic only; paired, so the design, dates and seed are
  identical across the three), `K` ∈ {3, 4, 6}, `L` ∈ {1, 2, 3, 6, 9, 12} months.
- **One run** is one (campaign, multiplier, K, L): a full shipped `wind-up` run. Nothing is shared
  between runs. That is 3 sites × 8 × 3 × 3 × 6 = 1296 synthetic runs plus 18 real ones.
- **The exclusion-channel A/B** (`"booleans"` vs `"nan"`) runs as a side arm at `K` = 4 only, on
  campaigns given synthetic reference exclusions: each seed draws 0–2 exclusions of 3–14 days per
  turbine.

### Running on the HPC

- `plan` writes the cell list (one line per run, a stable cell id).
- `run-cell <id>` runs one cell and writes `cells/<id>.json`, with the metrics' raw inputs and the
  diagnostics. It is idempotent: an existing result is skipped. Suited to a job array.
- `run-local --workers N` runs the same cells through a process pool.
- `merge` builds `candidate_baseline.json` from the cell files; `compare` diffs it against the
  committed `study_prepost_campaign_matrix_baseline.json`; `--accept-candidate` promotes it, as in
  `study_power_model_compare`.
- **Reanalysis** is pre-fetched once into the cache by a `prefetch` step, so cells run offline.
- **Determinism**: each worker runs LightGBM single-threaded and deterministic; parallelism is
  across cells. Whether that makes a baseline recorded on the HPC reproduce on the laptop is
  verified during the build (run a few cells on both and compare), not assumed. If it does not,
  the baseline stays machine-specific and the file says so, as the existing one does.

### What each cell records

Per run: each analysed turbine's estimate and truth (truth from the existing `truth_mask` over the
plan's post span), the farm uplift and its truth, every reference reading, and diagnostics: each
plan (span, lengths, power references, waking-only turbines, `pool_rule_met`), screen ejections,
northing changepoints found, and wall time.

### The committed baseline

Keyed by (site, K, L), plus a pooled `all` site. Real T13 rows are separate.

**Test turbines**, per multiplier, over the campaigns in the key:
- `bias`, `spread`, `score` on the per-turbine headline (the existing definitions);
- the same three on the farm uplift;
- **linearity**: per campaign and analysed turbine, fit `estimate = m · truth + c` across the three
  multipliers; report the mean and sd of `line_scale` (m, ideal 1) and `line_bias` (c, ideal 0).

**References** (truth 0), per multiplier, over every reference reading in the key:
- `mean` (the F2 headline), `median`, `sd`, `worst_abs`, `n`.

**Real HoT T13**: the reference metrics per (K, L); T13's own headline recorded for drift only.

**Leak check** (diagnostic, in the merge log): per reference, the largest change in its reading
across the three multipliers of one campaign. The upgrade should not move a reference's reading.

Pass/fail targets (for instance a reference mean within the 0.127 pp noise floor) belong to judging
the later F2 fix, not to recording the baseline.

### Tests

Cell ids are stable and unique; `run-cell` is idempotent; merge on synthetic cell files computes
each metric correctly, including `line_scale`/`line_bias` on a constructed exact line; compare flags
a moved cell; a tiny end-to-end run (one small campaign, one K, one L) completes under `slow`.

---

## Open questions (settle during the plans)

- The exact AeroUp Cp deltas: the shape above is a starting point; check its injected energy truth
  lands at a plausible AeroUp size (a few percent) before recording.
- Wall time per run on the HPC, and whether the matrix needs splitting across job arrays.
- Whether the pool rule's "nearest is eligible" is too strict on Kelmarsh's small layout, forcing
  the fallback everywhere there. If so, report it rather than loosen the rule.
