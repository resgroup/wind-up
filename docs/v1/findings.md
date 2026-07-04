# wind-up v1 — findings log

Empirical findings from the v1 benchmarking work. Newest first. Each entry records what was
observed, the evidence, the root cause, and what (if anything) it implies for the method design
or the issues list. Keep entries reproducible: name the study driver and the diagnostics they came
from, not just conclusions.

---

## F13 — removal ablation: dropping the availability feature + five ERA5 columns improves the benchmark; low importance ≠ removable

*2026-07-04 — follow-up to the Issue 9–11 additions: the same A/B protocol run in reverse (remove each
currently-accepted feature group, keep any removal that noticeably improves the score). Two new
ablation knobs on `PowerModelMethod`: `era5_exclude` (drops raw ERA5 columns + their sin/cos
companions; guarded against excluding `matching_vars` while `conditional_uplift` is on) and
`availability_feature` (drops the per-reference availability *feature*; `availability_col` stays
required for the downtime filter). Screens on the placebo, confirmation via two full sweeps
(`--method-overrides` on `study_power_model_compare.py`), all diffed against the post-F12 benchmark.*

### The accepted removal set (now the driver default; benchmark regenerated)
`availability_feature=False` + `era5_exclude = CURATED_ERA5_EXCLUDE = (apparent_temperature,
dew_point_2m, precipitation, rain, snowfall)`. Full-sweep deltas (pp; negative = better):
- **prepost overall**: Δ|bias| −0.19, Δspread −0.18, Δscore **−0.24** — the largest overall
  improvement of the whole Issue 9–11 campaign, and it came from *removing* features. Per campaign
  (placebo): 3 mo bias −0.59 → −0.12, 12 mo −0.21 → **0.00**, 6 mo overshoots mildly (−0.17 → +0.30).
- **toggle**: overall neutral (≤0.006); conditional Δ|bias| −0.98, Δscore **−1.47**.
- **prepost conditional**: the one cost, Δscore +0.54 (cells split 212 better / 183 worse) — accepted
  against the overall-P50 gains (Phase 1 is judged on P50 accuracy/precision first).
- Availability-alone (set A) shows nearly the same numbers; the ERA5 trims add a small consistent
  extra (prepost conditional Δ|bias| +0.07 → −0.17 vs A). The removals stack cleanly — unlike the
  F12 max+min interaction.

### Why removing availability helps
References are almost always available, and when one is not, its *power* column already carries the
fact (0/NaN) — so the counter added noise and a mild maintenance-calendar proxy rather than wake
information. The curated-feature physical argument ("the model should know whether a reference is
waking") was measured and lost to the data.

### Kept columns — low importance is not a removal licence
Removing bottom-of-the-ranking columns often *hurt*: `pressure_msl` removal cost +2.09 pp prepost
conditional score (the model evidently uses the msl-vs-surface pressure pair jointly), `weather_code`
removal +0.49, `cloud_cover` removal +0.39 toggle conditional, humidity removal worse everywhere.
Together with F12's rank-3/rank-4 accepted/rejected split, the lesson is symmetric: importance rank
predicts neither a feature's value nor its removability — only the benchmark gates do.

---

## F12 — reference active-power **minimum** accepted as a default feature; SD and max rejected (Issue 11)

*2026-07-04 — Issue 11 verdicts. Candidates A/B'd one field at a time per the Issue 9 protocol:
`study_power_model_compare.py --method-overrides '{"reference_stat_cols": [...]}'` against the committed
benchmark — placebo (`cp_0pct`, both modes) screen first, full 7-profile sweep for survivors. The HoT
loader now also unpacks `wtc_ActPower_max` / `wtc_ActPower_min` (the SD was already loaded); the fields
reach the model via `build_reference_features(..., extra_cols=...)` /
`PowerModelMethod.reference_stat_cols`.*

### Verdicts (deltas vs the pre-change benchmark, in pp; negative = better)
- **`wtc_ActPower_min` — ACCEPTED, now the default** (`reference_stat_cols=("wtc_ActPower_min",)` in the
  HoT drivers). Better on every gate in both modes: overall P50 Δ|bias| −0.01 (prepost) / −0.006 (toggle),
  Δspread ~0 / −0.009; conditional mean Δ|bias| **−0.46 (prepost)** / **−0.75 (toggle)**, Δscore −0.40 /
  −1.30. Interpretation: the within-period minimum tells the model when a reference dipped (gust lulls,
  brief curtailments) — a farm-sited variability signal the mean hides. **Benchmark JSON regenerated**
  from this configuration's full sweep.
- **`wtc_ActPower_stddev` — REJECTED.** The placebo screen alone disqualified it: conditional mean
  Δ|bias| +2.8 / Δscore +3.98 (prepost), +0.65 / +1.38 (toggle), and it jumped to importance rank 5.
  The within-period power SD is exactly the kind of channel the placebo gate exists for.
- **`wtc_ActPower_max` — REJECTED.** Full sweep: it shifts the prepost overall bias **uniformly +0.33 pp
  across all seven profiles** — a counterfactual level shift, not uplift tracking. That happens to offset
  the structural −0.4 pp prepost headline bias (|bias| improves) but costs spread (+0.08; 240/469 cells
  worse) and is an accidental cancellation — Issue 13 addresses that bias properly. Toggle-side it helps
  (−1.13 conditional), but combined with `min` (trio minus SD) it is toxic: prepost conditional
  Δ|bias| +3.4 / Δscore +5.2. Max and min together destabilise the matched conditional fits.
- **Reference nacelle wind speed / wind-speed SD — rejected without trial** (recorded per the issue): a
  reference anemometer is at high risk of calibration drift, which a prepost campaign reads as uplift;
  reference *power* is the calibration-stable channel, and same-type references degrade like the test
  turbine, giving a fairer counterfactual expectation.

### Method note
Importance rank alone was a poor red-flag here: `max`/`min` both ranked 3rd–4th (gain frac 3–4%), yet one
was accepted and one rejected — the benchmark gates (placebo bias, spread, conditional cells) did the
discriminating, not the ranking.

---

## F11 — explicit time features rejected: campaign-drift, season and solar all fail or add nothing (Issue 10)

*2026-07-04 — Issue 10 verdicts. `benchmarking/baselines/time_features.py` ships the features
(`days_since_campaign_start`, June-21-anchored `season` sin/cos, NOAA `solar` altitude/azimuth validated
against an ephem reference to <0.01°) behind `PowerModelMethod.time_features` (+ `latitude`/`longitude`),
default **off**. A/B'd one at a time on the placebo (`cp_0pct`, both modes) via
`study_power_model_compare.py --method-overrides`.*

### Verdicts (placebo deltas vs benchmark, pp)
- **`days_since_campaign_start` — REJECTED.** Prepost is the anticipated failure, measured: overall
  Δbias +0.23, Δspread **+1.08**, Δscore +0.97; conditional Δscore +5.3 with 54/67 cells worse. Trees
  cannot extrapolate the feature past the changeover (every upgraded-row value exceeds the training
  range, so predictions clamp at boundary leaves), which *adds* variance instead of absorbing drift.
  Toggle is ~neutral (−0.001 overall) — but prepost and toggle share one code path, so it stays out.
- **`season` — REJECTED.** Prepost overall slightly worse (Δscore +0.14, max Δ|bias| +0.26 at one
  campaign length), conditional worse in both modes (+0.39 / +0.16). Against a <12-month baseline the
  pair is a partial calendar proxy, as the issue warned.
- **`solar` — REJECTED.** Neutral overall (≤0.03) but conditional worse in both modes (+0.90 / +0.29
  Δscore) and negligible importance (rank 22–24 of 32, gain frac ≤0.0003) — the instantaneous weather
  columns already carry the diurnal signal at this site.

### Method note
The feared "time feature dominating the importance ranking" red flag never fired — all three sat far
down the ranking (gain frac ≤0.0006) *while still doing damage through spread*. The placebo benchmark
gate, not the importance watch, is the effective detector for drift-importing features. The module stays
in the tree for future sources (e.g. a site with genuine reference drift may re-litigate
`days_since_campaign_start` in toggle-only form).

---

## F10 — ERA5 derived quantities: utility shipped; hub-height wind speed validated standalone; no derivation earns a default place in the full model (Issue 9)

*2026-07-04 — Issue 9 verdicts. `benchmarking/baselines/era5_derived.py` is the shared derivation
utility (shear exponent, hub-height ws via the shear power law + `hub_height_m` — `HOT_HUB_HEIGHT_M =
59.0`, gust ratio, gust margin, veer, moist-air density), reused by
`inspect_era5_matching_importance.py` and available to the CEM matching step; features reach the model
via `PowerModelMethod.era5_derivations`, default **off**. Screened per candidate on the placebo, full
sweeps for survivors (`study_power_model_compare.py --method-overrides`).*

### The gust "TI proxy" is not one (measured against real SCADA TI)
- `gust_ratio = wind_gusts_10m / wind_speed_10m` correlates with the test turbine's measured TI at
  **Pearson +0.03** (Spearman +0.11; T01, ws>4 m/s, n≈195k) and implies TI ≈ 0.31 at the median vs the
  real 0.17 — ERA5's hourly grid-scale gustiness is a different quantity from local 10-min turbulence.
- Variants (per the issue's "play around" instruction): the absolute **gust margin** `gusts − ws_100m`
  is the best simple correlate (+0.22), shear exponent −0.21, |veer| −0.21, `gusts/ws_100m` +0.11.
  A LightGBM fit of TI on *all* ERA5 columns reaches held-out **R² = 0.38**, dominated by wind-direction
  sin/cos (~31% of gain — wake/terrain sectors) — much of site TI is direction-determined and the model
  already sees direction.

### A/B verdicts (vs benchmark, pp)
- **`wind_speed_hub` — validated, left opt-in.** With reference features removed (ERA5-only ranking, the
  Issue 9 exploration) it *dominates*: 63% of gain, permutation importance 0.48 vs 0.10 for raw
  `wind_speed_100m`. In the full model its full sweep improves overall P50 in both modes (prepost
  Δ|bias| −0.02, Δspread −0.035 uniformly across profiles; toggle conditional −1.02 score) at a small
  prepost conditional cost (+0.13). But combined with the accepted `wtc_ActPower_min` its marginal value
  disappears (combo no better than `min` alone, toggle conditional diluted), so it is **not defaulted**;
  it is the natural candidate for the F6 `matching_vars` revisit and for reference-poor sources.
- **`shear_exponent`, `veer` — REJECTED (mode-split).** Both help toggle conditional (−0.41 / −0.47
  score) and hurt prepost (+0.80 / +0.54); one code path, so out.
- **`gust_ratio`, `gust_margin`, `air_density` — REJECTED.** Overall neutral; conditional worse
  (gust_ratio prepost +1.33 score with the (6, ws) cell +5.9; gust_margin +0.34/+0.23; air_density
  +0.60 prepost, 25/35 cells worse). Consistent with the TI-proxy result: these columns add split noise,
  not cause.

### Interpretation
The reference active-power features already carry the site signal ERA5 derivations try to reconstruct —
in the full model every derivation lands at gain frac ≤0.0012. ERA5 derivations matter where references
are absent: the ERA5-only fit (R² 0.85) is where `wind_speed_hub` shines, which is exactly the CEM
matching / AEP-extrapolation context (Issues 8/15), not the counterfactual feature set.

---

## F9 — the matched two-direction conditional cross-prediction shipped as the sole conditional method, on by default

*2026-07-03 — Issue 8 ship. The F7/F8 development-time A/B flag `bias_correct` is **removed**; the
matched two-direction cross-prediction is now the sole conditional-uplift path, controlled by
`PowerModelMethod.conditional_uplift: bool = True` (**default on**). Current helpers:
`PowerModelMethod._estimate_conditional` (ERA5 match + forward/reverse fits) and `_conditional_by_bin`
(the re-leveled per-bin shape); the pure re-level helper `_relevel_conditional` is unchanged. Re-run
via `study_power_model_compare.py` (Issue 7), which overlays the committed benchmark vs the current run
vs truth per covered `(profile, condition)`.*

### What shipped (supersedes the F8 "still opt-in" decision)
- **Default flip.** F8 left the correction opt-in pending an A/B verdict; that verdict came in and it
  became the default. `conditional_uplift=False` still skips the expensive cross-prediction and returns
  overall-P50 only, so the opt-out remains for the ERA5-less / overall-only configuration.
- **Overall P50 unchanged, by construction.** The headline is still the single full-window fit
  (F8): the ship is bit-identical on overall P50 (max |Δ bias| ≤ 1e-4 pp both modes) — the correction
  is spent only on the per-condition decomposition.
- **Conditional accuracy roughly halved.** Over the condition-dependent + placebo profiles, mean per-bin
  |bias| falls **prepost 18.2 → 6.3 pp**, **toggle 13.0 → 4.1 pp** (score prepost 22.6 → 10.2, toggle
  18.2 → 8.0); ~87% of covered bins improve. The remaining worse bins are the rare sparse tails (tiny
  counts, both methods noise) — the F7 sparse-extreme overshoot, left unfloored by choice (F8).

### Packaging
- **One run folder**, not four: conditional CSVs under a `conditional/` subfolder, the implied-shrinkage
  diagnostic under `plots/7_conditional_uplift/`. The `implied_shrinkage` diagnostic stays on the public
  surface; the "bias correct(ed)" naming is gone.
- **Benchmark JSON regenerated** under the new default; `docs/v1/issues.md` Issue 8 updated to match.

### Implications / follow-ups (carried from F7/F8)
- A per-reporting-bin matched-count floor for the sparse-extreme overshoot (the handful of worse bins).
- Density-ratio weighting (WS4) to enlarge the short-campaign matched set.

---

## F8 — the self-consistent correction: keep the uncorrected full-data headline, re-level the matched decomposition onto it — overall now no worse

*2026-07-02 — Issue 8, the fix for F7's headline cost. Same A/B command
(`study_power_model_compare --modes prepost --profiles cp_0pct ti_dependent_cp ws_dependent_cp
--bias-correct`) and artefacts as F7. Estimator reworked in `PowerModelMethod._estimate_bias_corrected`
/ `_corrected_conditional`; new pure helper `_relevel_conditional`; `energy_ratio_by_bin` now also
returns per-bin `sum_actual` / `sum_counterfactual`.*

### The estimator (final)
- **Overall = the uncorrected full-data estimate.** Train on **all** baseline, predict **all** upgraded,
  one energy ratio — *identical* to the `bias_correct=False` headline (a unit test asserts exact
  equality). The whole-window shrinkage integrates to ≈ 0 (F5), so this is already the cleanest overall;
  the correction is spent only on the decomposition.
- **Per-bin = the matched two-direction shape, re-leveled onto the headline.** The shape
  `1+u_b = sqrt((1+r_fwd_b)/(1+r_rev_b))` still comes from the **CEM-matched** forward/reverse fits
  (matching is required — without it the reverse model predicts out-of-distribution across the prepost
  weather shift). Each condition is then rescaled by one factor `λ_c = (1+overall)/(1+u_agg)` whose
  **weights are the full-upgraded per-bin energy**, so the reported per-bin MWh partitions the full-data
  headline exactly ("overall = aggregation" self-consistency).

### Why not the two earlier variants (both measured)
Getting here took ruling out two tempting overalls, both worse than uncorrected:
- **Global `sqrt`-combine** (the F7 estimator): mean overall |bias| **0.49 pp** — not self-consistent
  (three separate nonlinear reductions) and still worse than uncorrected at short campaigns.
- **Matched forward-only** `r_fwd_all`: mean overall |bias| **0.82 pp** — *worse still*. The premise
  "the matched forward recovers the overall because shrinkage integrates out" was wrong: on the ≈ 11%
  matched subset the forward model is not energy-conserving, so `r_fwd_all` keeps a residual `(1+u)/s`
  shrinkage (~+1 pp at 3 mo). The `sqrt`-combine had actually been *cancelling* that. Lesson: the
  matched subset can't beat the full-data overall for the headline; only the per-bin *shape* needs the
  matched cross-prediction.

### Result
- **Overall: no worse — identical to uncorrected** at every profile × campaign (|Δ bias| and |Δ spread|
  ≤ 1e-4 pp, float noise). Contrast: uncorrected 0.35 pp, F7 combine 0.49, forward-only 0.82.
- **Per-bin de-tilt preserved:** 57 better / 9 worse / 3 ~ of 69 covered cells; mean per-bin |bias|
  ≈ 14 pp → ≈ 8.5 pp — the F5 tilt is gone across the populated range, and the ws & ti decompositions
  now energy-aggregate back to the (unchanged) headline.
- The 9 "worse" cells remain the sparsest TI/ws extremes (overshoot), left **unfloored** by choice; the
  re-level keeps them from moving the headline, so it is a tail display issue, not a headline bug.

### Decision / implications
- **`bias_correct` becomes a pure decomposition refinement on an unchanged headline.** The correction
  never touches the P50 — it only re-attributes the same total MWh across bins with the shrinkage tilt
  removed. That is a safe, Pareto-neutral property for the overall.
- **Still opt-in (`bias_correct=False` default);** whether to make it the default is a later call.
- **Follow-ups:** a per-reporting-bin matched-count floor for the sparse-extreme overshoot (one-line);
  density-ratio weighting (WS4) to enlarge the short-campaign matched set; a toggle-mode A/B (prepost is
  the F5/F7/F8 case reported here).

---

## F7 — the two-direction bias correction removes the F5 per-bin tilt but costs overall P50 at short campaigns, so it stays opt-in

*2026-07-02 — Issue 8, Component 6 A/B run. Command:
`study_power_model_compare --modes prepost --profiles cp_0pct ti_dependent_cp ws_dependent_cp --bias-correct`.
The committed benchmark is the **uncorrected** frozen run, so the run's before/after machinery reads as
corrected ("current") vs uncorrected ("benchmark") vs truth. Artefacts under the run's `comparison/`:
`conditional_before_after_<profile>_<condition>.png` (per-bin overlays),
`conditional_benchmark_comparison_prepost.csv` (per-bin |bias| verdict) and `benchmark_comparison_prepost.csv`
(overall). Single-case overlays also reproduced via `inspect_prepost_hard_case.py` (now runs uncorrected +
bias-corrected side by side).*

### Observation — the per-bin conditional bias is largely removed (the F5 target)
Across the two condition-dependent hard cases plus the placebo, at the 12-month campaign the correction
flattens the per-bin uplift toward truth in the large majority of bins: **57 better / 9 worse / 3 neutral of
69 covered cells**, mean per-bin |bias| ≈ **14 pp → ≈ 8.5 pp**. The F5 tilt is gone across the populated
range — e.g. `cp_0pct` placebo `ti (0.30,0.35]` 17.6 → 0.04 pp and `ws (2,4]` 45.6 → 12.7 pp;
`ws_dependent_cp` `ti (0.30,0.35]` 18.7 → 0.17 pp. The implied shrinkage the correction cancels is ≈ 0.99
overall on these cases (a small overall compression that integrates to ≈ 0, exactly as F5 predicted).

### Observation — the cost: overall P50 degrades at short campaigns
The correction is **not** free on the headline number. Overall |bias| / spread (pp), corrected vs uncorrected:

| campaign | corrected \|bias\| | uncorrected \|bias\| | Δ\|bias\| | Δ spread |
| --- | --- | --- | --- | --- |
| 3 mo  | 0.66–0.78 | 0.18–0.63 | **+0.06 … +0.16** (worse) | ≈ +0.6 (worse) |
| 6 mo  | 0.56–0.73 | 0.18–0.19 | **+0.38 … +0.54** (worse) | ≈ +0.05 (worse) |
| 12 mo | 0.05–0.10 | 0.23–0.24 | **−0.14 … −0.18** (better) | ≈ −0.15 (better) |

So the done-when "overall P50 no worse" holds **only at 12 months**; at 3/6 months the correction adds a
few tenths of a pp of bias and spread.

**Mechanism — finite-sample cost of a nonlinear two-ratio combine on a shrunken matched set.** The corrected
overall is a *different estimator*, not the same one on less data: the uncorrected path is one energy ratio
`Σactual/Σcf − 1` over **all** upgraded rows from **one** model trained on **all** baseline rows; the corrected
path is `sqrt((1+r_fwd)/(1+r_rev)) − 1` from **two** models trained on the **CEM-matched subset**. Two things
compound at short campaigns: (1) the matched subset collapses — the short upgraded window can only match a
sliver of the abundant baseline, so from the per-run CEM balance the fraction of baseline actually used falls
to **≈ 11% at 3 mo** (~11k rows/side) vs **≈ 47% at 12 mo** (~47k), i.e. each counterfactual model trains on
~9× less data at 3 mo; (2) the two noisier per-direction ratios are combined through a **nonlinear** function,
so by Jensen's inequality the expected combined value is offset from the noise-free value, and that offset
**grows as the inputs get noisier**. The campaign-length signature is the tell: a change of *estimand* (matching
to a different weather mix) would be roughly campaign-independent, but this cost **vanishes as data grows**
(worse at 3 mo → better than uncorrected at 12 mo) — the fingerprint of a finite-sample effect. The uncorrected
overall pays none of this and is already clean because the shrinkage integrates to ≈ 0 over the whole window
(F5): **the correction spends precision fixing a per-bin problem the headline never had.**

### The failure mode — sparse extreme bins overcorrect
All 9 "worse" per-bin cells are the sparsest condition extremes (lowest-TI `(0.0,0.05]`, highest-TI
`(0.40,0.50]`, lowest-ws `(2,4]`). There the two-direction ratio is estimated on very few matched rows, so
the correction overshoots — e.g. `ti (0.45,0.50]` swings −76.7 → +92.9 pp. This both drags the mean per-bin
|bias| up (hence ≈ 8.5 pp, not lower) and, via the extreme bins, perturbs the overall energy ratio at short
campaigns. Some extreme bins also drop to `NaN` (the non-positive-ratio / thin-bin guard), visible as gaps
in the single-case overlays.

### Decision / implications
- **Stays opt-in (`bias_correct=False` default); no default flip.** It decisively fixes the per-bin
  decomposition but is not a strict Pareto improvement on the overall P50, so it is not ready to be the
  default — exactly the A/B outcome the opt-in design was built to allow.
- **Candidate follow-ups** (being taken up next): the defect has two distinct sources needing two distinct
  levers, since the overall is a **global** energy ratio (sparse reporting bins carry little weight in it, so a
  per-bin fix does **not** touch it):
  1. **Per-bin extreme overshoot** → a **minimum matched-count floor per reporting bin**, below which that bin
     falls back to the uncorrected estimate — directly kills the sparse-bin overshoot (the 9 "worse" cells).
  2. **Overall short-campaign cost** → report the **uncorrected single-direction estimate as the headline P50**
     (F5: the whole-window shrinkage integrates to ≈ 0, so the two-direction combine only adds finite-sample
     noise there), keeping the two-direction correction for the per-bin *decomposition* where the shrinkage does
     **not** integrate out. This makes "overall no worse" hold by construction.
  Density-ratio weighting instead of hard CEM subsampling (earmarked for WS4) would additionally reclaim the
  short-campaign matched-sample size.
- Toggle mode not yet A/B'd; prepost is where F5 was diagnosed and is the case reported here.

---

## F6 — the ERA5 matching variables for Issue 8 bias-cancellation are `wind_speed_100m` + `wind_gusts_10m` + `wind_direction_100m`

*2026-07-02 — Issue 8, Component 1 matching-variable analysis. New one-off script
`benchmarking/baselines/inspect_era5_matching_importance.py` ranks the ERA5-only fields by how well
they predict the test turbine's real (un-upgraded) power, using LightGBM gain and held-out sklearn
permutation importance. Run on `T01`, ~253k normally-operating rows over the default 2016–2020 HoT
window; outputs (`feature_importance.png`, `predicted_vs_actual.png`, `era5_matching_importance.csv`)
under `<study-output-root>/inspection_era5_matching`.*

### Observation
The ERA5→test-power model predicts well (held-out **R²=0.84**, RMSE ≈ 287 kW), so the ranking is
trustworthy, not just precise. Wind-magnitude fields dominate: `wind_speed_100m`, `wind_speed_10m`
and `wind_gusts_10m` together carry ≈ 89% of the gain; every direction / thermodynamic field is
< 2% on both views. But `wind_speed_10m` and `wind_speed_100m` are strongly collinear, so **gain and
permutation disagreed** on the 2nd variable (gain favoured `wind_speed_10m`, permutation favoured
`wind_gusts_10m`) — the collinearity makes neither view a clean guide.

### What was done
Folded the two collinear speeds into one physical vertical-shear exponent
`alpha = ln(ws_100m / ws_10m) / ln(100/10)` (a stability / turbulence proxy that directly attacks the
F5 cause) and dropped `wind_speed_10m`. Held-out fit was **unchanged** (R²=0.844), confirming the two
speeds were substitutes, and with the redundancy gone **gain and permutation agree cleanly**:
`wind_speed_100m` ≫ `wind_gusts_10m` ≫ `wind_shear_exponent`, then a flat tail. The shear exponent is
a real independent signal (~2.5% on both views) but an order of magnitude below the two magnitude
fields at HoT.

### Decision
- **Matching set = `("wind_speed_100m", "wind_gusts_10m", "wind_direction_100m")`.** The first two are
  the top of the importance ranking (once the 10m/100m redundancy is folded out); **wind direction is
  added on physical grounds** — it governs the wake geometry between the test turbine and its
  references, so omitting it would leave a first-order confounder unmatched even though its *marginal*
  importance for predicting power is small.
- **Shear exponent deferred.** It ranks 3rd on importance and modest, and adopting it properly means
  adding the derivation to the method's real feature path (`power_model/features.py`), not just the
  analysis script where it currently lives. Left for another day; the shear derivation stays in
  `inspect_era5_matching_importance.py` only and changes no scored method or benchmark.

### Bin widths — verified on real HoT by the coverage sweep
A CEM coverage/sensitivity sweep (T01 prepost split at 2018-06-01, ~121k baseline / ~132k upgraded
normally-operating rows, via `benchmarking.baselines.power_model.matching.coarsened_exact_match`)
confirmed 3-var matching is affordable despite the nominal cell explosion — HoT weather concentrates
(prevailing SW ≈ 240°), so occupied cells are a small fraction of nominal and retention stays high:

| ws / gust / dir | one-sided dropped | matched/side | retained base / up |
| --- | --- | --- | --- |
| 2 / 3 / 30° | 101 | 111,166 | 91.7% / 84.4% |
| **2 / 3 / 20°** | **143** | **109,313** | **90.1% / 83.0%** |
| 2 / 3 / 10° | 271 | 105,511 | 87.0% / 80.1% |
| 1 / 3 / 5° | 950 | 95,930 | 79.1% / 72.9% |

**Chosen widths: `wind_speed_100m` = 2 m/s, `wind_gusts_10m` = 3 m/s, `wind_direction_100m` = 20°.**
ws = 2 m/s (generalises fine over that band and retains a touch more than 1 m/s); direction = 20°
because `wind_direction_100m` is a *reanalysis* direction — spatially smooth and coarse, so finer than
~20–30° is finer than the signal supports, and 5–10° roughly 3–7×'s the dropped one-sided cells for
little directional gain. 20° keeps ~90%/83% retention with ~120 matched rows per two-sided cell.

### Implications
Component 3 hard-codes the method's `matching_vars` default to the 3-var set and `matching_bin_edges`
to `{wind_speed_100m: 0..32 @ 2, wind_gusts_10m: 0..44 @ 3, wind_direction_100m: 0..360 @ 20}` (fixed
sectors, no wraparound — adjacent sectors are just separate cells, per Component 2). Per-farm re-tuning
and a proper shear feature are later steps.

---

## F5 — power_model's condition-dependent uplift error is the counterfactual model's own conditional bias (shrinkage), not a §3 post-treatment-conditioning artefact

*2026-07-01 — first result off the conditional-uplift instrument (per-(ws, TI)-bin scoring; see
`benchmarking/harness/conditions.py`, `scoring.py`, `PowerModelMethod._conditional_uplift`).
Diagnosed with a new per-segment residual diagnostic
`benchmarking/baselines/power_model/diagnostics.py::_plot_residual_binned` (writes
`residual_binned.png` and `residual_binned_pct.png` under `5_uplift_modelling/`), regenerated via
`benchmarking/baselines/inspect_prepost_hard_case.py`. The pinned case is the `cp_0pct` placebo on
`T07`, 6-month prepost, true uplift 0% — so in both the held-out baseline and the "upgraded" window
the residual is pure model error, an unusually clean read on model bias.*

### Observation
`power_model`'s overall P50 is excellent (F3/F4), but its **per-bin** uplift decomposition is badly
distorted at the condition extremes. On `ti_dependent_cp` the recovered uplift slopes to ≈ −75 pp in
the highest TI bin where the truth is roughly flat; on `ws_dependent_cp` the (2,4] m/s bin reads
≈ −30 pp against a +17 pp truth. The overall estimate is unaffected because these errors integrate
to ≈ 0.

### Evidence — three facts that localise the cause
1. **Not a §3 / binning-axis problem.** The synthetic upgrades for `ti_dependent_cp` and
   `ws_dependent_cp` use `ws_delta = 0` (`ws_factor = 1.0`) and never modify `wind_speed_sd`
   (`generator.py` `modified_columns = active_power, gen_rpm, wind_speed`). So the test turbine's
   **measured** ws/TI under treatment equals its **original untreated** ws/TI exactly — the method
   already bins on the same treatment-invariant axis the ground truth uses. (Binning the estimate on
   a reference-derived ws/TI instead would therefore change nothing; confirmed by reasoning, not
   pursued. Ground-truth binning was left unchanged.)
2. **Not confounding.** Toggle (concurrent reference, ≈ no temporal confounding) shows the *same*
   per-bin distortion shape as prepost.
3. **It is model shrinkage / conditional bias.** On the placebo, `residual_binned.png` shows the mean
   residual (actual − predicted) tilting from ≈ −25 kW at low power to ≈ +80 kW at high power — the
   tilt survives the Bland-Altman `mean(actual, predicted)` axis, so it is a real conditional bias,
   not just the errors-in-variables inflation from binning by actual. Across TI the same residual runs
   ≈ +30 kW → −38 kW (zero-crossing at TI ≈ 0.17), exactly the shape of the fake TI-uplift. The
   baseline and upgraded residual curves overlay (as they must at truth 0).

### Root cause
A regularised learner minimises squared error, so its prediction is pulled toward the conditional
mean: with imperfect features it **predicts smoother than reality** — over-predicts where power is low,
under-predicts where it is high (predicted-vs-actual slope < 1). The §3 rule forbids the test
turbine's own ws/TI as features, so the counterfactual model carries no direct turbulence
information; its residuals are therefore correlated with TI. Because power maps monotonically to wind
speed, and TI is inversely related to wind speed at fixed power, this single compression re-appears as
a negative residual at low ws / high TI and positive at high ws / low TI. The compression averages to
≈ 0 over the whole window (so the headline uplift is clean), but slicing the energy ratio
`Σactual / Σcounterfactual` **by condition** re-exposes the conditional bias as a spurious
condition-dependent uplift.

A **second, separate** failure mode inflates the visible excursions at the extremes: the low-ws /
high-TI bins hold very little energy, so a small kW residual over a tiny `Σcounterfactual` becomes a
huge *percentage*. The `residual_binned_pct.png` view (each bin's residual as a % of that bin's mean
power) makes this explicit — the well-populated bins sit within ±5–15% while the tiny-power bins blow
out to −160% (ws 5 m/s) / −330% (TI 0.35). So the extremes combine real conditional bias with ratio
instability.

### Interpretation
The per-bin instrument is faithfully measuring **model bias**, not a defect in the harness or the
conditioning axis. The overall-P50 verdict of F3/F4 stands; what F5 adds is that *conditional* P50 is
only trustworthy where (a) the counterfactual model's conditional bias is small and (b) the bin holds
enough energy for a stable ratio.

### Implications / candidate directions (not yet actioned)
1. **Reduce the counterfactual model's conditional bias** — the core lever. Leading candidate:
   **baseline-residual calibration** — estimate the model's per-condition mean residual `b(cond)` on
   untreated data (prepost baseline / toggle off-blocks, where the true uplift is 0) and subtract it
   from the upgraded per-bin ratio. On the placebo the baseline residual curve *is* an estimate of
   that bias and overlays the upgraded curve, so this should flatten the conditional-uplift curves
   toward truth. To be designed before coding.
2. **Give the model a treatment-invariant turbulence proxy** (each reference's own sd/ws, or ERA5
   gust/spread) so the counterfactual can learn the TI–power relation without touching the treated
   signal (§3-legal). Partial: only as good as the proxy's correlation with local TI.
3. **Guard the fragile tails** — suppress or flag bins below an energy/count floor; orthogonal to
   1–2 and fixes only the ratio-instability half.
4. **Reporting/tooling shipped this cycle:** the two `residual_binned*` diagnostics (shared y per row;
   percentage version normalised by each bin's own mean power, with the shared y-axis sized from bins
   within ±30% so tiny-power outliers clip rather than crush the scale). Widening TI bins 5%→10% was
   tried to tame the plot and **reverted** — it did not address the underlying bias and the y-axis
   sizing is the better fix.

---

## F4 — power_model beats v0 in prepost and at longer-toggle; v0's edge is only short-toggle, and v0 alone breaks on rated-power uprates

*2026-06-30 — the three-way comparison F3 flagged as the open question (power_model vs v0 in
*both* modes). Source: `benchmarking/baselines/study_power_model_compare.py`, which re-runs
**only** `power_model` over the current overnight cases and merges it with the frozen `v0_binned`
+ `naive_ratio` rows from the overnight run (`study_overnight_{prepost,toggle}.py`,
`include_v0=True`). Seven `overnight_profiles` (`cp_minus_10pct`, `cp_0pct`, `cp_plus_3pct`,
`cp_plus_10pct`, `ws_dependent_cp`, `ti_dependent_cp`, `rated_plus_5pct`), `n_replicates=4`,
seed 0; prepost campaigns 3/6/12 mo (84 cases/method), toggle 3/6/9/12 mo (112 cases/method). The
script's alignment guard confirmed all 84 + 112 fresh cases match the reference run's
method-independent ground truth exactly, so the merge compares identical cases.*

All numbers below are percentage points of fractional uplift (a 0.01 fraction = 1 pp). Bias = mean
signed error, spread = std of signed error, RMSE pooled over all cases for that mode/method.

### Observation — pooled over all profiles and campaign lengths

| mode | method | bias | spread | MAE | RMSE | within ±1pp | per-case win |
|---|---|---|---|---|---|---|---|
| prepost | naive_ratio | −1.11 | 4.62 | 3.83 | 4.73 | 17% | 0% |
| prepost | **power_model** | **−0.39** | **0.49** | **0.53** | **0.62** | **86%** | **73%** |
| prepost | v0_binned | −0.73 | 0.72 | 0.84 | 1.03 | 62% | 27% |
| toggle | naive_ratio | +0.15 | 0.16 | 0.19 | 0.22 | 100% | 39% |
| toggle | power_model | +0.16 | 0.19 | 0.20 | 0.24 | 100% | 22% |
| toggle | v0_binned | −0.01 | 0.33 | 0.23 | 0.32 | 98% | 38% |

("per-case win" = share of the N cases where that method has the smallest |error|.) In prepost,
**power_model's |error| is smaller than v0's in 73% of cases and smaller than naive's in 100%**.

### Observation — RMSE by campaign length (the short-data story, serves G2)

| mode | method | 3mo | 6mo | 9mo | 12mo |
|---|---|---|---|---|---|
| prepost | naive_ratio | 7.32 | 3.19 | — | 1.84 |
| prepost | **power_model** | **0.82** | **0.46** | — | **0.53** |
| prepost | v0_binned | 1.22 | 0.94 | — | 0.89 |
| toggle | naive_ratio | **0.30** | 0.23 | 0.17 | 0.14 |
| toggle | power_model | 0.38 | **0.22** | **0.18** | **0.11** |
| toggle | v0_binned | 0.32 | 0.31 | 0.33 | 0.32 |

The two structural facts: **(a)** in prepost power_model leads at every length, its biggest margin
at 3 months (0.82 vs v0 1.22 vs naive 7.32); **(b)** in toggle, power_model and naive both tighten
with more data (power_model 0.38 → 0.11), but **v0 does not improve with campaign length** — it
sits at ~0.32 RMSE from 3 to 12 months. So v0 only wins the shortest toggle campaign; from 6
months on, power_model is best in toggle too.

### Observation — profile spotlight (pooled over campaigns)

| profile | method | bias | RMSE | max |error| |
|---|---|---|---|---|
| prepost `rated_plus_5pct` | **power_model** | **−0.38** | **0.62** | **1.05** |
| prepost `rated_plus_5pct` | v0_binned | −1.24 | 1.42 | 2.23 |
| toggle `rated_plus_5pct` | **power_model** | +0.16 | **0.25** | **0.49** |
| toggle `rated_plus_5pct` | v0_binned | −0.63 | 0.68 | 1.10 |

power_model is **flat across all seven profiles** (prepost RMSE 0.61–0.63, bias ≈ −0.38 on every
one), whereas **v0 has a specific weak spot on the rated-power uprate** — its worst profile in both
modes (the only profile where v0's toggle RMSE, 0.68, is more than ~2× its others). A rated-power
change shifts power at high wind speeds where v0's binned power-curve has sparse, noisy bins;
power_model's continuous reference-conditioned fit has no such blind spot. On the placebo
(`cp_0pct`) all three are well-behaved (toggle v0 even edges power_model, 0.18 vs 0.24 RMSE).

### Interpretation
- **Prepost: power_model is the better method, decisively.** Lower bias (−0.39 vs −0.73 pp), lower
  spread (0.49 vs 0.72), ~40% lower RMSE than v0, and it wins the majority of cases head-to-head —
  the F1 contrast lever (expected power through the references) cancelling the common-mode drift
  that v0 corrects only through its detrend step. naive is not in contention (covariate shift).
- **Toggle: a near-tie that tips to power_model with data.** Naive and power_model are
  near-identical and both beat v0 overall on RMSE; v0's larger spread and its failure to improve
  with longer toggling are the cost of its binning. v0's only advantage is the 3-month toggle
  campaign, where power_model carries slightly more bias (+0.37) before its variance collapses.
- **v0's rated-power weakness is the clearest single result.** It is the one regime where v0 is
  both biased and high-variance in *both* modes, and where power_model's flatness is most valuable.

### Implications
1. **Answers F3's open question: power_model ≥ v0 in both modes for P50** — strictly better in
   prepost and at toggle ≥ 6 months, with v0 ahead only at the shortest toggle campaign. It is now
   the baseline to beat (G-level), not just vs naive.
2. **Short-toggle bias is power_model's one soft spot** — the +0.37 pp at 3-month toggle is the
   thing to chip at next (mirrors the residual prepost bias noted in F3 #3); candidates are the
   baseline-horizon / recency weighting already on the list.
3. **Add a rated-power-uprate case to any v0 regression framing** — it is v0's worst regime and a
   natural demonstrator for power_model's advantage; worth a dedicated diagnostic.
4. Reproduce/extend with `study_power_model_compare.py` (`--skip-run` to re-merge, `--modes` to
   restrict); it reuses the frozen slow v0 so each power_model iteration is cheap.

---

## F3 — A simple counterfactual power model halves prepost bias and spread vs naive; toggle is a wash

*2026-06-29 — new method `power_model` (the simplest-possible ML method: a single LightGBM
counterfactual power model). Source: `benchmarking/baselines/example_{prepost,toggle}_study.py`
run over the four
`example_profiles` (`constant_cp`, `wind_speed_cp`, `ti_cp`, `rated_power`), `n_replicates=4`,
campaign sweep 3/6/9/12 months, seed 0, scoring **naive + power_model** (oracle anchor; v0 off).
256 runs (64 per mode per method); the oracle's max |signed error| was 0.0, confirming harness
wiring. Single-case cross-check: `benchmarking/baselines/inspect_prepost_hard_case.py`.*

`power_model` fits the test turbine's power on curated reference-only features (each reference's
active power + availability, all raw ERA5 columns) over the baseline and predicts the
counterfactual over the upgraded window: `uplift = sum(actual)/sum(counterfactual) − 1`. It is the
contrast lever F1 recommended (expected power expressed *through* the references), now realised by
a much simpler estimator than the R-learner — no propensity, no cross-fit.

### Observation — bias (mean signed error) and spread (std signed error), fractional uplift
Overall, by campaign type:

| mode | method | bias | spread | MAE |
|---|---|---|---|---|
| prepost | naive_ratio | −0.63% | 1.20% | 0.97% |
| prepost | **power_model** | **−0.34%** | **0.54%** | **0.53%** |
| toggle | naive_ratio | +0.15% | 0.22% | 0.21% |
| toggle | power_model | +0.16% | 0.18% | 0.20% |

By campaign length (the short-campaign story, serves G2):

| mode | method | 3mo | 6mo | 9mo | 12mo |
|---|---|---|---|---|---|
| prepost | naive_ratio | −1.08% / 1.65% | −0.60 / 1.19 | −0.38 / 0.95 | −0.45 / 0.81 |
| prepost | **power_model** | **−0.69% / 0.54%** | −0.23 / 0.51 | −0.23 / 0.55 | −0.21 / 0.43 |
| toggle | naive_ratio | +0.31% / 0.22% | +0.16 / 0.23 | +0.06 / 0.21 | +0.08 / 0.12 |
| toggle | power_model | +0.36% / 0.08% | +0.12 / 0.18 | +0.10 / 0.17 | +0.05 / 0.10 |

(bias / spread per cell). Both methods are **flat across the four upgrade types** — in prepost
power_model holds bias ≈ −0.34% and spread ≈ 0.55% on *every* profile (naive ≈ −0.6% / 1.2% on
every profile), so neither method has a profile-specific weak spot.

Single hard case (F1's placebo: `cp_0pct`, `T07`, 6-month prepost, truth 0%): power_model reads
**−0.18%** where the cross-fit R-learner was ~−14% biased (naive +0.58%, v0 ~0%).

### Interpretation
- **Prepost is the clear win.** power_model roughly **halves both bias and spread** vs naive
  (spread 1.20%→0.54%, MAE 0.97%→0.53%), and its edge is largest at short campaigns — 3-month
  prepost spread 0.54% vs naive's 1.65% (~3× tighter), degrading far more gracefully as data
  shrinks. Expressing expected power through the references cancels the common-mode seasonal/
  long-term drift that naive's raw pre/post ratio carries — the F1 contrast lever, delivered by a
  method far simpler than the R-learner (which *amplified* prepost error instead).
- **Toggle is a wash.** Both methods are near-unbiased with ~0.2% spread. On interleaved on/off
  blocks there is little covariate shift for the model to correct, so the ML adds nothing over the
  naive ratio. power_model carries a touch more bias at 3-month toggle (+0.36%) but the tightest
  spread (0.08%).

### Implications
1. **power_model is the new baseline to beat** for prepost, and a credible drop-in for toggle.
   It is wired into both example study drivers and `inspect_prepost_hard_case.py` in place of the
   R-learner (which is kept as a comparator but no longer invested in).
2. **This is vs naive, not yet vs v0.** v0 was excluded for speed. The honest open question
   (G-level) is whether power_model beats v0 in *both* modes — v0 held prepost bias to ~−0.006 to
   −0.014 (F1), comparable to power_model's −0.0034 here, so a direct v0 vs power_model prepost
   run is the next comparison. Toggle is where neither naive nor the R-learner beat v0, so a v0
   toggle comparison is the priority there.
3. The small residual prepost bias (~−0.3%, slightly negative at every length) is worth a look —
   plausibly mild non-stationarity or filter asymmetry between the long baseline and the post
   season; a candidate for the future season-matched / recency-weighted baseline horizon.

---

## F2 — The prepost R-learner bias is driven by reactive-power and pitch reference features acting as calendar-time proxies

*2026-06-26 — Issue 5 (cross-fit R-learner). Source: the ablation driver
`benchmarking/baselines/inspect_prepost_feature_ablation.py`, which pins the F1 hard case
(`cp_0pct` placebo on `T07`, 6-month prepost, true uplift 0%) and the identical `MethodInput`,
then re-runs the R-learner with reference features removed before feature-building. Follows up F1.*

### Observation
On the F1 placebo case, dropping the reference **reactive-power** feature, then the **pitch**
features as well, removes almost all of the bias (truth = 0%, fixed seed, only the feature set
changes between arms):

| arm | dropped tags | R-learner estimate | error |
|---|---|---|---|
| full (all features) | — | **−22.26%** | −22.26% |
| no reactive power | `wtc_ReactPwr_mean` | **−5.81%** | −5.81% |
| no reactive power, no pitch | `wtc_ReactPwr_mean` + `wtc_PitcPos{A,B,C}_mean` | **−1.39%** | −1.39% |

Reactive power alone accounts for ~16 of the ~22 points of bias; adding pitch removes most of the
rest, leaving the placebo within ~1.4% of zero.

*(The full-feature estimate is −22.3% here vs the ~−14% quoted for this case in
`inspect_prepost_hard_case`'s docstring. The docstring predates the `mandatory availability filter`
commit `f8c3491`, which changed row selection; the cross-arm comparison is internally consistent
regardless of the absolute level.)*

### Interpretation — direct evidence for F1's overlap-failure root cause
F1 attributes the prepost bias to an overlap/positivity failure: the propensity model reconstructs
"is this the upgraded season?" from seasonally/temporally varying reference features. This ablation
localises *which* features carry that signal. Reactive power and pitch are the top propensity
features (F1 diagnostics), and the reactive-power diagnostic plots show its **control regime changes
over calendar time** — so it is a near-deterministic clock. Remove that clock and the propensity
model can no longer separate the long baseline from the upgraded season, so the `t_res → 0`
blow-up and the confounding it drives both shrink. This is the F1 root-cause #1 mechanism shown
end-to-end, with reactive power identified as the dominant temporal proxy and pitch as secondary.

**Caveat:** this is a diagnosis, not a fix. Dropping informative features only removes the proxy
*channel*; any reference feature with a time trend (or a post-treatment correlation) can re-open it,
and discarding genuinely predictive signal is the wrong long-term lever. The principled fixes remain
F1's: a test-vs-reference contrast (so common-mode temporal drift cancels before the ML sees it)
and/or a season-matched baseline window.

### To pick up Monday
1. **Isolate pitch** — run the missing arm (drop pitch only, keep reactive) to split their
   contributions cleanly; the current run only brackets them.
2. **Confirm it generalises** — repeat across the other hard cases / a couple of seeds / the
   non-placebo profiles (does removing the proxies also tame the +76% `cp_plus_10pct` overshoot?).
3. **Watch the propensity diagnostic** — re-check `propensity_std` per arm; the hypothesis predicts
   it falls back toward the flat base rate as the temporal proxies are removed.
4. **Decide the lever** — feed this into the F1 direction choice: feature hygiene/guarding vs the
   test-vs-reference contrast. The contrast is still expected to be the larger, more principled win.

---

## F1 — The R-learner is accurate in toggle but fails in prepost (overlap/confounding)

*2026-06-26 — Issue 5 (cross-fit R-learner). Source: the overnight studies
`benchmarking/baselines/study_overnight_{toggle,prepost}.py` over the seven shared
`overnight_profiles`, `n_replicates=4`, campaign sweep 3/6/(9)/12 months, incl. v0.*

### Observation
In **toggle** the R-learner is excellent — it tracks the oracle and matches or beats v0. In
**prepost** it is badly biased, and the error grows with the magnitude of the true effect (it
amplifies). Mean P50 estimate vs truth, 3-month campaign:

| profile | truth | toggle rlearner | prepost rlearner |
|---|---|---|---|
| cp_0pct (placebo) | 0.000 | +0.003 | **−0.158** |
| cp_plus_3pct | 0.020 | 0.023 | **−0.010** |
| cp_plus_10pct | 0.068 | 0.072 | **+0.763** |
| cp_minus_10pct | −0.068 | −0.066 | **−0.359** |
| ws_dependent_cp | 0.033 | 0.036 | +0.123 |

Toggle bias is ~0.001–0.003 with tiny spread; prepost overshoots (+10% → +76%, −10% → −36%) and
even the 0% placebo reads −16%. Longer campaigns shrink but do not fix it (cp_plus_10pct prepost:
0.76 → 0.33 → 0.19 at 3/6/12 months). By contrast v0 holds prepost bias to ~−0.006 to −0.014.

### Evidence — the propensity diagnostic
From the per-run `results` / `data_stats` CSVs (same test turbine, same upgrade window):

- **Toggle:** `propensity_mean ≈ 0.500`, `propensity_std ≈ 0.11`. Baseline and upgraded both span
  the same dates (interleaved 20-on/20-off), so they share a weather distribution. Propensity ≈ 0.5
  everywhere → the treatment residual `t − e_hat` is healthy → the R-learner collapses to clean
  regression adjustment. This is the regime it is designed for (design note §4).
- **Prepost:** `propensity_mean ≈ 0.11`, `propensity_std ≈ 0.27`. The `data_stats` show why:
  **baseline ≈ 2 years (2016→2018), upgraded ≈ one 3-month season (early 2018)**. Treatment is a
  deterministic function of calendar time, against a long, season-mismatched baseline.

### Root cause
This is an identifiability problem, not a bug. There are no timestamp features (by design — shuffled
K-fold cross-fitting assumes it), so in prepost the only contrast available is across time. That
breaks the estimator two ways, both visible in the diagnostics:

1. **Overlap / positivity failure.** The propensity model partially reconstructs "is this the
   upgraded season?" from seasonally-varying reference features (`std 0.27`, far from the flat 0.11
   base rate). Where `e_hat` drifts toward the upgraded window, `t_res → 0`, the pseudo-outcome
   `y_res / t_res` blows up, the `t_res²` weights concentrate on a few high-leverage rows, and the
   effect model extrapolates — producing the variance explosion and the magnitude-scaling bias.
2. **Unmodelled non-stationarity → confounding.** The outcome model `m(x) = E[Y|X]` is pooled over
   the long baseline but never forms a *test-vs-reference contrast*. Any drift in the test turbine's
   power relative to the references over those two years (seasonal `Y|X` shifts, air density, icing,
   direction/wake differences between a winter-heavy baseline and the specific post season) lands in
   the post-window residual and, because treatment ≡ time, is attributed straight to `tau`. The
   placebo −16% is pure confounding with no real effect.

v0 survives prepost precisely because it differences the test/reference power ratio and detrends,
cancelling the common-mode drift the R-learner leaves in.

### Implications / candidate directions (not yet actioned)
In rough order of expected leverage:

1. **Give the R-learner a test-vs-reference contrast** (the v0 lever): model the test/reference
   power *ratio* (or difference) as the outcome so common-mode seasonal/long-term drift cancels
   before the ML sees it. Largest expected win for prepost.
2. **Match the baseline window to the post window** (same season, comparable length) rather than
   pooling the full multi-year history — reduces non-stationarity, though it does not restore
   within-`X` overlap.
3. **Treat the R-learner as a randomised-treatment (toggle) method.** Consistent with the design
   note framing (it collapses to regression adjustment when the propensity is flat), the honest
   Phase-1 conclusion may be: R-learner for toggle, v0/reference-ratio for prepost — with the
   prepost overlap failure documented as a finding.

Relative to Issue 5's "done when" (P50 similar/better than v0 in both prepost and toggle): **met for
toggle, not met for prepost.**

### Secondary observations from these runs
- **Stale outputs intermixed.** The earlier output directories carried `leaderboard_all_profiles.csv`
  and several per-profile files from an older study (no rlearner rows, old profile names, a 9-month
  grid) alongside the fresh overnight outputs. Addressed by writing each run to a fresh timestamped
  folder with a `run.log` recording the git commit and study config
  (`benchmarking/baselines/overnight_common.py`, wired into both overnight scripts).
- **Both overnight runs were truncated** (ran out of wall-clock on the slow v0 step, not a crash):
  prepost completed 5 of 7 profiles, toggle 6 of 7. With `include_v0=True`, v0 dominates the budget
  (~hours per profile vs seconds for rlearner/naive) — see the "skip v0 in initial passes" note.
