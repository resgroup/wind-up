# wind-up v1 — findings log

Empirical findings from the v1 benchmarking work. Newest first. Each entry records what was
observed, the evidence, the root cause, and what (if anything) it implies for the method design
or the issues list. Keep entries reproducible: name the study driver and the diagnostics they came
from, not just conclusions.

---

## F3 — A simple counterfactual power model halves prepost bias and spread vs naive; toggle is a wash

*2026-06-29 — new method `power_model` (the simplest-possible ML method: a single LightGBM
counterfactual power model, design spec `docs/superpowers/specs/2026-06-29-power-model-design.md`).
Source: `benchmarking/baselines/example_{prepost,toggle}_study.py` run over the four
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
| prepost | naive | −1.08% / 1.65% | −0.60 / 1.19 | −0.38 / 0.95 | −0.45 / 0.81 |
| prepost | **power_model** | **−0.69% / 0.54%** | −0.23 / 0.51 | −0.23 / 0.55 | −0.21 / 0.43 |
| toggle | naive | +0.31% / 0.22% | +0.16 / 0.23 | +0.06 / 0.21 | +0.08 / 0.12 |
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
