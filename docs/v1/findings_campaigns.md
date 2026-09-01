# wind-up v1 — campaign findings log

Empirical findings from the realistic whole-farm campaigns tranche
([issues_campaigns.md](issues_campaigns.md)). Newest first. Each entry records what was
observed, the evidence, the root cause, and what it implies for the method design or the
issues list.

Entries are numbered **CF*n*** — a separate series from [findings.md](findings.md)'s
**F*n***, which belongs to the earlier, back-burnered effort.

Keep entries reproducible: name the driver and the exact configuration, not just conclusions.

---

## CF5 — T06 is the failure-mode fixture turbine: `power_model` holds it to 0.34% mean error with a 0.72 pp swing across seven windows while `naive_ratio` averages 3.3% and swings **13.3 pp**. T12 is the trap — stable but biased, with no headroom at all

*2026-09-01. Reproduce: `placebo_campaign(mode="prepost", upgraded=..., turbines=HOT_TURBINES)`
swept over seven windows (12- and 24-month baselines into post years 2017–2020), all 21 HoT
turbines, the six test candidates T07/T11/T12/T06/T16/T19. Truth is 0 by construction, so every
reading is method error.*

**The criterion.** A good R-series fixture needs `power_model` accurate *and* stable, and
`naive_ratio` inaccurate *or* unstable. The second half matters: the gap is the advanced
machinery doing visible work, and a failure mode that only bites advanced methods (a north jump,
say) has nothing to break unless that work is happening.

**Per-turbine `power_model` error (%), truth 0:**

| wtg | 12→2017 | 12→2018 | 12→2019 | 12→2020 | 24→2018 | 24→2019 | 24→2020 | mean abs |
|---|---|---|---|---|---|---|---|---|
| **T06** | +0.36 | +0.31 | +0.32 | −0.14 | +0.47 | +0.21 | +0.58 | **0.34** |
| T07 | +0.17 | +0.68 | −0.23 | +2.12 | +0.57 | +0.00 | −0.40 | 0.60 |
| T11 | +1.14 | +0.18 | +0.80 | +0.45 | +0.67 | +0.73 | +0.19 | 0.60 |
| T12 | +0.60 | +0.62 | +0.61 | +0.66 | +0.39 | +0.47 | +0.62 | 0.57 |
| T16 | −0.33 | −0.44 | −0.19 | +2.57 | −0.05 | −1.49 | +0.38 | 0.78 |
| T19 | +1.64 | −0.66 | +0.66 | +2.83 | −0.48 | +0.11 | +1.06 | 1.06 |

**The same turbines under `naive_ratio` (%):**

| wtg | 12→2017 | 12→2018 | 12→2019 | 12→2020 | 24→2018 | 24→2019 | 24→2020 | mean abs |
|---|---|---|---|---|---|---|---|---|
| **T06** | −5.53 | **+7.81** | −2.21 | +0.84 | +4.70 | +1.51 | −0.20 | **3.26** |
| T07 | +2.21 | −1.68 | +0.26 | +0.75 | −0.59 | −0.60 | +0.87 | 0.99 |
| T11 | +3.34 | −2.20 | +1.82 | +0.57 | −0.57 | +0.68 | +1.40 | 1.51 |
| T12 | +0.04 | +0.05 | +1.64 | −0.12 | +0.07 | +1.66 | +0.62 | 0.60 |
| T16 | −2.68 | +0.64 | −2.06 | −0.17 | −0.75 | −1.74 | −1.11 | 1.31 |
| T19 | +0.28 | −0.51 | +0.01 | −1.90 | −0.37 | −0.25 | −1.90 | 0.74 |

**Verdict.** **T06** is the fixture turbine — the most accurate under `power_model` *and* the
worst under `naive_ratio`, a 2.9 pp gap and 5x the next best. **T11** is the credible second
(0.60% vs 1.51%), for an independent second case.

**Two turbines to avoid, for opposite reasons.** **T19** is the only turbine where `naive_ratio`
beats `power_model` outright (0.74% vs 1.06%). **T12** looks attractive — the tightest
`power_model` spread on site, 0.27 pp — but reads **+0.6% in every single window** while
`naive_ratio` sits near zero. Reproducing to within 0.03 pp across unrelated baselines and post
years makes that a *structural* per-turbine bias, not sampling noise: something about T12's
relationship to its references that the model does not capture. With no headroom over the naive
floor, a fault injected there would have to overcome the model's own bias to be visible.

**Caveat.** Six turbines over seven **overlapping** windows: the window results are not
independent, and every ranking below T06 rests on tenths of a percent against swings of similar
size. T06's margin is large enough to act on; the T07/T16/T12 ordering is not.

---

## CF4 — Window choice matters much less than expected once the turbine set is right: six of seven windows put every turbine inside 1.7%. The one bad window (`12mo→2020`) is a **thin-baseline** artefact, not a bad year — widening 2019 to a 24-month baseline cures it

*2026-09-01. Same sweep as CF5, read per window rather than per turbine.*

| window | max abs | mean abs | spread |
|---|---|---|---|
| **24mo→2018** | 0.674 | 0.437 | 1.15 pp |
| **12mo→2018** | 0.684 | 0.482 | 1.34 pp |
| 12mo→2019 | 0.803 | 0.467 | 1.03 pp |
| 24mo→2020 | 1.060 | 0.539 | 1.46 pp |
| 24mo→2019 | 1.492 | 0.502 | 2.22 pp |
| 12mo→2017 | 1.644 | 0.708 | 1.98 pp |
| **12mo→2020** | **2.827** | **1.458** | 2.96 pp |

**The anomaly, and what it is not.** `12mo→2020` puts three turbines above 2% (T07 +2.12, T16
+2.57, T19 +2.83) when no other window exceeds 1.7%. The obvious reading — that 2020 is a
pathological year — is **wrong**: `24mo→2020` ranks fourth of seven and is entirely usable. The
difference is the baseline, not the post period. A 2019-only baseline is too thin for 2020;
widening it to 24 months moves T07 to −0.40, T16 to +0.38 and T19 to +1.06.

**Implication.** The committed placebo window (12 months of 2017 into 12 months of 2018) is
second of seven and within 0.01 pp of the best, so it stays. The 24-month baseline buys ~0.2 pp
of spread for an extra year of data — not worth it by default, but it is the lever to reach for
when a window looks unexpectedly bad.

---

## CF3 — Reaching the ±0.2% farm target is mostly about **reference count**, not test count: going from 3 to 20 references halved `power_model`'s error before any cancellation, and six test turbines then took the farm result to **+0.148%**

*2026-09-01. Reproduce: prepost placebo, 12 months of 2017 into 12 months of 2018, test turbines
taken as the first n of (T07, T11, T12, T06, T16, T19), every remaining turbine a reference.*

| n_test | 1 | 2 | 3 | 4 | 5 | **6** |
|---|---|---|---|---|---|---|
| `power_model` farm | +0.68% | +0.44% | +0.50% | +0.46% | +0.32% | **+0.148%** |
| `naive_ratio` farm | −1.68% | −1.94% | −1.27% | +0.34% | +0.39% | +0.23% |

**Two distinct levers, and the first is the bigger one.** At `n_test=1` no cancellation is
possible, yet `power_model` already reads +0.68% against +1.0% for the earlier six-turbine
configuration — going from 3 references to 20 roughly halved the error on its own. Test count
then removes what remains, but only because the residual is **two-sided**: at n=6 the per-turbine
errors are +0.68, +0.18, +0.62, +0.30, −0.44, −0.66, four positive and two negative.

**A superseded hypothesis, recorded because it was wrong in an instructive way.** With T01+T04 as
the test pair, `power_model` read +0.72% and +1.27% — both positive — which looked like a
common-mode bias that no amount of pooling could cancel, and pointed at the method (adaptive time
decay was the suspect). It was an artefact of those two turbines. With a set spread across the
site the errors are two-sided and cancellation works. **Turbine selection, not the method, was
the problem.**

**`naive_ratio`'s farm number is not comparable.** It reaches +0.23% at n=6, but on a **10 pp**
per-turbine spread — cancellation of large opposing errors, not accuracy. `power_model`'s 1.34 pp
spread is the real quality difference.

**Scatter.** Window-to-window variation is ~±0.5% (CF4), so a single farm number carries real
luck. +0.148% should be read as "inside target at this window", not as a settled figure.

---

## CF2 — `naive_ratio`'s prepost placebo bias is **inherent, not fixable by window choice**: every turbine reads ±0.6–3.2% with no step change anywhere, driven by inter-annual wind-direction shift redistributing wake exposure

*2026-09-01. Reproduce: 2017 vs 2018 energy ratio of each turbine against the sum of the others,
availability-filtered, on the six-turbine (T01–T05, T07) configuration.*

**Per-turbine naive prepost estimate, truth 0:** T01 −0.68%, T02 −0.58%, T03 −3.16%, T04 +2.52%,
T05 +2.11%. An independent hand calculation gave T04 = +2.52% against the runner's +2.46%, which
also cross-checks the C1 pipeline against a from-scratch computation.

**Not a data event.** Monthly energy ratios show no step anywhere in 2017–2018 — instead a
*relative redistribution*: T04 runs hot all through 2018 (monthly mean ≈ +2.9%) while T03 runs
cold (≈ −2.9%), with March 2018 extreme for both (+11% / −11%) during the easterly "Beast from
the East" spell. Hill of Towie's real AeroUp works are all 2021–2023, so the 2017–2018 window is
genuinely upgrade-free and the placebo is a true placebo.

**Root cause.** `naive_ratio` conditions on **nothing** — not wind speed, not direction. Between
years the direction distribution shifts, changing each turbine's wake exposure relative to its
neighbours by a few percent. No campaign window removes that, because it is not seasonal
composition (see CF1) but inter-annual climate. Only conditioning fixes it, which is what
`power_model` and v0 do and what the C-series is for.

**Implication.** `naive_ratio` is the leaderboard floor by construction. Its placebo reading is
a measure of how much work conditioning has to do on this site, not a defect to be tuned away.

---

## CF1 — The placebo separates prepost from toggle by an order of magnitude, and seasonal composition explains part of the prepost gap: matching baseline and post to the same twelve months cut the per-turbine spread from 5.4 pp to 3.1 pp

*2026-09-01. Reproduce: `uv run python -m benchmarking.campaigns.placebo`, Hill of Towie, both
modes, `upgrades=[]` so truth is 0 by construction.*

**Truth is exactly 0.0 in both modes** — the C1 pipeline's first end-to-end confirmation on real
SCADA.

**Toggle beats prepost by an order of magnitude.** On the original six-turbine configuration
(T01/T04 upgraded, 3-month baseline into a 3-month post): prepost per-turbine +2.49% / −5.66%
with an 8.2 pp spread, against toggle −0.79% / +1.24% at 2.0 pp. Moving toggle to six months of
50-minute blocks tightened it further to −0.10% / +0.29%, **0.39 pp** — a 5x improvement, because
finer interleaving makes the on and off blocks sample the same weather.

**Seasonal composition is part of the prepost gap.** A full year of 2017 baseline against only
Jan–Jun 2018 leaves the two periods holding *different seasons*, confounded with the (absent)
effect. Extending the post period to a full 12 months so both span the same months of the year
moved T01 from −1.71% to −0.64% and T04 from +3.71% to +2.46%, cutting the spread from 5.4 pp to
**3.1 pp**. The residual is CF2's inter-annual effect, which this does not touch.

**Implication for the campaign shape.** Prepost runs a 12-month campaign on 12 months of
baseline; toggle needs only six months, since its blocks interleave within whatever period it is
given. This is why `PLACEBO_CAMPAIGN_MONTHS` is per-mode.
