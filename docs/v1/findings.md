# wind-up v1 — findings log

Empirical findings from the v1 benchmarking work. Newest first. Each entry records what was
observed, the evidence, the root cause, and what (if anything) it implies for the method design
or the issues list. Keep entries reproducible: name the study driver and the diagnostics they came
from, not just conclusions.

---

## F33 — A per-bin cell with 1-2 records reports a **confidently wrong** sigma (coverage 0.158, and one cell reported sigma **exactly 0** while being 14 pp out). The bootstrap cannot estimate a variance from records that share a block — so below 3 records per side it must report NaN, not a number

*2026-07-16. Prompted by the question "what does a bin with only one or two data points do?" — a case
F29/F31/F32 had all bucketed away. Reproduce: the 256-replicate `cases.csv`, per-bin cells binned on
`min(n_upgraded_records, n_baseline_records)`.*

**The failure.** Measured coverage by the cell's *thinner* side (either side starves the ratio):

| min(n_on, n_off) | 1 | 2 | 3-4 | 5-7 | 8-11 | 12-20 | 21-50 | >50 |
|---|---|---|---|---|---|---|---|---|
| coverage | **0.158** | **0.237** | 0.579 | 0.656 | 0.718 | 0.724 | 0.676 | 0.680 |
| SE from target | **-3.0** | **-2.5** | -0.7 | -0.2 | +0.3 | +0.6 | -0.2 | -0.3 |
| median sigma [pp] | 0.90 | 2.34 | 3.82 | 4.88 | 4.90 | 3.95 | 3.01 | 1.07 |
| median \|error\| [pp] | 7.18 | 4.16 | 2.77 | 2.89 | 2.31 | 2.09 | 1.78 | 0.63 |

At one record per side sigma is **8x too small**; **8 cells reported sigma exactly 0.0** with a median
error of 10 pp and a worst of 22.6 pp. A reader would see "uplift -14.3% ± 0.0%".

**Why, and why it is not merely imprecision.** The bootstrap resamples whole **blocks**. If a cell's
records sit inside one block, a resample either includes that block `k` times or not at all — and the
ratio is `k*test / k*ref`, which is **independent of `k`**. Every finite resample returns the
identical uplift, so the spread is zero. The bootstrap does not lose precision here; it reports
certainty. That is the worst possible failure mode for a number whose entire job is to say how much
to trust an estimate.

**Why the earlier rounds missed it, which is the more useful lesson.** F29/F31/F32 all bucketed
per-bin coverage at `(0, 30]` records, which averages to a reassuring **0.623** and hides a 0.158
subset inside it. Worse, `calibration_summary` requires `sigma > 0`, so it counted the exact-zero
cells as `n_unusable` and **excluded the very worst cases from the coverage metric** — the statistic
was structurally blind to the failure it most needed to see. `n_unusable` was reported in every table
and never analysed. **A metric that drops its own pathological cases will always look calibrated.**

**The fix: a validity floor, not an additive term.** This *is* the data-count effect anticipated at
design time (and repeatedly "rejected" in F29/F31 by looking at the wrong resolution), but it is not
a term to add in quadrature — below the floor there is no variance estimate to correct, so the honest
output is NaN. Two complementary guards, both in `block_bootstrap`:

1. `min_records = 3` per side (`_MIN_RECORDS_PER_SIDE`) — set where coverage recovers: 1-2 are
   -2.5..-3.0 SE, 3-4 is -0.7 SE, 5+ is at target.
2. a degeneracy check — a resample spread of exactly zero, or a single block spanning the campaign
   (`n_blocks < 2`, which previously returned ~1e-15 float residue and reads as "±0.0 pp"), also
   reports NaN. This catches the pathology even if the floor is lowered.

`frac_resamples_finite` is still reported alongside, so NaN says "no" and the diagnostic says why.

**Scope: 1,917 of 82,944 per-bin cells (2.3%), of which 603 previously reported a finite — and
false — sigma. Zero headline cells are affected**, and no uplift changes (verified: the compare
reports both methods UNCHANGED).

**A tension worth naming.** The brief was that uncertainty is never optional. A NaN sigma is
formally "no uncertainty" — but the alternative here is a *lie*, and "this bin has too little data to
quantify" is an actionable input to a decision in a way that "±0.0%" is not. The point estimate for
such a bin is equally meaningless; leaving it in place while NaN-ing its sigma is deliberate (uplift
must not change), but a future issue could reasonably suppress both.

---

## F32 — At 256 replicates the bootstrap-only sigma is **calibrated**: pooled coverage 0.682 vs a 0.683 target at 6h blocks, every campaign length from 1 week to 1 year within 0.5 SE. No further uncertainty component is justified, and the 6h default is confirmed rather than merely safe

*2026-07-16 (toggle-specialist uncertainty, round 3 — the power run F31 asked for). Reproduce:
`uv run python -m benchmarking.baselines.study_toggle_specialist_uncertainty --replicates 256
--block-hours 1 3 6` — 96,768 cells. **The decision rule below was fixed before looking at the
data**, so this is a test rather than a search for a justification.*

**The question F31 left open.** Coverage sat consistently at ~0.65 against 0.683 — never significant
(-1.26 SE), never above target either. That is the shape of either a real ~5% optimism or noise, and
64 replicates could not tell them apart. Guessing would have meant fitting noise; the fix is power,
not modelling.

**The decision rule, pre-registered:** an inflation `k` is justified only if it is *consistent* — it
must bring **every** campaign length closer to target (`n_worse == 0`) and lower the worst deviation.
Trading one length for another is not an improvement.

**Result: the bootstrap-only sigma is calibrated.** Pooled over 1-52 weeks (~873 independent draws,
SE 0.016):

| block | pooled coverage | SE from target |
|---|---|---|
| 1h | 0.758 | **+4.76** |
| 3h | 0.701 | +1.15 |
| **6h** | **0.682** | **-0.09** |

Per campaign length at 6h: **0.689 / 0.684 / 0.698 / 0.668 / 0.663 / 0.689** for 1/2/4/8/26/52 weeks
— every one within **0.5 SE** of target, worst deviation **0.020**. `std(z)` agrees independently at
**1.013 / 0.975 / 0.950 / 1.028 / 1.041 / 0.927** (target 1.0), and `median|z|` at 0.61-0.74 against
the 0.674 a calibrated normal gives. Coverage and the magnitude-sensitive read concur.

**No inflation is justified — `k = 1.0` is optimal.** It is the only value with `n_worse == 0`; every
`k >= 1.05` moves 4-6 of the 6 lengths away from target and raises the worst deviation from 0.020 to
0.045+. **F31's ~0.65 was noise**, confirmed at 4x the power.

**The 6h default is confirmed, not merely safe.** F28 chose it on robustness grounds when 2h/3h/6h
were statistically indistinguishable. At 256 replicates they are distinguishable and 6h is right:
**1h over-covers significantly (+4.76 SE)** and 3h is drifting high (+1.15 SE). The robustness
argument (9 toggle cycles per block; enough blocks even at 1 week) picked the value the data now
independently endorses.

**Where this leaves the uncertainty work.** The *scale* corrections anticipated at design time are
rejected on evidence — an inflation (above), the campaign-level systematic floor (F31), and the shape
correction (F31, itself an artefact). What remains is a plain circular block bootstrap at 6h blocks,
calibrated from one week to one year across placebo and +/-2% profiles.

> **Corrected by F33.** This section originally concluded "the right move is to add nothing",
> including the low-count term. That was wrong, and wrong for an instructive reason: every coverage
> read here pools per-bin cells at `(0, 30]` records, which averages a broken 0.158 subset (1-2
> records) into a reassuring 0.623 — and `calibration_summary` excludes `sigma <= 0` cells as
> `n_unusable`, so the metric structurally dropped the worst cases. A **validity floor** at 3 records
> per side *is* justified. The claim that survives is narrower and still holds: no *scale* correction
> is justified where a sigma is estimable at all.

The residual caveats are honest and small: 52-week coverage rests on only ~16 independent draws
(SE 0.116), and ~1.6% of 1-week campaigns report a very large sigma where the reference denominator
nears zero — correct behaviour, but it makes `mean_sigma` a misleading summary (use medians or
coverage).

---

## F31 — Extending the campaign grid to a year kills F29's leading hypothesis: there is **no campaign-level systematic floor**, and the platykurtosis was an artefact of a narrow start range. The bootstrap-only sigma keeps working on ample data, and **no further component is justified**

*2026-07-15 (toggle-specialist uncertainty, round 2). Reproduce:
`uv run python -m benchmarking.baselines.study_toggle_specialist_uncertainty` — now 64 replicates x 3
profiles x **1/2/4/8/26/52 weeks** x 7 block lengths (56,448 cells). Two config changes from F28/F29,
both deliberate: the grid reaches a year, and `min_pre_months=0` with the start range widened to the
whole dataset (2016-01-01..2020-01-01).*

**Why the config changed.** `toggle_specialist` drops pre-campaign rows (`restrict_to_campaign`), so
`min_pre_months` buys it nothing and only costs start-range span — and span is exactly what long
campaigns need, because **replicates stop being independent once their windows overlap**. A 52-week
campaign is 364d, so the old 730d range held only ~2 non-overlapping positions. Widening to 1461d
doubles that. `independent_draws()` now reports the honest count per length and the SE is quoted on
it: **1-8wk ~64 draws (SE 0.058), 26wk ~32 (0.082), 52wk ~16 (0.116)**. A 52-week coverage anywhere
in **0.45..0.92** is indistinguishable from calibrated — long campaigns are precise but their
*uncertainty* is weakly evidenced, because 5 years of SCADA holds few independent year-long windows.

**Long campaigns are the discriminating test, and they kill the floor.** F29's lead candidate was an
irreducible campaign-level systematic that a within-campaign bootstrap cannot see. `sigma_boot`
shrinks with data, so any such floor **must dominate** once sigma is small enough. It does not:

| campaign | 1w | 2w | 4w | 8w | 26w | 52w |
|---|---|---|---|---|---|---|
| median sigma [pp] | 1.156 | 0.784 | 0.549 | 0.387 | 0.197 | **0.135** |
| median \|error\| [pp] | 0.733 | 0.476 | 0.310 | 0.216 | 0.157 | **0.110** |
| coverage | 0.599 | 0.646 | 0.698 | 0.724 | 0.594 | **0.635** |
| implied floor [pp] | 0.00 | 0.60 | 0.00 | 0.18 | 0.18 | **0.07** |

At 52 weeks sigma is 0.135 pp. A 0.2 pp floor would swamp it and drive coverage to ~0.4; observed
**0.635**. The implied floor does not persist — it is *smaller* at 52wk (0.07) than at 8wk (0.18),
which is the signature of noise, not of a floor. **Hypothesis rejected.** Short campaigns could never
have decided this, because `sigma_boot` swamps any floor there.

**The platykurtosis was an artefact of the narrow start range.** F29 measured kurtosis -0.35..-0.76
(Shapiro p<0.05 everywhere) and read it as a real shape problem capping achievable coverage. With the
widened range it is **~0** (-0.25/-0.36/+0.04/-0.02/+0.08/-0.09). Drawing 64 windows from only 2
years clustered them; the flat-topped error distribution was the clustering, not the estimator.
**Hypothesis rejected** — and a caution that F29's shape reasoning was over-read.

**No count term, confirmed again.** Per-bin coverage by record-count decade at 6h blocks:
0.623 / 0.728 / 0.674 / 0.713 / 0.654 / 0.674 for `<=30 / 30-100 / 100-300 / 300-1k / 1k-3k / 3k-10k`.
No trend.

**`mean_sigma` is a trap here; use medians or coverage.** 1-week `mean_sigma` reads 3.39 pp against an
RMS error of 1.64 pp — apparently 2x too wide — while coverage says 0.599, slightly too *narrow*. The
mean is dominated by **3 cases of 192 (1.6%)** with sigma up to **203 pp**. Those are not a defect:
they are 1-week campaigns whose reference denominator approaches zero in a low-wind week, and the
ratio estimator is genuinely unstable there, so sigma correctly says "no idea". They are also **not**
from the newly-added years (all three start in 2018), and 2016 campaigns look like every other year
(median 2668 used records, median sigma 0.414 pp, coverage 0.611) — the widening introduced no junk.

**Block length matters less than F28 implied, once the grid reaches a year.** Pooled over 1-52wk
(~304 independent draws, SE 0.027): 1h **0.720**, 2h 0.657, 3h 0.661, 6h 0.649, 12h 0.656, 24h 0.649,
48h 0.641. Everything from 2h to 48h is flat within noise; only 1h stands out, by over-covering. F28's
sharp block-length gradient was real but **specific to a grid of only short campaigns**: at 26/52wk
`T/L` is large enough that block length is irrelevant, which dilutes it. The 6h default stands (F28's
short-campaign case is unaffected and 48h is still worst at 1 week, 0.573 vs 0.651), but the margin is
narrower than F28 suggested.

**Verdict: no further uncertainty component is justified by this data.** Every candidate either fits
noise or fixes one campaign length while breaking another — a floor of 0.10 pp lifts pooled coverage
to 0.674 but pushes 52wk to 0.729 while leaving 1wk at 0.599; a 1.10x inflation reaches 0.686 pooled
but spreads 0.609..0.776 across lengths. Nothing anywhere is more than 1.6 SE from target.

**The one open question is power, not modelling — and F32 settled it.** Coverage sits consistently at
~0.65 against 0.683 here: never significant, but never above target either, which is the shape of
either a real ~5% optimism or noise. This ensemble cannot tell them apart, so adding an inflation on
this evidence would be fitting noise. **F32 ran it at 256 replicates: it was noise** (pooled 0.682 at
6h blocks, -0.09 SE), and no inflation is justified.

---

## F30 — `power_model`'s benchmark is **machine-specific** (~0.7 pp cross-machine, 14x its same-machine noise); `toggle_specialist`'s is portable to 5e-07 pp, so the toggle benchmark splits into a shared file plus one per platform

*2026-07-15. Reproduce: `study_toggle_methods_compare` on a machine that did not record the
benchmark. The LightGBM behaviour is visible by fitting any `make_outcome_model` at `verbose=1`.*

**Observation.** Running the toggle compare on the Linux laptop against a benchmark recorded on the
Windows laptop, `power_model` reports **MOVED — 25 of 84 cells, max ~0.70 pp** — permanently, and
regardless of the code under test. `toggle_specialist` reports UNCHANGED at **5e-07 pp** on the same
run, against the same foreign benchmark.

**This is not run-to-run noise, and reading it as a stale baseline is wrong (I did).** The evidence
that looks damning:

- two runs of identical code on one machine agree with **each other** to **0.045 pp** — matching the
  ~0.05 pp the script documents;
- yet both deviate from the foreign benchmark by ~0.7 pp with their delta patterns correlated
  **0.994**;
- and a clean `git archive HEAD` extract, containing no local changes at all, reproduces the same
  0.697 pp.

That reads as "same deltas every run ⇒ deterministic ⇒ the code changed", and the conclusion drawn
was "the committed baseline does not reproduce on its own commit; regenerate it". **Wrong.** It is
deterministic **per machine**, not per code: both runs share this machine's LightGBM reduction order,
so both depart from the recording machine's identically. The disconfirming evidence was available and
ignored — the input data was three weeks stale and unchanged, and the recording predated the run by
40 minutes on nominally identical code. One question ("whose machine recorded it?") settled it.

**Mechanism**, confirmed live rather than inferred:

1. **LightGBM times the machine and picks its histogram strategy from the result** —
   `[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000365
   seconds.` `force_row_wise`/`force_col_wise` are unset in `_COMMON`, and row-wise and col-wise
   accumulate gradient/hessian histograms in **different orders**.
2. **Floating-point addition is not associative**, so a different order changes the last bits.
3. **`num_threads` is unset**, defaulting to all cores (12 here), and the thread count partitions the
   histogram reduction — another order change.
4. **Windows vs Linux** compounds it: different wheel, compiler (MSVC vs GCC), OpenMP runtime (vcomp
   vs libgomp), SIMD codegen, libm.

Then it **amplifies**: a split is an `argmax` over candidate gains, so a ~1e-16 difference flips a
near-tied split, changes a tree, and 600 boosted trees compound it into ~0.7 pp.
`toggle_specialist` is immune because it is a sum and a divide — no trees, no threads, no argmax.

**Resolution: split the benchmark by portability, not by machine.**
`study_toggle_methods_compare_baseline.json` (v2) becomes three v3 files — `..._portable.json`
(`toggle_specialist`), `..._linux.json` and `..._win32.json` (`power_model`) — keyed on
`sys.platform`. A run diffs portable + this platform, merged. Each laptop writes only its own platform
file plus the shared one, so they never conflict in git. Portability is a per-method fact
(`_REPRODUCIBILITY`) sitting beside the band it already had, defaulting to `portable=False` because
wrongly claiming portability is a permanent, confusing failure on the other machine.

**The portability invariant, and why the obvious version of it is wrong.** From one machine "the
numbers moved" is ambiguous: it means either the method changed or portability broke. Refusing on any
difference would block every legitimate re-record; a `--force` escape would just train the user past
the check. **The commit is the discriminator** — differ at the *same* commit ⇒ portability broke
(refuse); differ at a *different* commit ⇒ an accepted change (rewrite). The dirty-tree guard is what
makes the commit trustworthy enough to lean on. Two further subtleties, both found before they bit:

- the comparison must use the **band**, not bit-equality: recorded cells carry wall time (differs
  every run, so an exact test rewrites the shared file every recording and creates the very conflict
  the split avoids), and `round(8)`'s 1e-8 resolution is only 2x the measured 5e-9 cross-machine
  difference;
- the portable file's `git_commit` therefore records **when those numbers were last established**,
  not who last ran a recording. An unchanged re-record leaves the file untouched, commit and all.

**Provenance.** Each file now records `platform` / `cpu_count` / `python_version` /
`lightgbm_version`, and a mismatch warns (never fails). This is the direct fix for the hole above:
the file could not say where it came from, so the only way to find out was to ask a human.

**`study_power_model_compare`'s benchmark: the numbers were never the problem.** A full read-only
re-run on the Linux laptop (the machine that records it) reproduces it exactly — **2030 of 2030 cells
neutral across both modes**. Its `e2e21b0-dirty` stamp is **very likely a false alarm**: its
`_git_commit()` counted **untracked** files as dirty, unlike `study_toggle_methods_compare`, which
deliberately passes `--untracked-files=no` because an untracked file cannot make a run irreproducible
from its commit — `git checkout <commit>` would not have it. With a local `CLAUDE.md` sitting
untracked, that definition also made `--update-baseline` **impossible on this machine**, permanently.
Now aligned to the toggle script's definition, with tests pinning both directions.

Three further guards ported to it, all previously absent:

- it *labelled* a dirty tree but never **refused** one;
- `--accept-candidate` — the documented no-re-run accept, and so the likeliest route for a bad stamp
  — promoted candidates without checking they were recorded clean;
- `record_baseline` read HEAD at *write* time, so an hours-long sweep straddling a commit would stamp
  code that never ran. The commit is now captured before the sweep, as the toggle script already did.

That script stays **single-machine by design**: every cell in it is `power_model`, so there is nothing
portable to split off, and it is always run on the Linux laptop.

**Recording the Linux half of the toggle benchmark exercised the invariant for real, and it passed:**
`Portable baseline unchanged — portability confirmed, not rewritten`. `toggle_specialist`'s cells,
recorded on the Windows laptop, matched the Linux run within band, so the shared file was left
untouched — no churn, no conflict, and a live cross-machine confirmation rather than an assumption.

---

## F29 — The anticipated failures of a bootstrap-only uncertainty did not appear: at a well-chosen block length `toggle_specialist`'s sigma is statistically indistinguishable from calibrated everywhere, and the sparse-bin failure was mostly F28's block-length artefact

*2026-07-15 (toggle-specialist uncertainty, round 1). Reproduce:
`uv run python -m benchmarking.baselines.study_toggle_specialist_uncertainty` — 64 replicates x 3
profiles x 1/2/4/8 weeks x 5 block lengths; the per-cell table is its `cases.csv` and the reads are
its `calibration_*.csv`. Coverage target 0.683; **binomial SE on 64 independent draws = 0.058**.*

The prior going in was that a block bootstrap would work for long campaigns and well-populated bins
and fail for short campaigns and sparse bins, so a **data-count term** would be needed. Measured, at
6h blocks, headline coverage by campaign length is **1w 0.672, 2w 0.724, 4w 0.693, 8w 0.599** — every
one within 1.5 SE of target (`-0.19`, `+0.70`, `+0.17`, `-1.44` SE). Per-bin coverage by record
count is **0.576 / 0.667 / 0.648 / 0.662 / 0.672** for `<=30 / 30-100 / 100-300 / 300-1k / >1k`
upgraded records — no significant count trend.

**The count effect was mostly F28 in disguise.** Sparse-bin (`<=30` records) coverage runs
**0.422 at 48h blocks → 0.606 at 1h**. A sparse bin at a long block length is short of records *and*
of blocks; shorten the block and the bootstrap recovers. The count term is not (yet) justified: the
block length was.

**The anticipated short-campaign bias was an artefact of 4 replicates.** The committed compare
baseline (n=4) shows a 1-week bias of ~-0.7 pp, which motivated a bias component. At n=64 the 1-week
bias is **-0.15 pp** against a 1.29 pp spread (bias is 12% of RMS). Removing the bias entirely
*lowers* 1-week coverage (0.557 → 0.510 at 48h), so it is not what limits coverage. The n=4 figure
was noise, and this is the concrete payoff of the replicate count.

**Two real, smaller effects remain open.**

1. **The error distribution is platykurtic, not normal** — kurtosis `-0.59 / -0.59 / -0.35 / -0.76`
   at 1/2/4/8 weeks, Shapiro p < 0.05 at every length. It is flatter than a Gaussian, so coverage
   sits *below* what the scale alone predicts: at 8 weeks `std(z) = 1.03` implies 0.668 under
   normality but 0.599 is observed. A 1-sigma coverage target assumes a shape the errors do not have.
2. **8 weeks is the weak end, not 1 week** — coverage 0.56-0.60 at every block length, with
   `rms/mean_sigma` = 1.08-1.16. This is the *opposite* of the prior. It is only -1.44 SE, so it is
   suggestive rather than established. A campaign-level systematic (something constant within a
   campaign, which a within-campaign bootstrap is structurally blind to) would explain both this and
   the platykurtosis; the seasonal read is consistent (campaigns starting Jul/Aug are biased
   -0.31/-0.35 pp, Dec/Feb +0.09/+0.10 pp) but not established. Adding a floor `f` in quadrature
   lands pooled coverage at 0.685 for `f = 0.10 pp`, but the per-length implied floors are
   inconsistent (0.52 / 0.00 / 0.00 / 0.16 pp at 1/2/4/8 weeks), so **no floor is proposed on this
   evidence** — it would be fitting a term to 1.4 SE of noise.

**Design implications.** Do not add the count term: at the block length now defaulted to (F28) there
is no count trend left for it to correct. The residual candidates, in the order the evidence supports
them, are: (1) a campaign-level systematic floor, if a larger ensemble confirms the 8-week gap —
which needs replicates, since 64 leaves it at only -1.44 SE; (2) a shape correction, since the
platykurtosis is the most reproducible non-normality here and it caps achievable 1-sigma coverage
even when the scale is right.

**Method note.** The three profiles are near-redundant for calibration: their signed errors correlate
**0.977-0.995** (they reuse the same `(turbine, treatment_start)` draws and differ only in injected
Cp). 768 headline cells are ~64 independent draws. Quote coverage SE on replicates, never on rows.

---

## F28 — The 48h block prior is wrong for a fast toggle: the *paired* residual's correlation scale is ~1-3h, and a 48h block under-covers (0.622 pooled, 0.557 at 1 week) by starving the bootstrap of blocks

*2026-07-15 (toggle-specialist uncertainty, round 1). Reproduce:
`uv run python -m benchmarking.baselines.study_toggle_specialist_uncertainty --block-hours 1 2 3 6 12`
plus the default grid; 64 replicates x 3 profiles x 1/2/4/8 weeks. Coverage target 0.683, SE 0.058.*

`block_hours` defaulted to 48 on the prior that turbine-to-turbine relationships autocorrelate on
roughly that scale. **That prior is about the wrong quantity.** The bootstrap resamples the residual
of the *paired on-vs-off comparison*, and Hill of Towie's toggle alternates every 20 minutes
(`DEFAULT_TOGGLE_PERIOD` = 40 min), so on and off ride the same weather and nearly all the slow
structure — direction, wake state, density, season — cancels between numerator and denominator.

**Measured autocorrelation of the paired hourly test/reference ratio residual** (8 replicates,
8-week placebo campaigns), mean over replicates:

| lag | 1h | 3h | 6h | 12h | 24h | 48h |
|---|---|---|---|---|---|---|
| autocorr | 0.181 | 0.065 | 0.047 | 0.034 | 0.032 | **0.003** |

The correlation scale is **~1-3 hours**. At 48h there is nothing left to capture.

**Headline coverage is monotone decreasing in block length**, crossing the target at ~2-3h:

| block | 1h | 2h | 3h | 6h | 12h | 24h | 48h | 96h |
|---|---|---|---|---|---|---|---|---|
| pooled coverage | 0.775 | 0.695 | 0.707 | 0.672 | 0.665 | 0.642 | **0.622** | 0.577 |
| 1-week coverage | 0.766 | 0.719 | 0.724 | 0.672 | 0.667 | 0.620 | **0.557** | 0.438 |

**Root cause of the long-block failure: circular-block overlap collapse, not noise.** A long block
does not merely make sigma noisy — it biases it **low**. With `n_draw = ceil(T/L)` blocks drawn from
starts anywhere in the campaign, a large `L/T` makes every resample overlap almost completely, so
the resample spread collapses. It is worst where `T/L` is small: at 1 week with 96h blocks
`T/L = 1.75`, mean sigma 0.86 pp against an actual RMS error of 1.30 pp, coverage **0.438**. At 8
weeks the same 96h block gives `T/L = 14` and sigma falls only ~7% from its 6h value. A synthetic
control (iid noise, 8-week campaign, so `T/L >= 28` throughout) reproduces the expected flat
sigma-vs-L to within ±5%, confirming the collapse is a small-`T/L` effect rather than a bug.

**Consequence: there is no plateau to read.** Standard practice picks the block length where
sigma-vs-L flattens. Here sigma is flat-to-falling across the whole range, because there is no
autocorrelation to capture and the only remaining gradient is the collapse. **Block length must be
chosen by coverage against truth, not by the sigma curve** — which is why the sweep is scored rather
than plotted alone.

**Coverage degrades in _both_ directions, so the good region is bounded.** Below ~2h the bootstrap
**over-covers** (1h: headline 0.775, sigma ~1.2x the 12h value). The cause is not established: an
on/off imbalance hypothesis (a 1h block spans only 1.5 cycles of the 40-minute toggle) was **not**
reproduced by synthetic controls with either a constant or a varying reference. So 1h is not adopted
on the strength of its score alone.

**Applied: `DEFAULT_BLOCK_HOURS` 48 → 6.** The data does **not** distinguish 2h/3h/6h — mean
|coverage - 0.683| across the four reads (headline pooled, headline 1-week, per-bin pooled, per-bin
sparse) is **0.046 / 0.047 / 0.049**. The pick is therefore on robustness, not score: 6h spans 9
toggle cycles (the most margin from the ~2-cycle edge where the over-coverage sets in), is ~2-6x the
measured correlation scale, still leaves 28 distinct blocks in a 1-week campaign, and stays sane for
a slower toggle where 2h would be a single cycle. **The default is only a default**: block length is
properly a function of the toggle period and campaign length, so a campaign whose toggle period is a
large fraction of 6h must raise it.

Changing it does not touch any uplift: `study_toggle_methods_compare` still reports
`toggle_specialist: UNCHANGED` (max delta 5e-07 pp), and
`TestUncertaintyDoesNotChangeUplift::test_block_length_moves_sigma_and_nothing_else` asserts the
point estimate is bit-identical across block lengths.

**Generalisation.** The right block length is set by the toggle period and the campaign length, not
by an absolute number of hours: enough cycles per block that on/off stay balanced, and enough blocks
per campaign that circular overlap does not collapse the spread. A downstream project with a slower
toggle needs a proportionally longer block, so a rule of the form `L = k x toggle_period` (with a
guard on `T/L`) would travel better than any fixed default. Not implemented — it needs its own
evidence.

---

## F27 — `toggle_specialist` can report a per-power-bin uplift without a model, but only with a **per-bin** `rho_base` and a reference-derived bin label; the global-`rho_base` shortcut is ~17 pp wrong

*2026-07-15 (per-power-bin uplift for the toggle specialist). Reproduce: the estimator comparison is
`TestPerBinIsNotBiasedByTheBaselineRatio` in `tests/benchmarking/baselines/test_toggle_specialist.py`;
the regression baseline is `study_toggle_methods_compare --update-baseline`.*

`toggle_specialist` now accepts `conditions=("power",)` + `rated_power_kw` and returns a per-power-bin
uplift alongside its headline. Getting there required rejecting two estimators that both look
reasonable, and the F23/F24 reasoning for `power_model` carries over almost unchanged.

**The estimator.** Bin **both** segments on `rho_base_global * ref_total` — the test turbine's
*predicted untreated* power — then report `rho_up(b) / rho_base(b) - 1` per bin.

**Why not the global `rho_base`.** Using one global denominator, `u(b) = rho_up(b)/rho_base_global - 1`,
has an attractive property: the per-bin numbers then aggregate back to the headline exactly. It is also
badly wrong. The test-to-reference ratio genuinely varies with power (different turbines, different
wakes, saturation near rated), so the estimator reads that structure as uplift. Measured on a synthetic
case where `k = test/ref_total` falls 0.9 → 0.7 across the power range: **worst per-bin error 16.5 pp on
a placebo, 17.3 pp at a true +5%**. The exact aggregation is bought by smearing the baseline's
power-dependence across the bins.

**Why not the test turbine's own power as the bin label.** This is the direct analogue of F23/F24, and
its failure mode is instructive: it scores **0.05 pp on a placebo** — i.e. the obvious zero-uplift test
*passes it* — and only breaks once a real uplift exists, at **1.22 pp on a true +5%** (a 24% relative
error). The mechanism: a real uplift shifts the test turbine's power, so the treated rows in a bin
correspond to *lower* untreated power than the baseline rows in it; against a power-dependent `k` that
mismatch becomes bias. **A placebo cannot discriminate these estimators** — the discriminating test is a
*constant non-zero* uplift, which must read that same uplift in every bin. Worth remembering for any
future per-bin work.

**The chosen estimator scores 0.05 pp in both cases**, and holds under 2% noise.

**Two deliberate departures from `power_model`'s conditional.**
- **No re-levelling.** Per-bin numbers do not aggregate exactly to the headline, and are not rescaled to.
  `power_model` needs its λ because its per-bin shape comes from a `sqrt` of two fits that does not
  aggregate; here the per-bin numbers are direct energy ratios, and rescaling them onto an identity that
  the per-bin `rho_base` deliberately broke would be a fiction. `sum_actual` / `sum_counterfactual` are
  returned so the gap stays inspectable.
- **No imputation.** An uncovered bin reports NaN with `n_records = 0`, not a bfill-then-0-at-rated
  prior. This falls out by construction (no baseline rows → no `rho_base(b)` → NaN counterfactual). A
  downstream per-bin decision rule wants a sparse bin to land on "keep going"; an imputed value would
  manufacture confidence that is not in the data.

**`power` is the only axis this method will ever support.** It is reference-derived, so the treatment
cannot move a row between bins. Binning by the test turbine's ws/TI would condition on post-treatment
signals, which is exactly the property this method exists to keep (`conditions=("ws",)` raises).

### Side observations
- **A new regression harness:** `study_toggle_methods_compare.py` scores `toggle_specialist` +
  `power_model` on HoT over a placebo and a symmetric +/-2% Cp pair × 1/2/4/8 **weeks**, against a
  committed baseline. The per-bin change diffed clean on it: `toggle_specialist`'s 12 cells all moved by
  **~1e-9** (i.e. unchanged), confirming the per-bin work left the headline alone.
- **`power_model` is not reproducible run to run, despite `seed` — and this sets a floor on every A/B
  against it.** Measured directly (two runs of *identical* code, `--profiles cp_0pct`, separate output
  dirs): `power_model`'s bias moved **5.0e-4 (0.05 pp) at campaign_weeks=1**, and **exactly 0.0 at 2
  and 8 weeks**. `toggle_specialist` moved **exactly 0.0 in every cell**.
  - **Mechanism:** `seed` governs sampling, not LightGBM's threaded float reduction order. The noise is
    **sparsity-driven**, not uniform: with a week of data the model sits near a split boundary, so a
    tiny float difference flips a tree and visibly moves the estimate; with 8 weeks the split decisions
    are far from the margin and the result is bit-identical. Hence the noise appears *only* at the short
    campaigns — the very regime this study exists to probe.
  - **This retroactively explains `study_power_model_compare`'s `_MATERIAL_PP = 0.1`**: it is not an
    arbitrary comfort band, it sits at ~2x this measured floor.
  - **Consequence for the harness:** bands are **per method** (`_UNCHANGED_ATOL`): `toggle_specialist`
    1e-7 (effectively bit-exact), `power_model` 1e-3. A single shared band would either cry wolf on
    every power_model re-run or throw away toggle_specialist's much stronger detector.
  - **Process note:** the design asserted "an unchanged method must diff to **exactly 0.0**". That was
    wrong on two counts (this nondeterminism, plus `record_baseline` storing cells rounded to 8 dp), and
    the first real before/after run disproved it. The first correction guessed the floor at 1e-4 from
    indirect evidence — also wrong, by 5x, and only caught by running the same code twice. **Measure the
    noise floor before setting a band.**
- **`toggle_specialist` has a real short-campaign bias:** at 1 week its bias is ≈ **−0.7 pp** with ~1 pp
  spread, consistently across all three profiles (vs ≤0.2 pp at 2+ weeks). Systematic, not noise; it sets
  a floor on what a 1-week toggle campaign can resolve. Not investigated here.
- **Provenance flaw in both study scripts:** `_git_commit()` stamps HEAD when the baseline is *written*,
  not when the run *started*. On a ~20-minute sweep a commit landing mid-run mislabels the record (the
  first toggle baseline stamped `04f36d6-dirty` though it measured `f6b509b`'s method code). Worth
  capturing the commit at run start.

---

## F26 — Issue 17: unifying the toggle conditional onto the all-data training window (from campaign-only) blows up the tail bins — the campaign-only restriction is confirmed necessary; both conditional asymmetries are deliberately retained

*2026-07-14 (Issue 17, scope item 2 — training-window unification). Reproduce:
`study_power_model_compare --modes toggle --profiles cp_0pct ti_dependent_cp ws_dependent_cp` with the toggle
conditional match's `baseline_sel` switched from `campaign_baseline` (strict interleaved off-rows) to
`training_baseline` (pre-campaign + off-rows), vs the committed campaign-only; diffed against the committed
baseline (before = campaign-only, after = all-data). Separate output dir; no baseline change. Prepost is
unaffected by construction (there `campaign_baseline == training_baseline`), so this is a toggle-only test.*

**Result — a clear regression, worst in the tails.**
- **Overall P50: bit-identical** (the change touches only the conditional match) — correctness check passed.
- **Every conditional axis got worse in both |bias| and spread:** mean |bias| Δ ti **+2.12**, ws **+1.76**,
  power **+0.10** pp; mean spread Δ ti +1.57, ws +1.98, power +0.40 pp.
- **The tails explode.** Highest-TI `(0.4,0.45]` went from ~5–9 pp to **60–97 pp** bias; lowest-ws cut-in
  `(0,2]`/`(2,4]` swung to **−15…−21 pp**. These are exactly the drift-contaminated bins F15 warned about:
  matching temporally-distant pre-campaign rows against campaign on-rows reads reference/era drift as per-bin
  uplift.

**Root cause / why the F17 floor doesn't rescue it (and why decay wouldn't either).** Adding pre-campaign rows
*increases per-bin counts*, so tail bins that were below the F17 count floor — and therefore safely **imputed** —
now clear the floor and are **trusted** with a drift-contaminated two-direction shape. More data makes it worse
by promoting garbage bins from imputed to trusted. Adaptive-half-life decay (the issue's proposed mitigation)
would downweight those rows in the *fit* but they still *count* toward the floor and still populate the matched
pairs, so it would not remove the promotion mechanism. This matches the issue's own note that the genuine fix is
"weighted **or restricted** matching" — and the *restricted* matching is precisely what `campaign_baseline`
already does.

**Decision: rejected — keep the toggle conditional on `campaign_baseline`.** Combined with F25, **Issue 17
closes via its second branch:** both conditional/headline asymmetries are *deliberately retained*, each with
fresh post-floor evidence for why —
1. *Conditional match is campaign-only (toggle), not all-data* — because unifying it reads drift as uplift and
   the count floor amplifies rather than fixes it (this finding).
2. *Conditional fits are unweighted, headline is decay-weighted* — because recency weighting on the conditional
   only churns negligible tail bins (F25).

The energy-aggregation identity and overall P50 were preserved throughout (both bit-identical in every A/B).
**A better unification direction remains open and is the recommended next step:** F22 showed the *opposite* move —
make the toggle **headline** campaign-only (pull it toward the conditional) — is neutral-to-better on overall P50
and unifies the data path from the other side; that is genuine design work for a supervised session, not tried
here.

---

## F25 — Issue 17 re-trial: recency-weighting the conditional direction fits (now the F17 count floor exists) is a no-op for toggle and only churns negligible extreme-tail bins for prepost — rejected

*2026-07-14 (Issue 17, scope item 1 — the specific F14/F16-flagged re-trial). Reproduce:
`study_power_model_compare --modes prepost toggle --profiles cp_0pct ti_dependent_cp ws_dependent_cp` with the
adaptive-half-life time-decay `weights` threaded into `_fit_direction` (the matched forward/reverse conditional
fits), vs the committed unweighted conditional; diffed against the committed baseline (before = unweighted,
after = weighted). Separate output dir; no baseline change.*

**Context.** F16 held time-decay weights off the conditional direction fits because pre-floor they destabilised
sparse extreme-condition bins (a degenerate tail fit could read three-digit per-bin uplift). Issue 14 deferred
lifting that until the per-bin count floor existed; the floor shipped in F17, so this re-trials the weighting.

**Result.**
- **Overall P50: bit-identical** in both modes (weighting touches only the conditional fits, not the headline) —
  a correctness check that passed.
- **Toggle: every conditional cell bit-identical.** As predicted from the code: the conditional match is
  campaign-only, and those rows all sit *inside* the campaign interval where the decay weight is exactly 1, so
  weighting is a no-op for toggle.
- **Prepost: change confined to extreme sparse tail bins.** Mean |bias| moved only marginally (ti −0.28, ws
  −0.05, power −0.04 pp), but per-bin swings were large (ti −11.6→+1.6, ws −7.4→+2.5 pp) and **entirely in the
  tails**: the biggest "improvements" are the highest-TI `(0.4,0.45]` bins dropping from ~54 pp bias to ~42 pp
  (both garbage, negligible energy) and the lowest-ws cut-in bins `(0,2]`/`(2,4]` with ±7–16 pp ratio-instability
  bias (F5's second failure mode). Meaningful populated bins are essentially unchanged, and **ws spread got
  worse** (+0.37 pp mean).

**Decision: rejected — keep the conditional fits unweighted (F16 stands, now with post-floor evidence).** The
floor removes the three-digit blowups, but weighting still only reshuffles negligible-energy tail bins in both
directions; it buys no accuracy on the bins that carry the energy and costs a little ws spread. Closes Issue 17
scope item 1: the missing half-life on the conditional path is deliberately retained as an asymmetry, because
adding it does nothing useful. The plumbing (`weights=` on `_fit_direction`/`_estimate_conditional`) was reverted
along with this rejection.

---

## F24 — binning *both* conditional directions on real power (instead of on the prediction) is a wash vs the F23 fix on the tested profiles, and is theoretically less robust — rejected

*2026-07-14 (Issue 17, follow-up A/B to F23). Reproduce: `study_power_model_compare --modes prepost toggle
--profiles cp_0pct ti_dependent_cp ws_dependent_cp` with the power frame's bin labels temporarily set to
`fwd_cond=y[mu]`, `rev_cond=y[mb]` (both real power) vs the committed F23 default (`pred_up`/`pred_base`); scored
by `conditional_benchmark_comparison` against the accepted baseline, so before = F23 prediction-based, after =
both-real-power. Separate output dir; no baseline change.*

**Motivation.** With F23 fixed by moving the *reverse* label off its own numerator (`y[mb]`→`pred_base`), an
alternative symmetry is to move the *forward* label onto real power too (`pred_up`→`y[mu]`), i.e. bin both
directions by actual power. A-priori concern: the forward side's real power `y[mu]` is **post-treatment** (design
§3) and is *also the forward ratio's numerator*, so this re-introduces regression-to-the-mean on the forward side
and makes the forward axis shift with the treatment.

**Result — a wash.** Mean per-bin |bias| change across the power axis was **−0.00 pp in both modes** (prepost:
4 bins better / 5 worse / 9 neutral; toggle: 6 / 4 / 8), with only scattered ±0.3 pp per-bin moves and no
systematic winner. The predicted §3 degradation did **not** appear: on `cp_0pct`/`ti_dependent_cp`/`ws_dependent_cp`
the uplift is 0–small, so treated vs untreated power seldom crosses a 20%-of-rated bin edge and the
post-treatment bin-reassignment smearing is second-order. The one visible tell in the predicted direction: the
placebo's lowest-power bin got *worse* under both-real-power in prepost (−0.25→+0.53 pp), consistent with RTM
leaking back on the forward side.

**Decision: rejected — keep the F23 prediction-based labelling.** It is empirically no worse here and
theoretically more robust: no RTM on either side, and a treatment-invariant axis. **Caveat / where the difference
should actually show:** the covered profiles are exactly the low/zero-uplift cases; the profiles where §3
smearing would be largest — `rated_plus_5pct` (uprate) and `cp_plus_10pct` — are not in `COVERED_PROFILES`, so
this A/B cannot see them. The theoretical edge of prediction-based is expected to matter there, not on these
three. Worth a targeted check if a future cycle extends the conditional before/after view to a large-uplift
profile.

---

## F23 — the `power`-axis conditional uplift's "positive at low power, negative at high power" tilt was regression-to-the-mean from binning the reverse direction on its own noisy numerator; labelling both directions by the counterfactual prediction removes it

*2026-07-14 (Issue 17, power-axis correctness). Reproduce: `benchmarking.baselines.study_power_model_compare`
(`conditional_benchmark_comparison_<mode>.csv` + `benchmark_comparison_<mode>.csv`); first isolated on the
`cp_0pct` placebo prepost, then confirmed on the full both-mode sweep (7 profiles, campaigns {1,2,3,6,12} mo,
4 replicates, seed 0). One-line change in `method.py:_conditional_by_bin`'s power frame: the reverse-direction
bin label `rev_cond` moved from `y[mb]` to `pred_base`.*

**Observation.** On `cp_0pct` (true uplift 0 in every bin, so per-bin estimate = pure bias) the `power`-axis
conditional uplift ran monotonically **positive at low power, negative at high** — at 12 mo prepost: `(-230,230]`
**+8.24 pp**, `(230,690]` +2.52, `(690,1150]` +0.77, `(1150,1610]` −0.59, `(1610,2070]` −1.32, `(2070,2530]`
−0.98. A flat-zero truth read as a strong slope; the same shape appeared on the real-uplift profiles and in
toggle.

**Root cause — binning the reverse ratio on its own numerator.** A condition axis' per-bin shape is
`1+u_b = sqrt((1+r_fwd)/(1+r_rev))` (`_combine_uplift`), which cancels a *common* per-bin multiplicative
shrinkage `s` — valid only when both directions bin a given operating point into the same bin. For ws/TI both
directions read the same treatment-invariant signal, so `s` cancels. The `power` frame instead labelled the
**forward** side by its counterfactual prediction `pred_up` (a regressor) but the **reverse** side by the actual
baseline power `y[mb]` — which is *also the reverse energy ratio's numerator* (`Σy[mb]/Σpred_base`). Binning a
ratio on its own numerator selects each bin on that variable's noise: a low-power bin over-selects downward
noise (`Σy[mb] < Σpred_base` → `r_rev < 0`), a high-power bin over-selects upward noise (`r_rev > 0`). So
`r_rev` tilts negative→positive across power. The forward side, binned on a prediction, carries no matching
tilt, so nothing cancels it; and because `r_rev` sits in the **denominator** of the combine, the tilt inverts:
low power `1+r_rev<1 → shape>1 → +uplift`, high power `shape<1 → −uplift`. It is classic regression to the mean,
the same model-error signature F5 saw in the residual-vs-actual-power diagnostic — surfaced into the estimate
once `power` became a scored conditional axis (F17).

**Fix.** Label the reverse side by its counterfactual prediction `pred_base` too, so both directions bin on a
(treatment-invariant) *prediction of the same untreated power*: neither side bins on its own noisy numerator (no
RTM), and the per-bin shrinkage is common again so it cancels in the combine — which is exactly what the combine
was designed to assume. The old comment's objection (`pred_base` is "a treated estimate") does not bite:
`pred_base` carries the same shrinkage the forward side does, and cancelling that shrinkage is the whole point of
the two-direction combine. The bin label changed; the ratio contents (`y[mu]/pred_up`, `y[mb]/pred_base`) did not.

**Evidence.**
- *Placebo, isolated.* `cp_0pct` 12 mo prepost power bins collapsed to truth: `(-230,230]` +8.24→**−0.25 pp**,
  `(230,690]` +2.52→+0.09, `(690,1150]` +0.77→+0.18, `(1150,1610]` −0.59→+0.08, `(1610,2070]` −1.32→+0.23,
  `(2070,2530]` −0.98→−0.02. Every bin `better`; all six within ±0.25 pp of 0.
- *Full both-mode sweep, benchmark diff (mean per-cell |bias| change vs the pre-fix committed baseline, pp;
  Δ<0 = better):*

  | mode | overall | ws | ti | power |
  | --- | --- | --- | --- | --- |
  | prepost | 0.000 | −0.000 | 0.000 | **−2.177** |
  | toggle | −0.000 | −0.000 | −0.000 | **−1.665** |

  The change is fully isolated: overall/ws/ti move ≤0.005 pp (float noise) in both modes, so **overall P50 is
  unchanged** and the energy-aggregation identity is preserved. On `power`, 149/210 prepost cells improved by
  >0.5 pp, 11 worsened.

**Residual (not fixed, documented).** The worsening concentrates in one bin, `(1610,2070]` (≈0.7–0.9 of rated —
the power-curve knee), which moved from ~0 to **~+2 pp** across several profiles. This is genuine model
conditional-calibration error at the rated ceiling (asymmetric residuals where predictions clip), previously
*masked* by the larger RTM tilt — not a new artifact. It is a candidate for the F5 baseline-residual calibration
idea; left for a later cycle since the net axis |bias| still dropped ~2 pp.

**Decision: accepted.** `rev_cond=pred_base` is committed as the default; `study_power_model_compare_baseline.json`
regenerated from the full sweep (both modes) and promoted via `--accept-candidate`. Uncommitted (user does git).

---

## F22 — for toggle, a campaign-only `power_model` *headline* (drop pre-campaign data entirely) is neutral-to-better and leaves the conditional shape unchanged — bears on Issue 17

*2026-07-14 — a throwaway A/B while scoping Issue 17 (reconciling the conditional estimator's data
path with the headline). Reproduce: `benchmarking.harness.score_study`, toggle, profiles
`cp_0pct`/`ws_dependent_cp`/`ti_dependent_cp`, campaigns {1, 12} mo, 4 replicates, seed 0; default
`PowerModelMethod` vs the same method wrapped to drop rows before the toggle start first (reusing
`naive_ratio.restrict_to_campaign`). The "default" arm reproduced the committed toggle cells in
`study_power_model_compare_baseline.json` bit-identically, so the A/B is trustworthy.*

**Where pre-campaign data enters `power_model` at all (toggle):** only the **headline** counterfactual
fit (`baseline_sel` = pre-campaign rows + campaign-off rows, adaptive-half-life weighted, F20). The
**conditional** step already excludes pre-campaign rows (`baseline_sel & _campaign_mask`), so dropping
pre-campaign data only touches the headline.

**Overall P50** (pp; score = √(bias²+spread²), lower better), mean over the three profiles:

| campaign | arm | bias | spread | score |
| --- | --- | --- | --- | --- |
| 1 mo | default (all-data) | **−0.31** | 0.23 | 0.39 |
| 1 mo | campaign-only | **+0.02** | 0.34 | 0.34 |
| 12 mo | default (all-data) | +0.07 | **0.18** | 0.19 |
| 12 mo | campaign-only | +0.07 | **0.11** | 0.13 |

- **1 mo:** dropping pre-campaign data removes a small negative bias (~−0.3 pp → ~0) but raises spread
  (0.23→0.34) — the pre-campaign rows stabilise a data-starved short campaign at the cost of pulling
  the estimate down. Net score slightly better without them.
- **12 mo:** bias identical; campaign-only has **lower spread** (0.18→0.11) and better score
  (0.19→0.13). With a full campaign the extra history buys no bias reduction and only injects
  across-replicate variance (each replicate's pre-campaign window differs).

**Conditional distributions: essentially unchanged** — per-bin |bias| moved ≤ ~0.04 pp (noise) in
every ws/TI bin across all three profiles. Expected from the code: the two-direction CEM fits already
run on campaign-only matched data in both arms, so their *shape* is identical; only the re-level anchor
(the headline) moves, and at 12 mo the headline barely moves.

**Implication for Issue 17.** Issue 17 frames unification as pushing the *conditional* toward the
headline (all-data + adaptive half-life + weighted/restricted matching). This probe shows the
**opposite direction is also on the table and looks cleaner for toggle**: make the *headline*
campaign-only (matching the conditional), which unifies the data path, is neutral-to-better on overall
P50, and leaves the conditional untouched. It also **qualifies F21's parenthetical** ("the committed
benchmark already shows all-data winning decisively at 3–12 mo"): that was an inference from the F16
regime map, not a direct all-data-vs-campaign-only-drop A/B — measured directly here, campaign-only
*improves* toggle spread at 12 mo rather than hurting. Caveats: narrow probe (toggle only, 3 profiles,
2 campaign lengths, 4 replicates); prepost is untouched (its baseline *is* the pre-campaign data, so
there is no such knob there).

---

## F21 — prune the historic opt-in knobs: `power_model` now presents only the winning configuration (Issue 16)

*2026-07-10 — code hygiene, not a methodology change. Every removed knob lost its A/B and was off/absent
in the shipped default, so the committed benchmark is unchanged (the acceptance test: a default-config
`study_power_model_compare.py` sweep reproduces `study_power_model_compare_baseline.json` bit-identically,
no `--update-baseline`). The point is a smaller surface, less overfit temptation, and code that reads as
the successful approach.*

### Removed (each off/absent in the default; the findings verdict that retired it in brackets)
- `calibrate_slope` [F14], `calibrate_residuals` [F15] — headline calibrations that never transferred;
  with them go `fitting.py`'s `fit_calibration_line` / `cell_residual_calibration` / `CalibrationLine` /
  `early_stopped_n_estimators` and the method's `_oof_baseline_predictions` / `_fit_calibration` /
  `_residual_corrections`.
- `early_stopping` [F14 — neutral, +15% runtime], `n_seed_ensemble` [F14 — spread is weather-sampling,
  not seed noise, +75% runtime].
- `toggle_estimator="double_ratio"` **and** the temporary `rho_off_scope` knob [F16/F19 — never beats the
  counterfactual on score]. With `double_ratio` gone, `toggle_estimator` was single-valued → dropped; the
  counterfactual energy ratio is the sole toggle headline (`_fit_predict_double_ratio` / `_rho_off_mask`
  removed).
- `time_features` (+ `latitude`, `longitude`) [F11 — all rejected], `era5_derivations` (+ `hub_height_m`)
  [F10 — all rejected *as model features*]. `era5_derived.py` and `time_features.py` **stay as utilities**
  (CEM matching / `inspect_era5_matching_importance.py` use the derivations); only the model-feature wiring
  and the method-surface knobs went.
- The injectable **`model_factory` seam** and its `OUTCOME_MODEL_FACTORIES` (`hgb`/`linear`) registry —
  removed entirely: **no driver, example or inspection script used it** (the issue's "if nothing uses it,
  remove it too"). LightGBM `make_outcome_model` is the sole outcome model; a future Phase-2 learner
  re-introduces a seam when a driver actually needs one. `fitting.py` thinned to just `time_block_folds`.

### Kept (so it isn't re-litigated)
Settled defaults: `era5_exclude=CURATED_ERA5_EXCLUDE` (F13), `availability_feature=False` (F13),
`reference_stat_cols` (schema), `matching_vars`/`matching_bin_edges` (F6), the adaptive time-decay default
and its `time_decay_half_life_days` expert override (F20), `_MIN_BIN_MATCHED_COUNT` conditional floor (F17).

### `toggle_campaign_only` — confirmed redundant, removed from `power_model`
The F20 adaptive half-life is a *soft* campaign-only at short campaigns, so the hard knob was expected
redundant. One cheap A/B settled it (`inspect_short_campaigns.py`, toggle 1–2 mo, placebo `cp_0pct` +
recovery `cp_plus_3pct`, 4 replicates): adaptive default (`tco=False`) vs adaptive + `tco=True`, overall
score pp:

| profile | mo | default | tco_true |
| --- | --- | --- | --- |
| cp_0pct | 1 | 0.380 | 0.334 |
| cp_0pct | 2 | 0.201 | 0.191 |
| cp_plus_3pct | 1 | 0.391 | 0.343 |
| cp_plus_3pct | 2 | 0.206 | 0.194 |

Campaign-only is marginally better at 1–2 mo (≤0.05 pp, near-noise on 4 replicates; the default carries a
small −0.3 pp bias, campaign-only near-zero bias but higher spread), and the committed benchmark already
shows all-data winning decisively at 3–12 mo — where a hard `tco=True` would hurt. The soft adaptive
window captures the short-campaign benefit without the long-campaign cost, so the knob does not earn its
place: **removed from `PowerModelMethod`** (always all-data; the conditional step still matches within the
campaign via `_campaign_mask`). `naive_ratio` keeps its own `toggle_campaign_only` — a different method,
untouched.

### Mechanics
Deleted the knobs, their config-validation branches, their `_config_params` entries and plumbing, and
their unit tests; thinned `fitting.py`; the run-config YAML loses the removed keys (a per-run diagnostic,
not the committed benchmark). `inspect_short_campaigns.py` lost its `double_ratio*`/`tco_true` arms;
`study_power_model_compare.py` help/docstring examples updated off the removed `era5_derivations`. Public
`PowerModelMethod` surface dropped from 30 to 21 constructor params. `poe all-fast` green. Acceptance test
passed: the full default-config sweep is bit-identical to the committed benchmark — **0 better / 0 worse of
819 prepost + 791 toggle conditional cells** on spread, score and |bias|, leaderboard deltas all zero.

---

## F20 — the headline training window self-configures: an adaptive time-decay half-life (2 × campaign duration) is the new default, subsuming the fixed 548 d; big short-campaign wins, tied long, in both modes (Issue 15 deliverable 2)

*2026-07-10 — Issue 15's mechanism-justified fallback after F19 ruled out the `double_ratio` path. New
`PowerModelMethod` default `adaptive_time_decay: bool = True`; the headline fit's time-decay half-life
is now `_TIME_DECAY_CAMPAIGN_MULTIPLE (=2.0) × campaign_duration_days` via a new
`_effective_half_life`. `time_decay_half_life_days` is demoted to an **expert override** (default flipped
548.0 → `None`, used only when `adaptive_time_decay=False`; a guard forbids setting both). Conditional
two-direction fits stay unweighted (F16). A/B'd via `study_power_model_compare.py` (plain run =
adaptive-default vs the fixed-548 committed benchmark), placebo `cp_0pct` both modes for the coarse `k`
sweep, then the full 7-profile both-mode sweep for the ship.*

### Mechanism — a scale-free "trust window", not a lookup
The best fixed half-life is regime-dependent (F16: ~90 d helps short campaigns in **both** modes; ≥1 yr
is safe long; 548 d fixed was a compromise). Making the half-life **proportional to the campaign's own
duration** — `k × campaign_days` — gives a short half-life for a short campaign (down-weight the stale
pre-campaign era that dominates a sliver campaign → less bias *and*, in toggle, less spread) and a long
one for a long campaign (use the plentiful recent data). One dimensionless `k` = "trust pre-campaign
data within ~k campaign-durations". A scale-free multiple is chosen over a length→half-life lookup
precisely to avoid benchmark overfitting: it is mechanism-anchored and generalises to any campaign
length or dataset span, including F16's "20 yr of SCADA, 1-month campaign" case by construction.
`k=2` → 1mo≈60 d, 3mo≈182 d, 12mo≈730 d (≈ today's 548 d regime at 12 months, so a strict
generalisation of the accepted default).

### Coarse `k` sweep (placebo `cp_0pct`, both modes, overall score pp; all vs fixed-548 benchmark)
All three adaptive `k` beat fixed-548 on pooled mean in **both** modes (prepost 0.55 → ~0.50, toggle
0.34 → ~0.25). Per-length the best `k` is scattered across {1.5, 2, 3} — the signature of a flat region
where finer tuning would fit placebo noise. **k=2 chosen**: best prepost pooled mean (0.493), 2nd toggle
(0.248, a hair behind k1.5's 0.232), and it wins the headline **toggle-1mo** case (0.380 vs k1.5 0.400,
k3 0.482). It is the middle of the bracket (least overfit-prone) and matches the a-priori mechanism
anchor; chasing k1.5's toggle-2mo edge would cost prepost-1mo/6mo. `k` lives as the module constant
`_TIME_DECAY_CAMPAIGN_MULTIPLE`, **not** a user knob, so it is documented without inviting per-dataset
tuning.

### Full 7-profile sweep verdict (overall P50, pooled per length, dScore pp; <0 = adaptive better)
| mo | prepost Δ | toggle Δ | cells worse >0.1pp |
| --- | --- | --- | --- |
| 1 | +0.067 | **−0.368** (all 7) | 0 |
| 2 | **−0.338** (all 7) | **−0.144** (all 7) | 0 |
| 3 | +0.014 | +0.024 | 0 |
| 6 | −0.027 | −0.005 | 0 |
| 12 | −0.005 | +0.003 | 0 |

Pooled mean score **prepost 0.559 → 0.501, toggle 0.350 → 0.252**. **Zero cells regress by >0.1 pp**
anywhere; the only positive deltas (prepost-1mo +0.067, 3mo +0.01–0.02) are uniform across profiles and
inside the 0.1 pp materiality band — i.e. tied. The prepost-1mo +0.067 is a spread effect (the shortest
prepost campaign wants maximal baseline; a 60 d half-life thins it) and neither a longer `k=3` (+0.084)
nor shorter `k=1.5` (+0.114) helps it — an inherent short-prepost tension the adaptive rule accepts to
buy the large toggle short-campaign gains. Conditional decomposition (re-levels to the now-adaptive
headline; the two-direction shape is unweighted and unchanged) is neutral-to-better: prepost mean
|bias| −0.87 pp (46 better/17 worse of 69), toggle −0.10 pp (32/32), no cell flagged a material
regression. Benchmark JSON regenerated on the new default.

### Decisions / follow-ups
- `adaptive_time_decay=True` is the shipped default; the fixed half-life is the expert escape hatch.
- **`toggle_campaign_only` demotion not yet decided** — the adaptive half-life is a *soft* campaign-only
  at short campaigns, so whether `toggle_campaign_only` is now redundant is a separate confirmation
  (tracked, not done here).
- The temporary `double_ratio` `rho_off_scope` knob (F19) is still present; removing it (shipping
  era-local as `double_ratio`'s behaviour) is the remaining knob-cleanup deliverable.
- `inspect_short_campaigns.py`'s `hl90`/`hl365` arms updated to set `adaptive_time_decay=False`.

---

## F19 — the era-local `double_ratio` gate: no `double_ratio` variant beats the counterfactual default, so the self-configuring toggle default is not an estimator swap (Issue 15 deliverable 1)

*2026-07-10 — Issue 15's first, gated step. New temporary A/B knob `PowerModelMethod.rho_off_scope`
(`"campaign"` default / `"all"`): `double_ratio`'s calibration ratio `rho_off = Σy_off/Σpred_off` is now
measured over the **campaign-window off rows only** (era-local), the fold models still training on all
off rows; `"all"` reproduces the pre-Issue-15 behaviour. New `_rho_off_mask` + threaded campaign mask;
to be removed with the knob cleanup. A/B'd on the toggle `cp_0pct` placebo at 1/2/3/6/12 months via
`study_power_model_compare.py --method-overrides`.*

### Result — era-local fixes the 1-month bias but does not dominate
Toggle `cp_0pct` placebo, score = √(bias²+spread²) pp (cf = counterfactual default = the committed
benchmark; gl = global-`rho_off` DR; el = era-local `rho_off` DR):

| mo | bias cf/gl/el | spread cf/gl/el | score cf/gl/el |
| --- | --- | --- | --- |
| 1 | −0.56 / −0.71 / **+0.04** | 0.49 / 0.69 / 0.81 | **0.74** / 0.99 / 0.81 |
| 2 | −0.17 / −0.24 / −0.22 | 0.30 / 0.50 / **0.25** | 0.34 / 0.56 / **0.34** |
| 3 | +0.11 / **−0.01** / +0.41 | 0.20 / 0.31 / 0.34 | **0.23** / 0.31 / 0.53 |
| 6 | +0.15 / **+0.07** / +0.16 | 0.15 / 0.22 / 0.26 | **0.22** / 0.24 / 0.31 |
| 12 | +0.08 / −0.03 / −0.01 | 0.17 / 0.24 / **0.11** | 0.19 / 0.25 / **0.11** |

Era-local fixes `double_ratio`'s 1-month **bias** (−0.71 → +0.04) by keeping `rho_off` in the ON
window's era, but the campaign-window off set is tiny (**2150 of 101532 rows at 1mo**), so `rho_off`
becomes a noisy ±0.5 % multiplier (1.004/1.009/0.994 across replicates vs the stable global 0.99934)
and **spread balloons**. On the composite score it does **not** dominate: the plain counterfactual
default is best-or-tied at 1/2/3/6 months; era-local only wins at 12mo; global-`rho_off` DR is dominated
almost everywhere. Prepost is untouched (the overrides are toggle-only). Verified on current code, not
just F16's recorded numbers.

### Root cause — a well-calibrated model leaves `double_ratio` nothing to do
`double_ratio` = `rho_on/rho_off − 1`; its value is cancelling model *miscalibration* shared between
the on/off sides. But with **all-data training the model is already well-calibrated** (logged
`rho_off = 0.99934` ≈ 1), so the ratio-of-ratios only **injects estimation noise** — visible in the
spreads (global-DR spread exceeds the plain counterfactual sum's at every length ≥2mo). `double_ratio`
only earns its keep when the model *is* miscalibrated — i.e. campaign-only / small fits — but F16 found
campaign-only `double_ratio` overcorrects and its 3-month spread balloons. It is boxed between "nothing
to correct" (all-data) and "too noisy to correct" (campaign-only); a smoother `rho_off` does not free it.

### Decision
No `double_ratio` variant beats the counterfactual default on score → the self-configuring toggle
default is **not** an estimator swap. The mechanism-justified lever is the training window
(drift-vs-shrinkage; F16's crossover), pursued as the adaptive time-decay half-life (**F20**). Era-local
`rho_off` is kept as `double_ratio`'s behaviour (a strict fix to the opt-in); the `rho_off_scope` knob is
temporary and slated for removal in the knob-cleanup deliverable.

---

## F18 — the frozen reference dir regenerated on current code and verified; the compare default repointed at it; a clean four-method bias/spread read (F17 stale-reference follow-up)

*2026-07-09 — the F17 stale-reference flag closed out. A full overnight run
(`study_overnight_prepost` + `study_overnight_toggle`, both `include_v0=True`, seed 0, 4 replicates,
1/2/3/6/12 months, all seven `overnight_profiles`) was produced on current committed code as
`~/temp/wind-up-benchmarking/badass overnight 20260708/`. `study_power_model_compare.py` now defaults
`_DEFAULT_REFERENCE_DIR` to it (was the unreproducible "30 June" run) and its `_load_reference_methods`
tolerates the timestamped `<mode>/<YYYYmmdd_HHMMSS>/` subdir that `start_overnight_run` writes, so an
overnight run drops in as a reference with no manual flattening.*

### Consistency verified (the run is safe to use as the frozen reference)
- **Config is identical** to the compare grid and the committed benchmark: `campaign_months=[1,2,3,6,12]`,
  `n_replicates=4`, `seed=0`, all seven profiles, both modes.
- **v0 now spans the whole grid.** `v0_binned` is present at all of 1/2/3/6/12 months × 4 replicates in
  both modes — fuller than the 30-June reference, which had no v0 below 3 months. The compare script
  reuses only v0 (`REUSED_METHODS = ["v0_binned"]`), so this is exactly what it needs.
- **power_model matches the committed benchmark within noise.** Reproducing `power_model_leaderboard`
  from the run and diffing against `study_power_model_compare_baseline.json`: **toggle** bit-identical
  (max |Δ| < 1e-6 on bias/spread/score), **prepost** max |Δbias| = 0.019 pp — well under the 0.1 pp
  materiality band. (The reference run's own power_model carries only `overall`+`ws` conditional cells,
  no `ti`, because the overnight driver `example_prepost_study.py` omits `wind_speed_sd_col`; irrelevant
  to the compare workflow, which recomputes power_model fresh with TI and only reuses v0.)
- Ground truth is method-independent, shared across all four methods per case, so the compare script's
  `_check_alignment` guard passes on the full intersection.

### The four-method P50 read (overall condition, pooled 7 profiles × 5 lengths × 4 reps; pp)
`oracle` = 0.00 bias / 0.00 spread in both modes, confirming the truth anchor.

| method | prepost bias | prepost spread | toggle bias | toggle spread |
|---|---|---|---|---|
| naive_ratio | −1.02 | 1.78 | +0.14 | 0.44 |
| v0_binned | −0.96 | 1.01 | −0.06 | 0.37 |
| power_model | +0.04 | 0.64 | −0.08 | 0.40 |

- **Prepost is the hard regime; toggle is easy.** Every method is several-fold tighter and less biased in
  toggle. power_model dominates prepost on both axes; in toggle v0 (0.37) and power_model (0.40) are
  comparable and naive (0.44) is only slightly behind.
- **naive and v0 under-report by ~1 pp in prepost**, and that negative bias is roughly *constant across
  upgrade size* (naive −0.8…−1.3 pp, v0 −0.5…−1.4 pp over all seven profiles) — a floor/detrend offset,
  not a scaling error. **power_model removes it** (+0.04 pp, uniform across every upgrade incl. the
  `cp_0pct` placebo) — the key robustness result.
- **naive's toggle bias scales with the true uplift** (`cp_plus_10pct` +0.44, `cp_minus_10pct` −0.27 pp):
  a proportional over-read the ratio estimator has and the two models do not.
- **`rated_plus_5pct` is v0's consistent weak spot** — its largest |bias| in both modes (prepost −1.44,
  toggle −0.69 pp).
- **Precision improves with campaign length** for all methods, most steeply for power_model in prepost
  (1 mo 1.00 → 3 mo 0.23 → 6 mo 0.20 pp); at 12 months toggle spreads fall to ~0.14–0.17 pp across the
  board. The 1-month prepost cell is where everyone struggles (naive 1.76, v0 1.06, power_model 1.00 pp).

### Implications
- v0 comparison numbers are now trustworthy (F17's blocker cleared); the leaderboard's v0-vs-power_model
  gap in prepost is a genuine current-code result, not a stale-reference artefact.
- Any future overnight run is a drop-in reference (loader handles the timestamp subdir); pointing
  `--reference-dir` at a run holding two-or-more run subdirs per mode fails loudly rather than guessing.

---

## F17 — conditional decomposition hardened (count floor + physics imputation + corrected re-level); out-of-the-box method promoted onto the class and scored 1–12 months; the frozen reference is stale (Issue 14)

*2026-07-08 — Issue 14, both efforts. New module
`benchmarking/baselines/power_model/conditional.py` (`impute_uncovered_bins`, `relevel_conditional`).
New `PowerModelMethod` internals: the per-reporting-bin count floor `_MIN_BIN_MATCHED_COUNT = 50`,
imputation + corrected re-level wired into `_conditional_by_bin`, and a `covered` flag on the per-run
`conditional/` CSV. Class defaults promoted (see below). `study_power_model_compare.py` now scores
1/2/3/6/12 months in both modes with `naive_ratio` recomputed fresh; benchmark JSON regenerated on the
new grid under the new conditional default. A/B'd on the covered profiles (`cp_0pct`,
`ti_dependent_cp`, `ws_dependent_cp`) against the pure-B benchmark; full 7-profile regen for the ship.*

### Effort B — the benchmarked config is now the out-of-the-box config, scored 1–12 months
- **Defaults promoted onto the class** so a bare `PowerModelMethod` *is* the benchmarked method:
  `min_child_samples=50` (F14, merged under user `model_params` on the LightGBM path), 
  `availability_feature=False` (F13), and `era5_exclude=CURATED_ERA5_EXCLUDE` (F13) with the **untouched
  default** applied drop-if-present (a non-Open-Meteo frame is not broken; an explicitly-set exclude
  keeps the strict typo guard). `reference_stat_cols` stays driver-level — it names a source-specific
  SCADA tag (`wtc_ActPower_min`), i.e. schema description, not tuning.
- **Grid extended to 1/2/3/6/12 months both modes**; `naive_ratio` recomputed fresh (cheap, no wind_up
  pipeline) so the merge no longer depends on the reference run's naive; v0 stays reference-only and is
  simply absent below 3 months; the alignment guard now checks truth on the fresh∩reference
  intersection only.
- **Out-of-the-box power_model, overall P50, mean over 7 profiles [pp]** (regenerated benchmark). The
  short-campaign regime (F16) is now visible in the committed benchmark:
  | months | prepost bias / spread / score | toggle bias / spread / score |
  | --- | --- | --- |
  | 1 | −0.56 / 1.00 / 1.15 | −0.57 / 0.50 / 0.76 |
  | 2 | +0.04 / 0.53 / 0.54 | −0.17 / 0.30 / 0.35 |
  | 3 | +0.14 / 0.23 / 0.27 | +0.11 / 0.21 / 0.23 |
  | 6 | +0.44 / 0.20 / 0.49 | +0.15 / 0.16 / 0.22 |
  | 12 | +0.12 / 0.34 / 0.36 | +0.08 / 0.17 / 0.19 |

### Effort A — the conditional decomposition, hardened
- **Every per-bin number is now a trustworthy measured value or a flagged, physics-informed
  imputation.** A bin is `covered` only if its two-direction shape is finite **and** both directions
  have `≥ _MIN_BIN_MATCHED_COUNT` matched rows; otherwise it is imputed (ws: bfill from the nearest
  covered bin above, then 0 uplift above the last covered bin — 0-at-rated; ti: the overall uplift) and
  flagged `covered=False`. Never a bare NaN, so `summarize_errors` (which drops non-finite errors)
  cannot be gamed by abstention, and the imputation prior is itself benchmarked.
- **Corrected re-level (the pure-bug fix).** `relevel_conditional` pins imputed bins at their imputed
  uplift and solves one λ over the **measured** bins only
  (`λ = S_m / (Σactual/one_plus_overall − C_i)`), so measured + imputed together energy-aggregate to the
  headline **exactly** even when coverage is imperfect (the old re-level solved λ over covered bins only
  and silently absorbed the uncovered bins' MWh). Guards: no measured bins or a non-positive denominator
  → overall uplift in every bin.
- **Result (A/B vs the pure-B benchmark, 309/321 conditional cells over all lengths + covered
  profiles):** overall P50 **bit-identical** (Δbias/Δspread/Δscore = 0.000 pp — the headline is the
  single full-window fit, untouched by construction); conditional mean **Δscore −3.7 pp prepost / −2.6 pp
  toggle** (Δbias −2.0/−1.4, Δspread −2.7/−2.0). The before/after view (longest campaign, covered
  profiles) reads **12 better / 57 ~ / 0 worse** prepost and **15 better / 54 ~ / 0 worse** toggle — the
  F7/F9 sparse-extreme "worse" bins are gone.
- **Floor value chosen on evidence, not the one bad bin** (`floor_threshold_evidence.py`, 14.6k bins
  from the A/B run): the raw two-direction shape `|u_b|` p90 by per-side matched count is 13 pp (<10,
  mostly degenerate → imputed anyway), **60 pp (10–25), 55 pp (25–50)**, 52 pp (50–100), then falls to
  25 pp (100–200), 13 pp (200–500), 10 pp (500+). The combine is untrustworthy below ~50 (a floor of 25
  would leave the wild 25–50 bucket unfloored); 50 is the smallest value that catches it, and raising
  higher would impute away ~940 moderately-populated 50–100 bins for no done-when gain. Kish ESS was not
  needed (no balance reweighting shipped — see below), so the floor compares raw per-side counts.

### Decisions
- **Coverage stays method-internal (user decision, trims the issue's "coverage in the leaderboard"
  bullet).** The `covered` flag lives only on the per-run `conditional/` CSV and drives the re-level;
  the harness seam (`MethodOutput.p50_by_condition`) is unchanged `[condition, condition_bin,
  p50_uplift]`. Rationale: once imputation makes every bin finite, `summarize_errors` drops nothing, so
  the F16 "deltas dominated by a few exploding cells" problem — the reason a coverage view was wanted —
  is dissolved by the imputation itself. The method reports its single best per-bin estimate; the
  leaderboard scores that.
- **Per-bin balance (post-stratify each reporting bin to the intersection of the two directions'
  ERA5-cell supports) — DEFERRED, not shipped.** The floor + imputation + corrected re-level already
  clear every Issue-14 done-when with margin (0 worse bins, conditional score materially better, overall
  unchanged), so per the adopt-only-if-it-helps protocol the extra within-bin reweighting is not built —
  it would thread ERA5 cell codes and a custom per-bin reduction into the core conditional path for a
  residual imbalance the floor already tames (YAGNI). Tracked as a follow-up; the design is a pure
  intersect-cell-support mask over the matched fwd/rev rows within each reporting bin, with the floor
  switching to Kish ESS `(Σw)²/Σw²` if it is ever adopted.

### The frozen reference dir is stale (flagged for regeneration)
- The naive-consistency check (`naive_consistency_check.py`, requested to run once before trusting the
  reference) **failed**, but in the *good* direction: current-code `naive_ratio` on the `cp_0pct`
  placebo has a **much tighter prepost spread** than the frozen 30-June reference (3-mo score 1.91 vs
  7.19, spread 1.57 vs 7.15; 6/12-mo better too), toggle essentially identical. Every input to naive —
  `naive_ratio.py`, `filtering.py`, the study config, the turbine set, the data span, and (via the
  passing alignment guard) the ground truth — is byte-identical between the reference era and HEAD, and
  the reference predates power_model (methods `[naive, oracle, rlearner, v0]`, ~26–30 June), so the
  30-June run was produced by a **local/uncommitted state** git can't reproduce. The committed benchmark
  is power_model-only and alignment-guarded, so this does not affect it; naive is now recomputed fresh
  (the correct, current-code bar). **The frozen v0 in that reference is from the same stale state** and
  should be regenerated (run `study_overnight_prepost` / `study_overnight_toggle`, both `include_v0=True`,
  on current committed code) before v0 comparison numbers are trusted.
- **Resolved in F18** — the reference was regenerated on current code and verified consistent; the
  compare script now defaults to it.

---

## F16 — a finite time-decay half-life (548 d) is the default, applied to the headline fit only; the double-ratio toggle estimator validated as opt-in; the 1–2-month regime flips the training-window verdict (Issue 13 extension)

*2026-07-04/05 — three user-directed follow-ups to F15, A/B'd against the post-F15 benchmark.
New: `toggle_estimator="double_ratio"` on `PowerModelMethod` (the naive-adoption hybrid) and
`benchmarking/baselines/inspect_short_campaigns.py` (1–2-month campaigns are outside the committed
benchmark grid; oracle + naive anchor them — the oracle scores exactly 0 there, so the harness
itself is sound at those lengths).*

### ACCEPTED — `time_decay_half_life_days = 548` (1.5 years), headline fit only; benchmark regenerated
- **Why finite at all:** the method must work on any dataset; with 20 years of SCADA an unbounded
  training window would let ancient eras dominate the campaign era. `0.5^(days outside the
  campaign interval / 548)`: rows in the campaign weigh exactly 1, year-old data ~63%,
  decade-old ~1%.
- **Dose curve (cp_0pct, both modes):** 90/180/365/548/1096 days all leave the toggle overall
  neutral-or-better; prepost overall bias nudges up slightly (+0.1–0.2 pp at 3 months) and
  everything ≥365 d is indistinguishable on this 2.5-year dataset — the specific choice of 548
  rests on the bounding argument, not a HoT win. Short half-lives (90 d) measurably help
  1–3-month campaigns in **both** modes (prepost 2-month score 0.21 vs 0.59 unweighted; toggle
  best-or-near-best at 1/2/3 months) at a small long-prepost bias cost — the documented
  short-campaign tuning.
- **Headline fit only.** Weighting the conditional two-direction fits destabilises the sparse
  extreme-TI tail bins (a degenerate matched fit in one replicate read +1039 % in one bin; it
  appeared at hl365/548 and not at 180/1096 — replicate chance, the F7/F8 tail fragility that
  Issue 14's count floor addresses). The matched contrast is already era-insensitive (its common
  shrinkage cancels), so the weights buy nothing there. With weights confined to the headline
  path the full sweep is clean: prepost conditional ALL Δscore +5.4 (weighted) → **+0.03**
  (headline-only); every overall row in both modes inside the ±0.1 pp neutrality band, toggle
  slightly better everywhere.

### Validated opt-in — `toggle_estimator="double_ratio"` (the naive-adoption hybrid)
- The model is demoted to removing condition unfairness; the headline is a ratio of ratios,
  ``(Σy_on/Σpred_on) / (Σy_off/Σpred_off) − 1``. Off rows are predicted out-of-fold and on rows
  by the same fold-model ensemble (basis-consistent, the F15 rule), so the model's shrinkage /
  level bias cancels between the interleaved sides instead of needing to be zero.
- **On the benchmark grid it does exactly what it promises:** toggle placebo headline bias ≈ 0 at
  every campaign length (−0.00/+0.08/−0.06/−0.05 pp vs +0.12/+0.16/+0.08/+0.08 for the default),
  score neutral. The logged ρ_off (0.9995–1.0006 on ~100k-row fits) is precisely the residual OOF
  shrinkage, cancelled by construction.
- **Not the default, for two measured reasons:** (i) with campaign-only training it overcorrects
  and the 3-month spread balloons (0.26 → 0.80 — five fold models on ~6k rows are noise); (ii) at
  1–2-month campaigns it is *worse* than the default (score 1.05/0.59 vs 0.90/0.38): with
  all-data training, ρ_off is measured across two years of eras while the ON window is a
  one-month sliver, so the "cancellation" subtracts the wrong era's miscalibration. Right tool
  for ≥3-month toggle campaigns where headline-bias purity matters.

### The 1–2-month regime check (the reason the user asked for it)
- **The Issue 13 training-window verdict flips below ~3 months:** campaign-only training wins at
  1–2 months (toggle 1-month score 0.334 vs 0.895 for all-data; bias +0.02 vs −0.64) because the
  campaign is <5 % of the all-data training set and drift dominates; all-data wins at ≥3 months
  where shrinkage dominates. The decay weights interpolate between the regimes: hl90 is
  best-or-near-best at 1, 2 *and* 3 months. A campaign-length-adaptive training window (or
  half-life) is the natural future refinement — noted for the later-work list, not implemented.
- The 3–12-month verdicts (the committed benchmark grid) all stand: mcs=50, tco=False, and the
  F12/F13 feature set were re-checked where they can flip and none reversed on that grid.
- power_model beats the naive anchor at 1–2-month toggle only in its campaign-only configuration
  (0.334 vs 0.518 at 1 month); in the default all-data configuration naive wins there — worth
  remembering when quoting short-campaign capability.

### Method notes
- The weights default surfaced a seam conflict: sklearn ``Pipeline`` factories take no
  ``sample_weight``. ``_fit_kwargs`` now probes support (``has_fit_parameter``) and fits
  unweighted with a warning instead of crashing the factory seam.
- 1–2-month campaigns are deliberately **not** added to the committed benchmark grid (the frozen
  v0/naive reference runs don't cover them); `inspect_short_campaigns.py` is the reproducible
  driver for that regime.

---

## F15 — residual calibration rejected with a sharpened OOF-transfer rule; `toggle_campaign_only=False` accepted (all-data headline training + campaign-restricted conditional); time-decay weights validated as opt-in (Issue 13)

*2026-07-04 — Issue 13 verdicts. New `PowerModelMethod` knobs (default off): `calibrate_residuals`
(ERA5-cell residual calibration: full-baseline out-of-fold predictions via the shared time-blocked
folds, mean residual per F6 CEM cell — `matching.cell_codes` is now public — read out under the
upgraded window's occupancy) and `time_decay_half_life_days` (campaign-proximity sample weights
`0.5^(days outside the campaign interval / half-life)`: interleaved campaign rows weigh exactly 1,
pre-campaign rows decay; threaded through every fit). One structural change: with
`toggle_campaign_only=False` the conditional two-direction step now **matches within the campaign
only** — pre-campaign rows serve only the headline fit's training data. A/B'd on the placebo
(`cp_0pct`) against the post-F14 benchmark; acceptance via a full 7-profile sweep. Entering
Issue 13, the prepost placebo headline bias was already −0.02/+0.35/+0.05 pp at 3/6/12 months
(the F3-era uniform −0.4 pp is gone since F13/F14) and toggle +0.36 pp at 3 months (F14: small-fit
tree shrinkage).*

### REJECTED — ERA5-cell residual calibration, twice; the F14 OOF-transfer rule now covers *shape*
- **v1 (raw cell means)**: prepost bias up nearly uniformly (+0.32/+0.21/+0.13 pp) — the global OOF
  residual level (≈ −1 kW: fold models fit on 80% of the rows over-predict relative to the final
  100% fit) leaks into every cell mean and swamps the mix-shift signal. Toggle: right direction but
  a short-campaign spread cost (3-month Δspread +0.72 — the F7 failure mode; the level term is
  noise at small n).
- **v2 (centred: `cell_mean − global_mean`, the pure mix-shift differential; unseen cells → 0)**:
  prepost deltas nearly identical to v1 (+0.33/+0.20/+0.15). The logged corrections under the
  upgraded mix are systematically *negative* (−0.05…−4.9 kW) while the observed placebo bias is
  *positive* — **the estimated correction has the wrong sign vs the actual bias**. The fold
  models' conditional residual *structure* differs from the final refit model's, not just its
  level. Toggle adds a sparse-cell artifact: ~450 F6 cells over ~6k off rows ≈ 13 rows/cell of
  noise (uniform positive bias shifts decaying ~1/n).
- **The sharpened rule (extends the F14 method note): out-of-fold residuals cannot be transferred
  to a refit model — neither their level nor their conditional shape.** Any future residual
  calibration must be basis-consistent: estimate corrections for the model actually deployed
  (e.g. predict with the fold ensemble itself, or calibrate on a held-out era the final model
  never saw).
- **Corollary**: the remaining prepost 6-month +0.35 pp placebo bias is measurably *not* an
  ERA5-weather-mix-shift effect (the purpose-built correction moves it the wrong way). 3- and
  12-month prepost already meet the issue's ≲0.1–0.2 pp target; the 6-month anomaly stays open
  (candidate mechanism: a seasonal/drift interaction specific to half-year windows).

### ACCEPTED — `toggle_campaign_only=False` (all-data headline training; benchmark regenerated)
- **Where the all-data damage actually lives.** Every naive all-data arm wrecked the toggle
  conditional identically (3-month ti Δscore ≈ +14) regardless of decay weights or calibration —
  because the drift enters through the conditional CEM **matching** (pre-campaign rows matched
  against campaign on-rows at full weight in the two-direction contrast), not through the fits.
  Weighting cannot fix a matching problem; the structural fix (conditional matches within the
  campaign only, where its shared-distribution assumption holds) resolves it completely — the
  3-month conditional comes out *better* than the campaign-only benchmark (ti Δscore −1.7).
- **Full-sweep verdict (7 profiles)**: prepost bit-identical (the flip is toggle-only; 0.0 across
  all 469 cells). Toggle: 3-month headline bias +0.358 → **+0.121** (Δscore −0.070), pooled ALL
  Δscore −0.144, |bias| cells 144 better / 86 worse. Accepted cost: mild spread at 9/12 months
  (overall Δscore +0.11/+0.08) — the extra rows only add drift variance once the campaign is
  data-rich. Three-method picture: power_model now ties v0/naive at 3-month toggle (0.294 vs
  0.296/0.297), closing its last deficit vs v0 (F4); naive keeps the 9/12-month toggle lead.
- `naive_ratio` keeps campaign-only, per the issue — there the restriction *is* the method's
  distribution matching.

### Validated opt-in — `time_decay_half_life_days` (campaign-proximity training weights)
- On top of all-data + the conditional fix, 90-day half-life trades headline bias for spread:
  3-month bias +0.19 (vs +0.12 unweighted) but the best 3-month overall score of any arm (0.239
  vs 0.290 unweighted, 0.359 benchmark), and pooled ALL −0.142. The 45-day dose is flat vs 90
  (ALL 4.038 vs 4.045) — the response is insensitive in this range.
- **Not defaulted** because the weights are shared-path: in prepost they nudge the placebo
  headline bias *up* at all three campaign lengths (+0.09/+0.09/+0.02; score is a wash — 3-month
  −0.14 better, 6-month +0.12 worse) — the wrong direction for the issue's own target metric.
  Remains available where short-campaign spread matters more than bias purity.

### Method notes
- The refactor no-op was verified against the benchmark before any A/B (all deltas 0.0), and the
  uncentered→centred iteration was driven by the per-run correction logs — keep logging the
  mean correction and the centred-out level; they made the wrong-sign diagnosis possible.
- Issue 13's "time-blocked baseline cross-validation" item shipped across F14/F15: the holdout
  display is time-blocked (F14) and `_oof_baseline_predictions` gives every baseline row an
  out-of-fold prediction (F15), shared by both calibration paths.

---

## F14 — outcome-model fundamentals: `min_child_samples` 200→50 accepted; linear_tree, early stopping, calibration slope, seed ensembling and alternative learners rejected; the toggle headline bias localised to small-fit tree shrinkage (Issue 12)

*2026-07-04 — Issue 12 verdicts. New module `benchmarking/baselines/power_model/fitting.py`
(time-blocked folds, calibration line, early-stopped capacity pick, and the model-factory registry
`OUTCOME_MODEL_FACTORIES`) behind four new `PowerModelMethod` knobs — `model_factory` (str or
callable), `n_seed_ensemble`, `early_stopping`, `calibrate_slope` — all defaulting to today's
behaviour (a default-config re-run diffs the benchmark at ≤0.001 pp everywhere). The baseline
holdout *diagnostic* switched from a shuffled 20% split to a time-blocked fold (Issue 13's honest
display item; estimates unchanged). Candidates A/B'd per the Issue 9 protocol:
`study_power_model_compare.py --method-overrides` placebo screens (`cp_0pct`, both modes), full
7-profile sweep for the survivor, all diffed against the post-F13 benchmark.*

### Framing principles (recorded here so they aren't relitigated)
- **The objective targets the conditional mean.** The estimand is an energy ratio and energy is a
  sum of conditional means, so L2 stays (design note §2). Power conditional on features is skewed
  (near cut-in, around rated), so median-type objectives — MAE, Huber in its robust regime,
  quantile-0.5 — estimate the median and bias the energy sum. The F5 shrinkage is a
  *regularisation* artefact, not a loss artefact; changing objective does not fix it. Legitimate
  within the mean family: Tweedie / variance-weighted L2 (efficiency candidates, untried).
  Quantile objectives are out of scope for the point estimate (they return in Issue 19/WS4).
- **Tune on uplift metrics, never on prediction RMSE.** More regularisation can improve held-out
  RMSE while worsening shrinkage. Yardsticks: placebo bias on the harness, per-bin residual
  flatness, the predicted-vs-actual **calibration slope** (target ≈ 1) on a time-blocked held-out
  baseline, and replicate spread.

### ACCEPTED — `min_child_samples` 200→50 (now the driver default; benchmark regenerated)
- Placebo screen: prepost ALL Δscore **−0.62**, Δspread −0.75 (score cells 31 better / 9 worse);
  toggle overall better at every campaign. Dose-response check at `min_child_samples=20`
  overshoots (prepost ALL only −0.12, overall biases drift positive) — 50 is the sweet spot.
- Full 7-profile sweep: **prepost ALL Δscore −0.36, Δspread −0.41** (cells: score 216 better / 86
  worse of 469); overall P50 neutral-or-better at every campaign in both modes (prepost 3-month
  placebo |bias| 0.121 → 0.018). The accepted cost: a few large toggle ti cells at 9/12 months
  regress (9-month ti Δscore +3.2; toggle conditional mean +0.29 even though score cells split
  159 better / 145 worse) — accepted against the overall-P50/precision gains, the F13 precedent.
- Shipped as `TUNED_MODEL_PARAMS = {"min_child_samples": 50}` passed by the four HoT drivers;
  the design-note common params in `make_outcome_model` (shared with the R-learner) are unchanged.

### REJECTED — everything else, each with a one-screen placebo verdict
- **`linear_tree=True`** — mode-split (the F10 shear/veer pattern): toggle mildly better (ALL
  Δscore −0.36) but prepost overall bias worse at every campaign (+0.32/+0.28/+0.11 pp) and
  prepost conditional much worse (3-month ti Δscore +8.4). The hoped-for F5 edge-extrapolation fix
  adds bias/variance under the prepost weather shift instead.
- **`early_stopping`** (time-blocked valid split, refit at the picked capacity) — neutral overall
  in both modes *including the 3-month-toggle small-fit regime it was aimed at*; conditional
  slightly net-worse; +~15% runtime. The fixed design-note capacity is validated across the ~10×
  fit-size range.
- **`calibrate_slope`** — prepost neutral (nothing to correct), toggle **overcorrects**: bias
  flips sign (3-month +0.39 → −0.38) with a spread cost (Δspread +0.60) — the F7 short-campaign
  failure mode. Root cause: the line is fit on out-of-fold predictions from 80%-sized fits, which
  shrink more than the 100% fit it is applied to, and that gap is largest exactly where the
  correction is largest.
- **`n_seed_ensemble=4`** — overall deltas neutral in both modes (|Δspread| ≤ 0.02 pp) for ~75%
  more runtime: replicate spread is dominated by weather sampling across windows, not seed noise.
- **`model_factory="hgb"`** (sklearn HistGradientBoostingRegressor, capacity-matched) — prepost
  worse (ALL Δscore +0.26), toggle marginally better: LightGBM's behaviour is family-level, not an
  implementation quirk. Stays in the registry as the cross-implementation check.
- **`model_factory="linear"`** (impute→scale→Ridge structured baseline) — far worse overall as
  expected (prepost placebo bias +3.5/+2.5/+1.3 pp; misspecification dominates the hoped-for
  "shrinkage-free" property), **but the cross-check paid off** (next section).

### The toggle headline bias is small-fit tree shrinkage — three independent probes agree (Issue 13 hand-off)
- **Calibration slopes scale with fit size**: ~1.0005–1.005 at n≈100k (2-year prepost baseline),
  ~1.008–1.017 at n≈12–25k, **~1.013–1.027 at n≈6k** (3-month toggle off rows). The prepost
  baseline is essentially slope-calibrated, so the −0.4 pp prepost headline bias is *not* a global
  calibration artefact — consistent with Issue 13's covariate-shift mechanism for prepost.
- **The linear model's toggle headline reads ~0** (3-month +0.39 → −0.01, better at every
  campaign length): a learner with no tree-style shrinkage does not show the bias.
- **More training data removes it**: `toggle_campaign_only=False` (the Issue 13 2×2's
  {all-data, no-calibration} cell, measured) cuts the 3-month toggle headline bias to +0.07 —
  but without the Issue 13 calibration/time-feature guards it imports drift everywhere else
  (conditional 3-month ti Δscore +12.7; ≥6-month campaigns uniformly worse; cells 13 better / 58
  worse). Campaign-only stays the default; revisit inside Issue 13's full 2×2.
- Capacity is *not* the lever: the accepted `min_child_samples=50` barely moves the 3-month
  toggle headline (+0.39 → +0.35). The shrinkage that matters is intrinsic to boosted trees on
  ~6k rows, so Issue 13's data-side fixes (more rows, made safe by calibration) are the right
  attack.

### Method notes
- The placebo screen again did all the discriminating; nothing needed the full sweep to be
  rejected. One screen ≈ 13 min vs ≈ 55 min for a full sweep.
- Post-hoc OOF calibration applied to a refit-on-100% model is structurally biased toward
  overcorrection at small n (the OOF-vs-final shrinkage gap) — any future residual-calibration
  design (Issue 13) must estimate the correction against the *final* model's predictions, e.g.
  by calibrating on a window the final model genuinely never saw, not by recycling OOF folds.

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
