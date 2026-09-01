# wind-up v1

This folder holds the planning material for the **v1** major upgrade of wind-up.
v1 is developed on the `v1` branch; feature PRs target `v1` rather than `main`.

## Contents

- **[goals.md](goals.md)** — the north-star vision and goals for v1 (the "why").
- **[roadmap.md](roadmap.md)** — workstreams (epics), phasing, and how the work is
  managed (the "what" and "in what order").
- **[issues_campaigns.md](issues_campaigns.md)** — **the current tranche**:
  rounding out v1 for real-world use — realistic whole-farm campaigns with
  self-configuring methods (C-series), robustness to synthesized failure modes
  (R-series), and a productized `wind-up` method released as v1.0.0 (W-series). Start
  here.
- **[issues.md](issues.md)** — drafts of the earlier issues (Issues 1–19).
  **Back-burnered** in favour of `issues_campaigns.md`; still valuable for later.
- **[findings_campaigns.md](findings_campaigns.md)** — empirical findings log for the
  current tranche (CF-numbered).
- **[findings.md](findings.md)** — empirical findings log from the earlier work.
- **[references.md](references.md)** — related open-source tools (FLASC, OpenOA,
  DSWE) and key methodology references (Kanev TNO report) to investigate later.

## One-paragraph summary

v0 is a single-method tool: it measures turbine-upgrade uplift with a binned
power-curve, test-vs-reference method. v1 turns wind-up into a **platform** in
which **alternative uplift methods are pluggable and objectively benchmarked** on
synthetic datasets with known ground truth. The driving goals are *more accurate
results from shorter campaigns* and *richer conditional information* about how an
upgrade performs (in wakes vs free-stream, day vs night, by direction/stability).
The first wave of work is **methodology-first**: build the public evaluation
harness, wire the v0 method in as the baseline to beat, and develop the first new
candidate method — judged on **P50 accuracy and precision only**. An uncertainty
(P95) model is deferred until a winning P50 method is found.

## How this work is managed

- **Branch:** `v1` (published). PRs target `v1`.
- **Methodology prototyping:** hybrid model — the synthetic-data + evaluation
  harness is **public** (suitable as a WeDoWind exercise / open benchmark, in the
  spirit of the Hill of Towie Kaggle challenge). Individual candidate methods may
  be prototyped anywhere, and are **ported into wind-up v1 only once they beat the
  v0 baseline** on the harness.
