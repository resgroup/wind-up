# Design — robustness to real-world failure modes (widening the campaigns tranche)

**Date:** 2026-08-28
**Status:** approved design; extends
`2026-08-27-realistic-campaigns-design.md` and feeds new R-series issues in
`docs/v1/issues_campaigns.md`
**Branch of work:** the same realistic-campaigns tranche (developed off `v1`)

## Problem

The realistic-campaigns tranche (C-series) matures v1 so an analyst can *declare a
campaign spec and the method does the right thing*. That makes v1 realistic, but not
yet **robust**: real SCADA carries data pathologies that were never used to develop
v0 or the v1 methods, and `power_model` in particular has soft spots (it can key on
unstable per-turbine sensors, trusts its references, and assumes a fixed set of
signals). To be usable in the real world, wind-up must handle these on its own.

This design **widens the tranche's ambition** from "simulate realistic campaigns" to
"**round out v1 so it is usable in the real world**" — two pillars:

1. **Realism** — declare a campaign spec, the method self-configures (the existing C-series).
2. **Robustness** — wind-up handles, on its own, the data pathologies real SCADA
   throws at it. We *synthesize* each failure mode with known ground truth, so we can
   measure exactly how much it degrades a method and prove a fix closes the gap.

## The four failure modes (R-series)

Real pathologies wind-up must survive:

- **R1 — Northing errors.** Turbine direction references carry **step changes** in
  their offset over time (a recalibration, a sensor swap). Drifts are *not* simulated
  here — steps only.
- **R2 — Unstable sensors.** Per-turbine anemometer and (to a lesser extent)
  temperature channels are unstable over time — **both step changes and slow
  drifts**, primarily on wind speed. Using them as ML features silently biases an
  estimate across the baseline↔treatment boundary.
- **R3 — Invalid references.** A reference turbine develops its **own** performance
  shift (degradation, curtailment change) unrelated to the tested upgrade, appearing
  during the campaign — a common, biasing real situation.
- **R4 — Missing data.** Channels/turbines are absent for part or all of a campaign
  (a reference offline, a signal missing), so the input no longer matches any
  hardcoded feature list.

## Key decisions (from brainstorming)

1. **Success = invariance, not a race against v0.** For each fault the baseline is
   `power_model`'s **own** error on the *clean* version of the fixture; the target is
   that injecting the fault degrades that error by little
   (`power_model`-under-fault ≈ `power_model`-clean). v0 is **skipped** where
   possible (it is slow, and was not developed against these synthesized faults); an
   optional one-off sniff is allowed, never the yardstick.

2. **The fault must bite.** Every R-issue begins by calibrating the fault magnitude
   until it **significantly** throws off `power_model` on the clean fixture. If it
   does not bite, the invariance target is vacuous and there is nothing to fix.
   Making the fault bite may take iteration and is part of "done".

3. **Fix location follows how the concern is shared.**
   - **Northing (R1) is a shared feature-engineering step** in the runner /
     preprocessing, upstream of every method — every method benefits, and C3/C5
     inherit it instead of hand-rolling northing handling.
   - **The other three (R2/R3/R4) are `power_model`-internal.** Reference selection
     in particular is method-specific (v0 screens references one at a time;
     `power_model` uses them all at once), so a reference-validity screen belongs
     inside the method, not in a shared layer.

4. **Isolated fixtures first, then re-verify on campaigns (hybrid).** Each fault is
   developed on a **tiny purpose-built fixture** — one treated turbine + ~3
   references, a simple known AeroUp-shaped uplift — so the fault signal is isolated
   and iteration is fast. The best faults are then re-injected into the relevant
   whole-farm campaigns (R1/R3 ↔ C3/C5) as an in-context check.

5. **Every fault is evaluated in both prepost and toggle.** The tiny fixture is run in
   **both modes**, and the "bites" check is **per-mode**. A toggle campaign's rapid
   on/off switching means a fault present in both the on and off periods can partly or
   wholly **cancel**, so a fault that bites hard in prepost may bite little or not at
   all in toggle. Mitigation is applied **where it bites**; where a fault does not bite
   in toggle, that is **documented as "no mitigation needed there"** — determined
   empirically, never assumed.

6. **Naming.** A distinct **`R1–R4`** ("robustness") series alongside `C1–C7`,
   deliberately **not** `F#` (that collides with `findings.md`).

## Sequencing (Approach A: robustness-first, after the foundation)

```
C1 → C2 → [ R1  R2  R3  R4 ] → C3 → C4 → C5 → C6    (C7 already done)
```

- C1 (declaration + runner + fixture plumbing) and C2 (seam decision) land first —
  the R-fixtures reuse that plumbing.
- **R1 (northing) lands before C3** so the prepost campaign inherits the shared
  northing step; R2–R4 are independent `power_model` work and can run in any order
  within the block.
- C3–C6 then re-verify the relevant faults in-context (R1/R3 with C3/C5).

Alternatives considered: **(B) interleave** each fault next to its nearest campaign —
less rework than campaigns-first but scatters the robustness story; **(C)
campaigns-first, robustness after C6** — clean separation but C3/C5 build throwaway
northing handling that R1 then replaces. (A) front-loads the one genuinely shared fix
(northing) and matches the isolated-first/hybrid decision.

## Per-issue acceptance shape

Every R-issue carries a two-phase **Done when:**, evaluated in **both prepost and
toggle**:

1. **Bites (per-mode):** the fault demonstrably and significantly throws off
   `power_model` on the clean tiny fixture, calibrated per-mode (not assumed). A fault
   that does not bite in a mode needs no mitigation there — recorded explicitly.
2. **Fixed:** where it bites, the fix restores `power_model`-under-fault ≈
   `power_model`-clean; for R1, C3/C5 additionally drop bespoke northing wiring in
   favour of the shared step.

## Risks & open questions

- **Making faults bite realistically.** The fault must be strong enough to break the
  method yet plausible as real SCADA. Calibration is per-fault and may need iteration.
- **Reference-validity screen scope (R3).** Detecting a bad reference inside
  `power_model` (which uses the whole pool at once) is less understood than v0's
  one-at-a-time round robin; it may spawn follow-up work.
- **Missing-data surface (R4).** "Adapt to available signals" spans feature discovery,
  partial windows, and graceful degradation; keep the tiny fixture minimal so the
  requirement stays crisp.
