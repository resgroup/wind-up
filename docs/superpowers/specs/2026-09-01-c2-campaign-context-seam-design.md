# Design — C2: how campaign context reaches the methods

**Date:** 2026-09-01
**Status:** USER REVIEW NEEDED
**Issue:** C2 in `docs/v1/issues_campaigns.md`
**Parent design:** `docs/superpowers/specs/2026-08-27-realistic-campaigns-design.md`

## Problem

C1 landed `CampaignSpec` — the public facts of a campaign — but nothing reads most of
it. `benchmarking/campaigns/methods.py` reads `spec.mode` and `spec.rated_power_kw`;
`runner.py` reads the timing, `usable_mask`, `analysis_period` and `turbine_col`.
`candidate_references`, `coords`, `north_offsets` and `excluded_turbines` reach no
method at all.

That is not an oversight, it is the deferred question: the seam is
`MethodInput(scada_df, test_wtg, upgrade_timing, turbine_col)`, and every method
derives its references the same implicit way — *every turbine in `scada_df` that is not
`test_wtg`* — in nine places across four methods. Reference selection is therefore
**enacted** by the runner subsetting the frame, never **declared**.

The campaigns and failure modes queued behind C2 all need declared campaign facts:

- **C3** must exclude other upgraded turbines and honour a candidate list.
- **C5** must drop a reference *for some wind directions only*, from declared geometry.
- **R1** needs coords and north offsets in the runner, for a shared step whose output
  every method inherits.
- **C8** must use a reference over the part of the period its own change history allows.

Row removal cannot express any of the time-varying cases. A complete-case method such
as `naive_ratio` loses the whole timestamp when one reference is gated, rather than
dropping just that reference; and a method cannot tell an enacted gate from R4's
genuine missing data. Validity has to be **told** to methods, not enacted on the frame.

## Scope

- A `CampaignContext` — the narrow, per-test-turbine view of a campaign that a method
  may consult — and its default constructor.
- `MethodInput.context`, threaded from both input paths (`CampaignRunner` via
  `score_one`, and the study path in `harness/scoring.py`).
- Every method sourcing its references and its row validity from the context.
- `ColumnSchema.northed(role)`, so R1 fills a derived column whose name it does not
  have to invent.

Out of scope, left to the issues that own them: the northing algorithm (R1),
wake-direction gating (C5), method-internal reference screening (R3), missing-data
adaptation (R4). C2 plumbs the channel; those issues fill it.

## Key decisions

### 1. Methods see a derived view, never the declaration

`CampaignSpec` is the analyst's declaration. C8 generalizes it to per-turbine change
histories and W2 promotes it to public `src/wind_up` API. Coupling every method to it
means each of those churns reaches every method.

So the runner **derives** a `CampaignContext` from the spec, per test turbine, and that
is what rides the seam. The context asks and answers questions — *which turbines may I
use as references? which rows may I use?* — rather than exposing declaration fields. C8
then changes one translation function, and the study path can build a context with no
declaration at all.

This also settles the layering: the context is part of the **method contract**, so it
lives in `benchmarking/harness/context.py` alongside the seam, while the derivation
lives in `benchmarking/campaigns/context.py`. `campaigns` already depends on `harness`;
the reverse dependency would be a cycle.

### 2. Declared validity is shared; screened validity is method-internal

| | Decided by | Lives |
|---|---|---|
| **Declared validity** — exclusions, per-turbine change histories (C8), geometry + direction wake gating (C5) | the campaign, identically for every method | runner-computed, rides the context |
| **Screened validity** — "this reference looks broken in the data" (R3) | the method — v0 screens references one at a time, `power_model` uses them all at once | method-internal |
| **Method configuration** — ERA5, model params, bin widths, `save_plots` | the method | constructor, as today |

Declared validity is a fact about the campaign; methods should not each re-derive it
from geometry. Screened validity is a modelling choice and must stay method-internal,
as the robustness design already concluded. Conflating the two is what makes C5 look
hard.

### 3. Validity is named for its purpose: `valid_for_uplift`

The load-bearing member is a boolean frame, timestamps × turbines, answering *may this
turbine's data at this timestamp contribute to the uplift estimate?*

It is deliberately **not** a general "usable" flag. Validity is purpose-specific: a
one-off curtailment ruins the turbine's performance but not its yaw alignment, so those
rows are invalid for uplift and perfectly valid for a northing analysis. Naming the
purpose means a second purpose arrives as a sibling frame (`valid_for_northing`, …)
rather than as a silent reinterpretation of one overloaded frame. C2 names only the
purpose it has.

One frame serves C3 (which references), C5 (a reference in a steered wake is `False`
for those directions only), C8 (a reference valid for part of the period) and R4 —
and it lets each method reduce it its own way: `naive_ratio` can drop a reference
wholesale rather than lose the timestamp, `power_model` can NaN just that reference's
features, v0 can screen per reference.

### 4. The frame is precomputed, not a callable

`valid_for_uplift` is materialized by whoever builds the context, not computed lazily
on demand. C5's direction gating needs the SCADA to evaluate, and an accessor taking a
frame would give different answers for different sub-frames — the same campaign fact
must not depend on which slice a method happens to pass. The runner has the full
visible frame and computes once.

The invariant is **coverage, not equality**: `valid_for_uplift.index` covers the unique
timestamps of `mi.scada_df`. Methods narrow it with `context.valid_over(index)`, which
raises on an uncovered timestamp. Coverage rather than equality is what lets
`replace(mi, scada_df=...)` — used by `restrict_to_campaign` in `naive_ratio` and
`toggle_specialist` — keep working untouched.

### 5. The default context is the truth, not a fallback

```python
CampaignContext.from_frame(scada_df, test_wtg=..., timing=..., turbine_col=...)
#   candidate_references = every other turbine present in the frame
#   valid_for_uplift     = all True
```

This is today's implicit contract written down. It is *correct*, not degraded, so
methods have exactly one read path and never branch on `context is None`. The study and
sweep paths keep working with no `CampaignSpec` in sight.

`MethodInput` keeps `upgrade_timing` and `turbine_col` as **constructor shorthands**
that build the default context when `context` is omitted; reading them goes through
properties delegating to the context. All ~100 existing construction sites in tests and
drivers keep working unchanged, and there is one source of truth at read time.

`mode` comes from the context, closing the `isinstance` type-switch on `upgrade_timing`
that the C1 design already flagged as the wrong inference once timing is per-turbine.

The context's field is `timing`, not `upgrade_timing`. C8 retires the "upgrade"
vocabulary — not every campaign is an upgrade, some confirm stable performance or
quantify a loss event — and C1 already began the move with
`CampaignSpec.change_label()`. `CampaignContext` is a new type, so naming its field
`upgrade_timing` would add one more use of the word C8 is committed to removing. The
qualifier is also unnecessary here: the context is per-test-turbine and describes
exactly one change, whereas on `MethodInput` the name floats free among unrelated
fields and needs it. `change_timing` is the near alternative, but the neutral term is
C8's decision to settle; `timing` stays correct whichever way C8 goes and needs no
later rename. So `mi.upgrade_timing` survives as a legacy delegating property,
`context.timing` is the forward name, and C8 renames the property and the spec field
without touching the context.

### 6. Methods consume it now

All four methods take reference *membership* from `context.candidate_references` and
honour `valid_for_uplift`, replacing the nine independent
`[c for c in wide.columns if c != test_wtg]` derivations
(`naive_ratio` ×3, `toggle_specialist` ×3, `power_model/features.py` ×1 shared by two
callers, `v0_binned` ×1).

Doing this in C2 rather than declaring an unused type is the difference between
plumbing a channel and repeating C1's shape of shipping fields nothing reads. Under the
default context it is behaviour-identical by construction.

**Methods keep their own reference ordering** and take only membership from the
context — `power_model` sorts, the others follow wide-column order. LightGBM feature
order depends on it, so changing it would move the frozen benchmarks.

The consumption contract:

- A turbine in the frame but not in `candidate_references` is not a reference, even
  though its data is present. Such a turbine can still be valuable for feature engineering (eg waking state)
- A row where `valid_for_uplift[turbine]` is `False` contributes none of that turbine's
  data to the estimate; each method applies it as a per-turbine mask, composing with
  its existing availability and complete-case filtering.
- `valid_for_uplift[test_wtg]` being `False` excludes that timestamp entirely.

## The type

```python
@dataclass(frozen=True)
class CampaignContext:
    test_wtg: str
    timing: pd.Timestamp | ToggleSchedule | pd.DataFrame
    turbine_col: str
    candidate_references: list[str]
    valid_for_uplift: pd.DataFrame          # bool; index x [test_wtg, *candidate_references]

    @property
    def mode(self) -> Literal["prepost", "toggle"]: ...
    def valid_over(self, index: pd.DatetimeIndex) -> pd.DataFrame: ...
```

The context carries **only answers**. `coords`, `north_offsets` and `rated_power_kw`
are deliberately absent: no method reads the first two — R1's shared northing step and
C5's wake gating both run in the runner, which holds the `CampaignSpec` directly, and
v0 uses its own vendored YAML — while `rated_power_kw` already reaches
`PowerModelMethod` and `ToggleSpecialistMethod` through their constructors, and a
second channel for one fact is exactly the inconsistency this type exists to avoid. If
a method later needs geometry, a field plus a line in `context_for` adds it; a field
that sits unread across three issues drifts.

R1's shared step writes a **derived column**, `northed_<source name>` (e.g.
`northed_YawAngleMean`), and leaves the original untouched. Correcting in place would
make every existing plot and diagnostic lie about what it shows — a chart labelled
`YawAngleMean` that is no longer `YawAngleMean` — and the original is still wanted, by
the northing diagnostics themselves and by R1's own fault injection.

The derived name comes from the schema, not from string-building at each call site:
`ColumnSchema.northed(role)` returns `f"northed_{getattr(self, role)}"` and raises if
the role is unset, preserving the rule that methods take column names *only* from a
`ColumnSchema`. C2 adds that accessor; R1 fills the column. (The function that computes
the values already exists as `north_calibrated_direction` in
`benchmarking/synthetic/upgrades.py` — `northed_` is its short form as a column prefix,
so R1 should not coin a third term.)

**Which turbine's direction it is decides whether it may be a feature.** Design-note
§3 bars the *test* turbine's own nacelle position: it is post-treatment, and northing
does not change that. It says nothing against a **reference** turbine's northed
direction, which is treatment-invariant in the same way reference power is — and is
plainly useful, since knowing where each reference is pointing is much of what resolves
who is waking whom. R1/W1 may well try it as a feature and this design does not
foreclose it.

Two caveats travel with that. Reference *anemometer* signals (nacelle wind speed and
SD) were rejected as features on calibration drift; direction is a different signal and
northing is precisely the drift correction for it, which is what makes the northed
version a defensible candidate where the raw one is not. And under wake steering a
reference sitting in a changed wake stops being treatment-invariant — which is what
C5's gating is for, not a reason to bar the feature everywhere.

There is **no `northing_applied` flag**. Whether the step has run is already written
in the frame — `columns.northed(role) in scada_df.columns` — and a flag beside it is
second state that can disagree with the thing it describes. A method that needs
northing checks for the column and raises naming it, which is both accurate by
construction and a better error. v0 needs no guard either way: it does its own northing
from its vendored YAML and never reads the derived column, so the double-correction
risk that in-place correction would have carried does not arise.

## Architecture

```
CampaignSpec (analyst declaration; C8 generalizes, W2 promotes to public API)
     │
     │  campaigns/context.py: context_for(spec, turbine=..., scada_df=...)
     ▼
CampaignContext (harness/context.py; the method contract)
     │  candidate_references / valid_for_uplift / timing / mode
     ▼
MethodInput(scada_df, test_wtg, context)  ─────►  every method

Study path: Replicate ──► CampaignContext.from_frame(...)  (no declaration needed)
```

`CampaignRunner` derives one context per upgraded turbine and passes it to `score_one`,
which forwards it to `_method_input` instead of building the default. This is where
`candidate_references`, `excluded_turbines` and `usable_mask` stop being dead fields.

## Acceptance: the frozen benchmarks must read UNCHANGED

**This is the acceptance criterion for C2, not a nice-to-have.** C2 changes no
estimator mathematics whatsoever — it changes *where a method learns which turbines to
use*. Every existing path builds the default context, whose `candidate_references` is
every other turbine and whose `valid_for_uplift` is all `True`, so every number must
come out where it is today. **Any movement in a frozen benchmark is a bug in the
re-plumbing, not a result.**

Two committed benchmarks, both diffed on the machine that recorded them:

| Benchmark | Command | Required verdict |
|---|---|---|
| `benchmarking/baselines/study_power_model_compare_baseline.json` | `uv run python -m benchmarking.baselines.study_power_model_compare --reference-dir "~/temp/wind-up-benchmarking/badass overnight 20260708"` | `UNCHANGED` |
| `study_toggle_methods_compare_baseline_{portable,linux,win32}.json` | `uv run python -m benchmarking.baselines.study_toggle_methods_compare` | `UNCHANGED` |

**Never re-record.** `--update-baseline` and `--accept-candidate` are forbidden for the
whole of C2. Those flags exist to accept a deliberate improvement; C2 has no
improvement to accept, so reaching for them would be recording a bug as the new truth.
If a benchmark moves, fix the code.

**`toggle_specialist` is the sharp instrument.** Its reproducibility band is `1e-7`
(`_REPRODUCIBILITY` in `study_toggle_methods_compare.py`) because it is pure arithmetic
that reproduces exactly; a genuine behaviour change shows there unambiguously.
`power_model`'s band is `1e-3` — its ~0.05 pp same-machine LightGBM noise could mask a
small real change, so a clean `power_model` verdict alone is not sufficient evidence.

**Run both before touching any method**, to confirm the working tree reads `UNCHANGED`
on this machine to start with. Without that pre-check, a pre-existing drift gets
misattributed to C2 — and conversely a `MOVED` afterwards can be pinned on this work
with confidence. Both benchmarks are machine-specific for `power_model` (~0.7 pp false
`MOVED` cross-machine, 14x the same-machine noise), so this must be the Linux laptop
the committed files were recorded on; a cross-machine excuse is not available here.

The two things that could break this, both guarded in §6: methods must keep their **own
reference ordering** (LightGBM feature order depends on it — take only *membership*
from the context), and the default context must remain exactly today's implicit
contract rather than a subtly different one.

## Testing

- `CampaignContext.from_frame` reproduces today's implicit contract: every other
  turbine is a candidate reference, everything valid.
- `valid_over` raises on an index carrying a timestamp the context does not cover.
- Per method, with a hand-built non-default context: a reference excluded from
  `candidate_references` contributes nothing; a reference marked invalid over part of
  the period contributes nothing over that part and still contributes over the rest;
  a timestamp where the test turbine is invalid drops out entirely.
- `ColumnSchema.northed` derives the prefixed name from a set role and raises on an
  unset one.
- `context_for` exposes no upgrade magnitude, mirroring the existing `CampaignSpec`
  truth-leak test — the runner stays the single audited place where truth could leak.
- The placebo campaign still reports ~0 for every method in both modes.

## Done when

Every method takes its references and row validity from `CampaignContext`; both input
paths supply one; `poe all-fast` green; and **both frozen benchmarks read `UNCHANGED`,
re-recorded neither by `--update-baseline` nor `--accept-candidate`** (see
[Acceptance](#acceptance-the-frozen-benchmarks-must-read-unchanged)). C2 is pure
re-plumbing, and the benchmarks are how that claim is proved rather than asserted.

C2 also amends its own **Done when** in `docs/v1/issues_campaigns.md`, which currently
promises a decision the code reflects across the board; the northing, gating, screening
and missing-data work it hints at belongs to R1, C5, R3 and R4.
