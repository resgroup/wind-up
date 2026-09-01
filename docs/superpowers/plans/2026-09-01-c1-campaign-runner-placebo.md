# C1 — Campaign declaration + runner + farm uplift + placebo campaign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the whole declaration → runner → farm-uplift → reporting pipeline on a placebo (zero injected uplift) whole-farm campaign, so a campaign is *declared* rather than hand-wired and every method reports ~0.

**Architecture:** A private `SyntheticCampaign` (holds the injected upgrades = ground truth) derives a public `CampaignSpec` (facts a method may see). A `CampaignRunner` loops the upgraded turbines, runs each applicable method once through the harness's `score_one` (capturing the `MethodOutput` on the way past, so one estimate call serves both output shapes), aggregates per-turbine estimates into one headline via the pure `wind_up.farm_uplift`, and compares against an exact pooled truth from `original_df`. A report module writes tables and plots.

**Tech Stack:** Python 3.10+, pandas, numpy, matplotlib; `uv` + `poe`; pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-28-c1-campaign-runner-placebo-design.md`

## Global Constraints

- **Never perform git write actions.** Do not commit, branch, push, tag, or stage. Each task ends with a **Checkpoint** step that reports to the user, who commits. This deliberately replaces the `git commit` step the writing-plans template uses.
- Style: `ruff` `line-length = 120`, `select = ["ALL"]`; `mypy` enforced. `poe lint` then `poe test-fast` before each checkpoint. Never run plain `poe test` (>10 min of `slow` tests).
- Tests treat warnings as errors (`filterwarnings = ["error", ...]`).
- **Docstrings state behaviour and usage, not justification.** No design rationale, no measured evidence, no finding numbers in source. Err far shorter than instinct.
- **Keyword arguments:** at most 1–2 positional args; put `*` after the obvious leading positional so the rest are keyword-only.
- `src/wind_up/` (v1 product) may not import from `benchmarking/` or `wind_up_v0/`. `benchmarking/` may import from `wind_up`.
- New v1 source tests live in `tests/wind_up/`; benchmarking tests mirror `benchmarking/` under `tests/benchmarking/`.

---

### Task 1: `farm_uplift` — the pure headline function

**Files:**
- Create: `src/wind_up/farm.py`
- Modify: `src/wind_up/__init__.py`
- Create: `tests/wind_up/__init__.py`
- Test: `tests/wind_up/test_farm.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `wind_up.farm.TurbineUplift(turbine: str, uplift: float, treated_energy: float, n_records: int, rated_power_kw: float)`; `wind_up.farm.FarmUplift(uplift: float, turbines: pd.DataFrame, uplift_spread: float)`; `wind_up.farm.farm_uplift(turbines: Sequence[TurbineUplift]) -> FarmUplift`. Re-exported as `wind_up.farm_uplift`, `wind_up.TurbineUplift`, `wind_up.FarmUplift`.

- [ ] **Step 1: Write the failing tests**

Create `tests/wind_up/__init__.py` (empty) and `tests/wind_up/test_farm.py`:

```python
"""Tests for the farm-uplift headline and its guards."""

from __future__ import annotations

import math

import pytest

from wind_up.farm import FarmUplift, TurbineUplift, farm_uplift


def _t(
    name: str = "T1",
    *,
    uplift: float = 0.05,
    treated_energy: float = 1000.0,
    n_records: int = 100,
    rated_power_kw: float = 2300.0,
) -> TurbineUplift:
    return TurbineUplift(
        turbine=name,
        uplift=uplift,
        treated_energy=treated_energy,
        n_records=n_records,
        rated_power_kw=rated_power_kw,
    )


def test_equal_uplifts_reproduce_the_pooled_ratio() -> None:
    result = farm_uplift([_t("T1", treated_energy=1000.0), _t("T2", treated_energy=2000.0)])
    assert result.uplift == pytest.approx(0.05)
    assert result.turbines["used"].all()
    assert (result.turbines["guard"] == "").all()


def test_headline_weights_turbines_by_treated_energy() -> None:
    # T2 carries 9x the energy, so the headline sits close to T2's 0.0.
    result = farm_uplift(
        [_t("T1", uplift=0.10, treated_energy=110.0), _t("T2", uplift=0.0, treated_energy=900.0)]
    )
    counterfactual = 110.0 / 1.10 + 900.0
    assert result.uplift == pytest.approx((110.0 + 900.0) / counterfactual - 1.0)
    assert result.uplift < 0.02


def test_uplift_spread_reports_the_range_across_used_turbines() -> None:
    result = farm_uplift([_t("T1", uplift=0.02), _t("T2", uplift=0.08), _t("T3", uplift=0.05)])
    assert result.uplift_spread == pytest.approx(0.06)


def test_uplift_spread_is_nan_for_a_single_turbine() -> None:
    assert math.isnan(farm_uplift([_t("T1")]).uplift_spread)


def test_uplift_of_minus_one_is_dropped_not_divided_by_zero() -> None:
    result = farm_uplift([_t("T1", uplift=-1.0), _t("T2", uplift=0.05, treated_energy=2000.0)])
    row = result.turbines.set_index("turbine").loc["T1"]
    assert not row["used"]
    assert row["guard"] == "negative_counterfactual"
    assert result.uplift == pytest.approx(0.05)


def test_uplift_below_minus_one_is_dropped() -> None:
    result = farm_uplift([_t("T1", uplift=-1.5), _t("T2", uplift=0.05, treated_energy=2000.0)])
    assert not result.turbines.set_index("turbine").loc["T1", "used"]
    assert result.uplift == pytest.approx(0.05)


def test_implied_capacity_factor_above_rated_is_capped() -> None:
    # u=-0.9 implies a counterfactual of 1000 kW-records over 10 records = 100 kW/record > 50 rated.
    result = farm_uplift([_t("T1", uplift=-0.9, treated_energy=100.0, n_records=10, rated_power_kw=50.0)])
    row = result.turbines.set_index("turbine").loc["T1"]
    assert row["used"]
    assert row["guard"] == "capacity_cap"
    assert row["counterfactual_energy"] == pytest.approx(500.0)
    assert result.uplift == pytest.approx(100.0 / 500.0 - 1.0)


def test_negative_treated_energy_is_dropped() -> None:
    result = farm_uplift([_t("T1", treated_energy=-5.0), _t("T2", treated_energy=2000.0)])
    row = result.turbines.set_index("turbine").loc["T1"]
    assert not row["used"]
    assert row["guard"] == "negative_energy"


def test_turbine_with_no_records_is_dropped() -> None:
    result = farm_uplift([_t("T1", n_records=0, treated_energy=0.0), _t("T2")])
    assert result.turbines.set_index("turbine").loc["T1", "guard"] == "no_records"


def test_non_finite_uplift_is_dropped() -> None:
    result = farm_uplift([_t("T1", uplift=float("nan")), _t("T2")])
    assert result.turbines.set_index("turbine").loc["T1", "guard"] == "non_finite_uplift"


def test_headline_is_nan_when_no_turbine_is_usable() -> None:
    result = farm_uplift([_t("T1", uplift=float("nan"))])
    assert math.isnan(result.uplift)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one turbine"):
        farm_uplift([])


def test_result_is_a_farm_uplift() -> None:
    assert isinstance(farm_uplift([_t("T1")]), FarmUplift)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/wind_up/test_farm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wind_up.farm'`

- [ ] **Step 3: Write the implementation**

Create `src/wind_up/farm.py`:

```python
"""Combine per-turbine uplift estimates into one farm headline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TurbineUplift:
    """One turbine's uplift estimate and the treated-period energy behind it.

    :param turbine: turbine name
    :param uplift: the turbine's P50 uplift, as an energy-ratio fraction
    :param treated_energy: observed treated-period energy — the sum of finite active power over
        the treated records
    :param n_records: how many records that sum covers
    :param rated_power_kw: the rating the capacity-factor cap uses; where the rating changed over
        the campaign, pass the higher of the pre- and post-change values
    """

    turbine: str
    uplift: float
    treated_energy: float
    n_records: int
    rated_power_kw: float


@dataclass(frozen=True)
class FarmUplift:
    """The farm headline and the per-turbine detail behind it.

    :param uplift: the headline, ``(Σ treated energy) / (Σ counterfactual energy) − 1`` over the
        used turbines; NaN when none are usable
    :param turbines: one row per input turbine with ``turbine``, ``uplift``, ``treated_energy``,
        ``n_records``, ``rated_power_kw``, ``counterfactual_energy``, ``used`` and ``guard``
        (``""`` when no guard fired)
    :param uplift_spread: the max−min of the used turbines' uplifts; NaN below two used turbines
    """

    uplift: float
    turbines: pd.DataFrame
    uplift_spread: float


def farm_uplift(turbines: Sequence[TurbineUplift]) -> FarmUplift:
    """Aggregate per-turbine uplifts into one energy-weighted farm headline.

    Each turbine's counterfactual energy is estimated as ``treated_energy / (1 + uplift)`` and
    guarded: a turbine is dropped when its uplift is non-finite or ``<= -1``, when its treated
    energy is negative, or when it has no records; a counterfactual implying a mean power above
    ``rated_power_kw`` is clipped to that rating.
    """
    if not turbines:
        msg = "farm_uplift needs at least one turbine"
        raise ValueError(msg)

    rows = [_evaluate(t) for t in turbines]
    frame = pd.DataFrame(rows)
    used = frame[frame["used"]]

    treated_total = float(used["treated_energy"].sum())
    counterfactual_total = float(used["counterfactual_energy"].sum())
    uplift = treated_total / counterfactual_total - 1.0 if counterfactual_total else float("nan")

    spreads = used["uplift"]
    spread = float(spreads.max() - spreads.min()) if len(spreads) > 1 else float("nan")
    return FarmUplift(uplift=uplift, turbines=frame, uplift_spread=spread)


def _evaluate(turbine: TurbineUplift) -> dict[str, object]:
    """Return one turbine's row: its counterfactual energy, whether it is used, and any guard."""
    base: dict[str, object] = {
        "turbine": turbine.turbine,
        "uplift": turbine.uplift,
        "treated_energy": turbine.treated_energy,
        "n_records": turbine.n_records,
        "rated_power_kw": turbine.rated_power_kw,
    }
    guard = _drop_reason(turbine)
    if guard:
        return {**base, "counterfactual_energy": float("nan"), "used": False, "guard": guard}

    counterfactual = max(turbine.treated_energy / (1.0 + turbine.uplift), 0.0)
    cap = turbine.rated_power_kw * turbine.n_records
    if counterfactual > cap:
        return {**base, "counterfactual_energy": cap, "used": True, "guard": "capacity_cap"}
    return {**base, "counterfactual_energy": counterfactual, "used": True, "guard": ""}


def _drop_reason(turbine: TurbineUplift) -> str:
    """Name the guard that removes ``turbine`` from the weighting, or ``""`` to keep it."""
    if not math.isfinite(turbine.uplift):
        return "non_finite_uplift"
    if turbine.n_records <= 0:
        return "no_records"
    if turbine.treated_energy < 0:
        return "negative_energy"
    if 1.0 + turbine.uplift <= 0:
        return "negative_counterfactual"
    return ""
```

Modify `src/wind_up/__init__.py` — add below the existing `__version__` line:

```python
from wind_up.farm import FarmUplift, TurbineUplift, farm_uplift

__all__ = ["FarmUplift", "TurbineUplift", "__version__", "farm_uplift"]
```

(Keep the `from importlib.metadata import version` / `__version__` lines exactly as they are, and put the `farm` import after them.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/wind_up/test_farm.py -q`
Expected: PASS (14 tests)

- [ ] **Step 5: Lint**

Run: `uv run poe lint`
Expected: ruff format/check clean, mypy clean.

- [ ] **Step 6: Checkpoint — report to the user**

Do **not** commit. Report: files added (`src/wind_up/farm.py`, `tests/wind_up/__init__.py`, `tests/wind_up/test_farm.py`), file modified (`src/wind_up/__init__.py`), test count, lint status. Flag the new untracked files so the user can `git add` them.

---

### Task 2: `true_farm_uplift` — the exact pooled truth

**Files:**
- Modify: `benchmarking/synthetic/ground_truth.py`
- Modify: `benchmarking/synthetic/generator.py` (add `SyntheticDataset.true_farm_uplift`)
- Modify: `benchmarking/synthetic/__init__.py` (export)
- Test: `tests/benchmarking/synthetic/test_ground_truth.py` (append; create if absent)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `benchmarking.synthetic.true_farm_uplift(synthetic_df, original_df, *, test_wtgs: Sequence[str], masks: Mapping[str, npt.ArrayLike] | None = None, columns: ColumnSchema = HOT_COLUMNS) -> float`; `SyntheticDataset.true_farm_uplift(*, test_wtgs, masks=None) -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarking/synthetic/test_ground_truth.py` (create the file with this content if it does not exist, adding the imports at the top):

```python
"""Tests for the pooled farm-level ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.synthetic import HOT_COLUMNS, true_farm_uplift


def _frames(powers: dict[str, list[float]], uplifts: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (synthetic, original) long frames from per-turbine original powers and a flat uplift."""
    index = pd.date_range("2020-01-01", periods=len(next(iter(powers.values()))), freq="10min", tz="UTC")
    original = pd.concat(
        [
            pd.DataFrame({HOT_COLUMNS.turbine: wtg, HOT_COLUMNS.active_power: values}, index=index)
            for wtg, values in powers.items()
        ]
    )
    synthetic = original.copy()
    for wtg, factor in uplifts.items():
        rows = synthetic[HOT_COLUMNS.turbine] == wtg
        synthetic.loc[rows, HOT_COLUMNS.active_power] = synthetic.loc[rows, HOT_COLUMNS.active_power] * (1 + factor)
    return synthetic, original


def test_farm_truth_pools_energy_across_turbines() -> None:
    synthetic, original = _frames({"T1": [100.0, 100.0], "T2": [300.0, 300.0]}, {"T1": 0.10, "T2": 0.0})
    # (220 + 600) / (200 + 600) - 1 = 0.025
    assert true_farm_uplift(synthetic, original, test_wtgs=["T1", "T2"]) == pytest.approx(0.025)


def test_farm_truth_honours_per_turbine_masks() -> None:
    synthetic, original = _frames({"T1": [100.0, 100.0], "T2": [300.0, 300.0]}, {"T1": 0.10, "T2": 0.0})
    masks = {"T1": np.array([True, False]), "T2": np.array([False, True])}
    # (110 + 300) / (100 + 300) - 1 = 0.025
    assert true_farm_uplift(synthetic, original, test_wtgs=["T1", "T2"], masks=masks) == pytest.approx(0.025)


def test_farm_truth_ignores_records_with_non_finite_power() -> None:
    synthetic, original = _frames({"T1": [100.0, np.nan], "T2": [300.0, 300.0]}, {"T1": 0.10, "T2": 0.0})
    assert true_farm_uplift(synthetic, original, test_wtgs=["T1", "T2"]) == pytest.approx(
        (110.0 + 600.0) / (100.0 + 600.0) - 1.0
    )


def test_farm_truth_is_zero_for_an_unchanged_farm() -> None:
    synthetic, original = _frames({"T1": [100.0, 100.0], "T2": [300.0, 300.0]}, {})
    assert true_farm_uplift(synthetic, original, test_wtgs=["T1", "T2"]) == pytest.approx(0.0)


def test_farm_truth_is_nan_when_no_energy_survives() -> None:
    synthetic, original = _frames({"T1": [0.0, 0.0]}, {})
    assert np.isnan(true_farm_uplift(synthetic, original, test_wtgs=["T1"]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/benchmarking/synthetic/test_ground_truth.py -q`
Expected: FAIL — `ImportError: cannot import name 'true_farm_uplift'`

- [ ] **Step 3: Write the implementation**

In `benchmarking/synthetic/ground_truth.py`, add after `true_net_uplift` (and add `from collections.abc import Mapping, Sequence` to the `TYPE_CHECKING` block):

```python
def true_farm_uplift(
    synthetic_df: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    test_wtgs: Sequence[str],
    masks: Mapping[str, npt.ArrayLike] | None = None,
    columns: ColumnSchema = HOT_COLUMNS,
) -> float:
    """Pooled energy-ratio uplift across several upgraded turbines.

    ``(Σᵢ synthetic energy) / (Σᵢ original energy) − 1`` over each turbine's own selected finite
    records — the N-turbine form of :func:`true_net_uplift`.

    :param synthetic_df: method-facing synthetic SCADA
    :param original_df: untouched original SCADA (ground-truth reference)
    :param test_wtgs: the upgraded turbines to pool
    :param masks: per-turbine boolean selections over that turbine's rows (time order); a turbine
        absent from the mapping, or ``masks=None``, uses the records the upgrade actually changed
    :param columns: the source-native column schema the frames are keyed by
    """
    synthetic_total = 0.0
    original_total = 0.0
    for wtg in test_wtgs:
        synthetic_power = synthetic_df.loc[synthetic_df[columns.turbine] == wtg, columns.active_power].to_numpy(
            dtype=float
        )
        original_power = original_df.loc[original_df[columns.turbine] == wtg, columns.active_power].to_numpy(
            dtype=float
        )
        selection = None if masks is None else masks.get(wtg)
        row_mask = (
            changed_record_mask(synthetic_power, original_power)
            if selection is None
            else np.asarray(selection, dtype=bool)
        )
        effective = row_mask & np.isfinite(synthetic_power) & np.isfinite(original_power)
        synthetic_total += synthetic_power[effective].sum()
        original_total += original_power[effective].sum()
    return float(synthetic_total / original_total - 1.0) if original_total else float("nan")
```

In `benchmarking/synthetic/generator.py`, import `true_farm_uplift` alongside the existing ground-truth imports and add this method to `SyntheticDataset`, after `true_net_uplift`:

```python
    def true_farm_uplift(self, *, test_wtgs: list[str], masks: dict[str, np.ndarray] | None = None) -> float:
        """Derive the pooled farm uplift across ``test_wtgs`` (synthetic vs original)."""
        return true_farm_uplift(
            self.synthetic_df, self.original_df, test_wtgs=test_wtgs, masks=masks, columns=self.columns
        )
```

In `benchmarking/synthetic/__init__.py`, add `true_farm_uplift` to the `ground_truth` import line and to `__all__` (keep `__all__` alphabetically sorted — it goes immediately before `"true_net_uplift"`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarking/synthetic/ -q`
Expected: PASS, existing synthetic tests still green.

- [ ] **Step 5: Lint**

Run: `uv run poe lint`

- [ ] **Step 6: Checkpoint — report to the user**

Do not commit. Report the new function, the `SyntheticDataset` method, the export, and test results.

---

### Task 3: `CampaignSpec` and `SyntheticCampaign`

**Files:**
- Create: `benchmarking/campaigns/__init__.py`
- Create: `benchmarking/campaigns/declaration.py`
- Create: `tests/benchmarking/campaigns/__init__.py`
- Test: `tests/benchmarking/campaigns/test_declaration.py`

**Interfaces:**
- Consumes: `benchmarking.synthetic.{ColumnSchema, HOT_COLUMNS, SyntheticDataset, ToggleSchedule, generate_dataset, treated_mask}`.
- Produces:
  - `CampaignSpec(upgraded_turbines: list[str], upgrade_timing: pd.Timestamp | ToggleSchedule, candidate_references: list[str], excluded_turbines: list[str], coords: dict[str, tuple[float, float]], north_offsets: list[tuple[str, pd.Timestamp, float]], rated_power_kw: float, analysis_period: tuple[pd.Timestamp, pd.Timestamp], turbine_col: str)` with `.mode -> Literal["prepost", "toggle"]`, `.timing_for(wtg) -> pd.Timestamp | ToggleSchedule`, `.usable_mask(wtg, index) -> npt.NDArray[np.bool_]`, `.change_label() -> str`, `.treatment_start -> pd.Timestamp`.
  - `SyntheticCampaign(upgraded_turbines, upgrade_timing, candidate_references, upgrades, coords, north_offsets, rated_power_kw, analysis_period, excluded_turbines=[], columns=HOT_COLUMNS, seed=0)` with `.spec() -> CampaignSpec`, `.generate(scada_df) -> SyntheticDataset`, `.turbines -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/benchmarking/campaigns/__init__.py` (empty) and `tests/benchmarking/campaigns/test_declaration.py`:

```python
"""Tests for the campaign declaration and the public spec derived from it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignSpec, SyntheticCampaign
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange, ToggleSchedule

PERIOD = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-07-01", tz="UTC"))
CHANGEOVER = pd.Timestamp("2020-04-01", tz="UTC")


def _campaign(*, upgrades: list | None = None, upgrade_timing: object = CHANGEOVER) -> SyntheticCampaign:
    return SyntheticCampaign(
        upgraded_turbines=["T1", "T2"],
        upgrade_timing=upgrade_timing,
        candidate_references=["T3", "T4", "T5"],
        excluded_turbines=["T5"],
        upgrades=[] if upgrades is None else upgrades,
        coords={f"T{i}": (57.5 + i * 0.01, -3.25) for i in range(1, 6)},
        north_offsets=[("T1", pd.Timestamp("2020-01-01", tz="UTC"), 1.5)],
        rated_power_kw=2300.0,
        analysis_period=PERIOD,
    )


def _scada(turbines: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")) -> pd.DataFrame:
    index = pd.date_range(PERIOD[0], PERIOD[1], freq="1h", tz="UTC", inclusive="left")
    frames = [
        pd.DataFrame(
            {
                HOT_COLUMNS.turbine: wtg,
                HOT_COLUMNS.active_power: 900.0,
                HOT_COLUMNS.wind_speed: 8.0,
                HOT_COLUMNS.wind_speed_sd: 0.8,
                HOT_COLUMNS.gen_rpm: 1400.0,
                HOT_COLUMNS.availability: 3600.0,
            },
            index=index,
        )
        for wtg in turbines
    ]
    return pd.concat(frames)


def test_spec_exposes_no_upgrade_physics() -> None:
    campaign = _campaign(upgrades=[ConstantCpChange(delta=0.05)])
    spec = campaign.spec()
    fields = {f.name for f in dataclasses.fields(spec)}
    assert "upgrades" not in fields
    assert not any("upgrade" in f and "timing" not in f and "turbines" not in f for f in fields - {"upgraded_turbines"})
    # nothing on the spec, at any depth of its repr, leaks the injected magnitude
    assert "0.05" not in repr(spec)


def test_spec_carries_the_public_facts() -> None:
    spec = _campaign().spec()
    assert spec.upgraded_turbines == ["T1", "T2"]
    assert spec.candidate_references == ["T3", "T4", "T5"]
    assert spec.excluded_turbines == ["T5"]
    assert spec.rated_power_kw == 2300.0
    assert spec.analysis_period == PERIOD
    assert spec.turbine_col == HOT_COLUMNS.turbine


def test_mode_is_prepost_for_a_changeover_timestamp() -> None:
    assert _campaign().spec().mode == "prepost"


def test_mode_is_toggle_for_a_schedule() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=CHANGEOVER)
    assert _campaign(upgrade_timing=schedule).spec().mode == "toggle"


def test_timing_for_returns_the_same_timing_for_every_upgraded_turbine() -> None:
    spec = _campaign().spec()
    assert spec.timing_for("T1") == CHANGEOVER
    assert spec.timing_for("T2") == CHANGEOVER


def test_timing_for_rejects_a_turbine_that_is_not_upgraded() -> None:
    with pytest.raises(KeyError, match="T3"):
        _campaign().spec().timing_for("T3")


def test_usable_mask_keeps_every_record_of_a_participating_turbine() -> None:
    spec = _campaign().spec()
    index = pd.date_range(PERIOD[0], periods=5, freq="1h", tz="UTC")
    assert spec.usable_mask("T3", index).all()
    assert spec.usable_mask("T1", index).all()


def test_usable_mask_drops_every_record_of_an_excluded_turbine() -> None:
    spec = _campaign().spec()
    index = pd.date_range(PERIOD[0], periods=5, freq="1h", tz="UTC")
    assert not spec.usable_mask("T5", index).any()


def test_change_label_is_neutral() -> None:
    assert _campaign().spec().change_label() == "the change"


def test_treatment_start_is_the_changeover_for_prepost() -> None:
    assert _campaign().spec().treatment_start == CHANGEOVER


def test_treatment_start_is_the_schedule_start_for_toggle() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=CHANGEOVER)
    assert _campaign(upgrade_timing=schedule).spec().treatment_start == CHANGEOVER


def test_generate_returns_an_unchanged_dataset_when_there_are_no_upgrades() -> None:
    dataset = _campaign().generate(_scada())
    pd.testing.assert_frame_equal(dataset.synthetic_df, dataset.original_df)


def test_generate_injects_the_declared_upgrade() -> None:
    dataset = _campaign(upgrades=[ConstantCpChange(delta=0.05)]).generate(_scada())
    assert not dataset.synthetic_df[HOT_COLUMNS.active_power].equals(
        dataset.original_df[HOT_COLUMNS.active_power]
    )


def test_generate_restricts_the_data_to_the_analysis_period() -> None:
    wide = _scada()
    extra = wide.copy()
    extra.index = extra.index - pd.Timedelta(days=90)
    dataset = _campaign().generate(pd.concat([extra, wide]))
    assert dataset.synthetic_df.index.min() >= PERIOD[0]
    assert dataset.synthetic_df.index.max() < PERIOD[1]


def test_turbines_lists_every_declared_turbine() -> None:
    assert _campaign().turbines == ["T1", "T2", "T3", "T4", "T5"]


def test_spec_is_a_campaign_spec() -> None:
    assert isinstance(_campaign().spec(), CampaignSpec)


def test_usable_mask_length_matches_the_index(  ) -> None:
    spec = _campaign().spec()
    index = pd.date_range(PERIOD[0], periods=7, freq="1h", tz="UTC")
    assert spec.usable_mask("T3", index).shape == (7,)
    assert spec.usable_mask("T3", index).dtype == np.bool_
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/benchmarking/campaigns/test_declaration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarking.campaigns'`

- [ ] **Step 3: Write the implementation**

Create `benchmarking/campaigns/declaration.py`:

```python
"""What a campaign is: the private declaration and the public spec derived from it.

``SyntheticCampaign`` holds the injected upgrades and so is ground truth; ``CampaignSpec``
carries only the facts an analyst would know and is what methods are given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, generate_dataset

if TYPE_CHECKING:
    import numpy.typing as npt

    from benchmarking.synthetic import ColumnSchema, SyntheticDataset


@dataclass(frozen=True)
class CampaignSpec:
    """The public facts of a campaign — everything a method may see, and nothing else.

    Read per-turbine facts through :meth:`timing_for` and :meth:`usable_mask` rather than the
    flat fields, and the mode through :attr:`mode` rather than the type of ``upgrade_timing``.

    :param upgraded_turbines: the turbines whose uplift is being estimated
    :param upgrade_timing: the changeover timestamp (prepost) or the ``ToggleSchedule`` (toggle)
    :param candidate_references: turbines a method may use as references
    :param excluded_turbines: turbines whose data must not be used at all
    :param coords: turbine name → ``(latitude, longitude)`` in degrees
    :param north_offsets: step-applied northing corrections, ``(turbine, from, offset_deg)``
    :param rated_power_kw: the turbines' rated power
    :param analysis_period: ``(start, end)`` of the whole record, end exclusive
    :param turbine_col: the turbine-identifier column of the SCADA frame
    """

    upgraded_turbines: list[str]
    upgrade_timing: pd.Timestamp | ToggleSchedule
    candidate_references: list[str]
    excluded_turbines: list[str]
    coords: dict[str, tuple[float, float]]
    north_offsets: list[tuple[str, pd.Timestamp, float]]
    rated_power_kw: float
    analysis_period: tuple[pd.Timestamp, pd.Timestamp]
    turbine_col: str = HOT_COLUMNS.turbine

    @property
    def mode(self) -> Literal["prepost", "toggle"]:
        """``"toggle"`` for a scheduled campaign, ``"prepost"`` for a single changeover."""
        return "toggle" if isinstance(self.upgrade_timing, ToggleSchedule) else "prepost"

    @property
    def treatment_start(self) -> pd.Timestamp:
        """When treatment begins: the changeover, or when toggling starts."""
        if isinstance(self.upgrade_timing, ToggleSchedule):
            return self.upgrade_timing.start if self.upgrade_timing.start is not None else self.analysis_period[0]
        return self.upgrade_timing

    def timing_for(self, turbine: str) -> pd.Timestamp | ToggleSchedule:
        """The upgrade timing of one upgraded turbine."""
        if turbine not in self.upgraded_turbines:
            msg = f"{turbine!r} is not an upgraded turbine of this campaign"
            raise KeyError(msg)
        return self.upgrade_timing

    def usable_mask(self, turbine: str, index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """Boolean mask over ``index`` of the records ``turbine``'s data may be used over."""
        usable = turbine not in self.excluded_turbines
        return np.full(len(index), usable, dtype=bool)

    def change_label(self) -> str:
        """How report and plot titles refer to what is being assessed."""
        return "the change"


@dataclass
class SyntheticCampaign:
    """A declared campaign: its turbines and roles, its timing, and the upgrades to inject.

    Private to the benchmark — it holds the injected upgrades, which are the ground truth.

    :param upgraded_turbines: turbines to upgrade and estimate
    :param upgrade_timing: changeover timestamp (prepost) or ``ToggleSchedule`` (toggle)
    :param candidate_references: turbines offered to methods as references
    :param upgrades: the upgrade callables to inject; empty for a placebo
    :param coords: turbine name → ``(latitude, longitude)`` in degrees
    :param north_offsets: step-applied northing corrections, ``(turbine, from, offset_deg)``
    :param rated_power_kw: the turbines' rated power
    :param analysis_period: ``(start, end)`` of the whole record, end exclusive
    :param excluded_turbines: turbines whose data must not be used
    :param columns: the source-native column schema the SCADA is keyed by
    :param seed: recorded in the generated dataset's run metadata
    """

    upgraded_turbines: list[str]
    upgrade_timing: pd.Timestamp | ToggleSchedule
    candidate_references: list[str]
    upgrades: list
    coords: dict[str, tuple[float, float]]
    north_offsets: list[tuple[str, pd.Timestamp, float]]
    rated_power_kw: float
    analysis_period: tuple[pd.Timestamp, pd.Timestamp]
    excluded_turbines: list[str] = field(default_factory=list)
    columns: ColumnSchema = HOT_COLUMNS
    seed: int = 0

    @property
    def turbines(self) -> list[str]:
        """Every declared turbine, upgraded first, in declaration order and without duplicates."""
        seen: dict[str, None] = {}
        for wtg in [*self.upgraded_turbines, *self.candidate_references]:
            seen.setdefault(wtg, None)
        return list(seen)

    def spec(self) -> CampaignSpec:
        """Derive the public spec: the same campaign with the injected upgrades dropped."""
        return CampaignSpec(
            upgraded_turbines=list(self.upgraded_turbines),
            upgrade_timing=self.upgrade_timing,
            candidate_references=list(self.candidate_references),
            excluded_turbines=list(self.excluded_turbines),
            coords=dict(self.coords),
            north_offsets=list(self.north_offsets),
            rated_power_kw=self.rated_power_kw,
            analysis_period=self.analysis_period,
            turbine_col=self.columns.turbine,
        )

    def generate(self, scada_df: pd.DataFrame) -> SyntheticDataset:
        """Inject the declared upgrades into ``scada_df`` over the analysis period."""
        start, end = self.analysis_period
        in_period = (scada_df.index >= start) & (scada_df.index < end)
        declared = scada_df[self.columns.turbine].isin(self.turbines).to_numpy()
        return generate_dataset(
            scada_df=scada_df[in_period & declared],
            test_wtgs=list(self.upgraded_turbines),
            upgrades=list(self.upgrades),
            mode="toggle" if isinstance(self.upgrade_timing, ToggleSchedule) else "prepost",
            upgrade_timing=self.upgrade_timing,
            rated_power_kw=self.rated_power_kw,
            columns=self.columns,
            seed=self.seed,
        )
```

Create `benchmarking/campaigns/__init__.py`:

```python
"""Whole-farm campaigns: declare one, run it, and report against the known truth."""

from __future__ import annotations

from benchmarking.campaigns.declaration import CampaignSpec, SyntheticCampaign

__all__ = ["CampaignSpec", "SyntheticCampaign"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarking/campaigns/test_declaration.py -q`
Expected: PASS (18 tests)

- [ ] **Step 5: Lint**

Run: `uv run poe lint`

- [ ] **Step 6: Checkpoint — report to the user**

Do not commit. Report the new `benchmarking/campaigns/` package and the new test package, and flag both as untracked.

---

### Task 4: `carried_forward_methods` — the mode rule

**Files:**
- Create: `benchmarking/campaigns/methods.py`
- Modify: `benchmarking/campaigns/__init__.py`
- Test: `tests/benchmarking/campaigns/test_methods.py`

**Interfaces:**
- Consumes: `CampaignSpec` (Task 3).
- Produces: `carried_forward_methods(spec: CampaignSpec, *, out_dir: Path, era5_hourly_df: pd.DataFrame | None = None, include_power_model: bool = True) -> list[Method]`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarking/campaigns/test_methods.py`:

```python
"""Tests for the applicable-method rule."""

from __future__ import annotations

import pandas as pd

from benchmarking.campaigns import carried_forward_methods
from benchmarking.synthetic import ToggleSchedule

from .test_declaration import CHANGEOVER, _campaign


def _names(spec, tmp_path) -> list[str]:
    return [m.name for m in carried_forward_methods(spec, out_dir=tmp_path, include_power_model=False)]


def test_prepost_skips_the_toggle_specialist(tmp_path) -> None:
    assert "toggle_specialist" not in _names(_campaign().spec(), tmp_path)


def test_toggle_includes_the_toggle_specialist(tmp_path) -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=CHANGEOVER)
    assert "toggle_specialist" in _names(_campaign(upgrade_timing=schedule).spec(), tmp_path)


def test_naive_ratio_runs_in_both_modes(tmp_path) -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=CHANGEOVER)
    assert "naive_ratio" in _names(_campaign().spec(), tmp_path)
    assert "naive_ratio" in _names(_campaign(upgrade_timing=schedule).spec(), tmp_path)


def test_power_model_is_included_when_asked(tmp_path) -> None:
    methods = carried_forward_methods(_campaign().spec(), out_dir=tmp_path, include_power_model=True)
    assert "power_model" in [m.name for m in methods]


def test_each_method_writes_into_its_own_subfolder(tmp_path) -> None:
    methods = carried_forward_methods(_campaign().spec(), out_dir=tmp_path, include_power_model=False)
    out_dirs = {m.out_dir for m in methods}
    assert len(out_dirs) == len(methods)
    assert all(d.parent == tmp_path for d in out_dirs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarking/campaigns/test_methods.py -q`
Expected: FAIL — `ImportError: cannot import name 'carried_forward_methods'`

- [ ] **Step 3: Write the implementation**

Create `benchmarking/campaigns/methods.py`:

```python
"""Which methods a campaign runs, and how each is configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.baselines.toggle_specialist import ToggleSpecialistMethod
from benchmarking.synthetic import HOT_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method


def carried_forward_methods(
    spec: CampaignSpec,
    *,
    out_dir: Path,
    era5_hourly_df: pd.DataFrame | None = None,
    include_power_model: bool = True,
) -> list[Method]:
    """Build the methods applicable to ``spec``, each writing into its own subfolder of ``out_dir``.

    ``toggle_specialist`` accepts only toggle campaigns and is left out of a prepost one.

    :param spec: the campaign being run
    :param out_dir: the turbine's output folder; each method gets a subfolder named after it
    :param era5_hourly_df: reanalysis for the power model; omit to run it without ERA5 features
    :param include_power_model: build the power model (needs the ``ml`` dependency group)
    """
    methods: list[Method] = [
        NaiveRatioMethod(columns=HOT_COLUMNS, out_dir=out_dir / "naive_ratio", save_plots=True)
    ]
    if spec.mode == "toggle":
        methods.append(
            ToggleSpecialistMethod(
                columns=HOT_COLUMNS,
                out_dir=out_dir / "toggle_specialist",
                save_plots=True,
                conditions=("power",),
                rated_power_kw=spec.rated_power_kw,
            )
        )
    if include_power_model:
        methods.append(
            PowerModelMethod(
                columns=HOT_COLUMNS,
                baseline_rated_power_kw=spec.rated_power_kw,
                era5_hourly_df=era5_hourly_df,
                availability_feature=False,
                era5_exclude=CURATED_ERA5_EXCLUDE,
                model_params=dict(TUNED_MODEL_PARAMS),
                out_dir=out_dir / "power_model",
                save_plots=True,
            )
        )
    return methods
```

Add to `benchmarking/campaigns/__init__.py`:

```python
from benchmarking.campaigns.methods import carried_forward_methods
```

and add `"carried_forward_methods"` to `__all__`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/benchmarking/campaigns/test_methods.py -q`
Expected: PASS (5 tests). If `PowerModelMethod` rejects `era5_hourly_df=None` or requires `active_power_min`, adjust the constructor call to match its actual signature — read `benchmarking/baselines/power_model/method.py` and keep the test's `include_power_model=False` cases passing regardless.

- [ ] **Step 5: Lint**

Run: `uv run poe lint`

- [ ] **Step 6: Checkpoint — report to the user**

Do not commit. Report the mode rule and which methods each mode builds.

---

### Task 5: `CampaignRunner`

**Files:**
- Create: `benchmarking/campaigns/runner.py`
- Modify: `benchmarking/campaigns/__init__.py`
- Test: `tests/benchmarking/campaigns/test_runner.py`

**Interfaces:**
- Consumes: `wind_up.farm_uplift`, `wind_up.TurbineUplift` (Task 1); `SyntheticDataset.true_farm_uplift` (Task 2); `CampaignSpec` (Task 3); harness `score_one`, `truth_mask`, `Replicate`, `CampaignWindow`.
- Produces:
  - `CampaignResult(spec: CampaignSpec, scores: pd.DataFrame, farm: pd.DataFrame, farm_uplifts: dict[str, FarmUplift], truth_farm_uplift: float, outputs: dict[tuple[str, str], MethodOutput])`
  - `CampaignRunner(spec: CampaignSpec, dataset: SyntheticDataset, *, build_methods: Callable[[str], list[Method]])` with `.run() -> CampaignResult`
  - `per_turbine_table(result: CampaignResult) -> pd.DataFrame` (columns `method`, `test_wtg`, `estimate`, `truth`, `signed_error`)

- [ ] **Step 1: Write the failing tests**

Create `tests/benchmarking/campaigns/test_runner.py`:

```python
"""Tests for the campaign runner: both output shapes, and a placebo reading ~0."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

from .test_declaration import CHANGEOVER, PERIOD, _campaign, _scada

TOLERANCE = 1e-9


class _ZeroMethod:
    """Reports exactly zero uplift, whatever it is given."""

    name = "zero"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        return MethodOutput(p50_overall=0.0)


class _OffsetMethod:
    """Reports a fixed non-zero uplift, for checking the farm headline is not hard-wired to 0."""

    name = "offset"

    def __init__(self, offset: float) -> None:
        self._offset = offset

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        return MethodOutput(p50_overall=self._offset)


class _RecordingMethod:
    """Captures every MethodInput it sees."""

    name = "recording"

    def __init__(self) -> None:
        self.seen: list[MethodInput] = []

    def estimate(self, mi: MethodInput) -> MethodOutput:
        self.seen.append(mi)
        return MethodOutput(p50_overall=0.0)


def _run(methods, *, upgrade_timing=CHANGEOVER):
    campaign = _campaign(upgrade_timing=upgrade_timing)
    dataset = campaign.generate(_scada())
    runner = CampaignRunner(campaign.spec(), dataset, build_methods=lambda _wtg: list(methods))
    return runner.run()


def test_placebo_per_turbine_estimates_are_zero_prepost() -> None:
    result = _run([_ZeroMethod()])
    table = per_turbine_table(result)
    assert set(table["test_wtg"]) == {"T1", "T2"}
    assert table["truth"].abs().max() < TOLERANCE
    assert table["signed_error"].abs().max() < TOLERANCE


def test_placebo_per_turbine_estimates_are_zero_toggle() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=8), start=CHANGEOVER)
    table = per_turbine_table(_run([_ZeroMethod()], upgrade_timing=schedule))
    assert table["truth"].abs().max() < TOLERANCE
    assert table["signed_error"].abs().max() < TOLERANCE


def test_placebo_farm_headline_is_zero_in_both_modes() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=8), start=CHANGEOVER)
    for timing in (CHANGEOVER, schedule):
        result = _run([_ZeroMethod()], upgrade_timing=timing)
        assert abs(result.truth_farm_uplift) < TOLERANCE
        assert abs(result.farm_uplifts["zero"].uplift) < TOLERANCE
        assert result.farm["signed_error"].abs().max() < TOLERANCE


def test_farm_headline_follows_the_method_not_the_truth() -> None:
    result = _run([_OffsetMethod(0.04)])
    assert result.farm_uplifts["offset"].uplift == pytest.approx(0.04)
    row = result.farm.set_index("method").loc["offset"]
    assert row["truth"] == pytest.approx(0.0, abs=TOLERANCE)
    assert row["signed_error"] == pytest.approx(0.04)


def test_scores_are_the_tidy_harness_rows() -> None:
    result = _run([_ZeroMethod()])
    expected = {"method", "test_wtg", "estimate", "truth", "signed_error", "treatment_start", "activity_end"}
    assert expected <= set(result.scores.columns)
    overall = result.scores[result.scores["condition"] == "overall"]
    assert len(overall) == 2  # one per upgraded turbine, n=1 campaign


def test_each_method_is_estimated_once_per_upgraded_turbine() -> None:
    recording = _RecordingMethod()
    _run([recording])
    assert len(recording.seen) == 2
    assert {mi.test_wtg for mi in recording.seen} == {"T1", "T2"}


def test_methods_never_see_an_excluded_turbine() -> None:
    recording = _RecordingMethod()
    _run([recording])
    for mi in recording.seen:
        assert "T5" not in set(mi.scada_df[HOT_COLUMNS.turbine])


def test_methods_see_only_the_analysis_period() -> None:
    recording = _RecordingMethod()
    _run([recording])
    for mi in recording.seen:
        assert mi.scada_df.index.min() >= PERIOD[0]
        assert mi.scada_df.index.max() < PERIOD[1]


def test_outputs_are_kept_for_every_method_and_turbine() -> None:
    result = _run([_ZeroMethod()])
    assert set(result.outputs) == {("zero", "T1"), ("zero", "T2")}
    assert all(isinstance(o, MethodOutput) for o in result.outputs.values())


def test_farm_table_reports_the_spread_and_guard_count() -> None:
    result = _run([_ZeroMethod()])
    row = result.farm.set_index("method").loc["zero"]
    assert "uplift_spread" in result.farm.columns
    assert row["n_guarded"] == 0


def test_treated_energy_matches_the_records_the_truth_uses() -> None:
    # the estimate side sums finite synthetic power; the truth sums the same records
    result = _run([_ZeroMethod()])
    detail = result.farm_uplifts["zero"].turbines
    assert (detail["n_records"] > 0).all()
    assert np.isfinite(detail["treated_energy"]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/benchmarking/campaigns/test_runner.py -q`
Expected: FAIL — `ImportError: cannot import name 'CampaignRunner'`

- [ ] **Step 3: Write the implementation**

Create `benchmarking/campaigns/runner.py`:

```python
"""Run a declared campaign: per-turbine estimates, one farm headline, both output shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.harness import CampaignWindow, MethodOutput, Replicate, score_one, truth_mask
from benchmarking.synthetic import treated_mask
from wind_up import TurbineUplift, farm_uplift

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method, MethodInput
    from benchmarking.synthetic import SyntheticDataset
    from wind_up import FarmUplift


@dataclass
class CampaignResult:
    """Everything one campaign run produced.

    :param spec: the campaign that was run
    :param scores: the tidy harness rows, one set per upgraded turbine at n=1
    :param farm: one row per method — ``estimate``, ``truth``, ``signed_error``,
        ``uplift_spread`` and ``n_guarded``
    :param farm_uplifts: each method's full :class:`~wind_up.FarmUplift`, including per-turbine detail
    :param truth_farm_uplift: the exact pooled farm truth
    :param outputs: each ``(method, turbine)``'s raw :class:`~benchmarking.harness.MethodOutput`
    """

    spec: CampaignSpec
    scores: pd.DataFrame
    farm: pd.DataFrame
    farm_uplifts: dict[str, FarmUplift]
    truth_farm_uplift: float
    outputs: dict[tuple[str, str], MethodOutput]


class _Capturing:
    """Delegates to a method and keeps its output, so one estimate call serves both output shapes."""

    def __init__(self, method: Method) -> None:
        self._method = method
        self.name = method.name
        self.output: MethodOutput | None = None

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate via the wrapped method, retaining the output."""
        self.output = self._method.estimate(mi)
        return self.output


class CampaignRunner:
    """Turn a campaign spec plus its generated dataset into per-turbine and farm results.

    :param spec: the public campaign facts; methods see nothing else
    :param dataset: the generated dataset, whose ``original_df`` supplies the truth
    :param build_methods: given an upgraded turbine's name, the methods to run for it
    """

    def __init__(
        self,
        spec: CampaignSpec,
        dataset: SyntheticDataset,
        *,
        build_methods: Callable[[str], list[Method]],
    ) -> None:
        self._spec = spec
        self._dataset = dataset
        self._build_methods = build_methods

    def run(self) -> CampaignResult:
        """Run every applicable method on every upgraded turbine and aggregate to one headline."""
        spec = self._spec
        method_frame = self._method_facing_frame()
        window = self._window()

        score_rows: list[dict[str, object]] = []
        outputs: dict[tuple[str, str], MethodOutput] = {}
        estimates: dict[str, list[TurbineUplift]] = {}
        truth_masks: dict[str, np.ndarray] = {}

        for wtg in spec.upgraded_turbines:
            replicate = Replicate(
                dataset=self._subset_dataset(method_frame),
                test_wtg=wtg,
                treatment_start=spec.treatment_start,
                upgrade_timing=spec.timing_for(wtg),
            )
            mask = truth_mask(replicate, window)
            truth_masks[wtg] = mask
            truth = replicate.true_uplift(mask=mask).overall
            energy, n_records = self._treated_energy(method_frame, turbine=wtg, mask=mask)

            for method in self._build_methods(wtg):
                capturing = _Capturing(method)
                score_rows.extend(
                    score_one(
                        capturing,
                        replicate=replicate,
                        window=window,
                        truth=truth,
                        mask=mask,
                        profile_name=spec.change_label(),
                    )
                )
                assert capturing.output is not None  # noqa: S101 - score_one always estimates
                outputs[method.name, wtg] = capturing.output
                estimates.setdefault(method.name, []).append(
                    TurbineUplift(
                        turbine=wtg,
                        uplift=capturing.output.p50_overall,
                        treated_energy=energy,
                        n_records=n_records,
                        rated_power_kw=spec.rated_power_kw,
                    )
                )

        truth_farm = self._dataset.true_farm_uplift(test_wtgs=list(spec.upgraded_turbines), masks=truth_masks)
        farm_uplifts = {name: farm_uplift(rows) for name, rows in estimates.items()}
        farm = pd.DataFrame(
            [
                {
                    "method": name,
                    "estimate": result.uplift,
                    "truth": truth_farm,
                    "signed_error": result.uplift - truth_farm,
                    "uplift_spread": result.uplift_spread,
                    "n_guarded": int((result.turbines["guard"] != "").sum()),
                }
                for name, result in farm_uplifts.items()
            ]
        )
        return CampaignResult(
            spec=spec,
            scores=pd.DataFrame(score_rows),
            farm=farm,
            farm_uplifts=farm_uplifts,
            truth_farm_uplift=truth_farm,
            outputs=outputs,
        )

    def _method_facing_frame(self) -> pd.DataFrame:
        """The synthetic rows a method may see: within the analysis period, usable turbines only."""
        spec = self._spec
        frame = self._dataset.synthetic_df
        start, end = spec.analysis_period
        keep = np.asarray((frame.index >= start) & (frame.index < end))
        for turbine, rows in frame.groupby(frame[spec.turbine_col], sort=False):
            usable = spec.usable_mask(str(turbine), pd.DatetimeIndex(rows.index))
            keep[frame[spec.turbine_col].to_numpy() == turbine] &= usable
        return frame[keep]

    def _subset_dataset(self, method_frame: pd.DataFrame) -> SyntheticDataset:
        """The dataset restricted to the method-facing rows, truth frame kept aligned."""
        from dataclasses import replace  # noqa: PLC0415 - local to keep the module import list flat

        original = self._dataset.original_df
        aligned = original.loc[original.index.isin(method_frame.index)]
        aligned = aligned[aligned[self._spec.turbine_col].isin(method_frame[self._spec.turbine_col].unique())]
        return replace(self._dataset, synthetic_df=method_frame, original_df=aligned)

    def _window(self) -> CampaignWindow:
        """One window spanning the whole campaign, so the harness scores it at n=1."""
        start, end = self._spec.analysis_period
        treatment_start = self._spec.treatment_start
        months = (end.year - treatment_start.year) * 12 + (end.month - treatment_start.month)
        return CampaignWindow(
            length=months,
            unit="months",
            baseline_start=start,
            treatment_start=treatment_start,
            activity_end=end,
        )

    def _treated_energy(self, frame: pd.DataFrame, *, turbine: str, mask: np.ndarray) -> tuple[float, int]:
        """Observed treated-period energy and record count for one turbine, finite records only."""
        columns = self._dataset.columns
        power = frame.loc[frame[columns.turbine] == turbine, columns.active_power].to_numpy(dtype=float)
        selected = mask & np.isfinite(power)
        return float(power[selected].sum()), int(selected.sum())


def per_turbine_table(result: CampaignResult) -> pd.DataFrame:
    """The per-turbine headline rows: one per method and upgraded turbine."""
    overall = result.scores[result.scores["condition"] == "overall"]
    return overall[["method", "test_wtg", "estimate", "truth", "signed_error"]].reset_index(drop=True)
```

Note on `treated_mask`: it is imported for the report's use in Task 6; if ruff flags it unused here, drop the import from this module.

Add to `benchmarking/campaigns/__init__.py`:

```python
from benchmarking.campaigns.runner import CampaignResult, CampaignRunner, per_turbine_table
```

and extend `__all__` with `"CampaignResult"`, `"CampaignRunner"`, `"per_turbine_table"` (keep it sorted).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarking/campaigns/ -q`
Expected: PASS. If `Replicate` construction or `score_one` complains about the dataset's `run_metadata` (`_conditional_rows` reads `run_metadata["rated_power_kw"]`), confirm `generate_dataset` recorded it — it does — and that `_subset_dataset` preserves `run_metadata`, which `dataclasses.replace` does.

- [ ] **Step 5: Lint**

Run: `uv run poe lint`

- [ ] **Step 6: Checkpoint — report to the user**

Do not commit. Report both output shapes working and the placebo reading ~0 in both modes.

---

### Task 6: The inspection report

**Files:**
- Create: `benchmarking/campaigns/report.py`
- Modify: `benchmarking/campaigns/__init__.py`
- Test: `tests/benchmarking/campaigns/test_report.py`

**Interfaces:**
- Consumes: `CampaignResult`, `per_turbine_table` (Task 5); `conditional_truth_vs_estimate` from `benchmarking.baselines.inspect_prepost_hard_case`; `plot_conditional_uplift`, `condition_bins`, `CONDITIONS` from `benchmarking.harness`.
- Produces: `write_campaign_report(result: CampaignResult, dataset: SyntheticDataset, *, out_dir: Path) -> Path`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarking/campaigns/test_report.py`:

```python
"""Tests for the campaign inspection report."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import pandas as pd

from benchmarking.campaigns import CampaignRunner, write_campaign_report
from benchmarking.harness import MethodInput, MethodOutput

from .test_declaration import CHANGEOVER, _campaign, _scada


class _ZeroMethod:
    name = "zero"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        return MethodOutput(p50_overall=0.0)


def _result():
    campaign = _campaign(upgrade_timing=CHANGEOVER)
    dataset = campaign.generate(_scada())
    runner = CampaignRunner(campaign.spec(), dataset, build_methods=lambda _wtg: [_ZeroMethod()])
    return runner.run(), dataset


def test_report_writes_the_three_tables(tmp_path) -> None:
    result, dataset = _result()
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    assert (out / "per_turbine.csv").exists()
    assert (out / "farm_uplift.csv").exists()
    assert (out / "scores.csv").exists()


def test_farm_table_records_the_spread_and_guards(tmp_path) -> None:
    result, dataset = _result()
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    farm = pd.read_csv(out / "farm_uplift.csv")
    assert {"method", "estimate", "truth", "signed_error", "uplift_spread", "n_guarded"} <= set(farm.columns)


def test_per_turbine_detail_is_written_for_each_method(tmp_path) -> None:
    result, dataset = _result()
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    detail = pd.read_csv(out / "farm_uplift_detail.csv")
    assert set(detail["turbine"]) == {"T1", "T2"}
    assert "guard" in detail.columns


def test_report_returns_the_directory_it_wrote_to(tmp_path) -> None:
    result, dataset = _result()
    assert write_campaign_report(result, dataset, out_dir=tmp_path) == tmp_path


def test_report_skips_conditional_plots_when_no_method_reports_conditions(tmp_path) -> None:
    result, dataset = _result()
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    assert not list((out / "conditional").glob("*.png")) if (out / "conditional").exists() else True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarking/campaigns/test_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'write_campaign_report'`

- [ ] **Step 3: Write the implementation**

Create `benchmarking/campaigns/report.py`:

```python
"""The whole-farm inspection report: per-turbine and farm tables, plus diagnostic plots."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from benchmarking.baselines.inspect_prepost_hard_case import conditional_truth_vs_estimate
from benchmarking.campaigns.runner import per_turbine_table
from benchmarking.harness import CONDITIONS, condition_bins, plot_conditional_uplift
from benchmarking.synthetic import treated_mask

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.runner import CampaignResult
    from benchmarking.synthetic import SyntheticDataset

logger = logging.getLogger(__name__)


def write_campaign_report(result: CampaignResult, dataset: SyntheticDataset, *, out_dir: Path) -> Path:
    """Write the campaign's tables and plots under ``out_dir`` and return it.

    Writes ``per_turbine.csv``, ``farm_uplift.csv``, ``farm_uplift_detail.csv`` and ``scores.csv``,
    plus one conditional uplift plot per condition for each method that reports per-condition
    estimates.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_turbine = per_turbine_table(result)
    per_turbine.to_csv(out_dir / "per_turbine.csv", index=False)
    result.farm.to_csv(out_dir / "farm_uplift.csv", index=False)
    result.scores.to_csv(out_dir / "scores.csv", index=False)

    detail = pd.concat(
        [frame.turbines.assign(method=name) for name, frame in result.farm_uplifts.items()], ignore_index=True
    )
    detail.to_csv(out_dir / "farm_uplift_detail.csv", index=False)

    logger.info("Per-turbine results for %s:\n%s", result.spec.change_label(), per_turbine.to_string(index=False))
    logger.info("Farm uplift:\n%s", result.farm.to_string(index=False))

    _write_conditional_plots(result, dataset, out_dir=out_dir / "conditional")
    return out_dir


def _write_conditional_plots(result: CampaignResult, dataset: SyntheticDataset, *, out_dir: Path) -> None:
    """One conditional-uplift plot per condition, for every method that reports per-condition rows."""
    spec = result.spec
    for (method_name, wtg), output in result.outputs.items():
        if output.p50_by_condition is None:
            continue
        rows = dataset.synthetic_df[dataset.synthetic_df[spec.turbine_col] == wtg]
        mask = treated_mask(pd.DatetimeIndex(rows.index), spec.timing_for(wtg))
        truth_by_condition = {
            condition: dataset.true_uplift(
                test_wtg=wtg,
                mask=mask,
                by=condition,
                bins=condition_bins(condition, rated_power_kw=spec.rated_power_kw),
            ).by_condition
            for condition in CONDITIONS
        }
        clean = {c: frame for c, frame in truth_by_condition.items() if frame is not None}
        if not clean:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = conditional_truth_vs_estimate(output, clean, method_name=method_name)
        for condition in clean:
            fig = plot_conditional_uplift(
                frame,
                condition=condition,
                save_path=out_dir / f"conditional_uplift_{condition}_{wtg}_{method_name}.png",
                title=f"Conditional uplift ({condition}) — {wtg}, {method_name} vs truth",
            )
            plt.close(fig)
```

Add to `benchmarking/campaigns/__init__.py`:

```python
from benchmarking.campaigns.report import write_campaign_report
```

and add `"write_campaign_report"` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarking/campaigns/ -q`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `uv run poe lint`

- [ ] **Step 6: Checkpoint — report to the user**

Do not commit. Report the report contents and any plots produced.

---

### Task 7: The two placebo campaigns and their driver

**Files:**
- Create: `benchmarking/campaigns/placebo.py`
- Test: `tests/benchmarking/campaigns/test_placebo.py`

**Interfaces:**
- Consumes: everything above; `benchmarking.baselines.hot_context.{NORTHING_YAML, build_hot_v0_context}`; `benchmarking.synthetic.sources.hill_of_towie.{load_hot_scada, load_hot_metadata}`.
- Produces: `placebo_campaign(mode: Literal["prepost", "toggle"]) -> SyntheticCampaign`; `run_placebo(*, mode, include_power_model=True, include_v0=False, out_root=None) -> CampaignResult`; `PLACEBO_TURBINES`, `PLACEBO_PERIOD`, `PLACEBO_CHANGEOVER`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarking/campaigns/test_placebo.py`:

```python
"""Tests for the two declared placebo campaigns."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table
from benchmarking.campaigns.placebo import PLACEBO_CHANGEOVER, PLACEBO_PERIOD, PLACEBO_TURBINES, placebo_campaign
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

TOLERANCE = 1e-9


class _ZeroMethod:
    name = "zero"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        return MethodOutput(p50_overall=0.0)


def _fixture_scada() -> pd.DataFrame:
    """A tiny stand-in for the Hill of Towie download: flat power over the placebo period."""
    index = pd.date_range(PLACEBO_PERIOD[0], PLACEBO_PERIOD[1], freq="1h", tz="UTC", inclusive="left")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: wtg,
                    HOT_COLUMNS.active_power: 900.0,
                    HOT_COLUMNS.wind_speed: 8.0,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.availability: 3600.0,
                },
                index=index,
            )
            for wtg in PLACEBO_TURBINES
        ]
    )


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_placebo_injects_nothing(mode: str) -> None:
    campaign = placebo_campaign(mode)
    assert campaign.upgrades == []
    dataset = campaign.generate(_fixture_scada())
    pd.testing.assert_frame_equal(dataset.synthetic_df, dataset.original_df)


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_placebo_spec_mode_matches(mode: str) -> None:
    assert placebo_campaign(mode).spec().mode == mode


def test_placebo_period_is_january_to_june_inclusive() -> None:
    start, end = PLACEBO_PERIOD
    assert (start.month, start.day) == (1, 1)
    assert (end.month, end.day) == (7, 1)  # end-exclusive, so June is included
    assert PLACEBO_CHANGEOVER > start
    assert PLACEBO_CHANGEOVER < end


def test_placebo_uses_about_six_turbines() -> None:
    assert 5 <= len(PLACEBO_TURBINES) <= 7


def test_toggle_placebo_declares_a_schedule() -> None:
    assert isinstance(placebo_campaign("toggle").upgrade_timing, ToggleSchedule)


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_placebo_runs_end_to_end_to_zero(mode: str) -> None:
    campaign = placebo_campaign(mode)
    dataset = campaign.generate(_fixture_scada())
    result = CampaignRunner(campaign.spec(), dataset, build_methods=lambda _wtg: [_ZeroMethod()]).run()
    assert per_turbine_table(result)["signed_error"].abs().max() < TOLERANCE
    assert abs(result.farm_uplifts["zero"].uplift) < TOLERANCE
    assert abs(result.truth_farm_uplift) < TOLERANCE


def test_placebo_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        placebo_campaign("sideways")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarking/campaigns/test_placebo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarking.campaigns.placebo'`

- [ ] **Step 3: Write the implementation**

Create `benchmarking/campaigns/placebo.py`:

```python
"""The placebo campaigns: a whole farm with nothing injected, run end-to-end.

Both modes are declared once here. With no upgrade the synthetic data equals the original, so
every method's per-turbine and farm estimate should read ~0.

Run it::

    uv run python -m benchmarking.campaigns.placebo

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``placebo``/``<mode>_<timestamp>/``. The
first run downloads and caches the Hill of Towie SCADA (Zenodo) and ERA5 (Open-Meteo).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl

mpl.use("Agg")

import pandas as pd
import yaml

from benchmarking.baselines.hot_context import NORTHING_YAML, build_hot_v0_context
from benchmarking.campaigns.declaration import SyntheticCampaign
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.report import write_campaign_report
from benchmarking.campaigns.runner import CampaignRunner
from benchmarking.synthetic import HOT_RATED_POWER_KW, ToggleSchedule
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.runner import CampaignResult

logger = logging.getLogger(__name__)

PLACEBO_TURBINES = ("T01", "T02", "T03", "T04", "T05", "T07")
PLACEBO_WTG_NUMBERS = [1, 2, 3, 4, 5, 7]
PLACEBO_UPGRADED = ("T01", "T04")
PLACEBO_EXCLUDED = ("T07",)
PLACEBO_PERIOD = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2018-07-01", tz="UTC"))
PLACEBO_CHANGEOVER = pd.Timestamp("2018-04-01", tz="UTC")
PLACEBO_TOGGLE_PERIOD = pd.Timedelta(hours=12)


def default_output_root() -> Path:
    """The directory this driver writes under; override with ``WIND_UP_BENCHMARKING_OUTPUT_DIR``."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "placebo"


def _north_offsets(turbines: Sequence[str]) -> list[tuple[str, pd.Timestamp, float]]:
    """Step-applied north offsets for ``turbines`` from the vendored northing YAML (UTC)."""
    data = yaml.safe_load(NORTHING_YAML.read_text())
    return [
        (str(name), pd.Timestamp(ts, tz="UTC"), float(offset))
        for (name, ts, offset) in data
        if str(name) in set(turbines)
    ]


def _coords() -> dict[str, tuple[float, float]]:
    """Hill of Towie turbine coordinates for the placebo turbines."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(PLACEBO_TURBINES)
    }


def placebo_campaign(mode: Literal["prepost", "toggle"], *, coords: dict | None = None) -> SyntheticCampaign:
    """Declare the placebo campaign for ``mode``: a whole farm with no upgrade injected.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param coords: turbine coordinates; loaded from the Hill of Towie metadata when omitted
    """
    if mode == "prepost":
        timing: pd.Timestamp | ToggleSchedule = PLACEBO_CHANGEOVER
    elif mode == "toggle":
        timing = ToggleSchedule(period=PLACEBO_TOGGLE_PERIOD, start=PLACEBO_CHANGEOVER)
    else:
        msg = f"unknown mode {mode!r}; expected 'prepost' or 'toggle'"
        raise ValueError(msg)
    return SyntheticCampaign(
        upgraded_turbines=list(PLACEBO_UPGRADED),
        upgrade_timing=timing,
        candidate_references=[w for w in PLACEBO_TURBINES if w not in PLACEBO_UPGRADED],
        excluded_turbines=list(PLACEBO_EXCLUDED),
        upgrades=[],
        coords=coords if coords is not None else {w: (0.0, 0.0) for w in PLACEBO_TURBINES},
        north_offsets=_north_offsets(PLACEBO_TURBINES),
        rated_power_kw=HOT_RATED_POWER_KW,
        analysis_period=PLACEBO_PERIOD,
    )


def run_placebo(
    *,
    mode: Literal["prepost", "toggle"],
    include_power_model: bool = True,
    out_root: str | Path | None = None,
) -> CampaignResult:
    """Run one placebo campaign end-to-end and write its report.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param include_power_model: run the power model as well as the fast methods
    :param out_root: base output dir; defaults to :func:`default_output_root`
    """
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{mode}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scada_df, _ = load_hot_scada(
        start_dt=PLACEBO_PERIOD[0],
        end_dt_excl=PLACEBO_PERIOD[1],
        wtg_numbers=PLACEBO_WTG_NUMBERS,
        wtg_names=list(PLACEBO_TURBINES),
    )
    campaign = placebo_campaign(mode, coords=_coords())
    dataset = campaign.generate(scada_df)
    spec = campaign.spec()
    era5 = build_hot_v0_context(wtg_names=list(PLACEBO_TURBINES)).reanalysis_datasets[0].data

    runner = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: carried_forward_methods(
            spec,
            out_dir=run_dir / wtg,
            era5_hourly_df=era5,
            include_power_model=include_power_model,
        ),
    )
    result = runner.run()
    write_campaign_report(result, dataset, out_dir=run_dir)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for placebo_mode in ("prepost", "toggle"):
        run_placebo(mode=placebo_mode)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarking/campaigns/ -q`
Expected: PASS. The tests never download Hill of Towie data — they call `placebo_campaign` (which defaults `coords`) and the tiny fixture. If `_north_offsets` reading the vendored YAML is slow or the file is missing in CI, make `placebo_campaign` accept `north_offsets=None` and skip the read; keep the driver passing the real ones.

- [ ] **Step 5: Run the whole fast suite and lint**

Run: `uv run poe lint` then `uv run poe test-fast`
Expected: all green, no regressions in existing tests.

- [ ] **Step 6: Real-data smoke run (manual, optional but recommended)**

Run: `uv run python -m benchmarking.campaigns.placebo`
Expected: both campaigns complete; `per_turbine.csv` and `farm_uplift.csv` show every method's estimate near 0 and `truth` exactly 0. Record the actual numbers for the user — a real method on real SCADA will not be exactly 0, and how close it lands is the interesting result of C1.

- [ ] **Step 7: Checkpoint — report to the user**

Do not commit. Report: full file list added, `poe all-fast` status, the placebo numbers from the smoke run, and every new untracked path so the user can `git add` them.

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| §1 module split — `src/wind_up/farm.py` | 1 |
| §1 module split — `benchmarking/campaigns/` (`declaration.py`, `runner.py`, `report.py`, placebo driver) | 3, 5, 6, 7 |
| §2 `CampaignSpec` public facts; test asserts no upgrade physics | 3 |
| §2 future-proofing — `spec.mode`, `timing_for`, `usable_mask`, `change_label` | 3 (defined), 5 + 6 (consumed) |
| §3 farm uplift, both guards, spread, guard flags | 1 |
| §3 truth headline from `original_df` | 2 |
| §4 runner: per-turbine methods, mode rule, truth, farm estimate, n=1 harness rows | 4, 5 |
| §5 output shape 1 — inspection report with tables, spread, guards, plots | 6 |
| §5 output shape 2 — tidy `score_one` frame | 5 (`CampaignResult.scores`) |
| §6 two placebo campaigns, ~6 turbines, Jan–June, `upgrades=[]`, one excluded | 7 |
| Testing — guard unit tests, brief-has-no-physics, fast tiny-fixture placebo both modes, both output shapes | 1, 3, 5, 7 |

**Deliberately deferred, and why**
- **v0 is not wired in.** §4 says "v0 is included but optional (slow)". `V0BinnedMethod` needs a `HotV0Context` and a real HoT asset config, so it cannot run on the tiny fixture and would make the placebo driver's default path a multi-hour run. The seam accepts it unchanged — `carried_forward_methods` gains an `include_v0` branch in one place. **Raise this with the user at the Task 4 checkpoint**: if they want it in C1, it is one constructor call plus a driver flag.
- **No reference selection in the runner.** The seam bakes reference selection into each `Method`; automatic selection from the spec is explicitly C3.

**Placeholder scan:** no TBDs; every step carries runnable code or an exact command. Two steps name a fallback if a real signature differs (Task 4 Step 4, Task 7 Step 4) — those are contingencies with a stated action, not placeholders.

**Type consistency:** `TurbineUplift`/`FarmUplift`/`farm_uplift` (Task 1) are used with those exact names in Task 5. `true_farm_uplift` (Task 2) is called as `dataset.true_farm_uplift(test_wtgs=..., masks=...)` in Task 5. `CampaignSpec.timing_for`/`usable_mask`/`mode`/`change_label`/`treatment_start` (Task 3) are consumed in Tasks 4–6 under those names. `CampaignResult` field names match between Tasks 5, 6 and 7. `per_turbine_table` is defined in `runner.py` and imported by `report.py` and the tests.
