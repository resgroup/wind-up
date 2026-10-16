"""The level probe's analysis: where a placebo reading's level lives in the per-row dump."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.level_analysis import (
    RowDump,
    aipw_level,
    baseline_predictions,
    channel_difference_by,
    ess_fraction,
    label_direction_sector,
    label_held_back,
    label_month,
    label_operating_pattern,
    label_propensity_decile,
    label_upwind_offline,
    label_wind_band,
    level_by,
    propensity,
    top_group,
)

if TYPE_CHECKING:
    from pathlib import Path

_POWER = "power"
_DIR = "northed_dir"
_Q = " @ "
_FAST = {"n_estimators": 20, "learning_rate": 0.2, "num_leaves": 7, "min_child_samples": 10, "n_jobs": 1}


def _dump(
    *,
    n_base: int = 2000,
    n_up: int = 1000,
    references: tuple[str, ...] = ("R1", "R2"),
    bias_kw: np.ndarray | None = None,
    direction_deg: float | np.ndarray = 180.0,
    coords: dict | None = None,
    seed: int = 0,
) -> RowDump:
    """A toy dump: hourly rows, baseline then upgraded, counterfactual = actual - ``bias_kw``."""
    rng = np.random.default_rng(seed)
    n = n_base + n_up
    ts = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    actual = rng.uniform(200.0, 2000.0, n)
    upgraded = np.arange(n) >= n_base
    bias = np.zeros(n)
    if bias_kw is not None:
        bias[upgraded] = bias_kw
    counterfactual = np.where(upgraded, actual - bias, np.nan)
    rad = np.deg2rad(np.broadcast_to(np.asarray(direction_deg, dtype=float), (n,)))
    rows = pd.DataFrame(
        {
            "timestamp": ts,
            "actual_kw": actual,
            "counterfactual_kw": counterfactual,
            "selected": True,
            "baseline": ~upgraded,
            "upgraded": upgraded,
            "reference_mean_ws": rng.uniform(3.0, 15.0, n),
        }
    )
    for ref in references:
        rows[f"{_POWER}{_Q}{ref}"] = actual * rng.uniform(0.9, 1.1, n)
        rows[f"{_DIR}_sin{_Q}{ref}"] = np.sin(rad)
        rows[f"{_DIR}_cos{_Q}{ref}"] = np.cos(rad)
    return _with_meta(rows, references=list(references), coords=coords or {})


def _with_meta(rows: pd.DataFrame, **meta: object) -> RowDump:
    up = rows["selected"] & rows["upgraded"]
    sum_actual = float(rows.loc[up, "actual_kw"].sum())
    sum_counter = float(rows.loc[up, "counterfactual_kw"].sum())
    full = {
        "test_wtg": "T1",
        "references": [],
        "power_free": [],
        "wake_only": [],
        "uplift": sum_actual / sum_counter - 1,
        "sum_actual_kw": sum_actual,
        "sum_counterfactual_kw": sum_counter,
        "n_baseline_rows": int((rows["selected"] & rows["baseline"]).sum()),
        "n_upgraded_rows": int(up.sum()),
        "timebase_s": 3600.0,
        "active_power_col": _POWER,
        "northed_direction_col": _DIR,
        "waking_threshold_kw": 100.0,
        "baseline_rated_power_kw": 2000.0,
        "coords": {},
        **meta,
    }
    return RowDump(rows=rows.reset_index(drop=True), meta=full)


class TestRowDump:
    def test_a_missing_required_column_raises_naming_it(self) -> None:
        dump = _dump()
        with pytest.raises(ValueError, match="counterfactual_kw"):
            RowDump(rows=dump.rows.drop(columns="counterfactual_kw"), meta=dump.meta)

    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        dump = _dump()
        dump.save(tmp_path / "T1")
        loaded = RowDump.load(tmp_path / "T1")
        pd.testing.assert_frame_equal(loaded.rows, dump.rows)
        assert loaded.meta == dump.meta


class TestClosure:
    def test_every_grouping_closes_on_the_headline(self) -> None:
        rng = np.random.default_rng(1)
        dump = _dump(bias_kw=rng.normal(20.0, 30.0, 1000), direction_deg=rng.uniform(0, 360, 3000))
        for labels in (label_month(dump), label_direction_sector(dump), label_wind_band(dump)):
            table = level_by(dump, labels)
            assert table["contribution_pp"].sum() == pytest.approx(100 * dump.meta["uplift"], abs=1e-9)
            assert table["row_share"].sum() == pytest.approx(1.0)
            assert "(missing)" not in set(table["group"])

    def test_a_nan_label_forms_the_missing_group(self) -> None:
        dump = _dump(bias_kw=np.full(1000, 10.0))
        labels = label_month(dump).astype(object)
        labels.iloc[-5:] = np.nan
        table = level_by(dump, labels)
        assert "(missing)" in set(table["group"])
        assert table.set_index("group").loc["(missing)", "n_rows"] == 5
        assert table["contribution_pp"].sum() == pytest.approx(100 * dump.meta["uplift"], abs=1e-9)

    def test_a_grouping_that_loses_rows_raises(self) -> None:
        dump = _dump(bias_kw=np.full(1000, 10.0))
        bad = RowDump(rows=dump.rows, meta={**dump.meta, "uplift": dump.meta["uplift"] + 0.01})
        with pytest.raises(ValueError, match="uplift"):
            level_by(bad, label_month(bad))


class TestLocalisation:
    def test_a_planted_bias_is_found_in_its_month_and_pattern(self) -> None:
        dump = _dump()
        rows = dump.rows.copy()
        up = rows["upgraded"].to_numpy()
        month = rows["timestamp"].dt.strftime("%Y-%m")
        target_month = month[up].iloc[300]
        # R2 offline for 200 upgraded rows, half of them in the target month
        offline = np.zeros(len(rows), dtype=bool)
        offline[np.flatnonzero(up)[200:400]] = True
        rows.loc[offline, f"{_POWER}{_Q}R2"] = np.nan
        planted = offline & (month == target_month).to_numpy()
        rows.loc[planted, "counterfactual_kw"] -= 100.0
        dump = _with_meta(rows, references=["R1", "R2"])
        planted_pp = 100 * 100.0 * planted.sum() / dump.meta["sum_counterfactual_kw"]

        by_month = level_by(dump, label_month(dump))
        assert top_group(by_month) == target_month
        assert by_month.set_index("group").loc[target_month, "contribution_pp"] == pytest.approx(planted_pp)

        by_pattern = level_by(dump, label_operating_pattern(dump))
        assert top_group(by_pattern) == "10"
        assert by_pattern.set_index("group").loc["10", "contribution_pp"] == pytest.approx(planted_pp)

    def test_rare_patterns_collapse_into_other(self) -> None:
        dump = _dump()
        rows = dump.rows.copy()
        rows.loc[np.flatnonzero(rows["upgraded"])[:10], f"{_POWER}{_Q}R1"] = np.nan
        labels = label_operating_pattern(_with_meta(rows, references=["R1", "R2"]), min_rows=50)
        assert set(labels[rows["upgraded"]]) == {"11", "other"}

    def test_power_free_references_are_not_in_the_pattern(self) -> None:
        dump = _dump(references=("R1", "R2", "R3"))
        labels = label_operating_pattern(RowDump(rows=dump.rows, meta={**dump.meta, "power_free": ["R2"]}))
        assert set(labels) == {"11"}


class TestDirectionSector:
    def test_north_wraps_into_one_sector(self) -> None:
        dump = _dump(direction_deg=np.where(np.arange(3000) % 2 == 0, 359.0, 1.0))
        labels = label_direction_sector(dump)
        assert (labels == labels.iloc[0]).all()

    def test_falls_back_to_reanalysis_without_reference_directions(self) -> None:
        dump = _dump()
        rows = dump.rows.drop(columns=[c for c in dump.rows.columns if c.startswith(_DIR)])
        rows["wind_direction_100m"] = 95.0
        labels = label_direction_sector(RowDump(rows=rows, meta=dump.meta))
        assert set(labels) == {"090"}

    def test_skipped_with_a_warning_when_there_is_no_direction(self, caplog: pytest.LogCaptureFixture) -> None:
        dump = _dump()
        rows = dump.rows.drop(columns=[c for c in dump.rows.columns if c.startswith(_DIR)])
        with caplog.at_level(logging.WARNING):
            assert label_direction_sector(RowDump(rows=rows, meta=dump.meta)) is None
        assert "direction" in caplog.text


class TestWindBand:
    def test_two_metre_bands(self) -> None:
        dump = _dump()
        labels = label_wind_band(dump)
        ws = dump.rows["reference_mean_ws"]
        assert (labels[(ws >= 4) & (ws < 6)] == "04-06").all()

    def test_skipped_without_wind_speed(self) -> None:
        dump = _dump()
        rows = dump.rows.assign(reference_mean_ws=np.nan)
        assert label_wind_band(RowDump(rows=rows, meta=dump.meta)) is None


class TestUpwindOffline:
    # T1 at the origin; R1 500 m due north, W1 500 m due east; 100 m rotors, so 5 D apart and a
    # disturbed sector of about 53 degrees: the wind from the north puts R1 upwind and W1 not.
    _COORDS: ClassVar[dict[str, list[float]]] = {
        "T1": [56.0, -3.0],
        "R1": [56.0 + 500 / 111_320, -3.0],
        "W1": [56.0, -3.0 + 500 / (111_320 * np.cos(np.deg2rad(56.0)))],
    }

    def _layout_dump(self, *, r1_power: list[float], w1_waking: list[float]) -> RowDump:
        n = len(r1_power)
        rows = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
                "actual_kw": 1000.0,
                "counterfactual_kw": 1000.0,
                "selected": True,
                "baseline": False,
                "upgraded": True,
                "reference_mean_ws": 8.0,
                f"{_POWER}{_Q}R1": r1_power,
                f"{_DIR}_sin{_Q}R1": 0.0,
                f"{_DIR}_cos{_Q}R1": 1.0,
                f"waking_{_POWER}{_Q}W1": w1_waking,
            }
        )
        return _with_meta(rows, references=["R1"], wake_only=["W1"], coords=self._COORDS)

    def test_counts_an_upwind_turbine_that_is_not_waking(self) -> None:
        # R1 idle, running, unknown; W1 is off but outside the sector, so it never counts
        dump = self._layout_dump(r1_power=[0.0, 1500.0, np.nan], w1_waking=[0.0, 0.0, 0.0])
        labels = label_upwind_offline(dump, rotor_diameter_m=100.0)
        assert list(labels) == ["1", "0", "unknown"]

    def test_a_turbine_outside_the_sector_does_not_count(self) -> None:
        dump = self._layout_dump(r1_power=[1500.0], w1_waking=[np.nan])
        assert list(label_upwind_offline(dump, rotor_diameter_m=100.0)) == ["0"]

    def test_skipped_without_coordinates(self) -> None:
        dump = self._layout_dump(r1_power=[0.0], w1_waking=[0.0])
        no_coords = RowDump(rows=dump.rows, meta={**dump.meta, "coords": {}})
        assert label_upwind_offline(no_coords, rotor_diameter_m=100.0) is None


class TestPropensity:
    def test_out_of_fold_over_selected_rows(self) -> None:
        # the feature shifts with the period, so the classifier separates them
        dump = _dump(n_base=600, n_up=400)
        rows = dump.rows.copy()
        rows.loc[rows["upgraded"], f"{_POWER}{_Q}R1"] += 3000.0
        rows.loc[5, "selected"] = False
        dump = _with_meta(rows, references=["R1", "R2"])
        m_hat = propensity(dump, model_params=_FAST)
        assert m_hat.index.equals(rows.index)
        assert np.isnan(m_hat.iloc[5])
        sel = rows["selected"]
        assert m_hat[sel].between(0, 1).all()
        assert m_hat[sel & rows["upgraded"]].mean() > m_hat[sel & rows["baseline"]].mean()

    def test_refuses_a_fixture_sized_campaign(self) -> None:
        with pytest.raises(ValueError, match="upgraded"):
            propensity(_dump(n_base=500, n_up=10), model_params=_FAST)
        with pytest.raises(ValueError, match="baseline"):
            propensity(_dump(n_base=20, n_up=500), model_params=_FAST)

    def test_ess_of_a_constant_propensity_is_the_whole_baseline(self) -> None:
        dump = _dump()
        m_hat = pd.Series(0.3, index=dump.rows.index)
        assert ess_fraction(dump, m_hat) == pytest.approx(1.0)

    def test_deciles_keep_the_near_certain_rows_apart(self) -> None:
        dump = _dump()
        m_hat = pd.Series(np.linspace(0.0, 1.0, len(dump.rows)), index=dump.rows.index)
        labels = label_propensity_decile(dump, m_hat)
        up = dump.rows["upgraded"]
        assert (labels[up & (m_hat > 0.95)] == "0.95-1.00").all()
        assert "0.95-1.00" not in set(labels[up & (m_hat <= 0.95)])


class TestAipw:
    @staticmethod
    def _shifted() -> tuple[RowDump, pd.Series, pd.Series]:
        """x in {0, 1}; the baseline sits mostly at 0, the campaign mostly at 1; the model is 50 kW low at 1."""
        n_base, n_up = 1000, 1000
        x = np.r_[np.arange(n_base) < 200, np.arange(n_up) < 800].astype(float)  # 20% vs 80% at x=1
        dump = _dump(n_base=n_base, n_up=n_up, bias_kw=50.0 * x[n_base:])
        rows = dump.rows.assign(x=x)
        dump = _with_meta(rows, references=["R1", "R2"])
        base = rows["baseline"].to_numpy()
        # the true propensity of each cell, and the model's baseline residual: 50 kW at x=1
        m_hat = pd.Series(np.where(x == 1, 800 / 1000, 200 / 1000), index=rows.index)
        g_oof = pd.Series(np.where(base, rows["actual_kw"] - 50.0 * x, np.nan), index=rows.index)
        return dump, m_hat, g_oof

    def test_recovers_a_planted_covariate_shift_bias(self) -> None:
        dump, m_hat, g_oof = self._shifted()
        result = aipw_level(dump, m_hat, g_oof=g_oof, g_in_sample=g_oof)
        assert result["observed_level_pp"] == pytest.approx(100 * dump.meta["uplift"])
        assert result["predicted_level_pp"] == pytest.approx(result["observed_level_pp"], rel=1e-9)
        assert result["observed_level_pp"] > 0
        assert result["trimmed_rows"] == 0

    def test_an_interpolating_in_sample_fit_controls_to_zero(self) -> None:
        dump, m_hat, g_oof = self._shifted()
        g_in = pd.Series(np.where(dump.rows["baseline"], dump.rows["actual_kw"], np.nan), index=dump.rows.index)
        result = aipw_level(dump, m_hat, g_oof=g_oof, g_in_sample=g_in)
        assert result["in_sample_control_pp"] == pytest.approx(0.0, abs=1e-12)

    def test_trims_near_certain_baseline_rows(self) -> None:
        dump, m_hat, g_oof = self._shifted()
        m_hat = m_hat.copy()
        m_hat.iloc[:7] = 0.995
        assert aipw_level(dump, m_hat, g_oof=g_oof, g_in_sample=g_oof)["trimmed_rows"] == 7

    def test_baseline_predictions_cover_every_baseline_row(self) -> None:
        dump = _dump(n_base=600, n_up=400)
        g_oof, g_in = baseline_predictions(dump, model_params=_FAST)
        base = dump.rows["selected"] & dump.rows["baseline"]
        for g in (g_oof, g_in):
            assert np.isfinite(g[base]).all()
            assert g[~base].isna().all()
            assert (g[base] >= 0).all()
            assert (g[base] <= max(2000.0, dump.rows.loc[base, "actual_kw"].max())).all()


class TestChannelDifference:
    def _pair(self) -> tuple[RowDump, RowDump]:
        rng = np.random.default_rng(3)
        a = _dump(references=("R1", "R2", "R3"), bias_kw=rng.normal(10.0, 5.0, 1000))
        rows_a = a.rows.copy()
        rows_b = a.rows.copy()
        held = np.zeros(len(rows_a), dtype=bool)
        held[np.flatnonzero(rows_a["upgraded"])[100:250]] = True
        # R3 is held back: its power is gone over the exclusion in both, its booleans only in b
        for rows in (rows_a, rows_b):
            rows.loc[held, f"{_POWER}{_Q}R3"] = np.nan
        rows_a[f"waking_{_POWER}{_Q}R3"] = 1.0
        rows_b.loc[held, "counterfactual_kw"] += 40.0
        return _with_meta(rows_a, references=["R1", "R2", "R3"]), _with_meta(rows_b, references=["R1", "R2", "R3"])

    def test_closes_on_the_difference_of_the_headlines(self) -> None:
        a, b = self._pair()
        table = channel_difference_by(a, b, label_month(a))
        expected = 100 * (a.meta["uplift"] - b.meta["uplift"])
        assert table["difference_pp"].sum() == pytest.approx(expected, abs=1e-9)

    def test_labels_the_held_back_rows(self) -> None:
        a, b = self._pair()
        labels = label_held_back(a, b)
        table = channel_difference_by(a, b, labels).set_index("group")
        expected = 100 * (a.meta["uplift"] - b.meta["uplift"])
        assert table.loc["held back", "difference_pp"] == pytest.approx(expected, abs=1e-9)
        assert table.loc["held back", "n_rows"] == 150
        assert table.loc["both", "difference_pp"] == pytest.approx(0.0, abs=1e-9)

    def test_refuses_dumps_with_different_rows(self) -> None:
        a, b = self._pair()
        rows = b.rows.copy()
        rows.loc[np.flatnonzero(rows["upgraded"])[0], "selected"] = False
        with pytest.raises(ValueError, match="upgraded selected rows"):
            channel_difference_by(a, _with_meta(rows, references=["R1", "R2", "R3"]), label_month(a))
