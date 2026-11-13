"""Tests for the power-model residual diagnostics (the shrinkage-check plot)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest  # noqa: TC002 - caplog fixtures are runtime types

from benchmarking.baselines.power_model.diagnostics import (
    DiagnosticData,
    _energy_timeseries_figure,
    _feature_overview_figure,
    _histogram_figure,
    _histogram_groups,
    _importance_figure,
    _robust_bins,
    log_top_features,
    plot_uplift_shape,
)
from benchmarking.baselines.power_model.shape import ShapeResult
from benchmarking.diagnostics.features import category_shares_figure, site_feature_figure, write_feature_outputs
from benchmarking.harness.waking import WAKING_SCENARIO_COL


def test_plot_uplift_shape_draws_each_axis_each_subsets_family_and_the_drift(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "condition": ["predicted_power"] * 2 + ["direction"] * 2,
            "condition_bin": ["(0.0, 500.0]", "(500.0, 1000.0]", "(0.0, 30.0]", "(30.0, 60.0]"],
            "p50_uplift": [0.03, 0.01, 0.02, 0.02],
            "sigma_uplift": [0.01, 0.01, 0.01, np.nan],
            "hours": [100.0, 80.0, 150.0, 5.0],
            "thin": [False, False, False, True],
        }
    )
    subsets = pd.DataFrame(
        {
            "axis": "waked",
            "subset": ["(0.5, 1.5]"] * 2,
            "n_bins": 2,
            "power_bin": ["(0.0, 500.0]", "(500.0, 1000.0]"],
            "p50_uplift": [0.05, 0.0],
            "sigma_sampling": [0.01, 0.01],
            "overall_uplift": [0.03, 0.01],
            "overall_sigma": [0.01, 0.01],
            "z": [1.4, -0.7],
            "hours": [50.0, 40.0],
            "too_thin": False,
            "flagged": False,
        }
    )
    drift = pd.DataFrame(
        {
            "reference": ["R1", "R1"],
            "month": ["2020-01", "2020-02"],
            "ratio": [1.0, 1.05],
            "spread": 0.05,
            "flagged": True,
        }
    )
    shape = ShapeResult(frame=frame, subsets=subsets, labels=pd.DataFrame())
    plot_uplift_shape(tmp_path, shape=shape, drift=drift, test_wtg="T07")
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {
        "uplift_predicted_power.png",
        "uplift_direction.png",
        "uplift_subsets_waked.png",
        "reference_wind_speed_drift.png",
    }


if TYPE_CHECKING:
    from pathlib import Path


class TestFeatureHistogramFolders:
    """A signal carried by several turbines gets one folder; everything else stays at the top level."""

    FEATURES: ClassVar[list[str]] = [
        "wtc_ActPower_mean @ R1",
        "wtc_ActPower_mean @ R2",
        "northed_wtc_NacelPos_mean_sin @ R1",
        "northed_wtc_NacelPos_mean_cos @ R1",
        "northed_wtc_NacelPos_mean_sin @ R2",
        "northed_wtc_NacelPos_mean_cos @ R2",
        "cloud_cover",
        "wind_direction_100m_sin",
        "wind_direction_100m_cos",
        "wind_direction_100m",
    ]

    def _placed(self, root: Path) -> dict[str, str]:
        groups = _histogram_groups(self.FEATURES)
        return {f: groups[f](root).name for f in self.FEATURES}

    def test_each_turbine_s_copy_of_a_signal_shares_one_folder(self, tmp_path: Path) -> None:
        placed = self._placed(tmp_path)
        assert placed["wtc_ActPower_mean @ R1"] == placed["wtc_ActPower_mean @ R2"] == "wtc_ActPower_mean"

    def test_a_sine_and_cosine_pair_is_one_signal(self, tmp_path: Path) -> None:
        placed = self._placed(tmp_path)
        assert placed["northed_wtc_NacelPos_mean_sin @ R1"] == "northed_wtc_NacelPos_mean"
        assert placed["northed_wtc_NacelPos_mean_cos @ R1"] == "northed_wtc_NacelPos_mean"

    def test_a_lone_plot_stays_at_the_top_level(self, tmp_path: Path) -> None:
        assert self._placed(tmp_path)["cloud_cover"] == tmp_path.name

    def test_a_reanalysis_field_and_its_companions_stay_at_the_top_level(self, tmp_path: Path) -> None:
        placed = self._placed(tmp_path)
        assert placed["wind_direction_100m"] == placed["wind_direction_100m_sin"] == tmp_path.name


class TestFeatureDistributionFigures:
    def test_a_site_feature_histogram_names_its_units(self) -> None:
        vals = np.linspace(-10.0, 50.0, 50)
        fig = _histogram_figure(vals, treated=np.arange(50) >= 25, label="solar_elevation", test_wtg="T1")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "solar_elevation [°]"
        assert "solar_elevation" in ax.get_title()
        plt.close(fig)

    def test_a_categorical_feature_is_labelled_by_its_category_names(self) -> None:
        values = pd.Series(["normal"] * 6 + ["T03 not waking"] * 3 + [None], dtype="category")
        fig = category_shares_figure(values, segments={"baseline": np.arange(10) < 5}, title="T1")
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["normal", "T03 not waking", "(missing)"]
        plt.close(fig)

    def test_rare_categories_are_pooled(self) -> None:
        values = pd.Series([f"c{i:02d}" for i in range(30)] * 2, dtype="category")
        fig = category_shares_figure(values, segments={"baseline": np.arange(60) < 30}, title="T1")
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert len(labels) == 20
        assert labels[-1] == "11 other categories"
        plt.close(fig)


class TestWhatTheFeatureLogSays:
    """A line at INFO, the table at DEBUG, and a warning when the model leant on something odd."""

    def _importance(self, leader: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": [leader, "wtc_ActPower_mean @ R2", "wind_speed_100m @ ERA5"], "gain": [900.0, 50.0, 10.0]}
        )

    def test_a_neighbours_power_on_top_says_nothing_alarming(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            log_top_features(self._importance("wtc_ActPower_mean @ R1"), active_power_col="wtc_ActPower_mean")
        assert "leant on" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_the_gain_numbers_are_debug_not_info(self, caplog: pytest.LogCaptureFixture) -> None:
        # the dump is for someone who went looking; the INFO line just names what led
        with caplog.at_level(logging.INFO):
            log_top_features(self._importance("wtc_ActPower_mean @ R1"), active_power_col="wtc_ActPower_mean")
        assert "900" not in caplog.text

    def test_a_reanalysis_column_on_top_is_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            log_top_features(self._importance("wind_speed_100m @ ERA5"), active_power_col="wtc_ActPower_mean")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings
        assert "wind_speed_100m @ ERA5" in warnings[0].getMessage()

    def test_a_neighbours_other_channel_on_top_is_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        # its direction is not its power: the expected leader is the same weather, measured
        with caplog.at_level(logging.INFO):
            log_top_features(
                self._importance("northed_wtc_NacelPos_mean_sin @ R1"), active_power_col="wtc_ActPower_mean"
            )
        assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_step_6_outputs_cover_every_site_feature_and_turbine(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-01", periods=200, freq="6h", tz="UTC")
    scada = pd.concat(
        pd.DataFrame(
            {"turbine": t, WAKING_SCENARIO_COL: np.where(np.arange(200) % 7 == 0, "R2 not waking", "normal")}, index=idx
        )
        for t in ("T1", "T2")
    )
    site = pd.DataFrame({"solar_elevation": np.linspace(-10, 50, 200), "other": 1.5}, index=idx)
    write_feature_outputs(scada, site_features=site, turbine_col="turbine", out_dir=tmp_path)
    assert sorted(p.name for p in (tmp_path / "site").iterdir()) == ["other.png", "solar_elevation.png"]
    assert sorted(p.name for p in (tmp_path / "waking_scenarios").iterdir()) == [
        "waking_scenario_T1.png",
        "waking_scenario_T2.png",
    ]
    table = pd.read_csv(tmp_path / "site_features.csv")
    assert list(table["feature"]) == ["solar_elevation", "other"]
    assert list(table["unit"].fillna("")) == ["°", ""]


def test_a_site_feature_histogram_shows_every_value_and_piles_nothing_into_its_end_bins() -> None:
    idx = pd.date_range("2020-01-01", periods=10_000, freq="1h", tz="UTC")
    values = pd.Series(np.random.default_rng(0).normal(0.2, 0.05, len(idx)), index=idx)
    values.iloc[:5] = 1.0
    fig = site_feature_figure(values, label="solar_elevation")
    hist_ax = fig.axes[0]
    low, high = hist_ax.get_xlim()
    assert low < values.min()
    assert high > 1.0
    heights = np.array([patch.get_height() for patch in hist_ax.patches])
    assert 0 < heights[-1] < heights.max() / 100
    assert "max 1" in fig.texts[0].get_text()
    plt.close(fig)


def _upgraded_data(days: int = 70) -> DiagnosticData:
    """DiagnosticData with ``days`` of 10-minute upgraded records, actual 2% above the counterfactual."""
    index = pd.date_range("2021-01-04", periods=days * 144, freq="10min", tz="UTC")
    pred = np.full(len(index), 1000.0)
    return DiagnosticData(
        test_wtg="T07",
        mode="prepost",
        index=index,
        treated_all=np.ones(len(index), dtype=bool),
        selected_all=np.ones(len(index), dtype=bool),
        y_all=1.02 * pred,
        timebase=pd.Timedelta(minutes=10),
        upgraded_ts=index,
        y_upgraded=1.02 * pred,
        pred_upgraded=pred,
        y_baseline_valid=np.array([]),
        pred_baseline_valid=np.array([]),
        feature_names=[],
        feature_values=pd.DataFrame(),
        y_selected=np.array([]),
        outcome_model=None,
        overall_uplift=0.02,
        sum_actual_kw=0.0,
        sum_counterfactual_kw=0.0,
        n_refs=3,
    )


class TestTheEnergyTimeseries:
    def test_it_is_weekly_with_the_weekly_uplift_below(self) -> None:
        fig = _energy_timeseries_figure(_upgraded_data(days=70))
        top, bottom = fig.axes
        assert len(top.lines[0].get_xdata()) == 10
        assert "weekly" in top.get_ylabel()
        assert np.allclose(np.asarray(bottom.patches[0].get_height()), 2.0)
        plt.close(fig)


class TestTheImportancePlots:
    IMPORTANCE = pd.DataFrame({"feature": ["a @ T1", "b @ T2", "c @ ERA5"], "gain": [1e11, 1e8, 1e3]})

    def test_feature_importance_is_on_a_log_scale_with_a_plain_title(self) -> None:
        fig = _importance_figure(self.IMPORTANCE, test_wtg="T13")
        assert fig.axes[0].get_xscale() == "log"
        assert "expect" not in fig.axes[0].get_title()
        plt.close(fig)

    def test_features_left_out_of_the_model_are_named(self) -> None:
        fig = _importance_figure(self.IMPORTANCE, test_wtg="T13", left_out=["waking_scenario @ T04"])
        notes = [t.get_text() for t in fig.axes[0].texts]
        assert notes == ["left out, one value in every record used:\nwaking_scenario @ T04"]
        assert "left out" not in fig.axes[0].get_title()
        plt.close(fig)

    def test_the_feature_overview_is_on_a_log_scale(self) -> None:
        catalogue = self.IMPORTANCE.assign(coverage_pct=100.0)
        fig = _feature_overview_figure(catalogue)
        assert fig.axes[0].get_xscale() == "log"
        plt.close(fig)


def test_histogram_bins_span_every_value() -> None:
    values = np.array([*np.linspace(0.0, 1.0, 1000), 50.0])
    edges = _robust_bins(values)
    assert edges[0] <= 0.0
    assert edges[-1] >= 50.0
