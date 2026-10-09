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
    _as_percent_of_power,
    _binned_stats,
    _condition_diagnostic_figure,
    _histogram_groups,
    _plot_residual_binned,
    _set_ylim_from_inliers,
    log_top_features,
    plot_conditional_diagnostics,
)
from benchmarking.diagnostics import stages


def _toy_per_bin() -> pd.DataFrame:
    """A per-bin conditional frame spanning ws, ti and power with a covered/imputed mix."""
    rows = []
    specs = {
        "ws": ["(4.0, 6.0]", "(6.0, 8.0]", "(8.0, 10.0]"],
        "ti": ["(0.05, 0.1]", "(0.1, 0.15]"],
        "power": ["(230.0, 690.0]", "(690.0, 1150.0]"],
    }
    for cond, bins in specs.items():
        for i, b in enumerate(bins):
            rows.append(
                {
                    "condition": cond,
                    "condition_bin": b,
                    "r_fwd": 0.05 + 0.01 * i,
                    "r_rev": -0.04 + 0.01 * i,
                    "implied_shrinkage": 0.98 + 0.01 * i,
                    "p50_uplift": 0.05 + 0.005 * i,
                    "covered": i % 2 == 0,  # alternate covered / imputed
                }
            )
    return pd.DataFrame(rows)


def test_plot_conditional_diagnostics_writes_a_figure_per_condition(tmp_path: Path) -> None:
    plot_conditional_diagnostics(tmp_path, _toy_per_bin(), test_wtg="T07")
    for cond in ("ws", "ti", "power"):
        assert (tmp_path / f"conditional_{cond}.png").exists()


def test_condition_diagnostic_figure_has_four_panels() -> None:
    sub = _toy_per_bin().query("condition == 'power'")
    fig = _condition_diagnostic_figure(sub, condition="power", test_wtg="T07")
    assert len(fig.axes) == 4
    plt.close(fig)


def test_condition_diagnostic_figure_shades_covered_and_imputed_distinctly() -> None:
    # the uplift panel must visually separate measured (covered) bins from imputed ones
    sub = _toy_per_bin().query("condition == 'ws'")  # covered = [True, False, True]
    fig = _condition_diagnostic_figure(sub, condition="ws", test_wtg="T07")
    uplift_ax = fig.axes[2]  # panels: fwd, rev, uplift, shrinkage
    facecolors = {tuple(patch.get_facecolor()) for patch in uplift_ax.patches}
    assert len(facecolors) >= 2  # covered and imputed bins are not the same colour
    plt.close(fig)


if TYPE_CHECKING:
    from pathlib import Path


def test_binned_stats_mean_sd_and_count() -> None:
    x = np.array([0.5, 1.5, 1.6, 1.7, 10.0])  # last value falls outside the edges
    y = np.array([1.0, 10.0, 12.0, 14.0, 999.0])
    edges = np.array([0.0, 1.0, 2.0])
    centers, mean, sd, count = _binned_stats(x, y, edges)
    assert list(centers) == [0.5, 1.5]
    # bin [0,1): a single point -> below _MIN_BIN_COUNT, so NaN mean/SD but count recorded
    assert count[0] == 1
    assert np.isnan(mean[0])
    assert np.isnan(sd[0])
    # bin [1,2): three points {10,12,14} -> mean 12, sample SD 2
    assert count[1] == 3
    assert mean[1] == 12.0
    assert sd[1] == 2.0


def test_binned_stats_all_nan_input_is_safe() -> None:
    edges = np.array([0.0, 1.0, 2.0])
    centers, mean, sd, count = _binned_stats(np.full(3, np.nan), np.arange(3.0), edges)
    assert len(centers) == 2
    assert np.isnan(mean).all()
    assert np.isnan(sd).all()
    assert (count == 0).all()


def _diag_data(*, with_conditions: bool) -> DiagnosticData:
    """A minimal DiagnosticData carrying only what the residual-binned plot reads."""
    rng = np.random.default_rng(0)
    n = 400
    y_base = rng.uniform(0, 2000, n)
    pred_base = 0.7 * y_base + 300  # deliberate shrinkage: slope < 1
    y_up = rng.uniform(0, 2000, n)
    pred_up = 0.7 * y_up + 300
    cond_up = cond_base = None
    if with_conditions:
        cond_base = pd.DataFrame({"ws": rng.uniform(0, 25, n), "ti": rng.uniform(0, 0.4, n)})
        cond_up = pd.DataFrame({"ws": rng.uniform(0, 25, n), "ti": rng.uniform(0, 0.4, n)})
    return DiagnosticData(
        test_wtg="T07",
        mode="prepost",
        index=pd.DatetimeIndex([]),
        treated_all=np.array([]),
        selected_all=np.array([]),
        y_all=np.array([]),
        timebase=pd.Timedelta(minutes=10),
        upgraded_ts=pd.DatetimeIndex([]),
        y_upgraded=y_up,
        pred_upgraded=pred_up,
        y_baseline_valid=y_base,
        pred_baseline_valid=pred_base,
        feature_names=[],
        feature_values=pd.DataFrame(),
        y_selected=np.array([]),
        outcome_model=None,
        overall_uplift=0.0,
        sum_actual_kw=0.0,
        sum_counterfactual_kw=0.0,
        n_refs=3,
        era5_lag_rows=None,
        era5_corr=None,
        era5_sweep=None,
        cond_upgraded=cond_up,
        cond_baseline_valid=cond_base,
    )


def test_as_percent_of_power_divides_per_bin_and_drops_nonpositive() -> None:
    out = _as_percent_of_power(np.array([10.0, 5.0, -3.0]), np.array([100.0, 0.0, 60.0]))
    assert out[0] == 10.0  # 10 kW of 100 kW
    assert np.isnan(out[1])  # mean power 0 -> dropped
    assert out[2] == -5.0  # -3 kW of 60 kW


def test_set_ylim_from_inliers_ignores_out_of_range_points() -> None:
    _, ax = plt.subplots()
    # inliers within +/-30 are {-10, 20}; the -330 outlier must not stretch the limits
    _set_ylim_from_inliers(ax, [np.array([-10.0, 20.0, -330.0, np.nan])])
    lo, hi = ax.get_ylim()
    assert lo < -10.0  # a small margin below the min inlier
    assert lo > -20.0  # but nowhere near the -330 outlier
    assert 20.0 < hi < 30.0
    plt.close()


def test_set_ylim_from_inliers_noop_when_no_inliers() -> None:
    _, ax = plt.subplots()
    before = ax.get_ylim()
    _set_ylim_from_inliers(ax, [np.array([100.0, -330.0])])  # all outside +/-30
    assert ax.get_ylim() == before
    plt.close()


def test_plot_residual_binned_writes_both_png_with_conditions(tmp_path: Path) -> None:
    model_dir = tmp_path / stages.UPLIFT_MODELLING
    model_dir.mkdir()
    _plot_residual_binned(model_dir, _diag_data(with_conditions=True))
    assert (model_dir / "residual_binned.png").exists()
    assert (model_dir / "residual_binned_pct.png").exists()


def test_plot_residual_binned_writes_png_without_conditions(tmp_path: Path) -> None:
    # No ws/TI columns configured: the plot still renders the power-axis panels.
    model_dir = tmp_path / stages.UPLIFT_MODELLING
    model_dir.mkdir()
    _plot_residual_binned(model_dir, _diag_data(with_conditions=False))
    assert (model_dir / "residual_binned.png").exists()
    assert (model_dir / "residual_binned_pct.png").exists()


class TestFeatureHistogramFolders:
    """One signal per folder, so a farm's worth of nacelle positions does not bury the singletons."""

    FEATURES: ClassVar[list[str]] = [
        "wtc_ActPower_mean @ R1",
        "wtc_ActPower_mean @ R2",
        "northed_wtc_NacelPos_mean_sin @ R1",
        "northed_wtc_NacelPos_mean_cos @ R1",
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

    def test_a_reanalysis_field_groups_with_its_companions(self, tmp_path: Path) -> None:
        placed = self._placed(tmp_path)
        assert placed["wind_direction_100m"] == placed["wind_direction_100m_sin"] == "wind_direction_100m"


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
