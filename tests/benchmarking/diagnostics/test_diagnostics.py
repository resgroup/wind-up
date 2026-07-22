"""Smoke tests for the shared benchmarking diagnostics package.

These assert the plotting/config functions run clean (the suite treats warnings as errors) and
write the expected files on a tiny synthetic frame, and that plots needing an absent signal skip
gracefully. They do not assert pixel content — image fidelity is reviewed by eye.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from benchmarking.diagnostics import write_common_diagnostics, write_run_config
from benchmarking.diagnostics.context import ERA5_WD_COL, ERA5_WS_COL, DiagnosticContext
from benchmarking.diagnostics.coverage import exclusion_bucket, plot_excluded_fraction
from benchmarking.diagnostics.density import density_scatter
from benchmarking.synthetic import ColumnSchema

if TYPE_CHECKING:
    from pathlib import Path

_FULL_COLUMNS = ColumnSchema(
    turbine="turbine",
    active_power="power",
    wind_speed="ws",
    wind_speed_sd="ws_sd",
    gen_rpm="rpm",
    pitch="pitch",
    reactive_power="reactive",
    nacelle_position="nacelle",
    ambient_temp="temp",
    availability="avail",
)


def _long_scada(index: pd.DatetimeIndex, turbines: list[str], *, rng: np.random.Generator) -> pd.DataFrame:
    """A small long-format SCADA frame with every diagnostic signal, indexed by timestamp."""
    frames = []
    for turbine in turbines:
        ws = rng.uniform(3, 18, len(index))
        power = np.clip(0.5 * ws**3, 0, 2300) + rng.normal(0, 20, len(index))
        frames.append(
            pd.DataFrame(
                {
                    "turbine": turbine,
                    "power": power,
                    "ws": ws,
                    "ws_sd": rng.uniform(0.3, 2.0, len(index)),
                    "rpm": np.clip(ws * 90, 0, 1600),
                    "pitch": rng.uniform(-2, 25, len(index)),
                    "reactive": rng.normal(0, 50, len(index)),
                    "nacelle": rng.uniform(0, 360, len(index)),
                    "temp": rng.uniform(-5, 25, len(index)),
                    "avail": 600.0,
                },
                index=index,
            )
        )
    return pd.concat(frames)


def _context(
    tmp_path: Path,
    *,
    columns: ColumnSchema = _FULL_COLUMNS,
    with_era5: bool = True,
    excluded: np.ndarray | None = None,
) -> DiagnosticContext:
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=300, freq="10min", tz="UTC")
    scada = _long_scada(index, ["T1", "T2", "T3"], rng=rng)
    treated = np.asarray(index >= index[len(index) // 2])
    used = rng.random(len(index)) > 0.1
    era5 = None
    if with_era5:
        era5 = pd.DataFrame(
            {ERA5_WS_COL: rng.uniform(3, 18, len(index)), ERA5_WD_COL: rng.uniform(0, 360, len(index))}, index=index
        )
    return DiagnosticContext(
        run_dir=tmp_path / "run",
        test_wtg="T1",
        turbine_col="turbine",
        columns=columns,
        scada_df=scada,
        treated_ts=treated,
        used_ts=used,
        timebase=pd.Timedelta(minutes=10),
        mode="prepost",
        era5_df=era5,
        excluded_ts=excluded,
    )


# --- caller-flagged row exclusions --------------------------------------------------------------


def _excluded_mask(n: int = 300, *, every: int = 5) -> np.ndarray:
    return np.arange(n) % every == 0


class TestExclusionTimelineBucket:
    """A single averaged dot is not a timeline: the bucket has to suit the campaign's length."""

    def test_short_campaign_buckets_daily(self) -> None:
        index = pd.date_range("2020-01-01", periods=300, freq="10min", tz="UTC")
        assert exclusion_bucket(index) == "1D"

    def test_long_campaign_buckets_weekly(self) -> None:
        index = pd.date_range("2020-01-01", periods=3 * 365 * 24, freq="h", tz="UTC")
        assert exclusion_bucket(index) == "7D"

    def test_a_short_campaign_gets_more_than_one_point(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, excluded=_excluded_mask())  # the fixture spans ~2 days
        line = plot_excluded_fraction(ctx)
        assert line is not None
        assert len(pd.Series(_excluded_mask(), index=ctx.index).resample(exclusion_bucket(ctx.index)).mean()) > 1


class TestExcludedRowPlots:
    """The exclusion view is the *usual* 2x3 operating-curve figure, coloured kept vs excluded."""

    def test_written_when_rows_are_excluded(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, excluded=_excluded_mask())
        names = {p.name for p in write_common_diagnostics(ctx)}
        assert "ops_curves_excluded.png" in names
        assert "excluded_row_fraction.png" in names

    def test_lands_in_the_filter_stage_folder(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, excluded=_excluded_mask())
        written = {p.name: p for p in write_common_diagnostics(ctx)}
        assert written["ops_curves_excluded.png"].parent.name == written["filter_coverage.png"].parent.name

    def test_skipped_when_the_method_excludes_nothing(self, tmp_path: Path) -> None:
        """A clean campaign must not sprout an empty plot."""
        ctx = _context(tmp_path, excluded=np.zeros(300, dtype=bool))
        names = {p.name for p in write_common_diagnostics(ctx)}
        assert "ops_curves_excluded.png" not in names
        assert "excluded_row_fraction.png" not in names

    def test_skipped_when_the_method_has_no_exclusion_concept(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path)
        assert ctx.excluded_ts is None
        names = {p.name for p in write_common_diagnostics(ctx)}
        assert "ops_curves_excluded.png" not in names

    def test_a_misaligned_exclusion_mask_skips_every_diagnostic(self, tmp_path: Path) -> None:
        """Same contract as the other masks: a length mismatch is a caller bug, not a partial plot."""
        ctx = _context(tmp_path, excluded=np.zeros(7, dtype=bool))
        assert write_common_diagnostics(ctx) == []


def test_common_diagnostics_writes_expected_plots(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    written = write_common_diagnostics(ctx)
    names = {p.name for p in written}
    expected = {
        "input_data_timeline.png",
        "input_data_coverage.png",
        "filter_coverage.png",
        "condition_histograms.png",
        "ops_curves.png",
        "ops_curves_kept_only.png",
        "ops_curves_by_upgrade.png",
        "reactive_vs_active.png",
        "power_factor.png",
        "northing_error.png",
    }
    assert expected <= names
    assert all(p.exists() for p in written)


def test_optional_signals_skip_gracefully(tmp_path: Path) -> None:
    bare = ColumnSchema(
        turbine="turbine",
        active_power="power",
        wind_speed="ws",
        wind_speed_sd="ws_sd",
        gen_rpm="rpm",
        availability="avail",
    )
    ctx = _context(tmp_path, columns=bare, with_era5=False)
    written = write_common_diagnostics(ctx)
    names = {p.name for p in written}
    # reactive needs a reactive tag; northing needs nacelle + ERA5 — both absent here.
    assert "reactive_vs_active.png" not in names
    assert "northing_error.png" not in names
    # the core curves still render.
    assert "ops_curves.png" in names


def test_write_run_config_yaml(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    path = write_run_config(ctx, method_name="unit_test", method_params={"foo": 1}, extra={"era5_lag_rows": 3})
    assert path.exists()
    record = yaml.safe_load(path.read_text())
    assert record["method"] == "unit_test"
    assert record["mode"] == "prepost"
    assert record["test_wtg"] == "T1"
    assert record["references"] == ["T2", "T3"]
    assert record["extra"]["era5_lag_rows"] == 3


def test_density_scatter_degenerate_input_does_not_raise() -> None:
    _fig, ax = plt.subplots()
    # all-identical x (no spread) must fall back to a flat colour rather than erroring.
    density_scatter(np.ones(20), np.arange(20.0), ax=ax)
    plt.close("all")


@pytest.mark.parametrize("with_era5", [True, False])
def test_runs_without_era5(tmp_path: Path, *, with_era5: bool) -> None:
    ctx = _context(tmp_path, with_era5=with_era5)
    written = write_common_diagnostics(ctx)
    assert ("northing_error.png" in {p.name for p in written}) == with_era5
