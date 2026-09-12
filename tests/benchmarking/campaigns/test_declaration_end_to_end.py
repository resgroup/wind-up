"""A declared campaign run end to end: YAML in, a truth-free report out.

The whole Phase-2 path in one test -- loader, the composed ``wind-up``, the truth-free core and
the analyst report -- on a small synthetic farm, with reanalysis supplied rather than fetched so
the test needs no network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.composed import RESOLVED_FILENAME, WIND_UP, run_declaration
from benchmarking.synthetic import HOT_COLUMNS

# Runs the real power model on a year of data.
pytestmark = pytest.mark.slow

if TYPE_CHECKING:
    from pathlib import Path

TURBINES = ("T01", "T02", "T03", "T04", "T05")
PERIOD = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2019-01-01", tz="UTC"))
CHANGEOVER = pd.Timestamp("2018-07-01", tz="UTC")
RATED_KW = 2300.0
# The uplift injected into T01 after the changeover, and how close the estimate must land.
INJECTED_UPLIFT = 0.05
TOLERANCE = 0.02


def _scada(*, mode: str) -> pd.DataFrame:
    """A five-turbine hourly farm with a shared wind signal, per-turbine noise, and one upgrade."""
    rng = np.random.default_rng(0)
    index = pd.date_range(*PERIOD, freq="1h", tz="UTC", inclusive="left")
    hours = np.arange(len(index), dtype=float)
    shared_ws = 8.5 + 2.5 * np.sin(hours / (24 * 30)) + 1.5 * np.sin(hours / 12)
    treated = index >= CHANGEOVER if mode == "prepost" else (np.floor(hours / 4) % 2 == 1) & (index >= CHANGEOVER)
    frames = []
    for turbine in TURBINES:
        ws = shared_ws + rng.normal(0.0, 0.3, len(index))
        power = np.clip(RATED_KW * (ws / 13.0) ** 3, 0.0, RATED_KW)
        if turbine == "T01":
            power = power * np.where(treated, 1.0 + INJECTED_UPLIFT, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: turbine,
                    HOT_COLUMNS.active_power: power,
                    HOT_COLUMNS.active_power_min: power * 0.95,
                    HOT_COLUMNS.wind_speed: ws,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.availability: 3600.0,
                    HOT_COLUMNS.nacelle_position: (210.0 + 20.0 * np.sin(hours / 50)) % 360.0,
                },
                index=index,
            )
        )
    return pd.concat(frames)


def _era5() -> pd.DataFrame:
    """Hourly reanalysis spanning the campaign, with the direction the northing step anchors on."""
    index = pd.date_range(*PERIOD, freq="1h", tz="UTC", inclusive="left")
    hours = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "wind_speed_100m": 9.0 + 2.0 * np.sin(hours / (24 * 30)),
            "wind_direction_100m": (210.0 + 20.0 * np.sin(hours / 50)) % 360.0,
            "temperature_2m": 8.0 + 5.0 * np.sin(hours / (24 * 90)),
        },
        index=index,
    )


TIMING = {
    "prepost": "  mode: prepost\n  changeover: 2018-07-01T00:00:00Z",
    "toggle": "  mode: toggle\n  start: 2018-07-01T00:00:00Z\n  period: 8h",
}

TEMPLATE = """\
name: e2e_{mode}

data:
  scada: scada.parquet
  schema: hill_of_towie
  turbines: turbines.csv

turbines:
  upgraded:   [T01]
  references: [T02, T03, T04, T05]
  excluded:   []
  rated_power_kw: {rated_kw}

timing:
{timing}

analysis_period:
  start: 2018-01-01T00:00:00Z
  end:   2019-01-01T00:00:00Z

northing:
  discover: false
"""


def _write_campaign(tmp_path: Path, *, mode: str) -> Path:
    """Write a complete campaign folder -- declaration, turbines sidecar and SCADA -- and return the YAML."""
    _scada(mode=mode).to_parquet(tmp_path / "scada.parquet")
    pd.DataFrame({"Name": list(TURBINES), "Latitude": [57.5 + i * 0.01 for i in range(5)], "Longitude": -3.25}).to_csv(
        tmp_path / "turbines.csv", index=False
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(TEMPLATE.format(mode=mode, rated_kw=RATED_KW, timing=TIMING[mode]))
    return path


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_a_declared_campaign_runs_and_writes_its_report(tmp_path: Path, mode: str) -> None:
    out_dir = tmp_path / "out"
    report = run_declaration(_write_campaign(tmp_path, mode=mode), out_dir=out_dir, era5_hourly_df=_era5())

    assert list(report.per_turbine["method"]) == [WIND_UP]
    assert list(report.per_turbine["test_wtg"]) == ["T01"]
    for name in ("per_turbine.csv", "farm_uplift.csv", "farm_uplift_detail.csv", "reference_stability.csv"):
        assert (out_dir / name).exists(), name
    assert (out_dir / RESOLVED_FILENAME).exists()


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_the_report_of_a_declared_campaign_carries_no_truth(tmp_path: Path, mode: str) -> None:
    # the isolation guarantee, on the real product path: there is no truth here to leak
    out_dir = tmp_path / "out"
    run_declaration(_write_campaign(tmp_path, mode=mode), out_dir=out_dir, era5_hourly_df=_era5())
    written = sorted(out_dir.glob("*.csv"))
    assert written
    for path in written:
        assert not ({"truth", "signed_error"} & set(pd.read_csv(path).columns)), path.name


def test_the_declared_uplift_is_recovered(tmp_path: Path) -> None:
    # not a scoring test -- it shows the declared path reaches a real estimate, not just a file
    out_dir = tmp_path / "out"
    report = run_declaration(_write_campaign(tmp_path, mode="prepost"), out_dir=out_dir, era5_hourly_df=_era5())
    assert float(report.farm["estimate"].iloc[0]) == pytest.approx(INJECTED_UPLIFT, abs=TOLERANCE)


def test_the_references_read_near_zero(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    report = run_declaration(_write_campaign(tmp_path, mode="prepost"), out_dir=out_dir, era5_hourly_df=_era5())
    stability = report.reference_stability
    assert set(stability["turbine"]) == {"T02", "T03", "T04", "T05"}
    assert stability["uplift"].abs().max() < TOLERANCE, stability.to_string(index=False)
