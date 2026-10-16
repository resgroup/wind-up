"""A declared campaign run end to end: YAML in, a truth-free report out.

The whole Phase-2 path in one test -- loader, the composed ``wind-up``, the truth-free core and
the analyst report -- on a small synthetic farm. The wind is the real Hill of Towie ERA5 record
(tracked under ``tests/test_data/hot``, at the farm's own grid cell), supplied rather than fetched so
the test needs no network; each turbine's SCADA is derived from it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.composed import RESOLVED_FILENAME, WIND_UP, run_declaration
from benchmarking.synthetic import HOT_COLUMNS

# Runs the real power model on a year of data.
pytestmark = pytest.mark.slow

TURBINES = ("T01", "T02", "T03", "T04", "T05")
PERIOD = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2019-01-01", tz="UTC"))
ERA5_PATH = (
    Path(__file__).resolve().parents[2]
    / "test_data"
    / "hot"
    / "reanalysis_data"
    / "ERA5T_57.50N_-3.25E_100m_1hr_20241231.parquet"
)
ERA5_COLUMNS = {
    "100_m_hws_mean_mps": "wind_speed_100m",
    "100_m_hwd_mean_deg-n_true": "wind_direction_100m",
    "10_m_hws_mean_mps": "wind_speed_10m",
    "2_m_temp_mean_deg-c": "temperature_2m",
    "2_m_rh_mean_%": "relative_humidity_2m",
    "0_m_pres_mean_pa": "surface_pressure",
}
PA_PER_HPA = 100.0
# The record has no gust field; a fixed gust factor on the 10 m wind stands in for it.
GUST_FACTOR = 1.4
CHANGEOVER = pd.Timestamp("2018-07-01", tz="UTC")
RATED_KW = 2300.0
# The uplift injected into T01 after the changeover, and how close the estimate must land.
INJECTED_UPLIFT = 0.05
TOLERANCE = 0.02


def _era5() -> pd.DataFrame:
    """The real Hill of Towie hourly reanalysis over the campaign, one hour past its end so every period is covered."""
    era5 = pd.read_parquet(ERA5_PATH, columns=list(ERA5_COLUMNS)).rename(columns=ERA5_COLUMNS)
    era5 = era5.loc[PERIOD[0] : PERIOD[1]].rename_axis(None)
    era5["surface_pressure"] = era5["surface_pressure"] / PA_PER_HPA
    era5["wind_gusts_10m"] = era5["wind_speed_10m"] * GUST_FACTOR
    return era5


def _scada(*, mode: str) -> pd.DataFrame:
    """A five-turbine hourly farm blown by the real ERA5 wind, with per-turbine noise and one upgrade."""
    rng = np.random.default_rng(0)
    era5 = _era5()
    index = pd.date_range(*PERIOD, freq="1h", tz="UTC", inclusive="left")
    hours = np.arange(len(index), dtype=float)
    # Each hourly SCADA record is the mean over its period, so it averages the hour's two end points.
    hourly = era5["wind_speed_100m"].to_numpy(dtype=float)
    shared_ws = (hourly[:-1] + hourly[1:]) / 2
    shared_wd = era5["wind_direction_100m"].to_numpy(dtype=float)[:-1]
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
                    HOT_COLUMNS.nacelle_position: (shared_wd + rng.normal(0.0, 3.0, len(index))) % 360.0,
                },
                index=index,
            )
        )
    return pd.concat(frames)


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
    """Write a complete campaign folder -- declaration, turbines sidecar and SCADA -- and return the YAML.

    The folder is ``tmp_path/campaign``, so the step 1 plots written beside it stay inside ``tmp_path``.
    """
    folder = tmp_path / "campaign"
    folder.mkdir(exist_ok=True)
    _scada(mode=mode).to_parquet(folder / "scada.parquet")
    pd.DataFrame(
        {
            "Name": list(TURBINES),
            "Latitude": [57.5 + i * 0.002 for i in range(5)],
            "Longitude": -3.25,
            "rotor_diameter_m": 82.0,
        }
    ).to_csv(folder / "turbines.csv", index=False)
    path = folder / "campaign.yaml"
    path.write_text(TEMPLATE.format(mode=mode, rated_kw=RATED_KW, timing=TIMING[mode]))
    return path


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_a_declared_campaign_runs_and_writes_its_report(tmp_path: Path, mode: str) -> None:
    out_dir = tmp_path / "out"
    report = run_declaration(_write_campaign(tmp_path, mode=mode), out_dir=out_dir, era5_hourly_df=_era5())

    assert list(report.per_turbine["method"]) == [WIND_UP]
    assert list(report.per_turbine["test_wtg"]) == ["T01"]
    campaign = out_dir / "B_campaign"
    assert (campaign / "per_turbine.csv").exists()
    assert (campaign / "07_periods" / "periods.csv").exists()
    assert (campaign / "07_periods" / "analysis_timeline_T01.png").exists()
    assert (campaign / "08_candidates" / "candidates.csv").exists()
    assert (campaign / "09_screen" / "reference_stability.csv").exists()
    assert (campaign / "15_uplift_distributions" / "conditional.csv").exists()
    for name in ("farm_uplift.csv", "farm_uplift_detail.csv"):
        assert (out_dir / "D_farm" / name).exists(), name
    assert (out_dir / RESOLVED_FILENAME).exists()
    # one folder per measured turbine, with no method or run level under it
    turbine = out_dir / "C_turbine_measurement" / "T01"
    assert (turbine / "results.csv").exists()
    assert (turbine / "plots" / "01_changes" / "ops_relationships_T01.png").exists()
    assert (turbine / "plots" / "02_operating_states" / "operating_states_T01.png").exists()
    # steps 1, 2 and 4 for every turbine
    preparation = out_dir / "A_data_preparation"
    assert (preparation / "01_changes" / "ops_relationships_T05.png").exists()
    assert (preparation / "02_operating_states" / "operating_states_T05.png").exists()
    assert (preparation / "02_operating_states" / "operating_state_hours.csv").exists()


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_the_report_of_a_declared_campaign_carries_no_truth(tmp_path: Path, mode: str) -> None:
    # the isolation guarantee, on the real product path: there is no truth here to leak
    out_dir = tmp_path / "out"
    run_declaration(_write_campaign(tmp_path, mode=mode), out_dir=out_dir, era5_hourly_df=_era5())
    written = sorted(out_dir.rglob("*.csv"))
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
