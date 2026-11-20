"""A record labelled missing reads exactly as a record never provided.

Each test runs a campaign twice: on SCADA with records step 2 labels missing, and on the same SCADA with a random
share of those records dropped, a reference's whole leading and trailing stretches included. Everything from
step 3 on must be identical.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.make_operating_state_fixture import load_fixture
from benchmarking.campaigns.composed import run_declaration
from benchmarking.campaigns.declaration import CampaignSpec, layout_from_coords
from benchmarking.campaigns.loader import load_declaration, operating_state_config
from benchmarking.campaigns.plans import plans_for
from benchmarking.campaigns.preparation import prepare_data
from benchmarking.campaigns.real import HOT_OPERATING_STATE, HOT_STATE_COL, label_hot_site_states
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import HOT_COORDINATES, HOT_RATED_POWER_KW, HOT_ROTOR_DIAMETER_M
from wind_up.analysis_period import MONTH, PlanSettings
from wind_up.operating_state import MISSING, STATE_COL, OperatingStateConfig, label_operating_states

from .test_declaration_end_to_end import ERA5_COLUMNS, ERA5_PATH, PA_PER_HPA, RATED_KW, _era5, _scada, _write_campaign
from .test_operating_states_real_data import _fixture_available

if TYPE_CHECKING:
    from pathlib import Path

COLS = HOT_COLUMNS
TIMEBASE = pd.Timedelta(hours=1)
# T03 has no data for its first and last fortnight.
GAP_TURBINE = "T03"
GAP = pd.Timedelta(days=14)
# What step 2 describes: the provided records and why each is missing, not what later steps use.
DESCRIBES_THE_INPUT = ("01_changes", "label_checks.csv")
# When a run wrote its results, not what it found.
WALL_CLOCK = ["time_calculated"]


def with_missing_records(scada: pd.DataFrame) -> pd.DataFrame:
    """Return ``scada`` with records step 2 labels missing: gaps, out-of-range power, stuck runs and contradictions."""
    rng = np.random.default_rng(1)
    scada = scada.copy()
    power = scada[COLS.active_power].to_numpy(dtype=float, copy=True)
    counter = scada[COLS.availability].to_numpy(dtype=float, copy=True)
    index = pd.DatetimeIndex(scada.index)
    gap = (scada[COLS.turbine].to_numpy() == GAP_TURBINE) & ((index < index.min() + GAP) | (index > index.max() - GAP))
    power[gap] = np.nan
    power[rng.random(len(scada)) < 0.02] = 3 * RATED_KW
    contradicted = rng.random(len(scada)) < 0.01
    counter[contradicted] = 0.0
    power[contradicted] = 0.5 * RATED_KW
    scada[COLS.active_power] = power
    scada[COLS.availability] = counter
    # a stuck run: every signal repeats the record before it
    for start in rng.choice(len(scada) - 10, size=40, replace=False):
        scada.iloc[start + 1 : start + 4, 1:] = scada.iloc[start, 1:].to_numpy()
    return scada


def without_missing_records(
    scada: pd.DataFrame,
    *,
    config: OperatingStateConfig | None = None,
    timebase: pd.Timedelta = TIMEBASE,
    rated_power_kw: float = RATED_KW,
    gap_turbine: str = GAP_TURBINE,
) -> pd.DataFrame:
    """Return ``scada`` without ``gap_turbine``'s non-finite power and a random half of the other missing records."""
    labelled = label_operating_states(
        scada,
        columns=COLS,
        config=config or OperatingStateConfig(),
        timebase=timebase,
        rated_power_kw=rated_power_kw,
    )
    missing = labelled[STATE_COL].to_numpy() == MISSING
    gap = (scada[COLS.turbine].to_numpy() == gap_turbine) & ~np.isfinite(scada[COLS.active_power].to_numpy())
    rng = np.random.default_rng(2)
    drop = gap | (missing & (rng.random(len(scada)) < 0.5))
    assert drop.sum() > 1000
    return scada[~drop]


def test_part_a_reads_a_missing_record_as_absent(tmp_path: Path) -> None:
    full = with_missing_records(_scada(mode="prepost"))
    spec = replace(load_declaration(_write_campaign(tmp_path, mode="prepost")).spec, north_offsets=None)
    a, b = (
        prepare_data(scada, spec=spec, columns=COLS, era5_hourly_df=_era5())
        for scada in (full, without_missing_records(full))
    )
    pd.testing.assert_frame_equal(
        a.scada_df.drop(columns=_signals(a.scada_df)), b.scada_df.drop(columns=_signals(b.scada_df))
    )
    assert a.north_tables.keys() == b.north_tables.keys()
    for turbine, table in a.north_tables.items():
        pd.testing.assert_frame_equal(table, b.north_tables[turbine])
    pd.testing.assert_frame_equal(a.possible_changes, b.possible_changes)
    pd.testing.assert_frame_equal(a.aligned_era5, b.aligned_era5)


def _signals(frame: pd.DataFrame) -> list[str]:
    """The provided signal columns, which differ on a dropped record by construction."""
    return [c for c in frame.columns if c in set(_scada(mode="prepost").columns) - {COLS.turbine}]


@pytest.mark.slow
def test_a_composed_run_reads_a_missing_record_as_absent(tmp_path: Path) -> None:
    full = with_missing_records(_scada(mode="prepost"))
    reports, outputs = [], []
    for name, scada in (("full", full), ("dropped", without_missing_records(full))):
        (tmp_path / name).mkdir()
        path = _write_campaign(tmp_path / name, mode="prepost")
        path.write_text(path.read_text().replace("discover: false", "discover: true"))
        scada.to_parquet(path.parent / "scada.parquet")
        out_dir = tmp_path / name / "out"
        reports.append(run_declaration(path, out_dir=out_dir, era5_hourly_df=_era5()))
        outputs.append(out_dir)
    a, b = reports
    for table in ("per_turbine", "farm", "reference_stability", "conditional"):
        pd.testing.assert_frame_equal(getattr(a, table), getattr(b, table), obj=table)
    assert a.warnings == b.warnings
    assert {t: p.summary() for t, p in a.plans.items()} == {t: p.summary() for t, p in b.plans.items()}
    written = sorted(p.relative_to(outputs[0]) for p in outputs[0].rglob("*.csv"))
    assert written == sorted(p.relative_to(outputs[1]) for p in outputs[1].rglob("*.csv"))
    for relative in written:
        if any(part in DESCRIBES_THE_INPUT for part in relative.parts):
            continue
        pd.testing.assert_frame_equal(
            pd.read_csv(outputs[0] / relative).drop(columns=WALL_CLOCK, errors="ignore"),
            pd.read_csv(outputs[1] / relative).drop(columns=WALL_CLOCK, errors="ignore"),
            obj=str(relative),
        )


def _hot_era5(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The tracked Hill of Towie hourly reanalysis over ``[start, end]``, in the Open-Meteo layout."""
    era5 = pd.read_parquet(ERA5_PATH, columns=list(ERA5_COLUMNS)).rename(columns=ERA5_COLUMNS)
    era5 = era5.loc[start:end].rename_axis(None)
    era5["surface_pressure"] = era5["surface_pressure"] / PA_PER_HPA
    era5["wind_gusts_10m"] = era5["wind_speed_10m"] * 1.4
    return era5


@pytest.mark.slow
@pytest.mark.skipif(not _fixture_available(), reason="Hill of Towie operating-state fixture not available")
def test_real_hill_of_towie_data_reads_a_missing_record_as_absent() -> None:
    # six months of every Hill of Towie turbine, with a placebo change on T13 halfway through
    full = load_fixture()
    full = full.assign(**{HOT_STATE_COL: label_hot_site_states(full)})
    config = operating_state_config(HOT_OPERATING_STATE)
    timebase = pd.Timedelta(minutes=10)
    dropped = without_missing_records(
        full, config=config, timebase=timebase, rated_power_kw=HOT_RATED_POWER_KW, gap_turbine="T13"
    )
    spec = CampaignSpec(
        upgraded_turbines=["T13"],
        upgrade_timing=pd.Timestamp("2018-11-01", tz="UTC"),
        candidate_references=[t for t in HOT_COORDINATES if t != "T13"],
        excluded_turbines=[],
        layout=layout_from_coords(HOT_COORDINATES, rotor_diameter_m=HOT_ROTOR_DIAMETER_M),
        north_offsets=None,
        rated_power_kw=HOT_RATED_POWER_KW,
        analysis_period=None,
    )
    era5 = _hot_era5(pd.Timestamp("2018-07-31", tz="UTC"), pd.Timestamp("2019-02-02", tz="UTC"))
    a, b = (
        prepare_data(scada, spec=spec, columns=COLS, era5_hourly_df=era5, operating_state=config)
        for scada in (full, dropped)
    )
    provided = set(full.columns) - {COLS.turbine}
    pd.testing.assert_frame_equal(
        a.scada_df.drop(columns=[c for c in a.scada_df.columns if c in provided]),
        b.scada_df.drop(columns=[c for c in b.scada_df.columns if c in provided]),
    )
    for turbine, table in a.north_tables.items():
        pd.testing.assert_frame_equal(table, b.north_tables[turbine])
    pd.testing.assert_frame_equal(a.possible_changes, b.possible_changes)
    settings = PlanSettings(min_side=2 * MONTH)
    plans = [
        plans_for(prepared.with_north_offsets(spec), prepared.scada_df, columns=COLS, settings=settings)
        for prepared in (a, b)
    ]
    assert plans[0].plans
    assert plans[0].extents == plans[1].extents
    assert {t: p.summary() for t, p in plans[0].plans.items()} == {t: p.summary() for t, p in plans[1].plans.items()}
