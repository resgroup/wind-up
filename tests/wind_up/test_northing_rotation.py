"""Northing must not care where north is: rotating every direction input rotates nothing out.

Hill of Towie's wind is mostly south-westerly, so its raw data rarely crosses 0/360. Adding the same
angle to every direction -- the turbines' yaw and the reanalysis alike -- leaves every residual, and
so every north offset, unchanged; but a rotation of 150 deg moves the bulk of the record onto north,
so each 360-0 crossing becomes an opportunity for a wrap bug to show.

Rotations that are multiples of the 30 deg veer sector keep every row in an equivalent sector, so
those must reproduce the unrotated tables exactly. Other rotations move sector boundaries relative
to the data, which legitimately changes the veer signature a little; those are held to the same
changepoints (same count, each within a few days -- the sector regrouping jitters a timestamp by a
fraction of a day), but not to the same offsets.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wind_up.circular_math import circ_diff
from wind_up.northing import DEFAULT_NORTHING, NorthingSettings, estimate_north_table, north_farm

FIXTURE = Path(__file__).parents[1] / "test_data" / "hot" / "northing" / "northing_inputs.parquet"
ALL_TURBINES = tuple(f"T{n:02d}" for n in range(1, 22))
WEST = tuple(f"T{n:02d}" for n in range(1, 16))

SECTOR_ALIGNED = (90.0, 150.0, 270.0)
SECTOR_MISALIGNED = (137.0, 223.0)


def _is_parquet(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_parquet(FIXTURE), reason="Hill of Towie northing fixture not available (git-lfs not pulled)"
)


@pytest.fixture(scope="module")
def hot() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def _arrays(hot: pd.DataFrame, turbines: tuple[str, ...], start: str, end: str, *, rotate_deg: float) -> tuple:
    """``(index, direction, usable, reanalysis)`` for a window, every direction rotated by ``rotate_deg``."""
    rows = hot[
        hot["turbine"].isin(turbines)
        & (hot["timestamp"] >= pd.Timestamp(start, tz="UTC"))
        & (hot["timestamp"] < pd.Timestamp(end, tz="UTC"))
    ]
    index = pd.DatetimeIndex(sorted(rows["timestamp"].unique()))
    direction, usable, reanalysis = {}, {}, None
    for turbine in sorted(rows["turbine"].unique()):
        one = rows[rows["turbine"] == turbine].drop_duplicates("timestamp").set_index("timestamp").reindex(index)
        wd = (one["era5_wd_deg"].to_numpy(dtype=float) + rotate_deg) % 360.0
        reanalysis = wd if reanalysis is None else np.where(np.isfinite(reanalysis), reanalysis, wd)
        yaw = (one["yaw_deg"].to_numpy(dtype=float) + rotate_deg) % 360.0
        direction[str(turbine)] = yaw
        usable[str(turbine)] = np.isfinite(yaw) & np.isfinite(wd)
    return index, direction, usable, reanalysis


_FARM_RUNS: dict[tuple, dict[str, pd.DataFrame]] = {}


def test_the_rotations_really_put_the_record_on_north(hot: pd.DataFrame) -> None:
    """Guard the premise: at +150 deg most of the yaw record sits within 60 deg of north."""
    rotated = (hot["yaw_deg"].dropna().to_numpy() + 150.0) % 360.0
    near_north = np.abs(circ_diff(rotated, 0.0)) < 60.0
    assert near_north.mean() > 0.5


def _compare(base: pd.DataFrame, rotated: pd.DataFrame, *, offsets_too: bool, time_tol_days: float = 0.0) -> list[str]:
    problems = []
    if len(base) != len(rotated):
        problems.append(f"{len(base) - 1} changepoints unrotated vs {len(rotated) - 1} rotated")
        return problems
    shift_days = np.abs(
        (pd.DatetimeIndex(rotated["timestamp"]) - pd.DatetimeIndex(base["timestamp"])).total_seconds()
    ) / 86400.0
    if (shift_days > time_tol_days).any():
        rows = np.flatnonzero(shift_days > time_tol_days).tolist()
        problems.append(f"changepoint times differ by up to {shift_days.max():.2f} d at rows {rows}")
    if offsets_too:
        worst = float(np.abs(circ_diff(base["north_offset"], rotated["north_offset"])).max())
        if worst > 1e-6:
            problems.append(f"offsets differ by up to {worst:.6f} deg")
    return problems


# A sector-misaligned rotation regroups rows across the fixed veer-sector boundaries, which nudges a
# changepoint's timestamp by a fraction of a day; the count never changes. Allow that jitter so the
# test still fails on a real wrap bug (which moves changepoints by weeks or changes their count).
_MISALIGNED_TIME_TOL_DAYS = 3.0


# Each case is (rotation, check_offsets). Sector-aligned rotations keep every row in an equivalent
# veer sector, so the offsets must match to the last decimal. A misaligned rotation regroups rows
# across the fixed 0/30/60... deg sector boundaries and can change the veer signature a little -- a
# legitimate offset difference, not a 0/360 wrap bug -- so its offsets are not pinned. The
# changepoints must survive either way, and those are always asserted (never hidden behind an xfail),
# so a rotation that moves a changepoint fails the test.
_ROTATIONS = [
    *(pytest.param(r, id=f"aligned-{r:.0f}") for r in SECTOR_ALIGNED),
    *(pytest.param(r, id=f"misaligned-{r:.0f}") for r in SECTOR_MISALIGNED),
]
NO_VEER = replace(DEFAULT_NORTHING, veer_sector_deg=None)


def _tolerances(rotate_deg: float) -> tuple[bool, float]:
    """``(offsets_too, time_tol_days)`` for a rotation: exact for sector-aligned, lenient otherwise."""
    aligned = rotate_deg in SECTOR_ALIGNED
    return aligned, 0.0 if aligned else _MISALIGNED_TIME_TOL_DAYS


def _farm_tables(
    hot: pd.DataFrame,
    turbines: tuple[str, ...],
    start: str,
    end: str,
    rotate_deg: float,
    settings: NorthingSettings = DEFAULT_NORTHING,
) -> dict[str, pd.DataFrame]:
    key = (turbines, start, end, rotate_deg, settings)
    if key not in _FARM_RUNS:
        index, direction, usable, reanalysis = _arrays(hot, turbines, start, end, rotate_deg=rotate_deg)
        _FARM_RUNS[key] = north_farm(
            index, direction_deg=direction, usable=usable, reanalysis_deg=reanalysis, settings=settings
        )
    return _FARM_RUNS[key]


def _lone(hot: pd.DataFrame, rotate_deg: float, settings: NorthingSettings = DEFAULT_NORTHING) -> pd.DataFrame:
    index, direction, usable, reanalysis = _arrays(hot, ("T16",), "2017-01-01", "2019-01-01", rotate_deg=rotate_deg)
    return estimate_north_table(
        index, direction["T16"], reference_deg=reanalysis, usable=usable["T16"], settings=settings
    )


@pytest.mark.parametrize("rotate_deg", _ROTATIONS)
def test_a_lone_turbine_against_reanalysis(hot: pd.DataFrame, rotate_deg: float) -> None:
    offsets_too, tol = _tolerances(rotate_deg)
    problems = _compare(_lone(hot, 0.0), _lone(hot, rotate_deg), offsets_too=offsets_too, time_tol_days=tol)
    assert problems == [], f"T16 rotated by {rotate_deg}: {problems}"


@pytest.mark.slow
@pytest.mark.parametrize("rotate_deg", _ROTATIONS)
def test_the_whole_farm(hot: pd.DataFrame, rotate_deg: float) -> None:
    base = _farm_tables(hot, ALL_TURBINES, "2019-01-01", "2021-01-01", 0.0)
    rotated = _farm_tables(hot, ALL_TURBINES, "2019-01-01", "2021-01-01", rotate_deg)
    offsets_too, tol = _tolerances(rotate_deg)
    problems = {
        name: p
        for name in ALL_TURBINES
        if (p := _compare(base[name], rotated[name], offsets_too=offsets_too, time_tol_days=tol))
    }
    assert problems == {}, f"rotated by {rotate_deg}: {problems}"


@pytest.mark.slow
@pytest.mark.parametrize("rotate_deg", _ROTATIONS)
def test_half_the_farm(hot: pd.DataFrame, rotate_deg: float) -> None:
    base = _farm_tables(hot, WEST, "2017-01-01", "2019-01-01", 0.0)
    rotated = _farm_tables(hot, WEST, "2017-01-01", "2019-01-01", rotate_deg)
    offsets_too, tol = _tolerances(rotate_deg)
    problems = {
        name: p
        for name in WEST
        if (p := _compare(base[name], rotated[name], offsets_too=offsets_too, time_tol_days=tol))
    }
    assert problems == {}, f"rotated by {rotate_deg}: {problems}"


class TestWithoutVeerSectors:
    """With the absolute-direction sectors switched off, every rotation must be exact."""

    @pytest.mark.parametrize("rotate_deg", SECTOR_ALIGNED + SECTOR_MISALIGNED)
    def test_a_lone_turbine_against_reanalysis(self, hot: pd.DataFrame, rotate_deg: float) -> None:
        problems = _compare(_lone(hot, 0.0, NO_VEER), _lone(hot, rotate_deg, NO_VEER), offsets_too=True)
        assert problems == [], f"T16 rotated by {rotate_deg}: {problems}"

    @pytest.mark.slow
    @pytest.mark.parametrize("rotate_deg", SECTOR_ALIGNED + SECTOR_MISALIGNED)
    def test_the_whole_farm(self, hot: pd.DataFrame, rotate_deg: float) -> None:
        base = _farm_tables(hot, ALL_TURBINES, "2019-01-01", "2021-01-01", 0.0, NO_VEER)
        rotated = _farm_tables(hot, ALL_TURBINES, "2019-01-01", "2021-01-01", rotate_deg, NO_VEER)
        problems = {name: p for name in ALL_TURBINES if (p := _compare(base[name], rotated[name], offsets_too=True))}
        assert problems == {}, f"rotated by {rotate_deg}: {problems}"
