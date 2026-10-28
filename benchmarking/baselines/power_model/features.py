"""Build the power-model's feature matrix (method step 6).

Every feature is upgrade-invariant: it comes from the power references, from waking scenarios that
do not depend on the test turbine's own signals, or from the site's position.

* per power reference: active power and, optionally, the north-calibrated direction as ``sin``/``cos``;
* the waking scenario (step 5) of the test turbine and of each power reference, as a categorical;
* the sun's elevation at the site, :data:`SOLAR_ELEVATION`.

Reference columns are named ``"<tag>{QUALIFIER}<turbine>"``; the test turbine's waking scenario is
named ``"waking_scenario{QUALIFIER}test"``. :func:`check_reference_only` rejects any column qualified
with the test turbine's name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pvlib

from benchmarking.harness.waking import WAKING_SCENARIO_COL

if TYPE_CHECKING:
    from collections.abc import Sequence

# Separator between a source-native tag and the turbine it came from in a feature name.
QUALIFIER = " @ "
# The qualifier of the test turbine's own waking scenario.
TEST_LABEL = "test"
# The prefix the shared northing step puts on a north-calibrated column.
_NORTHED_PREFIX = "northed_"

SOLAR_ELEVATION = "solar_elevation"
# Units of the features that carry no turbine.
SITE_FEATURE_UNITS = {SOLAR_ELEVATION: "°"}

logger = logging.getLogger(__name__)


def _checked_references(references: Sequence[str]) -> list[str]:
    """Return the references sorted and deduped; raises when there are none.

    Sorted so the feature matrix does not depend on the order the references were listed in.
    """
    refs = sorted(dict.fromkeys(str(r) for r in references))
    if not refs:
        msg = (
            "no references supplied: the power model needs at least one. The pool is the campaign's candidate "
            "references, so a turbine present in scada_df is not thereby available as a reference."
        )
        raise ValueError(msg)
    return refs


def build_reference_features(
    scada_df: pd.DataFrame,
    *,
    test_wtg: str,
    references: Sequence[str],
    turbine_col: str,
    active_power_col: str,
    extra_cols: Sequence[str] = (),
    direction_col: str | None = None,
) -> pd.DataFrame:
    """Return each reference's active power, ``extra_cols`` and northed direction, one row per timestamp.

    Columns are ``"<tag>{QUALIFIER}<turbine>"`` in sorted reference order. NaNs are kept.

    :param references: the power references; a name with no data in ``scada_df`` contributes no columns
    :param extra_cols: further per-reference value columns to carry
    :param direction_col: the north-calibrated direction column, carried as ``sin``/``cos``; raises
        when absent. A raw direction listed in ``extra_cols`` is dropped in its favour.
    """
    refs = _checked_references(references)
    extra_cols, direction_frame = _direction_features(
        scada_df, refs=refs, turbine_col=turbine_col, direction_col=direction_col, extra_cols=extra_cols
    )
    value_cols = list(dict.fromkeys([active_power_col, *extra_cols]))
    missing = sorted(c for c in value_cols if c not in scada_df.columns)
    if missing:
        msg = f"scada_df is missing required reference-feature columns {missing}; have {list(scada_df.columns)}"
        raise ValueError(msg)
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()

    tmp = scada_df[[turbine_col, *value_cols]].copy()
    tmp["_ts"] = scada_df.index
    wide = tmp.pivot_table(index="_ts", columns=turbine_col, values=value_cols, aggfunc="first")
    keep = [(col, r) for col in value_cols for r in refs if (col, r) in wide.columns]
    features = wide.loc[:, keep]
    features.columns = [f"{col}{QUALIFIER}{r}" for col, r in keep]
    features = features.reindex(index)
    features.index.name = index.name
    if direction_frame is not None:
        features = features.join(direction_frame.reindex(index), how="left")
    check_reference_only(features.columns.tolist(), test_wtg=test_wtg)
    return features


def without_constant_scenarios(features: pd.DataFrame, *, rows: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Return ``features`` without the waking-scenario columns holding one value over ``rows``, and those columns.

    :param features: the model's feature frame
    :param rows: the records the model is fitted and predicts on
    """
    prefix = f"{WAKING_SCENARIO_COL}{QUALIFIER}"
    used = features.iloc[np.flatnonzero(rows)]
    constant = [c for c in features.columns if str(c).startswith(prefix) and len(pd.unique(used[c])) <= 1]
    return features.drop(columns=constant), constant


def waking_scenario_features(
    scada_df: pd.DataFrame, *, test_wtg: str, references: Sequence[str], turbine_col: str
) -> pd.DataFrame:
    """Return the waking scenario of the test turbine and of each reference, as categoricals.

    The columns share one categorical dtype. A timestamp a turbine has no row for is NaN. Raises
    when ``scada_df`` has no :data:`WAKING_SCENARIO_COL`.
    """
    if WAKING_SCENARIO_COL not in scada_df.columns:
        msg = (
            f"scada_df has no {WAKING_SCENARIO_COL!r} column; the waking scenarios (method step 5) must be added first"
        )
        raise ValueError(msg)
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    columns = {}
    for turbine, label in ((test_wtg, TEST_LABEL), *((r, r) for r in _checked_references(references))):
        rows = scada_df[scada_df[turbine_col] == turbine]
        series = pd.Series(rows[WAKING_SCENARIO_COL].to_numpy(dtype=object), index=pd.DatetimeIndex(rows.index))
        columns[f"{WAKING_SCENARIO_COL}{QUALIFIER}{label}"] = series[~series.index.duplicated()].reindex(index)
    frame = pd.DataFrame(columns, index=index)
    values = pd.unique(frame.to_numpy().ravel())
    dtype = pd.CategoricalDtype(sorted(str(v) for v in values if isinstance(v, str)))
    frame = frame.astype(dtype)
    check_reference_only(frame.columns.tolist(), test_wtg=test_wtg)
    return frame


def _direction_features(
    scada_df: pd.DataFrame,
    *,
    refs: list[str],
    turbine_col: str,
    direction_col: str | None,
    extra_cols: Sequence[str],
) -> tuple[tuple[str, ...], pd.DataFrame | None]:
    """Return ``extra_cols`` with raw directions removed, and the per-reference sin/cos frame.

    ``direction_col=None`` returns ``extra_cols`` unchanged and no frame.
    """
    if direction_col is None:
        return tuple(extra_cols), None
    if direction_col not in scada_df.columns:
        msg = (
            f"the north-calibrated direction column {direction_col!r} is not in scada_df; the shared "
            f"northing step must run before the power model, which reads the northed direction rather "
            f"than the raw one. Columns present: {sorted(scada_df.columns)}"
        )
        raise ValueError(msg)
    raw = direction_col.removeprefix(_NORTHED_PREFIX)
    kept = tuple(c for c in extra_cols if c not in {raw, direction_col})
    if len(kept) != len(extra_cols):
        logger.info("dropping raw direction %r from features in favour of %r", raw, direction_col)

    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    columns = {}
    for ref in refs:
        rows = scada_df[scada_df[turbine_col] == ref]
        series = pd.Series(rows[direction_col].to_numpy(dtype=float), index=pd.DatetimeIndex(rows.index))
        series = series[~series.index.duplicated()].reindex(index)
        rad = np.deg2rad(series.to_numpy(dtype=float))
        columns[f"{direction_col}_sin{QUALIFIER}{ref}"] = np.sin(rad)
        columns[f"{direction_col}_cos{QUALIFIER}{ref}"] = np.cos(rad)
    return kept, pd.DataFrame(columns, index=index)


def solar_elevation_frame(
    index: pd.DatetimeIndex, *, timebase: pd.Timedelta, latitude: float, longitude: float
) -> pd.DataFrame:
    """Return :data:`SOLAR_ELEVATION`, the sun's apparent elevation in degrees, at the middle of each period.

    ``index`` holds timezone-aware period starts, each period ``timebase`` long. Negative values are
    below the horizon. Raises on a timezone-naive index.
    """
    if index.tz is None:
        msg = "solar elevation needs timezone-aware timestamps"
        raise ValueError(msg)
    middle = pd.DatetimeIndex(index).tz_convert("UTC") + timebase / 2
    position = pvlib.solarposition.get_solarposition(middle, latitude, longitude)
    return pd.DataFrame({SOLAR_ELEVATION: position["apparent_elevation"].to_numpy(dtype=float)}, index=index)


def extract_outcome(
    scada_df: pd.DataFrame,
    *,
    test_wtg: str,
    turbine_col: str,
    active_power_col: str,
) -> pd.Series:
    """Return the outcome ``y`` (the test turbine's active power) on the unique sorted index."""
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    test_rows = scada_df[scada_df[turbine_col] == test_wtg]
    y = pd.Series(test_rows[active_power_col].to_numpy(dtype=float), index=pd.DatetimeIndex(test_rows.index))
    return y[~y.index.duplicated()].reindex(index)


def reference_mean_wind_speed(
    scada_df: pd.DataFrame,
    *,
    references: Sequence[str],
    turbine_col: str,
    wind_speed_col: str,
) -> pd.Series:
    """Mean wind speed across the reference pool on the unique index, for the diagnostics only.

    This is **not** a model feature. Averaged over ``references`` only, so it stays upgrade-invariant.
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    refs = _checked_references(references)
    if wind_speed_col not in scada_df.columns:
        return pd.Series(np.nan, index=index)
    cols = []
    for r in refs:
        rows = scada_df[scada_df[turbine_col] == r]
        series = pd.Series(rows[wind_speed_col].to_numpy(dtype=float), index=pd.DatetimeIndex(rows.index))
        cols.append(series[~series.index.duplicated()].reindex(index))
    return pd.concat(cols, axis=1).mean(axis=1)


def check_reference_only(feature_names: list[str], *, test_wtg: str) -> None:
    """Raise if any feature is qualified with the test turbine (violating the §3 rule)."""
    offenders = [f for f in feature_names if f.endswith(f"{QUALIFIER}{test_wtg}")]
    if offenders:
        msg = (
            f"reference-only rule violated: features derived from the test turbine {test_wtg!r} "
            f"are not allowed (the upgrade distorts its signals, design note §3): {offenders}"
        )
        raise ValueError(msg)
