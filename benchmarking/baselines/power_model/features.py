"""Build the power-model's feature matrix (method step 6).

Every feature is upgrade-invariant: it comes from the power references, from waking scenarios that
do not depend on the test turbine's own signals, or from ERA5.

* per power reference: active power and, optionally, the north-calibrated direction as ``sin``/``cos``;
* the waking scenario (step 5) of the test turbine and of each power reference, as a categorical;
* ERA5: :data:`ERA5_FEATURES`.

Reference columns are named ``"<tag>{QUALIFIER}<turbine>"``; the test turbine's waking scenario is
named ``"waking_scenario{QUALIFIER}test"``. :func:`check_reference_only` rejects any column qualified
with the test turbine's name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.baselines.era5_derived import era5_derived_frame
from benchmarking.harness.reanalysis import ERA5_WD_RAW, ERA5_WS_RAW
from benchmarking.harness.waking import WAKING_SCENARIO_COL

if TYPE_CHECKING:
    from collections.abc import Sequence

# Separator between a source-native tag and the turbine it came from in a feature name.
QUALIFIER = " @ "
# The qualifier of the test turbine's own waking scenario.
TEST_LABEL = "test"
# The prefix the shared northing step puts on a north-calibrated column.
_NORTHED_PREFIX = "northed_"

ERA5_DERIVED = ("air_density", "shear_exponent", "gust_ratio")
ERA5_FEATURES = (ERA5_WS_RAW, f"{ERA5_WD_RAW}_sin", f"{ERA5_WD_RAW}_cos", *ERA5_DERIVED)
ERA5_UNITS = {
    ERA5_WS_RAW: "m/s",
    f"{ERA5_WD_RAW}_sin": "-",
    f"{ERA5_WD_RAW}_cos": "-",
    "air_density": "kg/m³",
    "shear_exponent": "-",
    "gust_ratio": "-",
}

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


def era5_feature_frame(aligned_era5: pd.DataFrame) -> pd.DataFrame:
    """Return :data:`ERA5_FEATURES` from aligned ERA5 with Open-Meteo column names.

    Raises when a column they are derived from is missing.
    """
    missing = sorted({ERA5_WS_RAW, ERA5_WD_RAW} - set(aligned_era5.columns))
    if missing:
        msg = f"aligned ERA5 is missing columns {missing}; have {list(aligned_era5.columns)}"
        raise ValueError(msg)
    rad = np.deg2rad(aligned_era5[ERA5_WD_RAW].to_numpy(dtype=float))
    out = pd.DataFrame(
        {
            ERA5_WS_RAW: aligned_era5[ERA5_WS_RAW].to_numpy(dtype=float),
            f"{ERA5_WD_RAW}_sin": np.sin(rad),
            f"{ERA5_WD_RAW}_cos": np.cos(rad),
        },
        index=aligned_era5.index,
    )
    return out.join(era5_derived_frame(aligned_era5, derivations=ERA5_DERIVED))


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


def test_condition_signals(
    scada_df: pd.DataFrame,
    *,
    test_wtg: str,
    turbine_col: str,
    wind_speed_col: str,
    wind_speed_sd_col: str | None,
) -> pd.DataFrame:
    """Test turbine's MEASURED ws and ti on the unique sorted index (post-treatment, accepted §3).

    ``ti`` is omitted when no SD column is configured.
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    rows = scada_df[scada_df[turbine_col] == test_wtg]
    ws = pd.Series(rows[wind_speed_col].to_numpy(dtype=float), index=pd.DatetimeIndex(rows.index))
    ws = ws[~ws.index.duplicated()].reindex(index)
    out = pd.DataFrame({"ws": ws})
    if wind_speed_sd_col is not None and wind_speed_sd_col in scada_df.columns:
        sd = pd.Series(rows[wind_speed_sd_col].to_numpy(dtype=float), index=pd.DatetimeIndex(rows.index))
        sd = sd[~sd.index.duplicated()].reindex(index)
        ws_arr = ws.to_numpy()
        out["ti"] = np.divide(sd.to_numpy(), ws_arr, out=np.full(len(ws_arr), np.nan), where=ws_arr != 0)
    return out


def check_reference_only(feature_names: list[str], *, test_wtg: str) -> None:
    """Raise if any feature is qualified with the test turbine (violating the §3 rule)."""
    offenders = [f for f in feature_names if f.endswith(f"{QUALIFIER}{test_wtg}")]
    if offenders:
        msg = (
            f"reference-only rule violated: features derived from the test turbine {test_wtg!r} "
            f"are not allowed (the upgrade distorts its signals, design note §3): {offenders}"
        )
        raise ValueError(msg)
