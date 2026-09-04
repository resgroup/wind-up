"""Build the power-model's curated, reference-only feature matrix.

The discipline (design note §3): every model feature must be *upgrade-invariant* — derived from
reference turbines (or ERA5), never the test turbine's own signals, which the upgrade distorts.
This matrix is deliberately **curated** to features known to relate to the *cause* of the test
turbine's power — weather and wakes:

* per **reference turbine**: active power (the primary stable weather-driven measurement), the
  availability counter (whether the reference is operating, hence whether it is making a wake),
  and optionally the **north-calibrated** direction as ``sin``/``cos`` (where each reference is
  pointing is much of what resolves who is waking whom);
* all raw **ERA5** columns, passed through under their original Open-Meteo names (no renaming),
  with derived ``sin``/``cos`` companions for the circular wind-direction fields.

Features that are not expected to add value and risk the model learning coincidences rather than
cause-effect (reactive power, blade pitch, …) are intentionally excluded.

Feature columns from references are named ``"<tag>{QUALIFIER}<turbine>"`` so the original tag is
preserved verbatim in importance diagnostics. :func:`check_reference_only` rejects any
test-turbine-qualified column (the §3 guard).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.baselines.era5_sync import ERA5_WD, ERA5_WS

if TYPE_CHECKING:
    from collections.abc import Sequence

# Separator between a source-native tag and the turbine it came from in a feature name.
QUALIFIER = " @ "
# The prefix the shared northing step puts on a north-calibrated column.
_NORTHED_PREFIX = "northed_"

logger = logging.getLogger(__name__)


def _references(scada_df: pd.DataFrame, *, test_wtg: str, turbine_col: str) -> list[str]:
    """Sorted reference turbine names (every turbine present except the test turbine)."""
    refs = sorted(t for t in scada_df[turbine_col].unique() if t != test_wtg)
    if not refs:
        msg = (
            f"no reference turbines available for test_wtg {test_wtg!r}: scada_df contains only "
            f"{sorted(scada_df[turbine_col].unique())}. The power model needs at least one reference turbine."
        )
        raise ValueError(msg)
    return refs


def build_reference_features(
    scada_df: pd.DataFrame,
    *,
    test_wtg: str,
    turbine_col: str,
    active_power_col: str,
    availability_col: str,
    extra_cols: Sequence[str] = (),
    include_availability: bool = True,
    direction_col: str | None = None,
) -> pd.DataFrame:
    """Wide, curated reference features: each reference turbine's active power (+ optional extras).

    By default each reference contributes its active power and availability; ``extra_cols`` adds
    more per-reference channels and ``include_availability=False`` drops the availability feature.
    Columns are ``"<tag>{QUALIFIER}<turbine>"`` keeping the original tag name. The test turbine
    contributes nothing (its power is the outcome, extracted separately). NaNs are preserved (no
    complete-case dropping) — LightGBM handles them natively. Raises if no reference turbine is
    present, or (defensively) if any test-turbine column would leak in.

    :param extra_cols: additional per-reference value columns to carry as features (Issue 11's
        active-power max/min/SD statistics); must be present in ``scada_df`` like the primary two
    :param include_availability: when ``False``, drop the per-reference availability *feature*
        (removal-ablation knob); ``availability_col`` must still exist in ``scada_df`` (the
        downtime filter needs it), so its presence is validated either way
    :param direction_col: the **north-calibrated** direction column each reference contributes as
        ``sin``/``cos`` companions (the raw degrees are never a feature: a tree cannot see that
        359 degrees is next to 1). Must be the column the shared northing step writes; raises
        naming it when absent. A raw direction listed in ``extra_cols`` is dropped in favour of
        it, so a reference never contributes both.
    """
    refs = _references(scada_df, test_wtg=test_wtg, turbine_col=turbine_col)
    extra_cols, direction_frame = _direction_features(
        scada_df, refs=refs, turbine_col=turbine_col, direction_col=direction_col, extra_cols=extra_cols
    )
    value_cols = [active_power_col, *([availability_col] if include_availability else []), *extra_cols]
    # availability_col stays validated even when not featured: it is a required input and the
    # docstring contract is that it exists for the downstream downtime filter.
    missing = sorted(c for c in {*value_cols, availability_col} if c not in scada_df.columns)
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


def _direction_features(
    scada_df: pd.DataFrame,
    *,
    refs: list[str],
    turbine_col: str,
    direction_col: str | None,
    extra_cols: Sequence[str],
) -> tuple[tuple[str, ...], pd.DataFrame | None]:
    """Return ``extra_cols`` with raw directions removed, and the per-reference sin/cos frame.

    ``(None, ...)`` in, ``(extra_cols unchanged, None)`` out: a caller that asks for no direction
    keeps exactly the feature set it had before.
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
    """Turn aligned ERA5 into model features: all raw columns passed through + dir sin/cos companions.

    All raw Open-Meteo columns are kept under their original names (no renaming); the neutral
    ``era5_ws`` / ``era5_wd`` aliases the sync adds for back-compat are dropped here (they duplicate
    ``wind_speed_100m`` / ``wind_direction_100m``). Circular wind-direction fields additionally get
    derived ``<col>_sin`` / ``<col>_cos`` companions (LightGBM cannot see that 359° ≈ 1°); the raw
    degree columns are kept too.
    """
    raw_cols = [c for c in aligned_era5.columns if c not in (ERA5_WS, ERA5_WD)]
    out = aligned_era5[raw_cols].astype(float).copy()
    for col in raw_cols:
        if "direction" in col:
            rad = np.deg2rad(out[col].to_numpy(dtype=float))
            out[f"{col}_sin"] = np.sin(rad)
            out[f"{col}_cos"] = np.cos(rad)
    return out


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
    test_wtg: str,
    turbine_col: str,
    wind_speed_col: str,
) -> pd.Series:
    """Mean wind speed across reference turbines on the unique index (used only for ERA5 lag sync).

    This is **not** a model feature — it is the site wind-speed signal the ERA5 correlation sweep
    locks onto. Computed from references only so it stays upgrade-invariant.
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    refs = _references(scada_df, test_wtg=test_wtg, turbine_col=turbine_col)
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
