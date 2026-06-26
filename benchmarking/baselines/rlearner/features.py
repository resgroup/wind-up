"""Build the R-learner's upgrade-invariant feature matrix from long SCADA.

The discipline (design note §3): every model feature must be *upgrade-invariant* — derived
from reference turbines (or ERA5 / met-mast / LiDAR), never the test turbine's own signals,
which the upgrade distorts. Within that rule the matrix is **maximal, not curated**: every
source-native column of every reference turbine is used as-is, original tag names intact, so
the model sees all the data holistically and the feature-importance diagnostics name real
tags.

The only column that must be identified by config is the **test turbine's active power**
(the outcome ``Y``); reference turbines need no per-column configuration.

Feature columns are named ``"<tag>{QUALIFIER}<turbine>"`` (e.g. ``"wtc_ActPower_mean @ R1"``)
so the original tag name is preserved verbatim. :func:`check_upgrade_invariant` is the
enforcement guard that rejects any test-turbine-qualified column — exercised by the
bias-guard regression test (design note §8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.baselines.rlearner.era5_sync import ERA5_WD, ERA5_WS
from benchmarking.synthetic import ToggleSchedule, treated_mask

# Separator between a source-native tag and the turbine it came from in a feature name.
QUALIFIER = " @ "


def _references(scada_df: pd.DataFrame, *, test_wtg: str, turbine_col: str) -> list[str]:
    """Sorted reference turbine names (every turbine present except the test turbine)."""
    refs = sorted(t for t in scada_df[turbine_col].unique() if t != test_wtg)
    if not refs:
        msg = (
            f"no reference turbines available for test_wtg {test_wtg!r}: scada_df contains only "
            f"{sorted(scada_df[turbine_col].unique())}. The R-learner needs at least one reference turbine."
        )
        raise ValueError(msg)
    return refs


def build_reference_features(scada_df: pd.DataFrame, *, test_wtg: str, turbine_col: str) -> pd.DataFrame:
    """Wide upgrade-invariant features: every reference turbine's every value column, NaN-preserving.

    Columns are ``"<tag>{QUALIFIER}<turbine>"`` keeping the original tag name. The test turbine
    contributes nothing here (its outcome is extracted separately). NaNs are preserved (no
    complete-case dropping) — LightGBM handles them natively. Includes the (currently no-op)
    engineered-feature seam. Raises if no reference turbine is present, or (defensively) if any
    test-turbine column would leak in.
    """
    refs = _references(scada_df, test_wtg=test_wtg, turbine_col=turbine_col)
    value_cols = [c for c in scada_df.columns if c != turbine_col]
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()

    # One pivot over all value columns (MultiIndex columns: (value_col, turbine)) rather than one
    # pivot_table per column — much cheaper on wide SCADA frames. Then keep reference turbines only,
    # in (value_col, ref) order, and flatten to the "<tag> @ <turbine>" names.
    tmp = scada_df.copy()
    tmp["_ts"] = scada_df.index
    wide = tmp.pivot_table(index="_ts", columns=turbine_col, values=value_cols, aggfunc="first")
    keep = [(col, r) for col in value_cols for r in refs if (col, r) in wide.columns]
    features = wide.loc[:, keep]
    features.columns = [f"{col}{QUALIFIER}{r}" for col, r in keep]
    features = features.reindex(index)
    features = pd.concat(
        [features, engineered_reference_features(scada_df, test_wtg=test_wtg, turbine_col=turbine_col)], axis=1
    )
    features.index.name = index.name
    check_upgrade_invariant(features.columns.tolist(), test_wtg=test_wtg)
    return features


def engineered_reference_features(scada_df: pd.DataFrame, *, test_wtg: str, turbine_col: str) -> pd.DataFrame:  # noqa: ARG001
    """Feature-engineering seam — currently a no-op (returns no columns).

    This is the obvious home for *derived* upgrade-invariant features. The prime future
    candidate is **north-corrected reference yaw position** (an accurate per-record wind
    direction and waking relationship — a key v0 value-add), alongside shear, stability
    proxies and air density. Not implemented now: ERA5 wind direction already gives reasonable
    directional information, and proper northing would pull in v0's machinery. Returns an empty
    frame on the data's unique timestamps so callers can concatenate it unconditionally.
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    return pd.DataFrame(index=index)


def extract_outcome_and_treatment(
    scada_df: pd.DataFrame,
    *,
    test_wtg: str,
    turbine_col: str,
    active_power_col: str,
    upgrade_timing: pd.Timestamp | ToggleSchedule,
) -> tuple[pd.Series, pd.Series]:
    """Return the outcome ``y`` (test turbine power) and upgrade flag ``t`` on the unique index.

    ``y`` is the test turbine's active power (may contain NaN downtime). ``t`` is the integer
    upgrade flag from :func:`treated_mask` (1 = upgraded record, 0 = baseline).
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    test_rows = scada_df[scada_df[turbine_col] == test_wtg]
    y = test_rows[active_power_col].copy()
    y.index = pd.DatetimeIndex(test_rows.index)
    y = y.reindex(index)
    t = pd.Series(np.asarray(treated_mask(index, upgrade_timing)).astype(int), index=index)
    return y, t


def era5_features(aligned_era5: pd.DataFrame) -> pd.DataFrame:
    """Turn aligned ERA5 (ws + wd degrees) into model features: ws passthrough, wd as sin/cos."""
    rad = np.deg2rad(aligned_era5[ERA5_WD].to_numpy(dtype=float))
    return pd.DataFrame(
        {
            ERA5_WS: aligned_era5[ERA5_WS].to_numpy(dtype=float),
            "era5_wd_sin": np.sin(rad),
            "era5_wd_cos": np.cos(rad),
        },
        index=aligned_era5.index,
    )


def check_upgrade_invariant(feature_names: list[str], *, test_wtg: str) -> None:
    """Raise if any feature is qualified with the test turbine (violating the §3 rule)."""
    offenders = [f for f in feature_names if f.endswith(f"{QUALIFIER}{test_wtg}")]
    if offenders:
        msg = (
            f"upgrade-invariant rule violated: features derived from the test turbine {test_wtg!r} "
            f"are not allowed (the upgrade distorts its signals, design note §3): {offenders}"
        )
        raise ValueError(msg)
