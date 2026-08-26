"""Comparison-derived ground-truth uplift for synthetic datasets.

The true uplift is never a declared constant: it is computed by comparing the test
turbine's synthetic power to its original power over exactly the records used, so it
changes with campaign length and any condition filter. Conditions for per-condition
breakdowns use the original (treatment-invariant) signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.synthetic.sources.hill_of_towie import HOT_COLUMNS

if TYPE_CHECKING:
    import numpy.typing as npt

    from benchmarking.synthetic.schema import ColumnSchema


def _condition_series(rows: pd.DataFrame, by: str, columns: ColumnSchema) -> npt.NDArray[np.float64]:
    """Resolve a treatment-invariant condition signal from the original rows."""
    if by == "ti":
        ws = rows[columns.wind_speed].to_numpy(dtype=float)
        sd = rows[columns.wind_speed_sd].to_numpy(dtype=float)
        # NaN (not inf/0-division warning) for calm rows; warnings are errors in tests.
        return np.divide(sd, ws, out=np.full_like(sd, np.nan), where=ws != 0)
    if by == "ws":
        return rows[columns.wind_speed].to_numpy(dtype=float)
    if by == "power":
        # The baseline operating point: bin on the ORIGINAL (pre-upgrade) active power so the axis is
        # upgrade-invariant.
        return rows[columns.active_power].to_numpy(dtype=float)
    return rows[by].to_numpy(dtype=float)


@dataclass
class UpliftResult:
    """Ground-truth uplift over a set of records."""

    overall: float
    by_condition: pd.DataFrame | None = None


def true_uplift(
    synthetic_df: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    test_wtg: str,
    mask: npt.ArrayLike | None = None,
    by: str | None = None,
    bins: npt.ArrayLike | None = None,
    columns: ColumnSchema = HOT_COLUMNS,
) -> UpliftResult:
    """Compute the true uplift of the test turbine, synthetic vs original.

    :param synthetic_df: method-facing synthetic SCADA
    :param original_df: untouched original SCADA (ground-truth reference)
    :param test_wtg: the upgraded turbine to measure
    :param mask: boolean selection over the test turbine's rows (time order); default is
        the records the upgrade actually changed
    :param by: optional treatment-invariant condition for a per-condition breakdown
        (``"ws"``, ``"ti"`` or an original column name)
    :param bins: bin edges for the ``by`` condition
    :param columns: the source-native column schema the frames are keyed by
    :return: the overall energy-ratio uplift, and a per-condition table when ``by`` is set
    """
    if by is not None and bins is None:
        msg = "bins must be provided when by is set (per-condition breakdown needs explicit edges)"
        raise ValueError(msg)

    original_wtg = original_df[original_df[columns.turbine] == test_wtg]
    synthetic_power = synthetic_df.loc[synthetic_df[columns.turbine] == test_wtg, columns.active_power].to_numpy(
        dtype=float
    )
    original_power = original_wtg[columns.active_power].to_numpy(dtype=float)

    row_mask = changed_record_mask(synthetic_power, original_power) if mask is None else np.asarray(mask, dtype=bool)
    # Real SCADA carries NaN power (downtime/missing); such records have no usable energy
    # and must not poison the sums, so the ratio is taken over finite records only.
    effective = row_mask & np.isfinite(synthetic_power) & np.isfinite(original_power)

    denom = original_power[effective].sum()
    overall = synthetic_power[effective].sum() / denom - 1.0 if denom else float("nan")

    by_condition = None
    if by is not None:
        by_condition = _uplift_by_condition(
            condition=_condition_series(original_wtg, by, columns)[effective],
            synthetic_power=synthetic_power[effective],
            original_power=original_power[effective],
            bins=bins,
        )
    return UpliftResult(overall=float(overall), by_condition=by_condition)


def true_net_uplift(
    synthetic_df: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    upstream: str,
    downstream: str,
    mask: npt.ArrayLike | None = None,
    columns: ColumnSchema = HOT_COLUMNS,
) -> float:
    """Production-weighted net uplift of a wake-steering upstream/downstream pair.

    Energy ratio over the union of both turbines' changed (or ``mask``-selected) finite records:
    ``(Σ syn_up + Σ syn_down) / (Σ orig_up + Σ orig_down) - 1``. This weights each turbine by its
    production (the campaign "average power of the pair" net), so a small loss on the windier
    steering turbine and a larger gain on the waked downstream turbine combine correctly.

    :param synthetic_df: method-facing synthetic SCADA
    :param original_df: untouched original SCADA (ground-truth reference)
    :param upstream: steering turbine name
    :param downstream: benefitting turbine name
    :param mask: optional boolean selection over the pair's timestamps (time order); default is the
        union of the two turbines' changed records
    :param columns: the source-native column schema the frames are keyed by
    """

    def _power(df: pd.DataFrame, wtg: str) -> npt.NDArray[np.float64]:
        return df.loc[df[columns.turbine] == wtg, columns.active_power].to_numpy(dtype=float)

    syn_up, orig_up = _power(synthetic_df, upstream), _power(original_df, upstream)
    syn_dn, orig_dn = _power(synthetic_df, downstream), _power(original_df, downstream)

    if mask is None:
        row_mask = changed_record_mask(syn_up, orig_up) | changed_record_mask(syn_dn, orig_dn)
    else:
        row_mask = np.asarray(mask, dtype=bool)
    # Both turbines are summed over the SAME timestamps so the pair energy ratio stays balanced;
    # require every power finite there (a NaN on either side would drop that timestamp for both).
    finite = np.isfinite(syn_up) & np.isfinite(orig_up) & np.isfinite(syn_dn) & np.isfinite(orig_dn)
    effective = row_mask & finite

    orig_den = orig_up[effective].sum() + orig_dn[effective].sum()
    syn_num = syn_up[effective].sum() + syn_dn[effective].sum()
    return float(syn_num / orig_den - 1.0) if orig_den else float("nan")


def changed_record_mask(
    synthetic_power: npt.NDArray[np.float64], original_power: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Boolean mask of records the upgrade actually changed (NaN-safe).

    A plain ``synthetic != original`` would flag downtime rows where both powers are NaN
    (since ``NaN != NaN``); those are excluded here so only genuinely modified records
    are treated as upgraded.
    """
    differs = synthetic_power != original_power
    both_nan = np.isnan(synthetic_power) & np.isnan(original_power)
    return differs & ~both_nan


def _uplift_by_condition(
    *,
    condition: npt.NDArray[np.float64],
    synthetic_power: npt.NDArray[np.float64],
    original_power: npt.NDArray[np.float64],
    bins: npt.ArrayLike | None,
) -> pd.DataFrame:
    """Energy-ratio uplift within bins of a condition signal."""
    grouped = pd.DataFrame(
        {
            "condition_bin": pd.cut(condition, bins=bins),
            "synthetic_energy": synthetic_power,
            "original_energy": original_power,
        }
    ).groupby("condition_bin", observed=False)
    table = grouped[["synthetic_energy", "original_energy"]].sum()
    table["n_records"] = grouped.size()
    table["true_uplift"] = table["synthetic_energy"] / table["original_energy"] - 1.0
    return table.reset_index()
