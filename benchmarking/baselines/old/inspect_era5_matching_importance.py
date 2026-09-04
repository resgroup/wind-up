"""One-off analysis: which ERA5 fields best predict the test turbine's power (Issue 8, Component 1).

The bias-cancellation correction (Issue 8) matches the baseline and upgraded periods on **ERA5-only**
weather so a common per-bin multiplicative shrinkage cancels between the two train/predict directions
(design + ``docs/v1/findings.md``). CEM cell count explodes with dimension, so we can only afford to
match on a *few* ERA5 variables — and they must be the ones that actually drive the test turbine's
power, or the matched shrinkage factor is not the one that distorts the estimate.

This script ranks the ERA5 fields by how well they predict the test turbine's (un-upgraded, real HoT)
power, using two views so gain alone is not over-trusted:

* **LightGBM gain importance** from the same outcome-model factory the method uses;
* **sklearn permutation importance** on a held-out split (model-agnostic, guards against gain quirks).

ERA5-only (not the full reference+ERA5 matrix) because the reference active-power features otherwise
dominate and mask the ERA5 signal, and only ERA5 has the full-coverage, temporally-stable columns we
can actually match on. The whole default window is genuine no-upgrade HoT SCADA, so every
normally-operating row is a "baseline" row for this purpose.

Outputs (under ``<out-root>/inspection_era5_matching``):

* ``feature_importance.png`` — gain and permutation rankings side by side (the selection view);
* ``predicted_vs_actual.png`` — held-out predicted vs actual test power with R²/RMSE/MAE. This is a
  **gate**: if ERA5 predicts test power poorly the whole ranking is suspect, not just imprecise.
* ``era5_matching_importance.csv`` — the merged ranking table.

The chosen matching set + rationale (citing these metrics) is recorded in
``docs/v1/findings.md`` and hard-coded as the method default; this script does not edit anything.

Run from the repo root::

    uv run python -m benchmarking.baselines.old.inspect_era5_matching_importance
    uv run python -m benchmarking.baselines.old.inspect_era5_matching_importance --test-wtg T07
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from benchmarking.baselines.era5_derived import shear_exponent
from benchmarking.baselines.era5_sync import sync_era5
from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
    default_output_root,
)
from benchmarking.baselines.filtering import NormalOperationFilter
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.power_model.features import (
    era5_feature_frame,
    extract_outcome,
    reference_mean_wind_speed,
)
from benchmarking.baselines.power_model.fitting import make_outcome_model
from benchmarking.diagnostics.density import density_scatter
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# Prefer a small matching set; flag the strongest few above a fraction-of-the-top-feature floor.
_SELECTION_FLOOR_FRAC = 0.05
_PREFERRED_N = 2


def _infer_timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Analysis timebase as the median spacing of the sorted unique timestamps (≈10 min for HoT)."""
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    if len(unique) < 2:  # noqa: PLR2004
        return pd.Timedelta(minutes=10)
    return pd.Timedelta(np.median(np.diff(unique.to_numpy())))


def add_shear_exponent(features: pd.DataFrame) -> pd.DataFrame:
    """Fold the collinear 10m/100m wind speeds into one physical vertical-shear exponent.

    The two ERA5 wind speeds are strongly correlated, so gain splits credit between them while
    permutation discounts whichever is redundant — neither view then cleanly reflects the *shear* they
    jointly encode. The power-law exponent (see
    :func:`benchmarking.baselines.era5_derived.shear_exponent`, the shared Issue 9 utility) captures
    that shear in a single column (a stability / turbulence proxy that directly attacks the cause),
    so we keep ``wind_speed_100m`` as the magnitude and drop the now-redundant ``wind_speed_10m``.
    """
    alpha = shear_exponent(features["wind_speed_10m"], features["wind_speed_100m"])
    return features.assign(wind_shear_exponent=alpha.to_numpy()).drop(columns=["wind_speed_10m"])


def build_era5_and_outcome(scada_df: pd.DataFrame, *, test_wtg: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (ERA5-only feature frame, test-turbine power) over normally-operating, finite rows.

    ERA5 is synced to the SCADA grid via the reference-turbine mean wind speed (upgrade-invariant, the
    same signal the method's lag sweep locks onto). No synthetic upgrade is injected, so the whole
    window is baseline; the normal-operation filter drops curtailed/down rows that would otherwise
    corrupt the target. The 10m/100m speeds are folded into a vertical-shear exponent
    (:func:`add_shear_exponent`).
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    timebase = _infer_timebase(index)
    context = build_hot_v0_context(wtg_names=list(scada_df[HOT_COLUMNS.turbine].unique()))

    y = extract_outcome(
        scada_df, test_wtg=test_wtg, turbine_col=HOT_COLUMNS.turbine, active_power_col=HOT_COLUMNS.active_power
    )
    reference_ws = reference_mean_wind_speed(
        scada_df, test_wtg=test_wtg, turbine_col=HOT_COLUMNS.turbine, wind_speed_col=HOT_COLUMNS.wind_speed
    )
    synced = sync_era5(
        context.reanalysis_datasets[0].data, target_index=index, reference_ws=reference_ws, timebase=timebase
    )
    raw = era5_feature_frame(synced.aligned)
    logger.info("ERA5 synced: lag=%d rows, corr=%.3f", synced.best_lag_rows, synced.best_corr)

    test_rows = scada_df[scada_df[HOT_COLUMNS.turbine] == test_wtg].sort_index()
    keep = NormalOperationFilter(
        active_power_col=HOT_COLUMNS.active_power,
        wind_speed_col=HOT_COLUMNS.wind_speed,
        availability_col=HOT_COLUMNS.availability,
    ).keep_mask(test_rows, timebase=timebase)
    keep = keep[~keep.index.duplicated()].reindex(index, fill_value=False).to_numpy()
    # Finite-check on the raw ERA5 columns (all present in reanalysis) so the derived shear NaN on rare
    # calm rows does not shrink the row set — keeping this comparable to the pre-shear run.
    selected = keep & np.isfinite(y.to_numpy(dtype=float)) & raw.notna().all(axis=1).to_numpy()

    features = add_shear_exponent(raw)
    return features.loc[selected], y.loc[selected]


@dataclass
class RankingResult:
    """The ERA5 importance ranking plus the held-out slice it was scored on.

    :param table: one row per ERA5 feature with gain / gain_frac / permutation importance, gain-sorted
    :param y_valid: actual test power on the held-out slice (feeds the predicted-vs-actual gate)
    :param pred_valid: model prediction on the held-out slice
    :param n_train: rows the ranking model was fitted on
    """

    table: pd.DataFrame
    y_valid: np.ndarray
    pred_valid: np.ndarray
    n_train: int


def rank_features(features: pd.DataFrame, y: pd.Series, *, seed: int) -> RankingResult:
    """Rank ERA5 features by LightGBM gain + held-out permutation importance on one seeded split.

    Fits on a seeded 80% split; gain comes from the fitted booster, permutation importance and the
    held-out predictions (returned for the predicted-vs-actual gate) both come off the untouched 20%
    so nothing is scored in-sample.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_valid = max(1, len(y) // 5)
    valid_idx, train_idx = order[:n_valid], order[n_valid:]
    x_train, x_valid = features.iloc[train_idx], features.iloc[valid_idx]
    y_train, y_valid = y.to_numpy(dtype=float)[train_idx], y.to_numpy(dtype=float)[valid_idx]

    model = make_outcome_model(random_state=seed)
    model.fit(x_train, y_train)
    gain = model.booster_.feature_importance(importance_type="gain").astype(float)
    perm = permutation_importance(model, x_valid, y_valid, n_repeats=10, random_state=seed, scoring="r2")

    table = pd.DataFrame(
        {
            "feature": list(features.columns),
            "gain": gain,
            "gain_frac": gain / gain.sum() if gain.sum() else np.nan,
            "perm_importance": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values("gain", ascending=False, ignore_index=True)

    pred_valid = np.asarray(model.predict(x_valid), dtype=float)
    return RankingResult(table=table, y_valid=y_valid, pred_valid=pred_valid, n_train=len(y_train))


def _fit_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """R², RMSE, MAE over the finite pairs (the predicted-vs-actual gate numbers)."""
    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[finite], predicted[finite]
    resid = actual - predicted
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot else float("nan")
    return {"r2": r2, "rmse": float(np.sqrt(np.mean(resid**2))), "mae": float(np.mean(np.abs(resid)))}


def _select_matching_vars(table: pd.DataFrame) -> list[str]:
    """Suggested matching set: features above a fraction-of-top gain floor, capped at the preferred count."""
    if table.empty:
        return []
    floor = _SELECTION_FLOOR_FRAC * float(table["gain"].iloc[0])
    above = table[table["gain"] >= floor]["feature"].tolist()
    return above[:_PREFERRED_N]


def plot_feature_importance(table: pd.DataFrame, *, test_wtg: str, out_dir: Path, top_n: int = 20) -> None:
    """Gain and permutation-importance rankings side by side — the matching-variable selection view."""
    top = table.head(top_n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6.0, 0.4 * len(top))))
    axes[0].barh(top["feature"], top["gain"], color="C0")
    axes[0].set_xlabel("LightGBM gain")
    axes[0].set_title("gain importance")
    apply_grid(axes[0])
    perm_order = table.head(top_n).sort_values("perm_importance")
    axes[1].barh(
        perm_order["feature"], perm_order["perm_importance"], xerr=perm_order["perm_importance_std"], color="C1"
    )
    axes[1].set_xlabel("permutation importance (Δ R², held-out)")
    axes[1].set_title("permutation importance")
    apply_grid(axes[1])
    fig.suptitle(f"{test_wtg}: ERA5 → test power importance — match on the strongest, cheapest-to-match few")
    save_fig(fig, out_dir / "feature_importance.png")


def plot_predicted_vs_actual(y_valid: np.ndarray, pred_valid: np.ndarray, *, test_wtg: str, out_dir: Path) -> None:
    """Held-out predicted vs actual test power — the gate that the ERA5 model is good enough to trust."""
    metrics = _fit_metrics(y_valid, pred_valid)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    density_scatter(y_valid, pred_valid, ax=ax, s=6, colorbar=True)
    hi = float(np.nanmax(y_valid)) if len(y_valid) else 1.0
    ax.plot([0.0, hi], [0.0, hi], color="red", linewidth=1.2, label="1:1")
    ax.set_xlabel("actual test power [kW]")
    ax.set_ylabel("predicted test power [kW]")
    ax.set_title(
        f"{test_wtg}: ERA5-only held-out fit  "
        f"R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.0f} kW, MAE={metrics['mae']:.0f} kW, n={len(y_valid)}"
    )
    ax.legend(loc="upper left")
    apply_grid(ax)
    save_fig(fig, out_dir / "predicted_vs_actual.png")


def run(*, test_wtg: str, out_root: Path | None, seed: int) -> pd.DataFrame:
    """Load HoT + ERA5, rank the ERA5 fields for ``test_wtg``, and write the plots + ranking CSV."""
    out_dir = (out_root if out_root is not None else default_output_root()) / "inspection_era5_matching"
    out_dir.mkdir(parents=True, exist_ok=True)

    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    features, y = build_era5_and_outcome(scada_df, test_wtg=test_wtg)
    logger.info("Ranking %d ERA5 features on %d normally-operating rows for %s", features.shape[1], len(y), test_wtg)

    result = rank_features(features, y, seed=seed)
    table = result.table
    table.to_csv(out_dir / "era5_matching_importance.csv", index=False)
    plot_feature_importance(table, test_wtg=test_wtg, out_dir=out_dir)
    plot_predicted_vs_actual(result.y_valid, result.pred_valid, test_wtg=test_wtg, out_dir=out_dir)

    metrics = _fit_metrics(result.y_valid, result.pred_valid)
    logger.info(
        "Held-out ERA5→test-power fit: R²=%.3f, RMSE=%.0f kW, MAE=%.0f kW (n_train=%d, n_valid=%d)",
        metrics["r2"],
        metrics["rmse"],
        metrics["mae"],
        result.n_train,
        len(result.y_valid),
    )
    logger.info(
        "ERA5 feature ranking (top 12 by gain):\n%s",
        table.head(12)[["feature", "gain", "gain_frac", "perm_importance"]].round(4).to_string(index=False),
    )
    logger.info(
        "Suggested matching set (gain >= %.0f%% of top, capped at %d): %s — verify against the cell budget "
        "(Component 2) before hard-coding as the default.",
        100 * _SELECTION_FLOOR_FRAC,
        _PREFERRED_N,
        _select_matching_vars(table),
    )
    logger.info("Wrote ERA5 matching-importance outputs to %s", out_dir)
    return table


def main() -> None:
    """CLI: rank the ERA5 fields for one test turbine and write the analysis outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--test-wtg",
        default=DEFAULT_TURBINE_SUBSET[0],
        choices=DEFAULT_TURBINE_SUBSET,
        help="test turbine whose power is the prediction target (default: the first study turbine)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="base output dir; the run writes under <out-root>/inspection_era5_matching "
        "(default: the study output root)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the train/valid split and the model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    run(test_wtg=args.test_wtg, out_root=args.out_root.expanduser() if args.out_root else None, seed=args.seed)


if __name__ == "__main__":
    main()
