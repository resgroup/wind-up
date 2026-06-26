"""``RLearnerMethod``: the cross-fit R-learner behind the harness ``Method`` seam.

Orchestrates the pieces (design note §4): build upgrade-invariant reference features, sync ERA5
(optional), filter the test turbine to normal operation, cross-fit the R-learner, aggregate to a
single P50 uplift, and write diagnostics. It is v0-independent — ERA5 is supplied as a plain
hourly DataFrame (the driver fetches it), so this package imports nothing from ``wind_up``.

Aggregation: the overall uplift is ``sum(tau) / sum(mu0)`` over the **upgraded** rows, which
equals the ground-truth definition ``sum(upgraded power)/sum(baseline power) - 1`` because
``mu0`` is the baseline expected power and ``mu0 + tau`` the upgraded.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarking.baselines.rlearner import diagnostics as diag
from benchmarking.baselines.rlearner.era5_sync import sync_era5
from benchmarking.baselines.rlearner.features import (
    QUALIFIER,
    build_reference_features,
    era5_features,
    extract_outcome_and_treatment,
)
from benchmarking.baselines.rlearner.filtering import NormalOperationFilter
from benchmarking.baselines.rlearner.nuisance import make_effect_model, make_outcome_model, make_propensity_model
from benchmarking.baselines.rlearner.rlearner import cross_fit_rlearner
from benchmarking.diagnostics import DiagnosticContext, write_common_diagnostics, write_run_config
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

_MIN_FOLDS = 2
_MIN_POINTS_FOR_TIMEBASE = 2


def _infer_timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer the analysis timebase as the median spacing of the sorted unique timestamps."""
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    if len(unique) < _MIN_POINTS_FOR_TIMEBASE:
        return pd.Timedelta(minutes=10)
    return pd.Timedelta(np.median(np.diff(unique.to_numpy())))


def _upgrade_start(upgrade_timing: pd.Timestamp | ToggleSchedule, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the upgrade-start timestamp (changeover for prepost; toggle origin for toggle)."""
    if isinstance(upgrade_timing, ToggleSchedule):
        return upgrade_timing.start if upgrade_timing.start is not None else index.min()
    return pd.Timestamp(upgrade_timing)


def _restrict_to_campaign(mi: MethodInput, *, toggle_campaign_only: bool) -> MethodInput:
    """Drop pre-campaign rows for a toggle campaign so the on/off comparison shares a distribution.

    The harness toggle window also carries the pre-campaign baseline, whose distribution differs
    from the campaign and reintroduces the temporal confounding toggling exists to avoid. When
    ``toggle_campaign_only`` (the default), restrict a toggle input to records at/after the toggle
    start (the interleaved on/off blocks), giving a balanced propensity and pure variance
    reduction. No-op for prepost and when the flag is off.
    """
    timing = mi.upgrade_timing
    if not (toggle_campaign_only and isinstance(timing, ToggleSchedule) and timing.start is not None):
        return mi
    return replace(mi, scada_df=mi.scada_df.loc[mi.scada_df.index >= timing.start])


@dataclass
class RLearnerMethod:
    """Pluggable cross-fit R-learner uplift estimator (prepost and toggle).

    :param active_power_col: the test turbine's active-power column (the outcome ``Y``)
    :param wind_speed_col: the wind-speed tag, used for ERA5 sync (reference mean) and the
        stuck-filter low-wind exemption; required if ``era5_hourly_df`` is given
    :param availability_col: optional "ready to operate" counter for the downtime filter
    :param era5_hourly_df: optional raw hourly ERA5 (Open-Meteo columns); added as features when given
    :param columns: source-native column schema, used only by the shared diagnostics (not estimation)
    :param name: method name shown in the leaderboard
    :param out_dir: where per-run folders are written; a temp dir when ``None``
    :param save_plots: also write the diagnostic plots
    :param n_folds: cross-fitting folds
    :param seed: cross-fitting seed
    :param model_params: LightGBM overrides passed to every nuisance/effect model
    :param timebase: analysis timebase; inferred from the data when ``None``
    :param toggle_campaign_only: for a toggle campaign, fit only on the interleaved on/off blocks
        (drop the pre-campaign baseline) so the propensity is balanced and there is no temporal
        confounding; no-op for prepost (whose baseline is the pre-campaign data)
    """

    active_power_col: str
    wind_speed_col: str | None = None
    availability_col: str | None = None
    era5_hourly_df: pd.DataFrame | None = None
    columns: ColumnSchema = HOT_COLUMNS
    name: str = "rlearner"
    out_dir: Path | None = None
    save_plots: bool = False
    n_folds: int = 5
    seed: int = 0
    model_params: dict[str, Any] = field(default_factory=dict)
    timebase: pd.Timedelta | None = None
    toggle_campaign_only: bool = True

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate the test turbine's P50 uplift for one campaign and write diagnostics."""
        mi = _restrict_to_campaign(mi, toggle_campaign_only=self.toggle_campaign_only)
        scada = mi.scada_df
        index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
        timebase = self.timebase if self.timebase is not None else _infer_timebase(scada.index)
        n_refs = scada[mi.turbine_col].nunique() - 1

        y, t = extract_outcome_and_treatment(
            scada,
            test_wtg=mi.test_wtg,
            turbine_col=mi.turbine_col,
            active_power_col=self.active_power_col,
            upgrade_timing=mi.upgrade_timing,
        )
        features = build_reference_features(scada, test_wtg=mi.test_wtg, turbine_col=mi.turbine_col)
        reference_ws = self._reference_mean_ws(features, index=index)
        own_ws = self._own_ws(scada, mi=mi, index=index)
        features, era5 = self._add_era5(features, index=index, reference_ws=reference_ws, timebase=timebase)

        selected = self._select_rows(scada, mi=mi, index=index, y=y, timebase=timebase)
        x_sel = features.loc[index[selected]]
        y_sel = y.to_numpy(dtype=float)[selected]
        t_sel = t.to_numpy(dtype=float)[selected]
        n_folds = min(self.n_folds, int(selected.sum()))
        if n_folds < _MIN_FOLDS:
            msg = f"too few normally-operating rows ({int(selected.sum())}) to cross-fit the R-learner."
            raise ValueError(msg)

        fit = cross_fit_rlearner(x_sel, y=y_sel, t=t_sel, n_folds=n_folds, seed=self.seed, **self._factories())
        overall = _aggregate_uplift(tau=fit.tau, mu0=fit.mu0, upgraded=t_sel.astype(bool))

        self._write(
            mi,
            index=index,
            timebase=timebase,
            t=t,
            selected=selected,
            y=y,
            fit=fit,
            x_sel=x_sel,
            own_ws=own_ws,
            overall=overall,
            n_refs=n_refs,
            era5=era5,
        )
        return MethodOutput(p50_overall=overall)

    def _factories(self) -> dict[str, Callable[[], Any]]:
        """Nuisance/effect model factories with the configured LightGBM overrides applied."""
        params = self.model_params
        return {
            "make_outcome": lambda: make_outcome_model(**params),
            "make_propensity": lambda: make_propensity_model(**params),
            "make_effect": lambda: make_effect_model(**params),
        }

    def _reference_mean_ws(self, features: pd.DataFrame, *, index: pd.DatetimeIndex) -> pd.Series:
        """Mean wind speed across reference turbines (used only for ERA5 lag sync)."""
        if self.wind_speed_col is None:
            return pd.Series(np.nan, index=index)
        ws_cols = [c for c in features.columns if c.startswith(f"{self.wind_speed_col}{QUALIFIER}")]
        if not ws_cols:
            return pd.Series(np.nan, index=index)
        return features[ws_cols].mean(axis=1)

    def _own_ws(self, scada: pd.DataFrame, *, mi: MethodInput, index: pd.DatetimeIndex) -> np.ndarray | None:
        """Return the test turbine's own wind speed aligned to ``index`` (for SCADA diagnostics), or None."""
        if self.wind_speed_col is None:
            return None
        test_rows = scada[scada[mi.turbine_col] == mi.test_wtg]
        series = pd.Series(
            test_rows[self.wind_speed_col].to_numpy(dtype=float), index=pd.DatetimeIndex(test_rows.index)
        )
        return series[~series.index.duplicated()].reindex(index).to_numpy(dtype=float)

    def _add_era5(
        self, features: pd.DataFrame, *, index: pd.DatetimeIndex, reference_ws: pd.Series, timebase: pd.Timedelta
    ) -> tuple[pd.DataFrame, Any]:
        """Sync ERA5 (if supplied) and append its features; return the (features, sync-result)."""
        if self.era5_hourly_df is None:
            return features, None
        if self.wind_speed_col is None:
            msg = "wind_speed_col is required to sync ERA5 (it provides the reference wind speed)."
            raise ValueError(msg)
        result = sync_era5(self.era5_hourly_df, target_index=index, reference_ws=reference_ws, timebase=timebase)
        return pd.concat([features, era5_features(result.aligned)], axis=1), result

    def _select_rows(
        self, scada: pd.DataFrame, *, mi: MethodInput, index: pd.DatetimeIndex, y: pd.Series, timebase: pd.Timedelta
    ) -> np.ndarray:
        """Boolean over ``index``: normally-operating test rows (cause-not-effect) with finite outcome."""
        test_rows = scada[scada[mi.turbine_col] == mi.test_wtg].sort_index()
        keep = NormalOperationFilter(
            active_power_col=self.active_power_col,
            wind_speed_col=self.wind_speed_col,
            availability_col=self.availability_col,
        ).keep_mask(test_rows, timebase=timebase)
        keep = keep.reindex(index, fill_value=False)
        return keep.to_numpy() & np.isfinite(y.to_numpy(dtype=float))

    def _write(
        self,
        mi: MethodInput,
        *,
        index: pd.DatetimeIndex,
        timebase: pd.Timedelta,
        t: pd.Series,
        selected: np.ndarray,
        y: pd.Series,
        fit: Any,  # noqa: ANN401
        x_sel: pd.DataFrame,
        own_ws: np.ndarray | None,
        overall: float,
        n_refs: int,
        era5: Any,  # noqa: ANN401
    ) -> None:
        """Assemble the diagnostic data and write the CSVs (+ plots), logging the top features."""
        upgrade_start = _upgrade_start(mi.upgrade_timing, index)
        run_name = f"rlearner_{mi.test_wtg}_{upgrade_start:%Y%m%d}_{index.max():%Y%m%d}"
        out_root = Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="rlearner_"))
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        y_arr = y.to_numpy(dtype=float)
        data = diag.DiagnosticData(
            test_wtg=mi.test_wtg,
            mode="toggle" if isinstance(mi.upgrade_timing, ToggleSchedule) else "prepost",
            index=index,
            treated_all=t.to_numpy(),
            selected_all=selected,
            y_all=y_arr,
            timebase=timebase,
            tau=fit.tau,
            m_hat=fit.m_hat,
            e_hat=fit.e_hat,
            mu0=fit.mu0,
            y_selected=y_arr[selected],
            condition_ws=own_ws[selected] if own_ws is not None else None,
            condition_ws_label=f"{self.wind_speed_col} @ {mi.test_wtg}" if self.wind_speed_col is not None else None,
            feature_names=list(x_sel.columns),
            feature_values=x_sel,
            outcome_model=fit.outcome_model,
            propensity_model=fit.propensity_model,
            effect_model=fit.effect_model,
            overall_uplift=overall,
            n_refs=n_refs,
            era5_lag_rows=era5.best_lag_rows if era5 is not None else None,
            era5_corr=era5.best_corr if era5 is not None else None,
            era5_sweep=era5.sweep if era5 is not None else None,
        )
        importance = diag.write_csvs(run_dir, run_name, ts, data)
        diag.log_top_features(importance)
        if self.save_plots:
            diag.save_plots(run_dir / "plots", data, importance)
            self._write_shared_diagnostics(mi, run_dir=run_dir, t=t, selected=selected, timebase=timebase, era5=era5)

    def _write_shared_diagnostics(
        self,
        mi: MethodInput,
        *,
        run_dir: Path,
        t: pd.Series,
        selected: np.ndarray,
        timebase: pd.Timedelta,
        era5: Any,  # noqa: ANN401
    ) -> None:
        """Emit the shared cross-method diagnostics (coverage/curves/histograms) and the run config."""
        # Align the schema's active-power / wind-speed roles to the columns this method was
        # configured to read, so the shared plots use the right columns even if they differ.
        columns = replace(self.columns, active_power=self.active_power_col)
        if self.wind_speed_col is not None:
            columns = replace(columns, wind_speed=self.wind_speed_col)
        ctx = DiagnosticContext(
            run_dir=run_dir,
            test_wtg=mi.test_wtg,
            turbine_col=mi.turbine_col,
            columns=columns,
            scada_df=mi.scada_df,
            treated_ts=t.to_numpy().astype(bool),
            used_ts=np.asarray(selected, dtype=bool),
            timebase=timebase,
            mode="toggle" if isinstance(mi.upgrade_timing, ToggleSchedule) else "prepost",
            era5_df=era5.aligned if era5 is not None else None,
        )
        write_common_diagnostics(ctx)
        extra = {
            "n_folds": self.n_folds,
            "seed": self.seed,
            "era5_lag_rows": era5.best_lag_rows if era5 is not None else None,
            "era5_corr": era5.best_corr if era5 is not None else None,
        }
        write_run_config(ctx, method_name=self.name, method_params=self._config_params(), extra=extra)

    def _config_params(self) -> dict[str, Any]:
        """Return the R-learner configuration recorded in the run-config YAML."""
        return {
            "active_power_col": self.active_power_col,
            "wind_speed_col": self.wind_speed_col,
            "availability_col": self.availability_col,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "toggle_campaign_only": self.toggle_campaign_only,
            "has_era5": self.era5_hourly_df is not None,
            "model_params": self.model_params,
        }


def _aggregate_uplift(*, tau: np.ndarray, mu0: np.ndarray, upgraded: np.ndarray) -> float:
    """Overall uplift = sum(tau)/sum(mu0) over the upgraded rows (the ground-truth row set)."""
    if not upgraded.any():
        return float("nan")
    denom = float(np.sum(mu0[upgraded]))
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return float(np.sum(tau[upgraded]) / denom)
