"""The v0 binned power-curve method behind the harness ``Method`` seam.

``V0BinnedMethod`` adapts a thin harness ``MethodInput`` to a full, faithful wind_up
pre/post power-performance run and returns its P50 uplift. For each campaign it renders a
per-campaign YAML and loads it with :meth:`WindUpConfig.from_yaml` (the standard way v0
assessments are configured), then runs the real ``run_wind_up_analysis`` + ``combine_results``.

Configuration is deliberately faithful: plots off, bootstrap untouched (its uncertainty is
required by ``combine_results``), ``ignore_turbine_anemometer_data`` / ``clip_rated_power_pp``
at v0 defaults. Two choices are specific to this benchmarking exercise:

* ``use_lt_distribution: False`` — we study *campaign* uplift (the injected ground truth),
  not long-term uplift.
* ``combine_results(..., auto_choose_refs=False)`` — an analyst normally reviews/chooses refs.

With ``years_offset_for_pre_period: 1`` and ``years_for_{lt_distribution,detrend}: 1``,
``from_yaml`` derives a seasonally-matched pre period (the post window shifted back one year)
and a one-year detrend window; the harness only provides ~12 months of pre data, so the pre
and detrend windows are constrained accordingly — the campaign-length degradation the harness
exists to measure. Prepost mode only; a ``ToggleSchedule`` input raises ``NotImplementedError``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import ToggleSchedule
from wind_up.combine_results import combine_results
from wind_up.constants import DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig

if TYPE_CHECKING:
    from benchmarking.baselines.hot_context import HotV0Context

_CAMPAIGN_YAML_TEMPLATE = """\
assessment_name: {assessment_name}
test_wtgs:
  - {test_wtg}
ref_wtgs:
{ref_lines}
upgrade_first_dt_utc_start: {upgrade}
analysis_last_dt_utc_start: {analysis_last}
years_offset_for_pre_period: 1
years_for_lt_distribution: 1
years_for_detrend: 1
use_lt_distribution: false
ws_bin_width: {ws_bin_width}
reanalysis_method: {reanalysis_method}
optimize_northing_corrections: false
northing_corrections_utc: !include {northing_yaml}
asset: !include {asset_yaml}
"""


def _subset_turbines(scada_df: pd.DataFrame) -> list[str]:
    """Return the sorted unique turbine names present in ``scada_df``."""
    return sorted(scada_df[DataColumns.turbine_name].unique().tolist())


def _extract_p50(tdf: pd.DataFrame, test_wtg: str) -> float:
    """Return the combined P50 uplift fraction for the (non-reference) test turbine."""
    row = tdf.loc[(tdf["test_wtg"] == test_wtg) & (~tdf["is_ref"])]
    if len(row) != 1:
        msg = f"expected exactly one non-ref combined result row for test_wtg {test_wtg!r}, found {len(row)}"
        raise ValueError(msg)
    return float(row["p50_uplift"].iloc[0])


@dataclass
class V0BinnedMethod:
    """Pluggable v0 binned power-curve baseline.

    :param context: the shared HoT source-context (metadata, reanalysis, vendored asset/northing)
    :param name: method name shown in the leaderboard
    :param ws_bin_width: power-curve wind-speed bin width in m/s
    :param reanalysis_method: wind_up reanalysis-node selection method
    :param scratch_dir: where per-campaign YAML + wind_up output go; a temp dir when ``None``
    :param save_plots: if True, save wind_up's per-campaign plots under ``<out_dir>/plots`` (each
        campaign has its own unique out dir); off by default since plots are slow and unused for
        scoring, but useful for manually inspecting a run
    """

    context: HotV0Context
    name: str = "v0_binned"
    ws_bin_width: float = 1.0
    reanalysis_method: str = "node_with_best_ws_corr"
    scratch_dir: Path | None = None
    save_plots: bool = False

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Run a faithful v0 pre/post analysis for one campaign and return its P50 uplift."""
        if isinstance(mi.upgrade_timing, ToggleSchedule):
            msg = "V0BinnedMethod supports prepost mode only; toggle schedules are not implemented"
            raise NotImplementedError(msg)

        cfg = self._build_config(mi)
        plot_cfg = PlotConfig(show_plots=False, save_plots=self.save_plots, plots_dir=cfg.out_dir / "plots")
        inputs = AssessmentInputs.from_cfg(
            cfg=cfg,
            plot_cfg=plot_cfg,
            scada_df=mi.scada_df,
            metadata_df=self.context.metadata_df,
            reanalysis_datasets=self.context.reanalysis_datasets,
            cache_dir=None,
        )
        trdf = run_wind_up_analysis(inputs)
        tdf = combine_results(trdf, auto_choose_refs=False, plot_config=None)
        tdf.to_csv(
            cfg.out_dir
            / f"{cfg.assessment_name}_combined_results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        return MethodOutput(p50_overall=_extract_p50(tdf, mi.test_wtg))

    def _build_config(self, mi: MethodInput) -> WindUpConfig:
        """Render and load the per-campaign WindUpConfig, with the asset filtered to the subset."""
        subset = _subset_turbines(mi.scada_df)
        refs = [t for t in subset if t != mi.test_wtg]
        if not refs:
            msg = (
                f"no reference turbines available for test_wtg {mi.test_wtg!r}: scada_df contains only "
                f"{subset}. The v0 binned method needs at least one reference turbine."
            )
            raise ValueError(msg)
        upgrade = pd.Timestamp(mi.upgrade_timing)
        analysis_last = pd.Timestamp(mi.scada_df.index.max())
        assessment_name = f"v0_{mi.test_wtg}_{upgrade:%Y%m%d}_{analysis_last:%Y%m%d}"

        scratch = Path(self.scratch_dir) if self.scratch_dir is not None else Path(tempfile.mkdtemp(prefix="v0_"))
        scratch.mkdir(parents=True, exist_ok=True)
        yaml_text = _CAMPAIGN_YAML_TEMPLATE.format(
            assessment_name=assessment_name,
            test_wtg=mi.test_wtg,
            ref_lines="\n".join(f"  - {r}" for r in refs),
            upgrade=upgrade.strftime("%Y-%m-%d %H:%M:%S"),
            analysis_last=analysis_last.strftime("%Y-%m-%d %H:%M:%S"),
            ws_bin_width=self.ws_bin_width,
            reanalysis_method=self.reanalysis_method,
            northing_yaml=self.context.northing_yaml.as_posix(),
            asset_yaml=self.context.asset_yaml.as_posix(),
        )
        yaml_path = scratch / f"{assessment_name}.yaml"
        yaml_path.write_text(yaml_text)

        cfg = WindUpConfig.from_yaml(yaml_path)
        cfg.out_dir = scratch / assessment_name
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        # The asset YAML lists all 21 HoT turbines, but only the subset has SCADA here. Restrict
        # it so wind farm coverage (e.g. reanalysis correlation) is computed over the right count.
        cfg.asset.wtgs = [w for w in cfg.asset.wtgs if w.name in subset]
        # similar subsetting logic for northing_corrections_utc
        cfg.northing_corrections_utc = [n for n in cfg.northing_corrections_utc if n[0] in subset]
        return cfg
