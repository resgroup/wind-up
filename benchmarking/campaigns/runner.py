"""The benchmark layer over the truth-free core: the same run, scored against the known truth."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.campaigns.data_screen import DEFAULT_DATA_SCREEN, DataScreenSettings
from benchmarking.campaigns.run import CampaignReport, estimate_campaign, visible_mask
from benchmarking.harness import CampaignWindow, Replicate, score_output, truth_mask
from benchmarking.harness.northing import DEFAULT_NORTHING_ROLES
from wind_up.analysis_period import DEFAULT_PLAN_SETTINGS
from wind_up.northing import DEFAULT_NORTHING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import numpy as np

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method, MethodOutput
    from benchmarking.synthetic import SyntheticDataset
    from wind_up import FarmUplift
    from wind_up.analysis_period import AnalysisPlan, PlanSettings
    from wind_up.northing import NorthingSettings


# The farm table's columns, named here so a campaign with nothing to aggregate still returns a
# frame consumers can index.
_FARM_COLUMNS = (
    "method",
    "estimate",
    "truth",
    "signed_error",
    "uplift_spread",
    "n_guarded",
    "reference_uplift_before_screen",
    "reference_uplift_after_screen",
    "reference_uplift_spread",
)


@dataclass
class CampaignResult:
    """Everything one campaign run produced, truth included.

    :param spec: the campaign that was run
    :param scores: the tidy harness rows, one set per upgraded turbine at n=1
    :param farm: one row per method -- ``estimate``, ``truth``, ``signed_error``, ``uplift_spread``,
        ``n_guarded`` and the combined reference uplift before and after the screen
    :param farm_uplifts: each method's :class:`~wind_up.FarmUplift`, including per-turbine detail
    :param truth_farm_uplift: the exact pooled farm truth
    :param outputs: each ``(method, turbine)``'s raw :class:`~benchmarking.harness.MethodOutput`
    :param report: the truth-free :class:`~benchmarking.campaigns.run.CampaignReport` this run is
        built on -- the analyst-facing tables, carrying no truth
    """

    spec: CampaignSpec
    scores: pd.DataFrame
    farm: pd.DataFrame
    farm_uplifts: dict[str, FarmUplift]
    truth_farm_uplift: float
    outputs: dict[tuple[str, str], MethodOutput]
    report: CampaignReport

    @property
    def unplanned(self) -> dict[str, str]:
        """Each upgraded turbine the period selector could not plan, with its reason; truth leaves them out too."""
        return self.report.unplanned


class CampaignRunner:
    """Turn a campaign spec plus its generated dataset into per-turbine and farm results.

    :param spec: the public campaign facts; methods see nothing else
    :param dataset: the generated dataset, whose ``original_df`` supplies the truth
    :param build_methods: given an upgraded turbine's name, the methods to run for it
    :param era5_wd: reanalysis wind direction covering the campaign, the anchor the shared northing
        step discovers against. Required when ``spec.north_offsets`` is ``None``; a declared table
        needs none.
    :param northing_roles: the direction roles the shared step corrects
    :param northing_settings: how the shared step's changepoint search is bounded
    :param northing_out_dir: where the shared step writes its table and plots when it discovers corrections
    :param northing_plots: write the shared step's plots as well as its table
    :param plan_settings: how a planning campaign's spans and power references are chosen
    :param possible_changes: each turbine's indicators of an undeclared change, reported as warnings
        against the plans they fall inside
    :param data_screen: how each plan's power references are screened on their data; ``None`` skips it
    """

    def __init__(
        self,
        spec: CampaignSpec,
        dataset: SyntheticDataset,
        *,
        build_methods: Callable[[str], list[Method]],
        era5_wd: pd.Series | None = None,
        northing_roles: Sequence[str] = DEFAULT_NORTHING_ROLES,
        northing_settings: NorthingSettings = DEFAULT_NORTHING,
        northing_out_dir: Path | None = None,
        northing_plots: bool = True,
        plan_settings: PlanSettings = DEFAULT_PLAN_SETTINGS,
        possible_changes: Mapping[str, Sequence[pd.Timestamp]] | None = None,
        data_screen: DataScreenSettings | None = DEFAULT_DATA_SCREEN,
    ) -> None:
        """Store the campaign, its data and the per-turbine method factory."""
        self._spec = spec
        self._dataset = dataset
        self._build_methods = build_methods
        self._era5_wd = era5_wd
        self._northing_roles = tuple(northing_roles)
        self._northing_settings = northing_settings
        self._northing_out_dir = northing_out_dir
        self._northing_plots = northing_plots
        self._plan_settings = plan_settings
        self._possible_changes = possible_changes
        self._data_screen = data_screen

    def run(self) -> CampaignResult:
        """Estimate the campaign on the truth-free core, then score what it produced against truth."""
        spec = self._spec
        report = estimate_campaign(
            spec,
            self._dataset.synthetic_df,
            build_methods=self._build_methods,
            columns=self._dataset.columns,
            era5_wd=self._era5_wd,
            northing_roles=self._northing_roles,
            northing_settings=self._northing_settings,
            northing_out_dir=self._northing_out_dir,
            northing_plots=self._northing_plots,
            plan_settings=self._plan_settings,
            possible_changes=self._possible_changes,
            data_screen=self._data_screen,
        )
        visible = self._visible_dataset(report)

        score_rows: list[dict[str, object]] = []
        truth_masks: dict[str, np.ndarray] = {}
        analysed = sorted(t for t in spec.upgraded_turbines if t not in report.unplanned)
        for wtg in analysed:
            window = self._window(report.plans[wtg])
            replicate = Replicate(
                dataset=visible,
                test_wtg=wtg,
                treatment_start=window.treatment_start,
                upgrade_timing=spec.timing_for(wtg),
            )
            synthetic = visible.synthetic_df
            test_index = pd.DatetimeIndex(synthetic.index[(synthetic[spec.turbine_col] == wtg).to_numpy()])
            mask = truth_mask(replicate, window) & spec.usable_mask(wtg, test_index)
            truth_masks[wtg] = mask
            truth = replicate.true_uplift(mask=mask).overall
            for (method_name, turbine), output in report.outputs.items():
                if turbine != wtg:
                    continue
                score_rows.extend(
                    score_output(
                        output,
                        method_name=method_name,
                        replicate=replicate,
                        window=window,
                        truth=truth,
                        mask=mask,
                        profile_name=spec.change_label(),
                        wall_time_s=report.wall_time_s[method_name, turbine],
                    )
                )

        truth_farm = visible.true_farm_uplift(test_wtgs=analysed, masks=truth_masks)
        references = report.farm.set_index("method")
        farm = pd.DataFrame(
            [
                self._farm_row(name, result, visible=visible, masks=truth_masks, references=references.loc[name])
                for name, result in report.farm_uplifts.items()
            ],
            columns=_FARM_COLUMNS,
        )
        return CampaignResult(
            spec=spec,
            scores=pd.DataFrame(score_rows),
            farm=farm,
            farm_uplifts=report.farm_uplifts,
            truth_farm_uplift=truth_farm,
            outputs=report.outputs,
            report=report,
        )

    def _farm_row(
        self,
        method: str,
        result: FarmUplift,
        *,
        visible: SyntheticDataset,
        masks: dict[str, np.ndarray],
        references: pd.Series,
    ) -> dict[str, object]:
        """One method's farm row, with its truth pooled over the turbines it actually used.

        A guard can drop a turbine from a method's estimate; pooling the truth over every
        upgraded turbine would then compare two different estimands. ``n_guarded`` flags the rows
        where that happened, since a method that dropped turbines is not directly comparable with
        one that used them all.
        """
        used = [str(w) for w in result.turbines.loc[result.turbines["used"], "turbine"]]
        truth = visible.true_farm_uplift(test_wtgs=used, masks={w: masks[w] for w in used})
        return {
            "method": method,
            "estimate": result.uplift,
            "truth": truth,
            "signed_error": result.uplift - truth,
            "uplift_spread": result.uplift_spread,
            "n_guarded": int((result.turbines["guard"] != "").sum()),
            "reference_uplift_before_screen": float(references["reference_uplift_before_screen"]),
            "reference_uplift_after_screen": float(references["reference_uplift_after_screen"]),
            "reference_uplift_spread": float(references["reference_uplift_spread"]),
        }

    def _visible_dataset(self, report: CampaignReport) -> SyntheticDataset:
        """Pair the frame the methods saw with the matching slice of the ground-truth original."""
        return replace(
            self._dataset,
            synthetic_df=report.scada_df,
            original_df=self._dataset.original_df[visible_mask(self._spec, self._dataset.original_df)],
        )

    def _window(self, plan: AnalysisPlan) -> CampaignWindow:
        """Return one window spanning the planned turbine's campaign, so the harness scores it at n=1.

        The window is the plan's span, treated from the start of its after period. ``length`` is
        the activity span in whole months; it labels the result rows and is not used to select records.
        """
        start, end = plan.start, plan.end
        treatment_start = plan.after[0]
        months = (end.year - treatment_start.year) * 12 + (end.month - treatment_start.month)
        return CampaignWindow(
            length=months,
            unit="months",
            baseline_start=start,
            treatment_start=treatment_start,
            activity_end=end,
        )


def per_turbine_table(result: CampaignResult) -> pd.DataFrame:
    """Return the per-turbine headline rows: one per method and upgraded turbine."""
    overall = result.scores[result.scores["condition"] == "overall"]
    return overall[["method", "test_wtg", "estimate", "truth", "signed_error"]].reset_index(drop=True)
