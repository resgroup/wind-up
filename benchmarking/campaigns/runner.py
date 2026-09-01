"""Run a declared campaign: per-turbine estimates, one farm headline, both output shapes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.harness import CampaignWindow, Replicate, score_one, truth_mask
from wind_up import TurbineUplift, farm_uplift

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method, MethodInput, MethodOutput
    from benchmarking.synthetic import SyntheticDataset
    from wind_up import FarmUplift


# The farm table's columns, named here so a campaign with nothing to aggregate still returns a
# frame consumers can index.
_FARM_COLUMNS = ("method", "estimate", "truth", "signed_error", "uplift_spread", "n_guarded")


@dataclass
class CampaignResult:
    """Everything one campaign run produced.

    :param spec: the campaign that was run
    :param scores: the tidy harness rows, one set per upgraded turbine at n=1
    :param farm: one row per method -- ``estimate``, ``truth``, ``signed_error``, ``uplift_spread``
        and ``n_guarded``
    :param farm_uplifts: each method's :class:`~wind_up.FarmUplift`, including per-turbine detail
    :param truth_farm_uplift: the exact pooled farm truth
    :param outputs: each ``(method, turbine)``'s raw :class:`~benchmarking.harness.MethodOutput`
    """

    spec: CampaignSpec
    scores: pd.DataFrame
    farm: pd.DataFrame
    farm_uplifts: dict[str, FarmUplift]
    truth_farm_uplift: float
    outputs: dict[tuple[str, str], MethodOutput]


class _Capturing:
    """Delegates to a method and keeps its output, so one estimate call serves both output shapes."""

    def __init__(self, method: Method) -> None:
        self._method = method
        self.name = method.name
        self.output: MethodOutput | None = None

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate via the wrapped method, retaining the output."""
        self.output = self._method.estimate(mi)
        return self.output


class CampaignRunner:
    """Turn a campaign spec plus its generated dataset into per-turbine and farm results.

    :param spec: the public campaign facts; methods see nothing else
    :param dataset: the generated dataset, whose ``original_df`` supplies the truth
    :param build_methods: given an upgraded turbine's name, the methods to run for it
    """

    def __init__(
        self,
        spec: CampaignSpec,
        dataset: SyntheticDataset,
        *,
        build_methods: Callable[[str], list[Method]],
    ) -> None:
        """Store the campaign, its data and the per-turbine method factory."""
        self._spec = spec
        self._dataset = dataset
        self._build_methods = build_methods

    def run(self) -> CampaignResult:
        """Run every applicable method on every upgraded turbine and aggregate to one headline."""
        spec = self._spec
        visible = self._visible_dataset()
        window = self._window()

        score_rows: list[dict[str, object]] = []
        outputs: dict[tuple[str, str], MethodOutput] = {}
        estimates: dict[str, list[TurbineUplift]] = {}
        truth_masks: dict[str, np.ndarray] = {}

        for wtg in spec.upgraded_turbines:
            replicate = Replicate(
                dataset=visible,
                test_wtg=wtg,
                treatment_start=spec.treatment_start,
                upgrade_timing=spec.timing_for(wtg),
            )
            mask = truth_mask(replicate, window)
            truth_masks[wtg] = mask
            truth = replicate.true_uplift(mask=mask).overall
            energy, n_records = self._actual_energy(visible, turbine=wtg, mask=mask)

            for method in self._build_methods(wtg):
                capturing = _Capturing(method)
                score_rows.extend(
                    score_one(
                        capturing,
                        replicate=replicate,
                        window=window,
                        truth=truth,
                        mask=mask,
                        profile_name=spec.change_label(),
                    )
                )
                if capturing.output is None:  # pragma: no cover - score_one always estimates
                    msg = f"{method.name} produced no output for {wtg}"
                    raise RuntimeError(msg)
                outputs[method.name, wtg] = capturing.output
                estimates.setdefault(method.name, []).append(
                    TurbineUplift(
                        turbine=wtg,
                        uplift=capturing.output.p50_overall,
                        actual_energy=energy,
                        n_records=n_records,
                        rated_power_kw=spec.rated_power_kw,
                    )
                )

        truth_farm = visible.true_farm_uplift(test_wtgs=list(spec.upgraded_turbines), masks=truth_masks)
        farm_uplifts = {name: farm_uplift(rows) for name, rows in estimates.items()}
        farm = pd.DataFrame(
            [
                {
                    "method": name,
                    "estimate": result.uplift,
                    "truth": truth_farm,
                    "signed_error": result.uplift - truth_farm,
                    "uplift_spread": result.uplift_spread,
                    "n_guarded": int((result.turbines["guard"] != "").sum()),
                }
                for name, result in farm_uplifts.items()
            ],
            columns=_FARM_COLUMNS,
        )
        return CampaignResult(
            spec=spec,
            scores=pd.DataFrame(score_rows),
            farm=farm,
            farm_uplifts=farm_uplifts,
            truth_farm_uplift=truth_farm,
            outputs=outputs,
        )

    def _visible_dataset(self) -> SyntheticDataset:
        """Return the dataset cut to what a method may see: analysis period, usable turbines only."""
        synthetic = self._dataset.synthetic_df
        keep = self._visible_mask(synthetic)
        return replace(
            self._dataset,
            synthetic_df=synthetic[keep],
            original_df=self._dataset.original_df[self._visible_mask(self._dataset.original_df)],
        )

    def _visible_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows of ``frame`` inside the analysis period whose turbine may be used."""
        spec = self._spec
        start, end = spec.analysis_period
        keep = np.asarray((frame.index >= start) & (frame.index < end))
        turbines = frame[spec.turbine_col].to_numpy()
        for turbine in pd.unique(turbines):
            is_turbine = turbines == turbine
            rows = pd.DatetimeIndex(frame.index[is_turbine])
            keep[is_turbine] &= spec.usable_mask(str(turbine), rows)
        return keep

    def _window(self) -> CampaignWindow:
        """Return one window spanning the whole campaign, so the harness scores it at n=1.

        ``length`` is the activity span in whole months; it labels the result rows and is not
        used to select records.
        """
        start, end = self._spec.analysis_period
        treatment_start = self._spec.treatment_start
        months = (end.year - treatment_start.year) * 12 + (end.month - treatment_start.month)
        return CampaignWindow(
            length=months,
            unit="months",
            baseline_start=start,
            treatment_start=treatment_start,
            activity_end=end,
        )

    def _actual_energy(self, dataset: SyntheticDataset, *, turbine: str, mask: np.ndarray) -> tuple[float, int]:
        """Return the energy one turbine actually produced over its upgraded records.

        Finite records only, with the record count that sum covers.
        """
        columns = dataset.columns
        frame = dataset.synthetic_df
        power = frame.loc[frame[columns.turbine] == turbine, columns.active_power].to_numpy(dtype=float)
        selected = mask & np.isfinite(power)
        return float(power[selected].sum()), int(selected.sum())


def per_turbine_table(result: CampaignResult) -> pd.DataFrame:
    """Return the per-turbine headline rows: one per method and upgraded turbine."""
    overall = result.scores[result.scores["condition"] == "overall"]
    return overall[["method", "test_wtg", "estimate", "truth", "signed_error"]].reset_index(drop=True)
