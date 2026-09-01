"""The campaign facts a method may consult, for one test turbine.

Derived from the campaign declaration by the runner (see
:mod:`benchmarking.campaigns.context`), so a method never reads the declaration itself. It
carries answers -- which turbines may serve as references, which rows may be used -- rather
than declaration fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from benchmarking.harness.toggle import is_toggle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from benchmarking.synthetic import ToggleSchedule


@dataclass(frozen=True)
class CampaignContext:
    """What a method may know about the campaign it is estimating one turbine's uplift for.

    :param test_wtg: the turbine whose uplift is being estimated
    :param timing: this turbine's changeover timestamp (prepost), or its ``ToggleSchedule`` /
        explicit ``toggle_df`` (toggle). Read :attr:`mode` rather than switching on the type.
    :param turbine_col: the turbine-identifier column of the SCADA frame
    :param candidate_references: the turbines a method may use as references. A turbine present
        in the frame but absent here is not a reference, whatever its data looks like.
    :param valid_for_uplift: boolean, timestamps x ``[test_wtg, *candidate_references]`` -- may
        this turbine's data at this timestamp contribute to the uplift estimate? A ``False``
        reference cell drops that reference for that row only, leaving the row usable; a
        ``False`` test-turbine cell drops the row outright. Deliberately named for its purpose:
        a curtailed record is invalid for uplift while staying valid for a northing analysis.
        Its index **covers** the frame's timestamps rather than matching them, so narrowing the
        frame does not invalidate it; narrow with :meth:`valid_over`.
    """

    test_wtg: str
    timing: pd.Timestamp | ToggleSchedule | pd.DataFrame
    turbine_col: str
    candidate_references: list[str]
    valid_for_uplift: pd.DataFrame

    @property
    def mode(self) -> Literal["prepost", "toggle"]:
        """``"toggle"`` for a scheduled campaign, ``"prepost"`` for a single changeover."""
        return "toggle" if is_toggle(self.timing) else "prepost"

    def valid_over(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return :attr:`valid_for_uplift` narrowed to ``index``. Raises on a timestamp not covered."""
        missing = pd.DatetimeIndex(index).difference(self.valid_for_uplift.index)
        if len(missing) > 0:
            msg = f"valid_for_uplift does not cover {len(missing)} of the requested timestamps, first {missing[0]}"
            raise ValueError(msg)
        return self.valid_for_uplift.loc[index]

    def references_among(self, columns: Iterable[str]) -> list[str]:
        """Return the entries of ``columns`` that are candidate references, in ``columns``' order.

        The context supplies membership only; the caller's ordering is preserved, since a method's
        feature order is its own business.
        """
        offered = set(self.candidate_references)
        return [str(c) for c in columns if str(c) in offered]

    def mask_invalid(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a timestamps x turbines ``frame`` with cells not valid for uplift set to NaN.

        Columns the context says nothing about are left untouched.
        """
        valid = self.valid_over(pd.DatetimeIndex(frame.index))
        shared = [c for c in frame.columns if c in valid.columns]
        if not shared:
            return frame
        return frame.mask(~valid[shared].reindex(columns=frame.columns, fill_value=True))

    def select(self, scada_df: pd.DataFrame, *, also: Iterable[str] = ()) -> pd.DataFrame:
        """Return the rows of a long-format ``scada_df`` the campaign allows this estimate to use.

        Drops turbines the campaign does not offer, and each remaining turbine's rows that are not
        valid for uplift. The long-frame counterpart of :meth:`references_among` plus
        :meth:`mask_invalid`, for methods that work before pivoting.

        :param also: turbines to keep besides the test turbine and its candidate references -- the
            other test turbines of a method that analyses several at once.
        """
        valid = self.valid_over(pd.DatetimeIndex(scada_df.index.unique()))
        turbines = scada_df[self.turbine_col].to_numpy()
        keep = np.zeros(len(scada_df), dtype=bool)
        for wtg in dict.fromkeys([self.test_wtg, *self.candidate_references, *also]):
            is_turbine = turbines == wtg
            if not is_turbine.any():
                continue
            rows = pd.DatetimeIndex(scada_df.index[is_turbine])
            keep[is_turbine] = valid[wtg].reindex(rows).to_numpy() if wtg in valid.columns else True
        return scada_df[keep]

    @classmethod
    def from_frame(
        cls,
        scada_df: pd.DataFrame,
        *,
        test_wtg: str,
        timing: pd.Timestamp | ToggleSchedule | pd.DataFrame,
        turbine_col: str,
    ) -> CampaignContext:
        """Return the context implied by ``scada_df`` alone: every other turbine a reference, all rows valid.

        The campaign-free default, used wherever there is no declaration to derive from (the study
        and sweep paths, and a ``MethodInput`` built without a context).
        """
        references = sorted({str(t) for t in scada_df[turbine_col].unique()} - {test_wtg})
        index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
        return cls(
            test_wtg=test_wtg,
            timing=timing,
            turbine_col=turbine_col,
            candidate_references=references,
            valid_for_uplift=pd.DataFrame(data=True, index=index, columns=[test_wtg, *references]),
        )
