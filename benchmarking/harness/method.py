"""The method seam the harness scores against: ``MethodInput -> MethodOutput``.

The input carries the data plus a :class:`~benchmarking.harness.context.CampaignContext` -- the
campaign facts for one test turbine, which is where a method learns its candidate references and
which rows are valid for uplift. Method-specific config (models, bin widths, plots) stays on the
``Method`` itself, so the seam carries answers rather than configuration. An input built without
a context uses the frame's own implicit contract: every other turbine is a candidate reference
and every row is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from benchmarking.harness.context import CampaignContext

if TYPE_CHECKING:
    import pandas as pd

    from benchmarking.synthetic import ToggleSchedule


@dataclass
class MethodInput:
    """The windowed data a method sees for one campaign.

    The frame carries **source-native** SCADA column names (the real tag names of the data
    source). Identifying turbines is a property of the data, not a method choice, so the seam
    names the turbine-identifier column here; methods read any value columns (e.g. active power)
    by their own source-native config.

    :param scada_df: windowed synthetic SCADA (all subset turbines, baseline + activity)
    :param test_wtg: the upgraded turbine to estimate
    :param upgrade_timing: changeover timestamp (prepost); a periodic ``ToggleSchedule`` or an
        explicit, possibly irregular ``toggle_df`` (a ``pd.DataFrame`` with boolean
        ``toggle_on``/``toggle_off`` on a ``DatetimeIndex``) for toggle. See
        :mod:`benchmarking.harness.toggle`.
    :param turbine_col: the turbine-identifier column in ``scada_df``
    :param campaign_context: the constructor slot for a campaign-derived
        :class:`~benchmarking.harness.context.CampaignContext`. Read :attr:`context` instead --
        it supplies the frame's own implicit contract when nothing was passed here. When it *is*
        passed, it is the source of ``upgrade_timing`` and ``turbine_col``, which are reconciled
        to it on construction.
    """

    scada_df: pd.DataFrame
    test_wtg: str
    upgrade_timing: pd.Timestamp | ToggleSchedule | pd.DataFrame | None = None
    turbine_col: str = "TurbineName"
    campaign_context: CampaignContext | None = None

    def __post_init__(self) -> None:
        """Reconcile the timing and turbine-column shorthands to a supplied context, and validate."""
        if self.campaign_context is None:
            if self.upgrade_timing is None:
                msg = (
                    "MethodInput needs either campaign_context or upgrade_timing: with neither there is "
                    "no changeover or schedule to estimate against."
                )
                raise ValueError(msg)
            return
        if self.campaign_context.test_wtg != self.test_wtg:
            msg = (
                f"campaign_context is for test_wtg {self.campaign_context.test_wtg!r} but this input estimates "
                f"{self.test_wtg!r}; the method would read another turbine's references and row validity."
            )
            raise ValueError(msg)
        self.upgrade_timing = self.campaign_context.timing
        self.turbine_col = self.campaign_context.turbine_col

    @property
    def context(self) -> CampaignContext:
        """The campaign facts for this turbine; the frame's own implicit contract by default.

        Built on first read rather than on construction, so an input that never consults it costs
        nothing and a frame without a turbine column is only a problem for a method that uses it.
        """
        if self.campaign_context is None:
            self.campaign_context = CampaignContext.from_frame(
                self.scada_df,
                test_wtg=self.test_wtg,
                timing=self.upgrade_timing,
                turbine_col=self.turbine_col,
            )
        return self.campaign_context


@dataclass
class MethodOutput:
    """A method's P50 uplift estimate, and optionally its uncertainty.

    The uncertainty fields are **additive**: they default to ``None``, so a method that reports
    only a P50 is unchanged and the harness records a NaN uncertainty for it.

    :param p50_overall: overall P50 uplift (energy-ratio fraction)
    :param p50_by_condition: optional per-condition estimates (columns ``condition``,
        ``condition_bin``, ``p50_uplift``); ``condition`` ∈ {"ws","ti","power"}; ``None`` when the
        method produces only an overall number. May also carry a ``sigma_uplift`` column, the
        per-bin counterpart of ``sigma_overall``.
    :param sigma_overall: optional 1-sigma on ``p50_overall``, a symmetric delta in energy-ratio
        fraction. Scored against the deviation from ground truth, so it is a *total* uncertainty.
    :param uncertainty_diagnostics: optional tidy frame keyed by ``(condition, condition_bin)``
        carrying whatever a method wants to say about how its uncertainty was reached. The harness
        never interprets these columns; it merges them onto the results rows and carries them
        through. Use ``("overall", "overall")`` for the headline row.
    :param labeled_rows: optional per-record frame exposing the row selection the estimate was
        actually built from: the test turbine's original SCADA columns, plus ``used`` (did the row
        survive the method's filtering), ``segment`` (``baseline`` / ``upgraded`` / ``excluded``),
        and one ``<condition>_bin`` column per condition the method was asked for. It lets a
        consumer compute a per-bin quantity the method does not itself report -- a mean pitch, say
        -- over exactly the rows and bins the uplift used, instead of re-deriving the filtering and
        binning and hoping the two agree. Deliberately row-level rather than pre-aggregated, so it
        serves consumers whose desired aggregate is not known here. The harness never interprets
        it.
    """

    p50_overall: float
    p50_by_condition: pd.DataFrame | None = None
    sigma_overall: float | None = None
    uncertainty_diagnostics: pd.DataFrame | None = None
    labeled_rows: pd.DataFrame | None = None


@runtime_checkable
class Method(Protocol):
    """A pluggable uplift estimator: a name plus ``estimate(MethodInput) -> MethodOutput``."""

    name: str

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Return a P50 uplift estimate for the campaign described by ``mi``."""
        ...
