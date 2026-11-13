"""Analysis warnings: what a run finds that an analyst should look at, ranked by severity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

Severity = Literal["high", "medium", "low"]
SEVERITIES: tuple[str, ...] = get_args(Severity)
COLUMNS: tuple[str, ...] = ("severity", "step", "turbine", "message")


@dataclass(frozen=True)
class AnalysisWarning:
    """One warning: its severity, the method step that raised it, the turbine it is against and the message.

    High questions the headline, medium changes the inputs of a measurement without questioning its
    result, low is for the analyst's attention only.
    """

    severity: Severity
    step: int
    turbine: str
    message: str

    def __post_init__(self) -> None:
        """Refuse a severity outside :data:`SEVERITIES`."""
        if self.severity not in SEVERITIES:
            msg = f"severity must be one of {SEVERITIES}, not {self.severity!r}"
            raise ValueError(msg)


def ranked(warnings: Iterable[AnalysisWarning]) -> list[AnalysisWarning]:
    """Return ``warnings`` highest severity first, then by step and turbine."""
    return sorted(warnings, key=lambda w: (SEVERITIES.index(w.severity), w.step, w.turbine, w.message))


def warnings_frame(warnings: Iterable[AnalysisWarning]) -> pd.DataFrame:
    """Return the ranked warnings as a frame with :data:`COLUMNS`."""
    rows = [(w.severity, w.step, w.turbine, w.message) for w in ranked(warnings)]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def summary(warnings: Iterable[AnalysisWarning]) -> str:
    """Return the count per severity on the first line, then one line per warning, ranked."""
    ordered = ranked(warnings)
    if not ordered:
        return "Analysis warnings: none"
    counts = ", ".join(f"{sum(w.severity == s for w in ordered)} {s}" for s in SEVERITIES)
    lines = [f"Analysis warnings: {counts}"]
    lines += [f"- {w.severity} (step {w.step}) {w.turbine}: {w.message}" for w in ordered]
    return "\n".join(lines)
