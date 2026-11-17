"""Shared per-run diagnostics for the v1 benchmarking methods.

A method adapts its internals to a :class:`~wind_up.diagnostics.context.DiagnosticContext`
and calls :func:`~wind_up.diagnostics.common.write_common_diagnostics` (plus
:func:`~wind_up.diagnostics.config_dump.write_run_config`) to emit a consistent, v0-grade
set of diagnostic plots and a run-config file. Project plotting conventions live in
:mod:`~wind_up.diagnostics.style` (grid on by default) and
:mod:`~wind_up.diagnostics.density` (density-coloured scatter).
"""

from __future__ import annotations

from wind_up.diagnostics.common import write_common_diagnostics
from wind_up.diagnostics.config_dump import write_run_config
from wind_up.diagnostics.context import DiagnosticContext, infer_timebase
from wind_up.diagnostics.density import density_scatter
from wind_up.diagnostics.style import apply_grid, save_fig

__all__ = [
    "DiagnosticContext",
    "apply_grid",
    "density_scatter",
    "infer_timebase",
    "save_fig",
    "write_common_diagnostics",
    "write_run_config",
]
