"""Test-turbine normal-operation filtering for the R-learner.

The filter now lives in :mod:`benchmarking.baselines.filtering` because more than one method uses
it; this module re-exports it for backwards compatibility.
"""

from __future__ import annotations

from benchmarking.baselines.filtering import NormalOperationFilter

__all__ = ["NormalOperationFilter"]
