"""Cross-fit R-learner uplift method (v1 Issue 5).

A pluggable, v0-independent treatment-effect estimator behind the harness ``Method`` seam.
See ``docs/v1/issues.md`` (Issue 5); module docstrings cite the ML uplift design note by section.
"""

from __future__ import annotations

from benchmarking.baselines.rlearner.method import RLearnerMethod

__all__ = ["RLearnerMethod"]
