"""Analysis-stage folder names for grouping the per-run diagnostic plots.

Plots are written into numbered stage subfolders of ``<run>/plots`` so a reviewer can tell at a
glance which step of the pipeline a plot describes (feedback 2026-06-26): raw inputs, filtering,
feature engineering, the data fed to the uplift model, the modelling itself, and the results.
"""

from __future__ import annotations

INPUTS = "1_inputs"
FILTER = "2_filter"
FEATURE_ENG = "3_feature_eng"
UPLIFT_INPUTS = "4_uplift_inputs"
UPLIFT_MODELLING = "5_uplift_modelling"
UPLIFT_RESULTS = "6_uplift_results"
