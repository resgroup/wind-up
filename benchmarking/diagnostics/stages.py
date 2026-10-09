"""Folder names for the per-run diagnostic plots, one per step of ``docs/v1/method.md``.

Plots are written into ``<run>/plots/<step>`` so a reviewer can relate each plot to the method step
it describes. The number is the step's number in the method.
"""

from __future__ import annotations

CHANGES = "01_changes"
OPERATING_STATES = "02_operating_states"
REANALYSIS = "03_reanalysis"
NORTHING = "04_northing"
WAKING = "05_waking"
FEATURES = "06_features"
VALID_RECORDS = "08_valid_records"
RELATE_REFERENCES = "09_relate_references"
UPLIFT = "10_uplift"
UPLIFT_DISTRIBUTIONS = "13_uplift_distributions"
