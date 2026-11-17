"""Folder names for the run's outputs, one per step of ``docs/v1/method.md``.

Part A writes each step's outputs under ``<run>/A_data_preparation/<step>``, Part B under
``<run>/B_campaign/<step>`` and Part C under ``<run>/C_turbine_measurement/<turbine>/plots/<step>``,
so a reviewer can relate each output to the method step it describes. The number is the step's
number in the method.
"""

from __future__ import annotations

CHANGES = "01_changes"
OPERATING_STATES = "02_operating_states"
REANALYSIS = "03_reanalysis"
NORTHING = "04_northing"
WAKING = "05_waking"
FEATURES = "06_features"
PERIODS = "07_periods"
CANDIDATES = "08_candidates"
SCREEN = "09_screen"
VALID_RECORDS = "11_valid_records"
RELATE_REFERENCES = "12_relate_references"
UPLIFT = "13_uplift"
UPLIFT_DISTRIBUTIONS = "15_uplift_distributions"
