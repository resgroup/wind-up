"""Percentile labels of a two-sided confidence interval."""

from __future__ import annotations


def percentile_labels(confidence_level: float) -> tuple[str, str]:
    """Return the labels of the low and high percentiles bounding a confidence interval.

    A level of 0.9 gives ("5", "95") and 0.95 gives ("2.5", "97.5").

    :param confidence_level: two-sided confidence level, greater than 0 and less than 1
    :return: tuple of low percentile label and high percentile label
    """
    if not 0 < confidence_level < 1:
        msg = f"confidence_level must be greater than 0 and less than 1, got {confidence_level}"
        raise ValueError(msg)
    p_low = (1 - confidence_level) / 2
    low, high = f"{p_low * 100:g}", f"{(1 - p_low) * 100:g}"
    if low == high:
        msg = f"confidence_level {confidence_level} is too small to give distinct percentile labels"
        raise ValueError(msg)
    return low, high
