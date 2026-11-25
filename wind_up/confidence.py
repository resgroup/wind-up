"""Validation and labels of a two-sided confidence level."""

from __future__ import annotations


def validate_confidence_level(confidence_level: float) -> None:
    """Raise a ValueError unless the confidence level is greater than 0 and less than 1."""
    if not 0 < confidence_level < 1:
        msg = f"confidence_level must be greater than 0 and less than 1, got {confidence_level}"
        raise ValueError(msg)


def percent_label(fraction: float) -> str:
    """Return a fraction as a percentage to 10 significant digits without trailing zeros, e.g. 0.975 gives "97.5"."""
    return f"{fraction * 100:.10g}"


def percentile_labels(confidence_level: float) -> tuple[str, str]:
    """Return the labels of the low and high percentiles bounding a confidence interval.

    A level of 0.9 gives ("5", "95") and 0.95 gives ("2.5", "97.5").

    :param confidence_level: two-sided confidence level, greater than 0 and less than 1
    :return: tuple of low percentile label and high percentile label
    """
    validate_confidence_level(confidence_level)
    p_low = (1 - confidence_level) / 2
    low, high = percent_label(p_low), percent_label(1 - p_low)
    if low == high:
        msg = f"confidence_level {confidence_level} is too small to give distinct percentile labels"
        raise ValueError(msg)
    return low, high
