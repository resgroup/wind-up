import pytest

from wind_up.confidence import percentile_labels


@pytest.mark.parametrize(
    ("confidence_level", "expected"),
    [
        (0.9, ("5", "95")),
        (0.8, ("10", "90")),
        (0.95, ("2.5", "97.5")),
        (0.99, ("0.5", "99.5")),
        (0.75, ("12.5", "87.5")),
    ],
)
def test_percentile_labels(confidence_level: float, expected: tuple[str, str]) -> None:
    assert percentile_labels(confidence_level) == expected


@pytest.mark.parametrize("confidence_level", [0, 1, -0.1, 1.5, float("nan")])
def test_percentile_labels_rejects_levels_outside_zero_to_one(confidence_level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level must be greater than 0 and less than 1"):
        percentile_labels(confidence_level)


def test_percentile_labels_rejects_a_level_too_small_for_distinct_labels() -> None:
    with pytest.raises(ValueError, match="too small to give distinct percentile labels"):
        percentile_labels(1e-9)
