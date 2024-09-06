import pytest

from wind_up.backporting import strict_zip


def test_equal_length_iterables() -> None:
    a = [1, 2, 3]
    b = ["a", "b", "c"]
    result = list(strict_zip(a, b, strict=True))
    assert result == [(1, "a"), (2, "b"), (3, "c")]


def test_unequal_length_raises_value_error() -> None:
    a = [1, 2, 3]
    b = ["a", "b"]

    with pytest.raises(ValueError):  # noqa PT011
        list(strict_zip(a, b, strict=True))


def test_unequal_length_non_strict() -> None:
    a = [1, 2, 3]
    b = ["a", "b"]

    result = list(strict_zip(a, b, strict=False))
    assert result == [(1, "a"), (2, "b")]  # Shorter iterable determines length


def test_empty_iterables() -> None:
    a = []
    b = []

    result = list(strict_zip(a, b, strict=True))
    assert result == []


def test_single_iterable() -> None:
    a = [1, 2, 3]

    result = list(strict_zip(a, strict=True))
    assert result == [(1,), (2,), (3,)]


def test_multiple_iterables() -> None:
    a = [1, 2, 3]
    b = ["a", "b", "c"]
    c = [True, False, True]

    result = list(strict_zip(a, b, c, strict=True))
    assert result == [(1, "a", True), (2, "b", False), (3, "c", True)]


def test_non_iterable_argument() -> None:
    a = [1, 2, 3]
    b = 5  # Not iterable

    with pytest.raises(TypeError):
        list(strict_zip(a, b, strict=True))


def test_tuple_and_generator() -> None:
    a = (1, 2, 3)
    b = (x for x in ["a", "b", "c"])  # Generator

    result = list(strict_zip(a, b, strict=True))
    assert result == [(1, "a"), (2, "b"), (3, "c")]


def test_nested_iterables() -> None:
    a = [[1], [2], [3]]
    b = [["a"], ["b"], ["c"]]

    result = list(strict_zip(a, b, strict=True))
    assert result == [([1], ["a"]), ([2], ["b"]), ([3], ["c"])]
