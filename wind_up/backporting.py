"""Backporting utilities for features in older Python versions."""

import sys
from collections.abc import Iterable, Iterator

if sys.version_info >= (3, 10):

    def strict_zip(*iterables: Iterable, strict: bool = False) -> Iterator:  # noqa: D103
        return zip(*iterables, strict=strict)
else:

    def strict_zip(*iterables: Iterable, strict: bool = False) -> Iterator:  # noqa: D103
        if strict:
            iterables = [list(it) for it in iterables]  # type: ignore[assignment]
            if not all(len(it) == len(iterables[0]) for it in iterables):  # type: ignore[arg-type]
                msg = "All iterables must have the same length"
                raise ValueError(msg)
        return zip(*iterables)  # type: ignore[call-overload]
