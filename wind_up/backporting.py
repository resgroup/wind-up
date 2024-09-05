import sys
from collections.abc import Iterable, Iterator

if sys.version_info >= (3, 10):

    def strict_zip(*iterables: Iterable, strict: bool = False) -> Iterator:
        return zip(*iterables, strict=strict)
else:

    def strict_zip(*iterables: Iterable, strict: bool = False) -> Iterator:
        if strict:
            iterables = [list(it) for it in iterables]
            if not all(len(it) == len(iterables[0]) for it in iterables):
                msg = "All iterables must have the same length"
                raise ValueError(msg)
        return zip(*iterables, strict=False)
