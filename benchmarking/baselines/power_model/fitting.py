"""Time-blocked fold assignment for the power model's baseline holdout fit.

:func:`time_block_folds` — contiguous-block, round-robin fold assignment for time-ordered rows.
A shuffled split leaks autocorrelation (held-out rows sit minutes from training rows), so its
residuals are optimistic; contiguous blocks confine the leakage to the block edges, which makes
the held-out fit-quality diagnostic honest.
"""

from __future__ import annotations

import numpy as np


def time_block_folds(n: int, *, n_folds: int = 5, n_blocks: int = 25) -> np.ndarray:
    """Assign ``n`` time-ordered rows to ``n_folds`` folds as round-robin contiguous blocks.

    Rows must already be in time order (the power model's row arrays are — they follow the sorted
    analysis index). The rows are cut into ``n_blocks`` contiguous, equal-length blocks and block
    ``i`` goes to fold ``i % n_folds``, so every fold samples all seasons while staying contiguous
    at the scale that matters for autocorrelation (only the block edges sit near training rows).
    Returns an int array of fold ids, one per row.
    """
    if n_folds < 2 or n_blocks < n_folds:  # noqa: PLR2004
        msg = f"need n_folds >= 2 and n_blocks >= n_folds, got n_folds={n_folds}, n_blocks={n_blocks}"
        raise ValueError(msg)
    block = np.minimum((np.arange(n) * n_blocks) // max(n, 1), n_blocks - 1)
    return (block % n_folds).astype(int)
