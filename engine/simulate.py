# Synthetic fault-injection simulation harness (clean case here in Milestone
# 1; noisy/flaky simulation added in Milestone 2/4).
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def run_clean(matrix: np.ndarray, faulty: Iterable[int]) -> np.ndarray:
    """Simulate noiseless pooled test outcomes for a given set of faulty items.

    Returns a boolean array of length T (one entry per pooled test/group):
    True = that pooled test fails (contains >=1 faulty item), False = it passes.
    """
    faulty = sorted(set(faulty))
    if not faulty:
        return np.zeros(matrix.shape[0], dtype=bool)
    return matrix[:, faulty].any(axis=1)
