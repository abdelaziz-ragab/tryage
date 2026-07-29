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


def run_noisy(
    matrix: np.ndarray,
    faulty: Iterable[int],
    flake_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate pooled test outcomes with injected flakiness.

    Models flakiness as a binary symmetric channel: the true (noiseless)
    outcome of each pooled test is computed as in run_clean, then each
    entry is independently flipped with probability flake_rate. This
    covers both directions real flaky tests exhibit - a clean pool
    observed as a spurious fail, or a genuinely faulty pool observed as a
    spurious pass - with a single, simple, configurable knob rather than
    separate false-positive/false-negative rates.
    """
    if not 0.0 <= flake_rate <= 1.0:
        raise ValueError("flake_rate must be in [0, 1]")
    true_outcomes = run_clean(matrix, faulty)
    flips = rng.random(matrix.shape[0]) < flake_rate
    return true_outcomes ^ flips
