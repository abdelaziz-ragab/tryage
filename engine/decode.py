# Exact decoder (Milestone 1); probabilistic/noisy decoding added in Milestone 2.
from __future__ import annotations

import itertools

import numpy as np

from simulate import run_clean


class DecodeTable:
    """Precomputed exact-decode lookup for one pooling matrix and a maximum
    of k simultaneous faults.

    Building the table also certifies whether the design is k-separable:
    every fault combination of size 0..k must produce a distinct pooled-test
    outcome pattern, or decoding would be ambiguous. This is a weaker (and
    much cheaper to satisfy) requirement than k-disjunctness, since it
    doesn't need to support the simple "eliminate items seen in a passing
    test" rule - it just needs each outcome to map back to one answer, which
    this table looks up directly.
    """

    def __init__(self, matrix: np.ndarray, k: int):
        self.matrix = matrix
        self.k = k
        n = matrix.shape[1]
        table: dict[bytes, frozenset[int]] = {}
        is_separable = True
        for size in range(k + 1):
            for combo in itertools.combinations(range(n), size):
                key = run_clean(matrix, combo).tobytes()
                if key in table:
                    is_separable = False
                else:
                    table[key] = frozenset(combo)
        self._table = table
        self.is_separable = is_separable

    def decode(self, outcomes: np.ndarray) -> set[int]:
        key = np.asarray(outcomes, dtype=bool).tobytes()
        faulty = self._table.get(key)
        if faulty is None:
            raise ValueError(
                f"pooled-test outcome doesn't match any fault combination "
                f"of size <= {self.k} - more simultaneous faults than this "
                "design tolerates, or noisy results (see Milestone 2)"
            )
        return set(faulty)
