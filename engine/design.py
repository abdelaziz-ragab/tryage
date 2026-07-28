# Pooling matrix construction (Milestone 1)
from __future__ import annotations

import itertools
from math import comb

import numpy as np


def _min_antichain_width(n: int) -> tuple[int, int]:
    """Smallest (t, w) such that the w-subsets of a t-element universe give
    at least n distinct codes, with w chosen to maximize C(t, w) (Sperner's
    theorem: same-size subsets never contain one another, so this is the
    smallest universe that fits n codes forming an antichain)."""
    t = 1
    while True:
        w = t // 2
        if comb(t, w) >= n:
            return t, w
        t += 1


def binary_encoding_matrix(n: int) -> np.ndarray:
    """k=1 construction: assign each item a distinct constant-weight binary
    code (Sperner-style antichain encoding).

    Constant weight guarantees no item's code is a subset of another's, which
    is exactly what decoding needs to pin down a single fault unambiguously:
    plain sequential numbering (1, 2, 3, ...) does NOT have this property
    (e.g. code 2 = 010 is a bit-subset of code 6 = 110) and produces
    ambiguous outcome patterns.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.ones((1, 1), dtype=bool)

    t, w = _min_antichain_width(n)
    codes = itertools.islice(itertools.combinations(range(t), w), n)
    matrix = np.zeros((t, n), dtype=bool)
    for item, bits in enumerate(codes):
        for bit in bits:
            matrix[bit, item] = True
    return matrix


def _greedy_separable_matrix(
    n: int, k: int, t: int, rng: np.random.Generator, tries_per_item: int
) -> np.ndarray | None:
    """Try to place n column-codes of width t, one item at a time, keeping
    the partial design k-separable at every step.

    A candidate code for the next item is accepted only if, combined with
    every subset of size < k among the already-placed items, it produces a
    pooled-test outcome never seen before. This only re-checks combinations
    that involve the new item (previously placed items were already mutually
    separable), so it's far cheaper than re-verifying the whole matrix from
    scratch on every attempt.
    """
    seen = {np.zeros(t, dtype=bool).tobytes()}
    columns: list[np.ndarray] = []
    for _ in range(n):
        placed = False
        for _ in range(tries_per_item):
            candidate = rng.random(t) < 0.5
            new_keys = []
            ok = True
            for size in range(k):
                for combo in itertools.combinations(range(len(columns)), size):
                    outcome = candidate.copy()
                    for c in combo:
                        outcome |= columns[c]
                    key = outcome.tobytes()
                    if key in seen or key in new_keys:
                        ok = False
                        break
                    new_keys.append(key)
                if not ok:
                    break
            if ok:
                columns.append(candidate)
                seen.update(new_keys)
                placed = True
                break
        if not placed:
            return None
    return np.array(columns, dtype=bool).T


def build_pooling_matrix(
    n: int,
    k: int,
    *,
    seed: int | None = None,
    tries_per_item: int = 1000,
) -> np.ndarray:
    """Build a T x N boolean pooling matrix (rows = pooled tests/groups,
    columns = items) that's k-separable: every combination of up to k
    simultaneously faulty items among n produces a distinct pooled-test
    outcome pattern, so decode.DecodeTable can recover it exactly under
    noiseless testing.

    k=1 uses the exact antichain encoding above. k>1 uses a randomized
    greedy construction - for a PoC, this is simpler than a structured
    combinatorial code (e.g. Kautz-Singleton) and just as effective, since
    each candidate design is checked for separability directly rather than
    assumed correct. Row count starts near the k=1 bound and grows only if
    greedy search fails to find a design at the current size. If no design
    below T=N is found, the identity matrix (one test per item) is used -
    for small N and k close to N, non-adaptive pooling genuinely can't beat
    testing every item individually, and identity is always a valid decode.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return binary_encoding_matrix(n)

    rng = np.random.default_rng(seed)
    t_start = binary_encoding_matrix(n).shape[0]
    for t in range(t_start, n):
        matrix = _greedy_separable_matrix(n, k, t, rng, tries_per_item)
        if matrix is not None:
            return matrix

    return np.eye(n, dtype=bool)
