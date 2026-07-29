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


def _min_hamming_distance(outcome: np.ndarray, seen_arr: np.ndarray) -> int:
    """Smallest Hamming distance from outcome to any row already in seen_arr
    (which is never empty - see _greedy_separable_matrix)."""
    return int((seen_arr != outcome).sum(axis=1).min())


def _greedy_separable_matrix(
    n: int,
    k: int,
    t: int,
    min_distance: int,
    rng: np.random.Generator,
    tries_per_item: int,
) -> np.ndarray | None:
    """Try to place n column-codes of width t, one item at a time, keeping
    the partial design k-separable - and, for min_distance > 1, noise-robust
    - at every step.

    A candidate code for the next item is accepted only if, combined with
    every subset of size < k among the already-placed items, it produces a
    pooled-test outcome at least min_distance Hamming-apart from every
    outcome already in play. min_distance=1 (the Milestone 1 default) is
    exactly "never produces an outcome seen before" - plain separability,
    with no margin against a flipped bit. Milestone 2 raises min_distance to
    buy back some of that margin, at the cost of needing more rows: see
    docs/build-plan.md Milestone 2 for why this exists - the Milestone 1
    construction's separability guarantee turned out to hold with a minimum
    Hamming distance of only 1-2, which a single flaky pooled-test result is
    enough to defeat.

    This only re-checks combinations that involve the new item (previously
    placed items were already mutually separable at the required distance),
    so it's far cheaper than re-verifying the whole matrix from scratch on
    every attempt.
    """
    seen_arr = np.zeros((1, t), dtype=bool)  # the all-pass "no faults" outcome
    columns: list[np.ndarray] = []
    for _ in range(n):
        placed = False
        for _ in range(tries_per_item):
            candidate = rng.random(t) < 0.5
            new_outcomes: list[np.ndarray] = []
            ok = True
            for size in range(k):
                for combo in itertools.combinations(range(len(columns)), size):
                    outcome = candidate.copy()
                    for c in combo:
                        outcome |= columns[c]
                    if _min_hamming_distance(outcome, seen_arr) < min_distance:
                        ok = False
                        break
                    if new_outcomes and _min_hamming_distance(
                        outcome, np.array(new_outcomes)
                    ) < min_distance:
                        ok = False
                        break
                    new_outcomes.append(outcome)
                if not ok:
                    break
            if ok:
                columns.append(candidate)
                seen_arr = np.vstack([seen_arr, np.array(new_outcomes)])
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
    min_distance: int = 1,
    max_t: int | None = None,
) -> np.ndarray:
    """Build a T x N boolean pooling matrix (rows = pooled tests/groups,
    columns = items) that's k-separable: every combination of up to k
    simultaneously faulty items among n produces a distinct pooled-test
    outcome pattern, so decode.DecodeTable can recover it exactly under
    noiseless testing.

    min_distance raises that guarantee from "distinct" (min_distance=1, the
    Milestone 1 default) to "at least min_distance Hamming-apart" - the
    margin the Milestone 2 probabilistic decoder needs to stay accurate once
    pooled-test outcomes carry noise. Larger min_distance costs more rows.

    k=1 uses the exact antichain encoding above when it already meets
    min_distance (it achieves distance 2); otherwise, and for k>1, a
    randomized greedy construction is used - for a PoC, this is simpler than
    a structured combinatorial/error-correcting code and just as effective,
    since each candidate design is checked directly rather than assumed
    correct. Row count starts near the k=1 bound and grows only if greedy
    search fails to find a design at the current size, up to max_t (default
    n).

    If min_distance=1 and no design at or below max_t is found, the
    identity matrix (one test per item) is used - for small N and k close to
    N, non-adaptive pooling genuinely can't beat testing every item
    individually, and identity is always a valid decode. Identity is NOT
    used as a fallback when min_distance > 1: it only achieves min_distance
    1 between "no faults" and "one fault", so silently returning it would
    violate the caller's requested noise margin - a RuntimeError is raised
    instead (see Milestone 2 findings: for some (n, k, min_distance)
    combinations, no design at or below max_t exists at all, or search just
    didn't find one - the caller must decide whether to raise max_t, accept
    a smaller min_distance, or accept paying more than brute force).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if min_distance < 1:
        raise ValueError("min_distance must be >= 1")
    if max_t is None:
        max_t = n
    if k == 1 and min_distance <= 2:
        return binary_encoding_matrix(n)

    rng = np.random.default_rng(seed)
    t_start = binary_encoding_matrix(n).shape[0]
    for t in range(t_start, max_t):
        matrix = _greedy_separable_matrix(n, k, t, min_distance, rng, tries_per_item)
        if matrix is not None:
            return matrix

    if min_distance <= 1:
        return np.eye(n, dtype=bool)
    raise RuntimeError(
        f"no {min_distance}-Hamming-separable design found for n={n}, k={k} "
        f"within max_t={max_t} rows and tries_per_item={tries_per_item} - "
        "raise max_t/tries_per_item, or accept a smaller min_distance"
    )
