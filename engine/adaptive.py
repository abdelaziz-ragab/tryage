# Adaptive fallback for exceeded fault tolerance (Milestone 3)
from __future__ import annotations

import numpy as np

from decode import DecodeTable


def compute_cleared_and_suspects(
    matrix: np.ndarray, outcomes: np.ndarray
) -> tuple[set[int], set[int]]:
    """Split items into "cleared" (definitely not faulty) and "suspects"
    (everyone else), using a rule that holds regardless of how many items
    are actually faulty: under clean/noiseless testing, any item that
    appears in a passing pool cannot be at fault (a pool only fails if it
    contains >=1 faulty item), full stop. This doesn't depend on the
    design's k-separability guarantee at all - it's true whether the real
    fault count is 0, k, or 2k - which is exactly why it's the right tool
    for narrowing down an exceeded-tolerance situation that DecodeTable's
    k-limited lookup table can't resolve on its own.

    suspects is therefore always a superset of the true fault set: it can
    only ever wrongly include innocent items that happen to share every one
    of their pools with a genuinely faulty item (never wrongly exclude a
    real fault), which is what makes brute-force-testing exactly the
    suspects set (see AdaptiveDecoder.decode) a safe, complete fallback.
    """
    outcomes = np.asarray(outcomes, dtype=bool)
    n = matrix.shape[1]
    cleared: set[int] = set()
    for row in range(matrix.shape[0]):
        if not outcomes[row]:
            cleared.update(matrix[row].nonzero()[0].tolist())
    suspects = set(range(n)) - cleared
    return cleared, suspects


class AdaptiveDecoder:
    """Wraps DecodeTable with a fallback for when the true fault count
    exceeds what the design was built to tolerate (k).

    A naive version of this class trusted any exact-table match outright,
    on the theory that a >k-sized fault set coincidentally producing an
    outcome identical to some <=k combo's outcome would be astronomically
    rare (~2^-T for a design with T rows). That assumption was wrong,
    measured directly: because T is chosen close to the *minimum* needed to
    separate combos up to size k, the outcome space is packed tightly
    enough that larger fault sets collide with a smaller combo's outcome
    routinely, not rarely - a first cut of this class was confidently wrong
    roughly half the time on exceeded-k trials before this was caught by
    test_milestone3.py's own correctness sweep (not by inspection).

    So every table match is verified with one extra pooled test before
    being trusted: the complement pool (every item NOT in the candidate
    set). Under clean testing, that pool can only fail if a true fault
    exists outside the candidate - which directly catches both an
    incomplete candidate (missing a fault) and a candidate that's a
    coincidental match for a completely different, larger fault set. If the
    complement passes, the candidate is trusted (this does not itself rule
    out the rarer case of the candidate over-including an innocent item
    whose row-pattern happens to add no new failing row - see
    docs/build-plan.md Milestone 3 for that residual caveat).

    If the table found no match at all, or the complement check fails, the
    candidate is discarded entirely and the full fallback runs: every
    suspect from compute_cleared_and_suspects (never a real fault missed,
    though it may over-include innocent items) is tested individually, in
    parallel, as one round - not full brute force over all n items, and
    not a sequential/multi-round bisection.
    """

    def __init__(self, matrix: np.ndarray, k: int):
        self.matrix = matrix
        self.k = k
        self.table = DecodeTable(matrix, k)

    def decode(self, outcomes, run_pool) -> tuple[set[int], bool]:
        """run_pool(subset: set[int]) -> bool must report whether a pool
        containing exactly that subset of items would fail (True) or pass
        (False) - the seam a real system would wire to CI orchestration
        (Milestone 7); tests/harness pass a simulator.

        Returns (decoded_faulty_set, used_fallback). used_fallback is True
        only when the (potentially large) suspects-based resolution ran -
        the one-test complement verification that runs on every successful
        table match is a small, fixed cost paid regardless (see
        AdaptiveSweepResult.avg_extra_tests_fraction in harness.py for its
        honestly-measured overhead), not counted as "the fallback" itself.
        """
        outcomes = np.asarray(outcomes, dtype=bool)
        n = self.matrix.shape[1]
        try:
            candidate = self.table.decode(outcomes)
        except ValueError:
            candidate = None

        if candidate is not None:
            complement = set(range(n)) - candidate
            if not complement or not run_pool(complement):
                return candidate, False

        _, suspects = compute_cleared_and_suspects(self.matrix, outcomes)
        confirmed = {i for i in suspects if run_pool({i})}
        return confirmed, True
