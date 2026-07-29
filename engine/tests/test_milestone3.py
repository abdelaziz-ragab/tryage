import itertools

import numpy as np
import pytest

from adaptive import AdaptiveDecoder, compute_cleared_and_suspects
from design import build_pooling_matrix
from harness import run_adaptive_sweep
from simulate import run_clean

BATCH_SIZES = (8, 16, 32)
FAULT_TOLERANCES = (1, 2, 3)


def _run_pool_for(faulty: set[int]):
    def run_pool(subset: set[int]) -> bool:
        return bool(subset & faulty)

    return run_pool


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("k", FAULT_TOLERANCES)
def test_within_tolerance_does_not_need_fallback(n, k):
    """Regression: everything Milestone 1/2 already guaranteed (fault count
    <= k decodes exactly) must keep working, and AdaptiveDecoder shouldn't
    need the expensive suspects-based resolution when the original design
    already resolves it correctly (it still pays one cheap verification
    test - see AdaptiveDecoder.decode - but that isn't "the fallback")."""
    matrix = build_pooling_matrix(n, k, seed=0)
    decoder = AdaptiveDecoder(matrix, k)
    for size in range(k + 1):
        for combo in itertools.islice(itertools.combinations(range(n), size), 15):
            faulty = set(combo)
            outcomes = run_clean(matrix, faulty)
            decoded, used_fallback = decoder.decode(outcomes, _run_pool_for(faulty))
            assert decoded == faulty
            assert not used_fallback


@pytest.mark.parametrize("n,k", [(16, 2), (32, 2), (16, 3), (32, 3)])
def test_exceeded_tolerance_resolves_correctly_via_fallback(n, k):
    """The core Milestone 3 claim: when the true fault count exceeds k, the
    decoder must not confidently return a wrong answer - it must detect the
    ambiguity and use the fallback round to arrive at the exact truth."""
    matrix = build_pooling_matrix(n, k, seed=0)
    decoder = AdaptiveDecoder(matrix, k)
    rng = np.random.default_rng(0)
    for excess in (1, 2):
        fault_count = k + excess
        if fault_count > n:
            continue
        for _ in range(30):
            faulty = set(rng.choice(n, size=fault_count, replace=False).tolist())
            outcomes = run_clean(matrix, faulty)
            decoded, used_fallback = decoder.decode(outcomes, _run_pool_for(faulty))
            assert decoded == faulty, f"n={n} k={k} faulty={faulty} decoded={decoded}"
            assert used_fallback


def test_suspects_always_a_superset_of_true_faults():
    """Fundamental correctness property compute_cleared_and_suspects relies
    on: it may over-include innocent items, but must never exclude a real
    fault, regardless of how many faults there actually are (0, k, or way
    past k) - this is what makes brute-force-testing the suspects a safe,
    complete fallback."""
    n, k = 16, 2
    matrix = build_pooling_matrix(n, k, seed=0)
    rng = np.random.default_rng(0)
    for fault_count in range(0, n + 1, 3):
        faulty = set(rng.choice(n, size=fault_count, replace=False).tolist())
        outcomes = run_clean(matrix, faulty)
        _, suspects = compute_cleared_and_suspects(matrix, outcomes)
        assert faulty <= suspects, f"fault_count={fault_count}: {faulty} not subset of {suspects}"


def test_all_items_faulty_degenerates_correctly_to_full_retest():
    n, k = 16, 2
    matrix = build_pooling_matrix(n, k, seed=0)
    decoder = AdaptiveDecoder(matrix, k)
    faulty = set(range(n))
    outcomes = run_clean(matrix, faulty)
    decoded, used_fallback = decoder.decode(outcomes, _run_pool_for(faulty))
    assert decoded == faulty
    assert used_fallback


def test_zero_faults_decodes_directly_without_fallback():
    n, k = 16, 2
    matrix = build_pooling_matrix(n, k, seed=0)
    decoder = AdaptiveDecoder(matrix, k)
    outcomes = run_clean(matrix, set())
    decoded, used_fallback = decoder.decode(outcomes, _run_pool_for(set()))
    assert decoded == set()
    assert not used_fallback


# --- fallback overhead: is the follow-up round actually "small"? ---------
#
# Milestone 3's own framing calls for "one small, targeted follow-up round"
# - not full brute force. compute_cleared_and_suspects guarantees
# *correctness* regardless of overhead (see test above), but "small" is a
# separate, empirical claim that needs its own measurement, in the same
# spirit as every other milestone's benchmark tables in this project.
def test_fallback_overhead_measured_honestly():
    num_trials = 200
    rows = []
    for n in (16, 32):
        for k in (2, 3):
            for excess in (1, 2):
                fault_count = k + excess
                sweep = run_adaptive_sweep(n, k, fault_count, num_trials, trial_seed=0)
                rows.append((n, k, fault_count, sweep))

    print(f"\n{'N':>4} {'k':>3} {'faults':>7} {'exact-match':>12} {'avg extra tests/N':>18}")
    for n, k, fault_count, sweep in rows:
        print(
            f"{n:>4} {k:>3} {fault_count:>7} "
            f"{sweep.exact_match_rate:>11.1%} {sweep.avg_extra_tests_fraction:>17.1%}"
        )

    # Correctness must hold in every case tested (this is the actual
    # Milestone 3 requirement) - overhead is measured, not assumed.
    for n, k, fault_count, sweep in rows:
        assert sweep.exact_match_rate == 1.0, f"n={n} k={k} faults={fault_count}"
        assert sweep.fallback_rate == 1.0  # every exceeded-k trial should trigger it
