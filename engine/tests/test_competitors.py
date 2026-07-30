import numpy as np
import pytest

from benchmark import simulate_bisection, simulate_brute_force
from competitors import simulate_github_queue, simulate_graphite, simulate_mergify


# --- sanity checks against hand-computed cases (don't trust a competitor
# simulator you haven't verified, same standard as Milestone 4) ----------


def test_github_queue_matches_hand_computed_case():
    """N=4, single fault at position 2: round 1 tests all 4 cumulative
    positions (4 runs), finds the first failure at offset 2 (culprit),
    removes it; round 2 retests the remaining tail [item3] (1 run), clean.
    Total: 5 ci_runs, 2 rounds."""
    rng = np.random.default_rng(0)
    r = simulate_github_queue(4, {2}, 0.0, rng)
    assert r.decoded == {2}
    assert r.ci_runs == 5
    assert r.logical_rounds == 2


def test_github_queue_fault_at_front_retests_whole_tail():
    """Fault at position 0: round 1 (4 runs) finds it immediately, but
    round 2 must retest the entire remaining tail [1,2,3] (3 runs) since
    everything was cumulative-contaminated. Total: 7 ci_runs, 2 rounds."""
    rng = np.random.default_rng(0)
    r = simulate_github_queue(4, {0}, 0.0, rng)
    assert r.decoded == {0}
    assert r.ci_runs == 7
    assert r.logical_rounds == 2


def test_github_queue_no_fault_is_one_round():
    rng = np.random.default_rng(0)
    r = simulate_github_queue(5, set(), 0.0, rng)
    assert r.decoded == set()
    assert r.ci_runs == 5
    assert r.logical_rounds == 1


def test_mergify_arity_2_matches_plain_bisection_exactly():
    """Mergify's own documented fallback when parallel speculative checks
    aren't configured is arity=2, i.e. plain binary bisection - must match
    benchmark.simulate_bisection exactly, not just approximately."""
    for seed in range(10):
        faulty = {seed % 8}
        r_bisect = simulate_bisection(8, faulty, 0.0, np.random.default_rng(seed))
        r_mergify = simulate_mergify(8, faulty, 0.0, np.random.default_rng(seed), arity=2)
        assert r_bisect.ci_runs == r_mergify.ci_runs
        assert r_bisect.decoded == r_mergify.decoded


def test_mergify_higher_arity_trades_ci_runs_for_fewer_rounds():
    rng4 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    r4 = simulate_mergify(8, {3}, 0.0, rng4, arity=4)
    r2 = simulate_mergify(8, {3}, 0.0, rng2, arity=2)
    assert r4.decoded == r2.decoded == {3}
    assert r4.logical_rounds <= r2.logical_rounds
    assert r4.ci_runs >= r2.ci_runs


def test_mergify_rejects_invalid_arity():
    with pytest.raises(ValueError):
        simulate_mergify(8, {0}, 0.0, np.random.default_rng(0), arity=1)


def test_mergify_test_all_pieces_terminates_with_simultaneous_faults():
    """Regression test for a real infinite-loop bug this milestone caught:
    with test_all_pieces=True and 2+ simultaneous faults landing in
    different pieces, all those pieces test as failing - taking their
    UNION (as an earlier version did) can reconstruct the exact pool being
    split, with no shrinkage, looping forever. Fixed by keeping only the
    first failing piece per level (matching simulate_bisection's
    tie-breaking), letting the outer repeat-until-clean loop catch any
    other simultaneous faults on a later pass."""
    rng = np.random.default_rng(0)
    r = simulate_mergify(2, {0, 1}, 0.0, rng, arity=2, test_all_pieces=True)
    assert r.decoded == {0, 1}

    # Broader sweep: many simultaneous-fault configurations, must all
    # terminate (this test itself hanging is the failure mode).
    for n in (4, 8, 16):
        for num_faults in range(1, min(n, 4) + 1):
            rng = np.random.default_rng(0)
            faulty = set(range(num_faults))
            r = simulate_mergify(n, faulty, 0.0, rng, arity=3, test_all_pieces=True)
            assert r.decoded == faulty


def test_graphite_small_batch_matches_brute_force_exactly():
    """Graphite's own documented threshold: <=8 stacks uses full parallel
    isolation (brute force)."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    assert simulate_graphite(6, {2}, 0.0, rng1) == simulate_brute_force(6, {2}, 0.0, rng2)
    assert simulate_graphite(8, {0, 5}, 0.0, np.random.default_rng(1)) == simulate_brute_force(
        8, {0, 5}, 0.0, np.random.default_rng(1)
    )


def test_graphite_large_batch_matches_bisection_exactly():
    """Above Graphite's threshold, their docs say bisection."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    assert simulate_graphite(16, {5}, 0.0, rng1) == simulate_bisection(16, {5}, 0.0, rng2)


def test_graphite_threshold_is_configurable():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    r = simulate_graphite(10, {2}, 0.0, rng1, parallel_threshold=12)
    rb = simulate_brute_force(10, {2}, 0.0, rng2)
    assert r == rb  # 10 <= 12 -> brute force even though 10 > default threshold of 8
