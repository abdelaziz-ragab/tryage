import math

import numpy as np
import pytest

from benchmark import (
    ComparisonResult,
    compare_approaches,
    run_benchmark_sweep,
    simulate_bisection,
    simulate_brute_force,
    simulate_combinatorial,
)
from design import build_pooling_matrix


# --- sanity checks against hand-computed small cases ---------------------
# Milestone 4's own instruction: don't trust a benchmarking tool you
# haven't verified.


def test_bisection_matches_hand_computed_round_count():
    """N=8, single fault, clean: 1 initial "still faulty?" check + 3 binary
    halving steps (8->4->2->1) to find it + 1 final check on the remaining
    7 confirming no more faults = 5 total sequential CI runs."""
    rng = np.random.default_rng(0)
    result = simulate_bisection(8, {3}, 0.0, rng)
    assert result.decoded == {3}
    assert result.ci_runs == 5
    assert result.logical_rounds == 5  # every step is sequential
    assert result.waves == 5


def test_bisection_round_count_scales_with_fault_count_and_log_n():
    """Two faults in N=32 (clean): 2 bisection passes (~log2(32)=5 steps
    each) plus a leading/trailing "still faulty?" check per pass - roughly
    2*(5+1) = 12, not exactly log2(32) alone, since each pass also pays for
    its own "is anyone left" checks."""
    rng = np.random.default_rng(0)
    result = simulate_bisection(32, {5, 20}, 0.0, rng)
    assert result.decoded == {5, 20}
    assert result.ci_runs <= 2 * (math.ceil(math.log2(32)) + 1) + 1


def test_brute_force_matches_hand_computed_cost():
    rng = np.random.default_rng(0)
    result = simulate_brute_force(16, {4, 9}, 0.0, rng)
    assert result.decoded == {4, 9}
    assert result.ci_runs == 16
    assert result.logical_rounds == 1
    assert result.waves == 1


def test_brute_force_waves_respect_parallelism_ceiling():
    rng = np.random.default_rng(0)
    result = simulate_brute_force(10, {1}, 0.0, rng, parallelism_ceiling=3)
    assert result.ci_runs == 10
    assert result.waves == math.ceil(10 / 3)


def test_bisection_ignores_parallelism_ceiling_by_construction():
    """Bisection's dependency chain can't be parallelized - waves must
    equal ci_runs regardless of ceiling, since each step depends on the
    previous one's result."""
    rng = np.random.default_rng(0)
    result = simulate_bisection(16, {4}, 0.0, rng)
    assert result.waves == result.ci_runs == result.logical_rounds


@pytest.mark.parametrize("n", (16, 32))
def test_combinatorial_clean_exceeded_k_is_always_exact(n):
    """Regression test for a real bug this milestone caught in itself: an
    earlier version of simulate_combinatorial used only the Milestone 2
    confidence-threshold check for every case, which cannot detect an
    exceeded-k situation (it measured 18-20% accuracy on clean N=16/32
    exceeded-k trials that Milestone 3's AdaptiveDecoder already proved
    should be 100%). Clean-case exceeded-k must be delegated to
    AdaptiveDecoder and always resolve exactly."""
    k, fault_count = 2, 3
    sweep = run_benchmark_sweep(n, k, fault_count, 0.0, 100, trial_seed=0)
    assert sweep.summary()["combinatorial"]["accuracy"] == 1.0


def test_combinatorial_matches_milestone1_known_row_counts():
    """Cross-check against Milestone 1's already-established, exhaustively-
    verified row counts for the default (seed=0) design."""
    rng = np.random.default_rng(0)
    result = simulate_combinatorial(16, 2, {3, 7}, 0.0, rng)
    expected_t = build_pooling_matrix(16, 2, seed=0).shape[0]
    assert result.ci_runs == expected_t
    assert result.decoded == {3, 7}
    assert result.logical_rounds == 1


# --- correctness of the comparison / sweep plumbing -----------------------


def test_compare_approaches_runs_all_three_on_the_same_fault_set():
    rng = np.random.default_rng(0)
    result = compare_approaches(16, 2, 2, 0.0, rng)
    assert isinstance(result, ComparisonResult)
    assert len(result.true_faulty) == 2
    # clean case: all three should recover the exact truth
    assert result.bisection.decoded == result.true_faulty
    assert result.brute_force.decoded == result.true_faulty
    assert result.combinatorial.decoded == result.true_faulty


def test_clean_case_all_approaches_are_100pct_accurate():
    sweep = run_benchmark_sweep(16, 2, 2, 0.0, 100, trial_seed=0)
    summary = sweep.summary()
    for approach in ("bisection", "brute_force", "combinatorial"):
        assert summary[approach]["accuracy"] == 1.0, approach


def test_combinatorial_uses_meaningfully_fewer_ci_runs_than_brute_force_at_n16():
    """Success criterion #2 (build-plan.md Section 4): meaningfully fewer
    CI runs than brute force at typical batch sizes with 1-2 faults."""
    sweep = run_benchmark_sweep(16, 2, 1, 0.0, 50, trial_seed=0)
    summary = sweep.summary()
    assert summary["combinatorial"]["avg_ci_runs"] < summary["brute_force"]["avg_ci_runs"]


def test_bisection_needs_far_more_logical_rounds_than_the_design():
    """Success criterion #1: ~k*log2(N) sequential rounds for bisection vs
    1 (occasionally 2) for the design."""
    sweep = run_benchmark_sweep(32, 2, 2, 0.0, 50, trial_seed=0)
    summary = sweep.summary()
    assert summary["combinatorial"]["avg_logical_rounds"] <= 2
    assert summary["bisection"]["avg_logical_rounds"] > summary["combinatorial"]["avg_logical_rounds"]
