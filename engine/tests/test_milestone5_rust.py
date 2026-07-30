import pytest

from replay_rust import (
    RustRollupCase,
    load_batch_sizes,
    load_cases,
    replay_case,
    run_full_replay,
    structural_savings_by_n,
)


def test_loads_real_cases():
    cases = load_cases()
    assert len(cases) == 162


def test_loads_real_batch_sizes():
    sizes = load_batch_sizes()
    assert len(sizes) == 249
    assert max(sizes) >= 30  # real rollups of this scale were observed


def test_structural_savings_hit_target_range_at_real_large_n():
    """Section 4 criterion #2 (40-60% fewer CI runs than brute force),
    tested against real batch sizes rust-lang/rust actually uses for its
    rollups - not synthetic N. This is the gap openebs/mayastor's data
    (real batches of only 2-4 PRs) couldn't test."""
    sizes = load_batch_sizes()
    savings = structural_savings_by_n(sizes)
    # N=25 and N=30 were both real, observed rollup sizes in this dataset.
    assert 25 in savings and 30 in savings
    _, ratio_25 = savings[25]
    _, ratio_30 = savings[30]
    assert 0.4 <= 1 - ratio_25 <= 0.6
    assert 0.4 <= 1 - ratio_30 <= 0.6


def test_savings_ratio_is_better_at_largest_real_n_than_at_16():
    """Row count isn't strictly monotonic in N (the per-N greedy search can
    land on a higher T at N=17 than N=16, then hold that T while N keeps
    growing - a real, expected property of searching each N independently,
    not a bug), so this checks the honest, non-trivial claim: the largest
    real batch sizes actually used end up meaningfully cheaper than N=16,
    not a claim of monotonic improvement at every step."""
    sizes = load_batch_sizes()
    savings = structural_savings_by_n(sizes)
    _, ratio_16 = savings[16]
    for n in (28, 30, 34):
        _, ratio_n = savings[n]
        assert ratio_n < ratio_16


def test_real_rust_replay_is_fully_correct():
    """162 real historical rollup failures, each with an explicit
    bot-named culprit (not inferred) - the engine must decode every one
    exactly right."""
    results = run_full_replay()
    assert len(results) == 162
    assert all(r.engine_correct for r in results)


def test_engine_rounds_never_exceed_two():
    results = run_full_replay()
    assert all(r.engine_rounds <= 2 for r in results)


def test_replay_case_rejects_culprit_outside_batch():
    case = RustRollupCase(pr=1, prs=[10, 20, 30], culprits=[99], actual_rounds=1)
    assert replay_case(case) is None


@pytest.mark.parametrize("n", [16, 20, 25, 30])
def test_engine_ci_runs_never_exceed_brute_force_at_these_real_sizes(n):
    sizes = load_batch_sizes()
    savings = structural_savings_by_n(sizes)
    assert n in savings
    t, ratio = savings[n]
    assert t <= n
