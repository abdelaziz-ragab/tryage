import itertools

import numpy as np
import pytest

from decode import DecodeTable, ProbabilisticDecoder
from design import build_pooling_matrix
from harness import run_sweep
from simulate import run_clean, run_noisy

BATCH_SIZES = (8, 16, 32)
FAULT_TOLERANCES = (1, 2, 3)
FLAKE_RATES = (0.0, 0.05, 0.15, 0.25)


# --- simulate.run_noisy ------------------------------------------------


def test_run_noisy_zero_flake_matches_clean():
    matrix = build_pooling_matrix(16, 2, seed=0)
    rng = np.random.default_rng(0)
    for faulty in ({}, {1}, {2, 5}):
        clean = run_clean(matrix, faulty)
        noisy = run_noisy(matrix, faulty, 0.0, rng)
        assert np.array_equal(clean, noisy)


def test_run_noisy_full_flake_inverts():
    matrix = build_pooling_matrix(16, 2, seed=0)
    rng = np.random.default_rng(0)
    faulty = {1, 5}
    clean = run_clean(matrix, faulty)
    noisy = run_noisy(matrix, faulty, 1.0, rng)
    assert np.array_equal(noisy, ~clean)


def test_run_noisy_empirical_flip_rate_matches_flake_rate():
    matrix = build_pooling_matrix(16, 2, seed=0)
    rng = np.random.default_rng(0)
    faulty = {1, 5}
    clean = run_clean(matrix, faulty)
    flake_rate = 0.2
    flips = 0
    trials = 5000
    for _ in range(trials):
        noisy = run_noisy(matrix, faulty, flake_rate, rng)
        flips += np.sum(noisy != clean)
    empirical_rate = flips / (trials * matrix.shape[0])
    assert empirical_rate == pytest.approx(flake_rate, abs=0.02)


def test_run_noisy_rejects_invalid_flake_rate():
    matrix = build_pooling_matrix(8, 1, seed=0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        run_noisy(matrix, {0}, 1.5, rng)


# --- decode.ProbabilisticDecoder ---------------------------------------


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("k", FAULT_TOLERANCES)
def test_combo_posterior_sums_to_one(n, k):
    matrix = build_pooling_matrix(n, k, seed=0)
    decoder = ProbabilisticDecoder(matrix, k, flake_rate=0.1)
    outcomes = run_clean(matrix, {0} if n > 0 else {})
    pairs = decoder.combo_posterior(outcomes)
    total = sum(p for _, p in pairs)
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n", (8, 16))
@pytest.mark.parametrize("k", (1, 2))
def test_zero_flake_recovers_exact_decode(n, k):
    """At flake_rate=0 the Bayesian decoder should agree with the exact
    Milestone 1 decoder on every fault combination it was built to tolerate,
    with the true combo's posterior essentially certain."""
    matrix = build_pooling_matrix(n, k, seed=0)
    exact_table = DecodeTable(matrix, k)
    assert exact_table.is_separable
    decoder = ProbabilisticDecoder(matrix, k, flake_rate=1e-9)

    for size in range(k + 1):
        for combo in itertools.islice(itertools.combinations(range(n), size), 20):
            faulty = set(combo)
            outcomes = run_clean(matrix, faulty)
            assert decoder.map_estimate(outcomes) == faulty
            posterior = decoder.posterior(outcomes)
            for i in faulty:
                assert posterior[i] > 0.99
            for i in set(range(n)) - faulty:
                assert posterior[i] < 0.01


def test_probabilistic_decoder_rejects_invalid_input():
    matrix = build_pooling_matrix(8, 1, seed=0)
    with pytest.raises(ValueError):
        ProbabilisticDecoder(matrix, 1, flake_rate=1.5)
    with pytest.raises(ValueError):
        ProbabilisticDecoder(matrix, 1, flake_rate=0.1, prior_fault_rate=0.0)


# --- design.py min_distance construction (the fix for the noise-collapse
# finding below) -----------------------------------------------------------


def test_min_distance_construction_achieves_requested_distance():
    """Regression test for the fix: build_pooling_matrix(..., min_distance=d)
    must actually deliver a design whose enumerated fault-combo outcomes are
    at least d Hamming-apart, not just distinct (d=1)."""
    n, k, min_distance = 16, 2, 3
    matrix = build_pooling_matrix(
        n, k, seed=0, min_distance=min_distance, max_t=30, tries_per_item=1500
    )
    codes = [
        run_clean(matrix, combo)
        for size in range(k + 1)
        for combo in itertools.combinations(range(n), size)
    ]
    min_observed = min(
        int(np.sum(a != b)) for a, b in itertools.combinations(codes, 2)
    )
    assert min_observed >= min_distance


def test_min_distance_unreachable_raises_instead_of_silently_falling_back():
    """Unlike the min_distance=1 case, identity is never a valid substitute
    for an unreachable min_distance>1 design - it only achieves distance 1 -
    so build_pooling_matrix must raise rather than quietly returning it."""
    with pytest.raises(RuntimeError):
        build_pooling_matrix(16, 2, seed=0, min_distance=3, max_t=17, tries_per_item=500)


# --- flake-rate sweep: THE critical Milestone 2 result ------------------
#
# Per docs/build-plan.md Milestone 2 & Section 4: >95% correct-culprit
# identification at realistic flake rates (10-25%) is the PoC's headline
# accuracy target, and a collapse here is a stop-and-reconsider signal, not
# something to quietly paper over with a lowered bar.
#
# Measured result at Milestone 1's designs (min_distance=1, "merely
# separable, no noise margin" - see engine/harness.py, run directly for the
# full table): exact fault-set identification is 100% at flake_rate=0 (as it
# must be - Milestone 1's clean-case guarantee), but degrades sharply well
# before reaching the 15-25% flake range - typically 25-65% exact-match at
# 15% flake and 25-55% at 25% flake, far short of the >=95% target. This
# held using both a marginal->0.5 threshold rule AND the single-best-combo
# (MAP) estimate, which rules out "bad decision threshold" as the
# explanation.
#
# Root cause (confirmed by direct measurement, not guessed): the Milestone 1
# pooling constructions are built to be merely *separable* - every fault
# combination up to k produces a distinct outcome pattern - with no
# minimum-Hamming-distance margin between those patterns. Measured minimum
# pairwise Hamming distance across enumerated fault combos was only 1 for
# every k>=2 design (2 for k=1's antichain code). With that little
# separation, a single flipped pooled-test result is often enough to make
# two fault hypotheses indistinguishable or to flip the decoder's answer.
#
# Fix applied: design.py's greedy construction now takes a min_distance
# parameter and rejects any candidate that doesn't keep every enumerated
# fault-combo outcome at least min_distance Hamming-apart (see
# test_min_distance_construction_achieves_requested_distance above). This
# does recover real accuracy - but at a steep, and highly batch-size-
# dependent, row cost. Measured (N=32, k=2): min_distance=3 -> T=28 (88% of
# brute force) reaches 82.7% at 15% flake; min_distance=7 -> T=47 (147% of
# brute force - MORE than brute force) is needed to reach 95.7% at 15%
# flake, and even then only reaches 73% at 25% flake. At N=64, k=2, the
# picture is meaningfully better: min_distance=7 -> T=55 (86% of brute
# force, i.e. still a real saving) reaches 96.7% at 15% flake - a genuine
# win - but 25% flake still needed min_distance=17 -> T=105 (164% of brute
# force) to cross 95%.
#
# Net honest conclusion: the min-distance fix is real and directionally
# correct (more redundancy -> more accuracy, exactly as coding theory
# predicts), and there IS a genuine window - larger batches (N>=64),
# moderate flake (~15%) - where a design can beat brute force in row count
# AND clear the 95% accuracy bar. But at small batches (N<=32) or high flake
# (~25%), closing the accuracy gap costs at or above brute-force rows,
# which undercuts the "meaningfully fewer CI runs than brute force" success
# criterion (build-plan.md Section 4, #2). A static, universal min_distance
# margin isn't the whole answer - see the conversation notes for the
# follow-up direction discussed (an adaptive, Milestone-3-style follow-up
# round triggered only on ambiguous/low-confidence decodes, so the
# redundancy cost is paid only when noise actually bites, not on every
# batch). This test intentionally keeps the min_distance=1 sweep below as
# the tracked baseline, since that's still what build_pooling_matrix's
# default produces.
def test_flake_rate_sweep_reports_honest_accuracy():
    num_trials = 300
    results = {}
    for n in BATCH_SIZES:
        for k in FAULT_TOLERANCES:
            for flake_rate in FLAKE_RATES:
                sweep = run_sweep(n, k, flake_rate, num_trials, trial_seed=0)
                results[(n, k, flake_rate)] = sweep

    print(f"\n{'N':>4} {'k':>3} {'flake':>7} {'exact(thr)':>11} {'exact(top-m)':>13}")
    for (n, k, flake_rate), sweep in results.items():
        print(
            f"{n:>4} {k:>3} {flake_rate:>7.0%} "
            f"{sweep.threshold_exact_match_rate:>10.1%} "
            f"{sweep.topm_exact_match_rate:>12.1%}"
        )

    # Clean case must still be perfect - this is Milestone 1's guarantee and
    # any regression here would mean the noisy decoder broke the noiseless
    # path, which would be a real bug.
    for n in BATCH_SIZES:
        for k in FAULT_TOLERANCES:
            assert results[(n, k, 0.0)].threshold_exact_match_rate == 1.0

    # Honest, currently-true property: accuracy degrades monotonically (does
    # not improve) as flake_rate increases, for every (n, k). This is the one
    # thing we can honestly assert about the noisy regime right now - NOT
    # that the >=95% target is met, because measured results show it isn't.
    for n in BATCH_SIZES:
        for k in FAULT_TOLERANCES:
            rates = [results[(n, k, fr)].threshold_exact_match_rate for fr in FLAKE_RATES]
            assert rates == sorted(rates, reverse=True), (
                f"n={n} k={k}: expected non-increasing accuracy as flake "
                f"rate rises, got {list(zip(FLAKE_RATES, rates))}"
            )


def test_min_distance_recovers_accuracy_under_noise():
    """Fast regression check for the fix direction itself (not the full
    honest table above, which is expensive): at a fixed, realistic flake
    rate, a min_distance-boosted design should decode meaningfully more
    accurately than the min_distance=1 baseline, confirming the row-count
    investment actually buys back accuracy rather than just satisfying the
    construction-level distance property in isolation."""
    n, k, flake_rate, num_trials = 32, 2, 0.15, 200
    baseline = run_sweep(
        n, k, flake_rate, num_trials, trial_seed=0, min_distance=1, max_t=32
    )
    boosted = run_sweep(
        n,
        k,
        flake_rate,
        num_trials,
        trial_seed=0,
        min_distance=5,
        max_t=80,
        tries_per_item=2000,
    )
    assert boosted.threshold_exact_match_rate > baseline.threshold_exact_match_rate + 0.2
