# Flake-rate calibration (Milestone 2.5)
#
# Milestone 2's flake_rate parameter is the probability a single pooled-test
# *outcome* is corrupted (flipped) - a worst-case, per-observation rate. But
# nobody hands you "flake_rate" directly: what's actually known (or
# estimable) about a customer's repo is closer to industry-published figures
# on individual *test* flakiness, plus how many retries their CI already
# does before reporting a final result. This module derives the former from
# the latter.
from __future__ import annotations

# Google's widely-cited flaky-test figures (Micco, GTAC 2016 / "Flaky Tests
# at Google and How We Mitigate Them"): ~1.5% of all individual test *runs*
# report a flaky (inconsistent) result, aggregated across the whole test
# population - and that flakiness is concentrated in ~16% of distinct tests
# (the other ~84% never flake). These are the two defaults below.
GOOGLE_AGGREGATE_PER_TEST_FLAKE_RATE = 0.015
GOOGLE_FRACTION_TESTS_EVER_FLAKY = 0.16


def calibrate_flake_rate(
    tests_per_run: int,
    *,
    retries: int = 0,
    per_test_flake_rate: float = GOOGLE_AGGREGATE_PER_TEST_FLAKE_RATE,
    fraction_tests_ever_flaky: float = GOOGLE_FRACTION_TESTS_EVER_FLAKY,
) -> float:
    """Derive the effective per-pooled-outcome corruption probability
    (Milestone 2's flake_rate) from industry test-flakiness figures and a
    retry policy, rather than assuming a number.

    Model: a test suite has `tests_per_run` tests executed per pooled CI
    run (typically the whole suite, since merge-queue CI usually reruns it
    for every pooled combination rather than a PR-specific subset). A
    fraction `fraction_tests_ever_flaky` of those tests are flaky; each
    flaky test misfires on any given attempt with probability
    `per_test_flake_rate / fraction_tests_ever_flaky` (the *conditional*
    rate among flaky tests - dividing out the population share recovers it
    from the *aggregate*, population-wide rate that's actually published).

    If the CI already retries a failing test `retries` times before
    accepting a result, a flaky test only corrupts the final pooled outcome
    if it misfires on the original attempt *and* every retry - probability
    conditional_rate ** (retries + 1) - which is what actually reaches
    Tryage's decoder. The returned flake_rate is the probability that at
    least one of the (fraction_tests_ever_flaky * tests_per_run) flaky
    tests in the pool produces such a persistent, retry-surviving misfire.

    This is a Poisson-style aggregate estimate, not an exact combinatorial
    count - fine for calibrating a single scalar knob, not intended as a
    precise per-repo prediction. It also assumes flaky-test misfires are
    independent across retries, which understates risk for "systemic"
    flakiness (e.g. a broken shared fixture that fails deterministically
    within a given run) - see docs/build-plan.md Milestone 2.5 for the
    caveat and worked examples.
    """
    if tests_per_run < 1:
        raise ValueError("tests_per_run must be >= 1")
    if retries < 0:
        raise ValueError("retries must be >= 0")
    if not 0.0 < fraction_tests_ever_flaky <= 1.0:
        raise ValueError("fraction_tests_ever_flaky must be in (0, 1]")
    if not 0.0 <= per_test_flake_rate <= fraction_tests_ever_flaky:
        raise ValueError(
            "per_test_flake_rate must be in [0, fraction_tests_ever_flaky] "
            "(it's an aggregate rate across ALL tests, so it can't exceed "
            "the flaky population's share)"
        )

    conditional_rate = per_test_flake_rate / fraction_tests_ever_flaky
    residual_rate = conditional_rate ** (retries + 1)
    num_flaky_tests = fraction_tests_ever_flaky * tests_per_run
    prob_no_persistent_misfire = (1 - residual_rate) ** num_flaky_tests
    return 1 - prob_no_persistent_misfire
