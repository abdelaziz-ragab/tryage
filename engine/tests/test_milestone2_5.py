import pytest

from calibration import calibrate_flake_rate


def test_more_retries_always_lowers_calibrated_flake_rate():
    rates = [calibrate_flake_rate(500, retries=r) for r in range(5)]
    assert rates == sorted(rates, reverse=True)


def test_more_tests_per_run_always_raises_calibrated_flake_rate():
    rates = [calibrate_flake_rate(s, retries=2) for s in (50, 200, 1000, 5000)]
    assert rates == sorted(rates)


def test_zero_retries_matches_hand_derived_values():
    # S=100, 0 retries: expect ~79.3% (see conversation notes / build-plan
    # Milestone 2.5 for the derivation this checks).
    assert calibrate_flake_rate(100, retries=0) == pytest.approx(0.793, abs=0.001)


def test_realistic_large_monorepo_with_modest_retries_lands_in_tested_range():
    """The concrete, honest finding this milestone produced: a large
    monorepo (S~2000 tests) with a modest (2-retry) mitigation policy - not
    even a weak one - calibrates to ~23% corruption probability, squarely
    inside the 15-25% range Milestone 2 already tested and found collapses
    accuracy. This is NOT a case where calibration rescues the picture."""
    calibrated = calibrate_flake_rate(2000, retries=2)
    assert 0.15 <= calibrated <= 0.30


def test_aggressive_retry_policy_recovers_a_low_flake_rate():
    # Same large suite, one more retry: drops below the tested range,
    # showing the lever that actually matters is retry depth, not the
    # suite-size assumption.
    calibrated = calibrate_flake_rate(2000, retries=3)
    assert calibrated < 0.05


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        calibrate_flake_rate(0)
    with pytest.raises(ValueError):
        calibrate_flake_rate(100, retries=-1)
    with pytest.raises(ValueError):
        calibrate_flake_rate(100, fraction_tests_ever_flaky=0.0)
    with pytest.raises(ValueError):
        # per_test_flake_rate can't exceed the flaky population's share
        calibrate_flake_rate(100, per_test_flake_rate=0.5, fraction_tests_ever_flaky=0.1)
