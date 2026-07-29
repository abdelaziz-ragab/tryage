# Noisy-decode simulation harness (Milestone 2; the fuller multi-approach
# comparison harness lands in Milestone 4 - this covers exactly what
# Milestone 2 needs to validate the probabilistic decoder before anything
# else gets built on top of it).
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adaptive import AdaptiveDecoder
from decode import ProbabilisticDecoder
from design import build_pooling_matrix
from simulate import run_clean, run_noisy


@dataclass
class TrialResult:
    true_faulty: frozenset[int]
    threshold_predicted: frozenset[int]
    topm_predicted: frozenset[int]


@dataclass
class SweepResult:
    n: int
    k: int
    flake_rate: float
    num_trials: int
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def threshold_exact_match_rate(self) -> float:
        hits = sum(1 for t in self.trials if t.threshold_predicted == t.true_faulty)
        return hits / self.num_trials

    @property
    def topm_exact_match_rate(self) -> float:
        hits = sum(1 for t in self.trials if t.topm_predicted == t.true_faulty)
        return hits / self.num_trials

    @property
    def item_precision_recall_f1(self) -> tuple[float, float, float]:
        """Per-item precision/recall/F1 for the threshold-based prediction,
        pooled across all trials (i.e. treating every (trial, item) pair as
        one binary classification instance)."""
        tp = fp = fn = 0
        for t in self.trials:
            tp += len(t.threshold_predicted & t.true_faulty)
            fp += len(t.threshold_predicted - t.true_faulty)
            fn += len(t.true_faulty - t.threshold_predicted)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return precision, recall, f1


def run_sweep(
    n: int,
    k: int,
    flake_rate: float,
    num_trials: int,
    *,
    prior_fault_rate: float | None = None,
    design_seed: int | None = 0,
    trial_seed: int | None = None,
    min_distance: int = 1,
    max_t: int | None = None,
    tries_per_item: int = 1000,
    restarts: int = 1,
) -> SweepResult:
    """Run num_trials randomized trials at a fixed flake_rate: for each
    trial, pick a random fault count in [0, k], inject that many random
    faults, simulate a noisy pooled-test outcome, decode it, and record how
    the prediction compares to the true injected fault set.

    One pooling matrix and decoder are built once per (n, k) and reused
    across all trials - that's the realistic scenario (the design is fixed
    per repo/batch-size, only which PRs are actually faulty and which tests
    flake varies trial to trial).

    min_distance (and max_t, tries_per_item, which only matter when
    min_distance > 1) are forwarded to build_pooling_matrix - see its
    docstring and design.py's Milestone 2 notes: min_distance=1 is the
    Milestone 1 default (merely-separable, no noise margin); raising it
    buys back decode accuracy under noise at the cost of more pooled tests.
    restarts is also forwarded - see design.py's notes on why a single
    search attempt can miss a cheaper design that more restarts would find,
    particularly in the small-N-relative-to-k regime.
    """
    matrix = build_pooling_matrix(
        n,
        k,
        seed=design_seed,
        min_distance=min_distance,
        max_t=max_t,
        tries_per_item=tries_per_item,
        restarts=restarts,
    )
    if prior_fault_rate is None:
        # Expected fault count if fault_count ~ Uniform(0, k) is k/2.
        prior_fault_rate = max(k / 2.0, 0.5) / n
    decoder = ProbabilisticDecoder(
        matrix, k, flake_rate, prior_fault_rate=prior_fault_rate
    )
    rng = np.random.default_rng(trial_seed)

    result = SweepResult(n=n, k=k, flake_rate=flake_rate, num_trials=num_trials)
    for _ in range(num_trials):
        fault_count = int(rng.integers(0, k + 1))
        true_faulty = frozenset(
            rng.choice(n, size=fault_count, replace=False).tolist()
        )
        outcomes = run_noisy(matrix, true_faulty, flake_rate, rng)

        posterior = decoder.posterior(outcomes)
        threshold_predicted = frozenset(
            i for i, p in posterior.items() if p > 0.5
        )
        topm_predicted = frozenset(
            sorted(posterior, key=lambda i: posterior[i], reverse=True)[:fault_count]
        )

        result.trials.append(
            TrialResult(
                true_faulty=true_faulty,
                threshold_predicted=threshold_predicted,
                topm_predicted=topm_predicted,
            )
        )
    return result


def print_flake_rate_report(
    n_values: list[int],
    k_values: list[int],
    flake_rates: list[float],
    num_trials: int,
    *,
    seed: int = 0,
) -> list[SweepResult]:
    """Run the sweep across every (n, k, flake_rate) combination and print a
    report table. Returns the raw results for further inspection/assertions."""
    header = (
        f"{'N':>4} {'k':>3} {'flake':>7} {'exact(thr)':>11} "
        f"{'exact(top-m)':>13} {'precision':>10} {'recall':>8} {'f1':>6}"
    )
    print(header)
    print("-" * len(header))
    results = []
    for n in n_values:
        for k in k_values:
            for flake_rate in flake_rates:
                sweep = run_sweep(
                    n, k, flake_rate, num_trials, trial_seed=seed
                )
                results.append(sweep)
                p, r, f1 = sweep.item_precision_recall_f1
                print(
                    f"{n:>4} {k:>3} {flake_rate:>7.0%} "
                    f"{sweep.threshold_exact_match_rate:>10.1%} "
                    f"{sweep.topm_exact_match_rate:>12.1%} "
                    f"{p:>10.1%} {r:>8.1%} {f1:>6.1%}"
                )
    return results


@dataclass
class AdaptiveTrialResult:
    true_faulty: frozenset[int]
    decoded: frozenset[int]
    used_fallback: bool
    extra_tests: int


@dataclass
class AdaptiveSweepResult:
    n: int
    k: int
    fault_count: int
    num_trials: int
    trials: list[AdaptiveTrialResult] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        hits = sum(1 for t in self.trials if t.decoded == t.true_faulty)
        return hits / self.num_trials

    @property
    def fallback_rate(self) -> float:
        return sum(1 for t in self.trials if t.used_fallback) / self.num_trials

    @property
    def avg_extra_tests_fraction(self) -> float:
        """Average real extra-test cost (beyond the original T-row round),
        as a fraction of n - counts every actual run_pool call, whether
        that's just the 1-test complement verification or the full
        suspects-based resolution."""
        return sum(t.extra_tests for t in self.trials) / (self.num_trials * self.n)


def run_adaptive_sweep(
    n: int,
    k: int,
    fault_count: int,
    num_trials: int,
    *,
    design_seed: int | None = 0,
    trial_seed: int | None = None,
) -> AdaptiveSweepResult:
    """Run num_trials randomized trials with exactly fault_count faults
    injected (typically fault_count > k, to exercise Milestone 3's
    fallback) against one fixed design, measuring both correctness and the
    real cost of the fallback (how many extra tests it actually needed
    beyond the original round, as a fraction of n) - the number that
    determines whether "one small, targeted follow-up round" holds up as a
    real claim or degenerates to full brute force in practice.
    """
    matrix = build_pooling_matrix(n, k, seed=design_seed)
    decoder = AdaptiveDecoder(matrix, k)
    rng = np.random.default_rng(trial_seed)

    result = AdaptiveSweepResult(n=n, k=k, fault_count=fault_count, num_trials=num_trials)
    for _ in range(num_trials):
        true_faulty = frozenset(rng.choice(n, size=fault_count, replace=False).tolist())
        outcomes = run_clean(matrix, true_faulty)

        call_count = 0

        def run_pool(subset: set[int], _faulty=true_faulty) -> bool:
            nonlocal call_count
            call_count += 1
            return bool(subset & _faulty)

        decoded, used_fallback = decoder.decode(outcomes, run_pool)

        result.trials.append(
            AdaptiveTrialResult(
                true_faulty=true_faulty,
                decoded=frozenset(decoded),
                used_fallback=used_fallback,
                extra_tests=call_count,
            )
        )
    return result


if __name__ == "__main__":
    print_flake_rate_report(
        n_values=[8, 16, 32],
        k_values=[1, 2, 3],
        flake_rates=[0.0, 0.05, 0.15, 0.25],
        num_trials=500,
    )
