# Formal simulation & benchmark harness (Milestone 4)
#
# Compares three approaches to resolving a merge-queue batch failure, side
# by side, on the same injected fault/flake scenario:
#   (a) sequential bisection - today's standard (every existing tool)
#   (b) brute-force parallel isolation - Graphite's current approach
#   (c) this engine's combinatorial design (Milestones 1-3)
#
# Three metrics are tracked per approach, deliberately kept distinct because
# they answer different questions:
#   - logical_rounds: the true sequential-dependency chain length (this is
#     THE number behind "1 parallel round vs k*log2(N) sequential rounds" -
#     bisection's rounds are inherently sequential, since each step's
#     composition depends on the previous step's result, so no amount of
#     parallelism helps it; brute force and the combinatorial design are 1
#     logical round each (2 for the design if its fallback triggers),
#     regardless of how many jobs are IN that round.
#   - ci_runs: total CI jobs actually run - the cost/efficiency metric.
#   - waves: ci_runs actually chunked by a parallelism ceiling, respecting
#     the logical dependency structure - the realistic wall-clock proxy
#     (Section 7's risk: "the cost advantage over brute force assumes you
#     can run several pooled groups in parallel" - this is where that
#     assumption gets modeled explicitly instead of assumed away).
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from adaptive import AdaptiveDecoder, compute_cleared_and_suspects
from decode import ProbabilisticDecoder
from design import build_pooling_matrix
from simulate import run_clean, run_noisy


@dataclass
class ApproachResult:
    decoded: frozenset[int]
    logical_rounds: int
    ci_runs: int
    waves: int


def _waves_for(job_counts: list[int], parallelism_ceiling: int | None) -> int:
    if not parallelism_ceiling:
        return len(job_counts)
    return sum(math.ceil(c / parallelism_ceiling) for c in job_counts if c > 0) or 1


def _noisy_read(is_fault_present: bool, flake_rate: float, rng: np.random.Generator) -> bool:
    return bool(is_fault_present) ^ bool(rng.random() < flake_rate)


def simulate_bisection(
    n: int, faulty: set[int], flake_rate: float, rng: np.random.Generator
) -> ApproachResult:
    """One full bisection pass finds a single culprit via binary search
    (test one half; if it fails, recurse into it, else the other half);
    repeated - find one, remove it, repeat - until a check of the whole
    remaining candidate set passes clean. Every single test here is
    sequential: the next test's composition depends on this one's result,
    so logical_rounds == ci_runs == waves (parallelism cannot help a
    dependency chain), matching "the sequential one-at-a-time bisection
    used by every existing tool today" (build-plan.md Section 1.1).

    Under noise, a single corrupted read can send the search down the
    wrong branch or "find" a phantom culprit - this isn't special-cased,
    it's simulated directly (a fresh noisy read every single test, exactly
    as a real flaky CI run would produce), which is the honest way to
    measure how fragile plain bisection actually is under real flakiness.
    """
    remaining = list(range(n))
    found: set[int] = set()
    ci_runs = 0

    def run_pool(subset: list[int]) -> bool:
        nonlocal ci_runs
        ci_runs += 1
        return _noisy_read(any(i in faulty for i in subset), flake_rate, rng)

    while remaining:
        if not run_pool(remaining):
            break
        pool = list(remaining)
        while len(pool) > 1:
            mid = len(pool) // 2
            left = pool[:mid]
            pool = left if run_pool(left) else pool[mid:]
        culprit = pool[0]
        found.add(culprit)
        remaining.remove(culprit)

    return ApproachResult(
        decoded=frozenset(found), logical_rounds=ci_runs, ci_runs=ci_runs, waves=ci_runs
    )


def simulate_brute_force(
    n: int,
    faulty: set[int],
    flake_rate: float,
    rng: np.random.Generator,
    parallelism_ceiling: int | None = None,
) -> ApproachResult:
    """Test every item individually, all in one logical round (no
    dependency between them) - Graphite's current approach. A low
    parallelism ceiling chunks that one round into multiple waves, but
    doesn't change the logical round count."""
    decoded = {i for i in range(n) if _noisy_read(i in faulty, flake_rate, rng)}
    return ApproachResult(
        decoded=frozenset(decoded),
        logical_rounds=1,
        ci_runs=n,
        waves=_waves_for([n], parallelism_ceiling),
    )


def simulate_combinatorial(
    n: int,
    k: int,
    faulty: set[int],
    flake_rate: float,
    rng: np.random.Generator,
    *,
    parallelism_ceiling: int | None = None,
    design_seed: int | None = 0,
    min_distance: int = 1,
    matrix: np.ndarray | None = None,
    prior_fault_rate: float | None = None,
    confidence_threshold: float = 0.9,
    suspect_posterior_floor: float = 0.05,
) -> ApproachResult:
    """Round 1: the pooling design's T tests.

    flake_rate == 0 is delegated entirely to Milestone 3's already-verified
    AdaptiveDecoder (exact table match + complement-pool verification,
    falling back to compute_cleared_and_suspects when needed) rather than
    reimplemented here - that's the one guaranteed-100%-correct path this
    engine has, clean or exceeded-k, and it would be dishonest to report a
    weaker ad hoc substitute for it in a benchmark whose whole point is
    accuracy comparison. See docs/build-plan.md Milestone 4 for why this
    split exists: a first cut of this function used only the Milestone 2
    confidence-threshold check for every case, which cannot detect an
    exceeded-k situation the way the exact-match+complement check can, and
    measured 18-20% accuracy on clean exceeded-k trials that Milestone 3
    already proved should be 100%.

    flake_rate > 0 uses the Milestone 2 Bayesian posterior instead - there
    is no verified noisy+exceeded-k decoder in this engine yet, so this is
    an honest, clearly-scoped approximation, not a hardened path: if the
    top hypothesis's posterior is below confidence_threshold, round 2
    individually re-tests every item whose marginal posterior exceeds
    suspect_posterior_floor (a small, targeted, confidence-driven
    follow-up, not full brute force), and that fresh, direct noisy read
    overrides round 1's fuzzier evidence for whichever items got re-tested.
    """
    if matrix is None:
        matrix = build_pooling_matrix(n, k, seed=design_seed, min_distance=min_distance)
    t = matrix.shape[0]

    if flake_rate == 0.0:
        outcomes = run_clean(matrix, faulty)
        adaptive = AdaptiveDecoder(matrix, k)

        def run_pool(subset: set[int]) -> bool:
            return bool(subset & faulty)

        decoded, used_fallback = adaptive.decode(outcomes, run_pool)
        if not used_fallback:
            return ApproachResult(
                decoded=frozenset(decoded),
                logical_rounds=1,
                ci_runs=t,
                waves=_waves_for([t], parallelism_ceiling),
            )
        _, suspects = compute_cleared_and_suspects(matrix, outcomes)
        extra = max(len(suspects), 1)
        return ApproachResult(
            decoded=frozenset(decoded),
            logical_rounds=2,
            ci_runs=t + extra,
            waves=_waves_for([t, extra], parallelism_ceiling),
        )

    outcomes = run_noisy(matrix, faulty, flake_rate, rng)
    decoder = ProbabilisticDecoder(
        matrix, k, flake_rate, prior_fault_rate=prior_fault_rate
    )
    pairs = decoder.combo_posterior(outcomes)
    best_combo, best_p = pairs[0]
    decoded = set(best_combo)

    if best_p >= confidence_threshold:
        return ApproachResult(
            decoded=frozenset(decoded),
            logical_rounds=1,
            ci_runs=t,
            waves=_waves_for([t], parallelism_ceiling),
        )

    posterior = decoder.posterior(outcomes)
    suspects = {i for i, p in posterior.items() if p >= suspect_posterior_floor}
    for i in suspects:
        if _noisy_read(i in faulty, flake_rate, rng):
            decoded.add(i)
        else:
            decoded.discard(i)

    return ApproachResult(
        decoded=frozenset(decoded),
        logical_rounds=2,
        ci_runs=t + len(suspects),
        waves=_waves_for([t, len(suspects)], parallelism_ceiling),
    )


@dataclass
class ComparisonResult:
    n: int
    k: int
    fault_count: int
    flake_rate: float
    true_faulty: frozenset[int]
    bisection: ApproachResult
    brute_force: ApproachResult
    combinatorial: ApproachResult


def compare_approaches(
    n: int,
    k: int,
    fault_count: int,
    flake_rate: float,
    rng: np.random.Generator,
    *,
    parallelism_ceiling: int | None = None,
    design_seed: int | None = 0,
    min_distance: int = 1,
    matrix: np.ndarray | None = None,
) -> ComparisonResult:
    faulty = set(rng.choice(n, size=fault_count, replace=False).tolist())
    bisection = simulate_bisection(n, faulty, flake_rate, rng)
    brute_force = simulate_brute_force(
        n, faulty, flake_rate, rng, parallelism_ceiling=parallelism_ceiling
    )
    combinatorial = simulate_combinatorial(
        n,
        k,
        faulty,
        flake_rate,
        rng,
        parallelism_ceiling=parallelism_ceiling,
        design_seed=design_seed,
        min_distance=min_distance,
        matrix=matrix,
    )
    return ComparisonResult(
        n=n,
        k=k,
        fault_count=fault_count,
        flake_rate=flake_rate,
        true_faulty=frozenset(faulty),
        bisection=bisection,
        brute_force=brute_force,
        combinatorial=combinatorial,
    )


@dataclass
class BenchmarkSweepResult:
    n: int
    k: int
    fault_count: int
    flake_rate: float
    parallelism_ceiling: int | None
    num_trials: int
    trials: list[ComparisonResult] = field(default_factory=list)

    def _accuracy(self, approach: str) -> float:
        hits = sum(
            1
            for c in self.trials
            if getattr(c, approach).decoded == c.true_faulty
        )
        return hits / self.num_trials

    def _avg(self, approach: str, attr: str) -> float:
        return sum(getattr(getattr(c, approach), attr) for c in self.trials) / self.num_trials

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for approach in ("bisection", "brute_force", "combinatorial"):
            out[approach] = {
                "accuracy": self._accuracy(approach),
                "avg_logical_rounds": self._avg(approach, "logical_rounds"),
                "avg_ci_runs": self._avg(approach, "ci_runs"),
                "avg_waves": self._avg(approach, "waves"),
            }
        return out


def run_benchmark_sweep(
    n: int,
    k: int,
    fault_count: int,
    flake_rate: float,
    num_trials: int,
    *,
    parallelism_ceiling: int | None = None,
    design_seed: int | None = 0,
    min_distance: int = 1,
    trial_seed: int | None = None,
) -> BenchmarkSweepResult:
    matrix = build_pooling_matrix(n, k, seed=design_seed, min_distance=min_distance)
    rng = np.random.default_rng(trial_seed)
    result = BenchmarkSweepResult(
        n=n,
        k=k,
        fault_count=fault_count,
        flake_rate=flake_rate,
        parallelism_ceiling=parallelism_ceiling,
        num_trials=num_trials,
    )
    for _ in range(num_trials):
        result.trials.append(
            compare_approaches(
                n,
                k,
                fault_count,
                flake_rate,
                rng,
                parallelism_ceiling=parallelism_ceiling,
                design_seed=design_seed,
                min_distance=min_distance,
                matrix=matrix,
            )
        )
    return result


def report_header() -> str:
    header = (
        f"{'N':>4} {'k':>3} {'faults':>7} {'flake':>6} {'ceil':>5} {'approach':<13} "
        f"{'accuracy':>9} {'rounds':>7} {'ci_runs':>8} {'waves':>6}"
    )
    return header + "\n" + "-" * len(header)


def print_sweep_rows(sweep: BenchmarkSweepResult) -> None:
    """Print one sweep's rows immediately - used both by the batch report
    below and by callers that want a result printed as soon as it's
    computed, not just after an entire (possibly slow) battery finishes."""
    summary = sweep.summary()
    ceil_str = str(sweep.parallelism_ceiling) if sweep.parallelism_ceiling else "inf"
    for approach, label in (
        ("bisection", "bisection"),
        ("brute_force", "brute-force"),
        ("combinatorial", "combinatorial"),
    ):
        s = summary[approach]
        print(
            f"{sweep.n:>4} {sweep.k:>3} {sweep.fault_count:>7} "
            f"{sweep.flake_rate:>6.0%} {ceil_str:>5} {label:<13} "
            f"{s['accuracy']:>8.1%} {s['avg_logical_rounds']:>7.2f} "
            f"{s['avg_ci_runs']:>8.2f} {s['avg_waves']:>6.2f}",
            flush=True,
        )


def print_benchmark_report(sweeps: list[BenchmarkSweepResult]) -> None:
    print(report_header())
    for sweep in sweeps:
        print_sweep_rows(sweep)


if __name__ == "__main__":
    print(report_header())
    for n in (8, 16, 32):
        for fault_count in (1, 2):
            for flake_rate in (0.0, 0.15):
                sweep = run_benchmark_sweep(n, 2, fault_count, flake_rate, 200, trial_seed=0)
                print_sweep_rows(sweep)
