# Real, named-competitor comparison (Milestone 4/5 extension) - replays
# every confirmed competitor method (engine/competitors.py) against the
# SAME real historical batches already used to validate this engine's own
# accuracy (engine/replay.py's mayastor cases, engine/replay_rust.py's
# rust-lang/rust rollup cases), not synthetic trials.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adaptive import AdaptiveDecoder, compute_cleared_and_suspects
from benchmark import ApproachResult, simulate_bisection, simulate_brute_force
from competitors import simulate_github_queue, simulate_graphite, simulate_mergify
from design import build_pooling_matrix
from replay import run_full_replay as run_mayastor_replay
from replay_rust import run_full_replay as run_rust_replay
from simulate import run_clean


def real_cases() -> list[tuple[int, set[int]]]:
    """Every real, already-validated historical case from both datasets:
    (batch size, culprit indices within that batch), using the same index
    mapping (position within the batch's PR list) already used for this
    engine's own replay - so every method is compared on identical inputs."""
    cases = []
    for r in run_mayastor_replay():
        prs = r.inference.failed_batch.prs
        pr_to_idx = {pr: i for i, pr in enumerate(prs)}
        cases.append((len(prs), {pr_to_idx[pr] for pr in r.inference.likely_culprits}))
    for r in run_rust_replay():
        prs = r.case.prs
        pr_to_idx = {pr: i for i, pr in enumerate(prs)}
        cases.append((len(prs), {pr_to_idx[pr] for pr in r.case.culprits}))
    return cases


def our_engine(n: int, culprits: set[int], k: int = 1, seed: int = 0) -> ApproachResult:
    """k=1 is the recommended default, not k=2 (Milestones 1-4's original
    default): real historical fault counts are overwhelmingly single-fault
    (171/178 = 96% across both real datasets - see
    test_real_fault_counts_are_overwhelmingly_single_fault). Sizing the
    design for the realistic common case instead of a worst-case tolerance
    target, and leaning on Milestone 3's already-validated AdaptiveDecoder
    fallback for the rare multi-fault cases, cuts total CI-run cost by
    ~41% (1526 -> 903 across the 178 real cases) while the average round
    count barely moves (1.01 -> 1.04, since the fallback triggers on only
    ~4% of real cases) - see docs/build-plan.md "Competitor methodology"
    for the full real-data comparison this produces.
    """
    matrix = build_pooling_matrix(n, k, seed=seed)
    outcomes = run_clean(matrix, culprits)
    decoder = AdaptiveDecoder(matrix, k)

    def run_pool(subset):
        return bool(subset & culprits)

    decoded, used_fallback = decoder.decode(outcomes, run_pool)
    t = matrix.shape[0]
    if used_fallback:
        _, suspects = compute_cleared_and_suspects(matrix, outcomes)
        extra = max(len(suspects), 1)
        return ApproachResult(
            decoded=frozenset(decoded), logical_rounds=2, ci_runs=t + extra, waves=2
        )
    return ApproachResult(decoded=frozenset(decoded), logical_rounds=1, ci_runs=t, waves=1)


METHODS = {
    "our_engine (k=2, original default)": lambda n, c: our_engine(n, c, k=2),
    "our_engine (k=1, recommended default)": lambda n, c: our_engine(n, c, k=1),
    "bisection (Aviator/bors-ng, optimistic)": lambda n, c: simulate_bisection(
        n, c, 0.0, np.random.default_rng(0)
    ),
    "bisection (Aviator/bors-ng, conservative)": lambda n, c: simulate_bisection(
        n, c, 0.0, np.random.default_rng(0), test_both_halves=True
    ),
    "mergify (arity=2, optimistic)": lambda n, c: simulate_mergify(
        n, c, 0.0, np.random.default_rng(0), arity=2
    ),
    "mergify (arity=2, conservative)": lambda n, c: simulate_mergify(
        n, c, 0.0, np.random.default_rng(0), arity=2, test_all_pieces=True
    ),
    "mergify (arity=4, optimistic)": lambda n, c: simulate_mergify(
        n, c, 0.0, np.random.default_rng(0), arity=4
    ),
    "graphite (dual-strategy, optimistic)": lambda n, c: simulate_graphite(
        n, c, 0.0, np.random.default_rng(0)
    ),
    "graphite (dual-strategy, conservative)": lambda n, c: simulate_graphite(
        n, c, 0.0, np.random.default_rng(0), test_both_halves=True
    ),
    "github native queue": lambda n, c: simulate_github_queue(n, c, 0.0, np.random.default_rng(0)),
    "brute force (naive baseline)": lambda n, c: simulate_brute_force(
        n, c, 0.0, np.random.default_rng(0)
    ),
}


@dataclass
class MethodTotals:
    correct: int = 0
    ci_runs: int = 0
    rounds: int = 0


def run_full_comparison(cases: list[tuple[int, set[int]]] | None = None) -> dict[str, MethodTotals]:
    if cases is None:
        cases = real_cases()
    totals = {name: MethodTotals() for name in METHODS}
    for n, culprits in cases:
        for name, fn in METHODS.items():
            result = fn(n, culprits)
            t = totals[name]
            t.correct += int(result.decoded == culprits)
            t.ci_runs += result.ci_runs
            t.rounds += result.logical_rounds
    return totals


def print_comparison_report(totals: dict[str, MethodTotals], n_cases: int) -> None:
    bf_ci = totals["brute force (naive baseline)"].ci_runs
    header = f"{'method':<45} {'correct':>9} {'avg_rounds':>11} {'total_ci':>9} {'vs_bf_ci':>9}"
    print(header)
    print("-" * len(header))
    for name, t in totals.items():
        avg_rounds = t.rounds / n_cases
        pct_ci_vs_bf = 100 * (1 - t.ci_runs / bf_ci)
        print(
            f"{name:<45} {t.correct:>6}/{n_cases:<3} {avg_rounds:>11.2f} "
            f"{t.ci_runs:>9} {pct_ci_vs_bf:>8.1f}%"
        )


if __name__ == "__main__":
    cases = real_cases()
    totals = run_full_comparison(cases)
    print(f"Total real cases: {len(cases)}\n")
    print_comparison_report(totals, len(cases))
