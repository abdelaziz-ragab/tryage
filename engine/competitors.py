# Real, named-competitor simulators, built from CONFIRMED public technical
# sources (not assumptions) - see docs/build-plan.md "Competitor
# methodology research" for the full sourced writeup. This corrects and
# extends benchmark.py's generic three-way comparison (bisection / brute
# force / combinatorial), which characterized "brute force" as simply
# "Graphite's approach" - research confirmed that's outdated: Graphite
# actually runs a documented dual-strategy (parallel isolation for small
# batches, bisection for large ones), not pure brute force.
#
# Confirmed via each vendor's own docs/engineering blog (sources in
# build-plan.md):
#   - Aviator: bisection (docs.aviator.co/mergequeue/concepts/batching) -
#     already exactly modeled by benchmark.simulate_bisection.
#   - bors-ng: bisection (github.com/bors-ng/bors-ng) - already exactly
#     modeled by benchmark.simulate_bisection, and already validated
#     against real historical data in engine/replay.py.
#   - Mergify: an n-ary generalization of bisection (n = configured
#     parallel speculative checks + 1; degrades to plain binary bisection,
#     i.e. n=2, when parallelism isn't configured) - modeled here as
#     simulate_mergify.
#   - Graphite: documented dual-strategy - full parallel isolation
#     (brute force) by default for batches at or below their own
#     documented threshold (<=8 stacks) or with high historical failure
#     rate, bisection above that - modeled here as simulate_graphite,
#     composed directly from benchmark.py's existing two arms rather than
#     duplicating their logic.
#   - GitHub native merge queue: NOT bisection (confirmed absence in
#     GitHub's own docs) - a cumulative/incremental queue-ordered scheme:
#     each PR is tested on (base + every PR ahead of it in queue); on a
#     failure, the failing PR is dropped and every PR behind it gets its
#     branch recreated and retested. Modeled here as simulate_github_queue.
#
# Not modeled (excluded honestly rather than guessed): Trunk.io offers a
# "bisection" feature but publishes no algorithm detail beyond the name;
# Google's internal FACF (Flake Aware Culprit Finding) and Uber's
# SubmitQueue speculation-tree are real, well-documented systems but are
# internal/academic, not market competitors sold to the same customers -
# see build-plan.md for how they relate to this engine's own approach.
from __future__ import annotations

import math

import numpy as np

from benchmark import ApproachResult, _noisy_read, _waves_for, simulate_bisection, simulate_brute_force


def simulate_mergify(
    n: int,
    faulty: set[int],
    flake_rate: float,
    rng: np.random.Generator,
    *,
    arity: int = 2,
    test_all_pieces: bool = False,
) -> ApproachResult:
    """n-ary generalization of bisection: split a failing batch into up to
    `arity` pieces (not just 2), test arity-1 of them in parallel and infer
    the last by elimination (the standard bisection trick, generalized -
    Mergify hasn't published exact pseudocode beyond a prose example, so
    this is the natural, standard generalization of confirmed binary
    bisection, not a verified-verbatim reimplementation). arity=2 is
    Mergify's own documented fallback when parallel speculative checks
    aren't configured; arity=(max_parallel_checks+1) when they are.

    test_all_pieces mirrors benchmark.simulate_bisection's
    test_both_halves: whether the last piece is inferred by elimination
    (False, optimistic) or explicitly tested too (True, conservative) -
    the same real ambiguity applies here (no public confirmation of which
    convention Mergify's implementation uses), reported as a range.

    Each split level is ONE parallel round (tested together, unlike plain
    bisection's single half-test per round only in the sense that more
    pieces are tested at once - still fully sequential round-to-round,
    since the next split depends on this one's result).
    """
    if arity < 2:
        raise ValueError("arity must be >= 2")
    remaining = list(range(n))
    found: set[int] = set()
    ci_runs = 0
    logical_rounds = 0

    def run_pool(subset: list[int]) -> bool:
        nonlocal ci_runs
        ci_runs += 1
        return _noisy_read(any(i in faulty for i in subset), flake_rate, rng)

    while remaining:
        logical_rounds += 1
        if not run_pool(remaining):
            break
        pool = list(remaining)
        while len(pool) > 1:
            chunk = math.ceil(len(pool) / arity)
            pieces = [pool[i : i + chunk] for i in range(0, len(pool), chunk)]
            logical_rounds += 1
            test_set = pieces if test_all_pieces else pieces[:-1]
            # Bisection-style tie-breaking: with 2+ simultaneous faults,
            # more than one piece can test as failing. Recursing into ALL
            # of them (rather than just one) can reconstruct the very pool
            # being split with no shrinkage - a real infinite-loop bug this
            # caught during testing (test_all_pieces=True, 2 faults each in
            # a different singleton piece: both pieces fail, their union is
            # the original pool). Every piece in test_set is still tested
            # (no short-circuiting - test_all_pieces means paying for all
            # of them, matching test_both_halves's unconditional both-sides
            # cost in simulate_bisection), but only the FIRST failing one
            # is kept per level; the outer while-loop's repeat-until-clean
            # structure finds any other simultaneous faults on a later
            # pass, exactly like benchmark.simulate_bisection already does.
            results = [run_pool(piece) for piece in test_set]
            failing = next(
                (piece for piece, failed in zip(test_set, results) if failed), pieces[-1]
            )
            pool = failing
        culprit = pool[0]
        found.add(culprit)
        remaining.remove(culprit)

    return ApproachResult(
        decoded=frozenset(found),
        logical_rounds=logical_rounds,
        ci_runs=ci_runs,
        waves=logical_rounds,
    )


def simulate_graphite(
    n: int,
    faulty: set[int],
    flake_rate: float,
    rng: np.random.Generator,
    *,
    parallel_threshold: int = 8,
    parallelism_ceiling: int | None = None,
    test_both_halves: bool = False,
) -> ApproachResult:
    """Graphite's own documented policy (graphite.com/docs/merge-queue-optimizations,
    graphite.com/blog/the-first-stack-aware-merge-queue): full parallel
    isolation (brute force) for batches at or below their threshold
    (default <=8 stacks) or when historical failure rate is high; bisection
    above that. Historical-failure-rate weighting isn't modeled (no public
    formula for it) - this uses the size threshold only, the confirmed,
    concrete half of their policy.
    """
    if n <= parallel_threshold:
        return simulate_brute_force(
            n, faulty, flake_rate, rng, parallelism_ceiling=parallelism_ceiling
        )
    return simulate_bisection(n, faulty, flake_rate, rng, test_both_halves=test_both_halves)


def simulate_github_queue(
    n: int, faulty: set[int], flake_rate: float, rng: np.random.Generator
) -> ApproachResult:
    """GitHub's confirmed mechanism (docs.github.com "Managing a merge
    queue"): PRs are tested in queue order, each on a branch containing the
    target plus every PR ahead of it. Modeled as one round of N parallel
    cumulative tests; the earliest failing position is the confirmed
    culprit (every position at or after a bad PR also fails, since it's
    cumulative), which is removed, and every PR behind it is retested in
    the next round - shrinking each round rather than a binary search.

    No bisection anywhere in this - confirmed absent from GitHub's own
    docs - so cost degrades far worse than bisection under 2+ simultaneous
    faults: each fault costs a full retest of everything behind it, not
    log2(n).
    """
    order = list(range(n))
    found: set[int] = set()
    ci_runs = 0
    logical_rounds = 0
    start = 0

    while start < len(order):
        logical_rounds += 1
        segment = order[start:]
        cumulative_faulty = False
        first_failure_offset = None
        for offset, _ in enumerate(segment):
            cumulative_faulty = cumulative_faulty or (segment[offset] in faulty)
            result = _noisy_read(cumulative_faulty, flake_rate, rng)
            ci_runs += 1
            if result and first_failure_offset is None:
                first_failure_offset = offset
        if first_failure_offset is None:
            break
        culprit = segment[first_failure_offset]
        found.add(culprit)
        start += first_failure_offset + 1

    return ApproachResult(
        decoded=frozenset(found),
        logical_rounds=logical_rounds,
        ci_runs=ci_runs,
        waves=logical_rounds,
    )
