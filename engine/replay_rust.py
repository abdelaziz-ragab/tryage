# Historical replay validation, second real data source (Milestone 5
# follow-up): rust-lang/rust's "rollup" system.
#
# openebs/mayastor (engine/replay.py) validated decode correctness and the
# rounds/speed advantage on real data, but its real batches were small
# (2-4 PRs) - well below Milestone 1's N>=16 threshold for CI-run savings,
# so it couldn't test that claim. rust-lang/rust deliberately batches many
# low-risk PRs into one combined "rollup" PR specifically to save CI cost,
# with real batch sizes regularly reaching into the 20s and 30s - exactly
# the range needed to test the CI-run efficiency claim (Section 4 #2) on
# real, actually-used batch compositions instead of only synthetic N.
#
# Two distilled data files (not the full raw comment/body text, which was
# used only to build these during initial research):
#   - rust_rollup_batch_sizes.json: every real batch size (N) observed
#     across 249 sampled rollup PRs (both failed and large-successful),
#     used for the structural T-vs-N efficiency comparison independent of
#     outcome/ground-truth.
#   - rust_rollup_cases.json: 162 real failed rollups with clean ground
#     truth - rust-bors's newer format explicitly names the culprit PR
#     ("PR #X, which is a member of this rollup, was unapproved... This
#     rollup was thus unapproved"), a direct signal rather than the
#     subset-inference heuristic mayastor's older/external-CI data needed.
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from adaptive import AdaptiveDecoder, compute_cleared_and_suspects
from decode import DecodeTable
from design import build_pooling_matrix
from simulate import run_clean

_DATA_DIR = os.path.join(os.path.dirname(__file__), "replay_data")
_CASES_PATH = os.path.join(_DATA_DIR, "rust_rollup_cases.json")
_SIZES_PATH = os.path.join(_DATA_DIR, "rust_rollup_batch_sizes.json")


def load_batch_sizes(path: str = _SIZES_PATH) -> list[int]:
    with open(path) as f:
        return json.load(f)


@dataclass
class RustRollupCase:
    pr: int
    prs: list[int]
    culprits: list[int]
    actual_rounds: int


def load_cases(path: str = _CASES_PATH) -> list[RustRollupCase]:
    with open(path) as f:
        raw = json.load(f)
    return [RustRollupCase(**c) for c in raw]


def structural_savings_by_n(
    n_values: list[int], k: int = 2, design_seed: int = 0
) -> dict[int, tuple[int, float]]:
    """For each distinct real batch size actually used, the row count (T)
    a k-separable design needs and its ratio to brute force (N) - the
    direct test of Section 4 criterion #2 ("meaningfully fewer CI runs")
    against real, not synthetic, batch sizes. Independent of any batch's
    outcome or ground truth - this is a structural design property."""
    results = {}
    for n in sorted(set(n_values)):
        if n < 2:
            continue
        matrix = build_pooling_matrix(n, k, seed=design_seed)
        assert DecodeTable(matrix, k).is_separable
        t = matrix.shape[0]
        results[n] = (t, t / n)
    return results


@dataclass
class RustReplayResult:
    case: RustRollupCase
    engine_decoded: frozenset[int]
    engine_correct: bool
    engine_rounds: int
    engine_ci_runs: int
    actual_rounds: int
    brute_force_ci_runs: int


def replay_case(case: RustRollupCase, k: int = 2, design_seed: int = 0) -> RustReplayResult | None:
    prs = case.prs
    n = len(prs)
    pr_to_idx = {pr: i for i, pr in enumerate(prs)}
    if any(c not in pr_to_idx for c in case.culprits):
        return None  # a named culprit wasn't in the parsed batch list
    culprit_idx = {pr_to_idx[c] for c in case.culprits}

    matrix = build_pooling_matrix(n, k, seed=design_seed)
    outcomes = run_clean(matrix, culprit_idx)
    decoder = AdaptiveDecoder(matrix, k)

    def run_pool(subset):
        return bool(subset & culprit_idx)

    decoded, used_fallback = decoder.decode(outcomes, run_pool)
    t = matrix.shape[0]
    if used_fallback:
        _, suspects = compute_cleared_and_suspects(matrix, outcomes)
        ci_runs = t + max(len(suspects), 1)
        rounds = 2
    else:
        ci_runs = t
        rounds = 1

    decoded_prs = frozenset(prs[i] for i in decoded)
    correct = decoded_prs == frozenset(case.culprits)
    return RustReplayResult(
        case=case,
        engine_decoded=decoded_prs,
        engine_correct=correct,
        engine_rounds=rounds,
        engine_ci_runs=ci_runs,
        actual_rounds=case.actual_rounds,
        brute_force_ci_runs=n,
    )


def run_full_replay(path: str = _CASES_PATH) -> list[RustReplayResult]:
    cases = load_cases(path)
    results = [replay_case(c) for c in cases]
    return [r for r in results if r is not None]
