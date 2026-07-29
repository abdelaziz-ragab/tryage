# Historical replay validation (Milestone 5)
#
# Real merge-queue history, distilled from openebs/mayastor's public GitHub
# PR history (a real open-source project that used bors-ng for its merge
# queue - see docs/build-plan.md Milestone 5 for how this repo was found
# and validated as a real, active bors-ng dogfooder). The raw source is
# bors[bot]'s comments on ~831 PRs; replay_data/mayastor_bors_events.json
# is a distilled (job_type, job_id, outcome, timestamp) extract per PR,
# not the full comment text, so it's small enough to check into the repo
# and rerun offline without hitting the GitHub API again.
#
# Key real-data lesson (found, not assumed): Jenkins "staging" job IDs get
# reused/reset over time - two PRs three years apart shared the same
# job_id. Grouping by job_id alone silently produces false batches, so
# reconstruction also requires comment timestamps to fall within a bounded
# window of each other.
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from adaptive import AdaptiveDecoder, compute_cleared_and_suspects
from design import build_pooling_matrix
from simulate import run_clean

_DATA_PATH = os.path.join(os.path.dirname(__file__), "replay_data", "mayastor_bors_events.json")


def load_distilled_events(path: str = _DATA_PATH) -> list[tuple[int, str, int, str, str]]:
    """Each event: (pr_number, job_type ["trying"|"staging"], job_id, outcome
    ["success"|"failure"|"canceled"|"unknown"], created_at)."""
    with open(path) as f:
        raw = json.load(f)
    return [tuple(e) for e in raw]


def _parse_ts(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


@dataclass
class Batch:
    job_id: int
    prs: list[int]
    outcome: str
    timestamps: list[str]


def reconstruct_staging_batches(
    events: list[tuple], max_gap_seconds: float = 6 * 3600
) -> list[Batch]:
    """Group staging events (the real batch-test attempts - "trying" events
    are bors's single-PR pre-approval checks, not batches, and are ignored
    here) by job_id, splitting on job_id reuse across unrelated time
    periods (see module docstring) rather than trusting job_id alone."""
    by_job: dict[int, list[tuple]] = {}
    for pr, job_type, job_id, outcome, ts in events:
        if job_type != "staging":
            continue
        by_job.setdefault(job_id, []).append((pr, outcome, ts, _parse_ts(ts)))

    batches = []
    for job_id, entries in by_job.items():
        entries.sort(key=lambda e: e[3])
        clusters = []
        current = [entries[0]]
        for e in entries[1:]:
            if e[3] - current[-1][3] > max_gap_seconds:
                clusters.append(current)
                current = [e]
            else:
                current.append(e)
        clusters.append(current)

        for cluster in clusters:
            prs = sorted(set(e[0] for e in cluster))
            outcomes = set(e[1] for e in cluster if e[1] != "unknown")
            outcome = next(iter(outcomes)) if len(outcomes) == 1 else "unknown"
            timestamps = sorted(set(e[2] for e in cluster))
            batches.append(Batch(job_id=job_id, prs=prs, outcome=outcome, timestamps=timestamps))
    batches.sort(key=lambda b: b.timestamps[0])
    return batches


@dataclass
class CulpritInference:
    failed_batch: Batch
    likely_culprits: set[int]
    confidence: str  # "clear" or "unresolved" - see infer_culprits
    evidence: str


def infer_culprits(batches: list[Batch]) -> list[CulpritInference]:
    """First-pass, automated ground-truth inference for failed multi-PR
    batches: if a later batch succeeds with an exact subset of the failed
    batch's PRs, the dropped PR(s) are the likely culprit(s).

    This is explicitly NOT trusted blindly - manual audit of a sample
    (see docs/build-plan.md Milestone 5) found real cases where this
    heuristic's "clear" answer is murkier than it looks (e.g. a PR that
    needed 13 more failed solo retries over 4 days before finally passing
    isn't a clean single-culprit story - the "surviving" PR may have had
    real issues too). Report results with that caveat, don't paper over it.
    """
    results = []
    for i, batch in enumerate(batches):
        if batch.outcome != "failure" or len(batch.prs) <= 1:
            continue
        failed_prs = set(batch.prs)
        failed_ts = batch.timestamps[0]
        later = [b for b in batches[i + 1:] if b.timestamps[0] > failed_ts]

        found = None
        for b in later:
            b_prs = set(b.prs)
            if b.outcome == "success" and b_prs and b_prs < failed_prs:
                dropped = failed_prs - b_prs
                found = CulpritInference(
                    failed_batch=batch,
                    likely_culprits=dropped,
                    confidence="clear",
                    evidence=(
                        f"later successful batch (job {b.job_id}) retained exact "
                        f"subset {sorted(b_prs)}, dropping {sorted(dropped)}"
                    ),
                )
                break
        results.append(
            found
            or CulpritInference(
                failed_batch=batch,
                likely_culprits=set(),
                confidence="unresolved",
                evidence="no clean subsequent-batch evidence found",
            )
        )
    return results


def count_actual_rounds(batch: Batch, all_batches_sorted: list[Batch]) -> tuple[int, int]:
    """Two distinct real costs (see docs/build-plan.md Milestone 5 for why
    they're kept separate rather than reported as one number):

    rounds_to_isolate: rounds from the failed batch (inclusive) until bors
    first tests a proper subset of the original batch - the moment it
    stops testing the full original set together. Fair, apples-to-apples
    comparison to what this engine replaces.

    rounds_to_full_resolution: rounds until a strict subset is confirmed
    successful - can include many rounds of a human iterating real code
    fixes on the isolated culprit (found in real data: one PR failed alone
    13 more times over 4 days before finally passing) - a real cost, but
    not bisection overhead, and not something either bors or this engine's
    decoding can shortcut.
    """
    failed_prs = set(batch.prs)
    failed_ts = batch.timestamps[0]
    rounds_to_isolate = None
    rounds = 1
    for b in all_batches_sorted:
        if b.timestamps[0] <= failed_ts:
            continue
        if set(b.prs) & failed_prs:
            rounds += 1
            if rounds_to_isolate is None and set(b.prs) < failed_prs:
                rounds_to_isolate = rounds
            if b.outcome == "success" and set(b.prs) < failed_prs:
                return rounds_to_isolate or rounds, rounds
    return rounds_to_isolate or rounds, rounds


@dataclass
class ReplayResult:
    inference: CulpritInference
    engine_decoded: frozenset[int]
    engine_correct: bool
    engine_rounds: int
    engine_ci_runs: int
    rounds_to_isolate: int
    rounds_to_full_resolution: int


def replay_batch(
    inference: CulpritInference, all_batches_sorted: list[Batch], k: int = 2, design_seed: int = 0
) -> ReplayResult:
    """Replay one real, reconstructed batch (composition + inferred ground
    truth) through the actual engine (Milestones 1-3) and compare against
    bors's real historical resolution cost for the same incident."""
    batch = inference.failed_batch
    prs = batch.prs
    n = len(prs)
    pr_to_idx = {pr: i for i, pr in enumerate(prs)}
    culprit_idx = {pr_to_idx[pr] for pr in inference.likely_culprits if pr in pr_to_idx}

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
    correct = decoded_prs == frozenset(inference.likely_culprits) & set(prs)
    rounds_to_isolate, rounds_to_full = count_actual_rounds(batch, all_batches_sorted)

    return ReplayResult(
        inference=inference,
        engine_decoded=decoded_prs,
        engine_correct=correct,
        engine_rounds=rounds,
        engine_ci_runs=ci_runs,
        rounds_to_isolate=rounds_to_isolate,
        rounds_to_full_resolution=rounds_to_full,
    )


def run_full_replay(path: str = _DATA_PATH) -> list[ReplayResult]:
    events = load_distilled_events(path)
    batches = reconstruct_staging_batches(events)
    inferences = infer_culprits(batches)
    return [
        replay_batch(inf, batches) for inf in inferences if inf.confidence == "clear"
    ]
