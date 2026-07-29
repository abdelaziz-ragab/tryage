from replay import (
    Batch,
    infer_culprits,
    load_distilled_events,
    reconstruct_staging_batches,
    replay_batch,
    run_full_replay,
)


def test_loads_real_snapshot():
    events = load_distilled_events()
    assert len(events) > 1000  # 2371 at time of writing - sanity floor


def test_job_id_reuse_across_years_is_not_grouped_as_one_batch():
    """Regression test for a real data-quality bug this milestone caught:
    Jenkins staging job IDs get reused/reset over time - job_id=6 matched
    a PR from 2020 and a PR from 2023 in the raw data, three years apart.
    Grouping by job_id alone silently produces a false multi-PR batch."""
    events = [
        (100, "staging", 6, "success", "2020-09-22T10:15:16Z"),
        (200, "staging", 6, "success", "2023-08-15T14:01:47Z"),
    ]
    batches = reconstruct_staging_batches(events)
    assert len(batches) == 2
    assert all(len(b.prs) == 1 for b in batches)


def test_same_job_id_close_in_time_is_grouped_as_one_batch():
    events = [
        (100, "staging", 42, "failure", "2021-01-21T17:44:24Z"),
        (200, "staging", 42, "failure", "2021-01-21T17:44:30Z"),
    ]
    batches = reconstruct_staging_batches(events)
    assert len(batches) == 1
    assert batches[0].prs == [100, 200]
    assert batches[0].outcome == "failure"


def test_trying_events_are_not_treated_as_batches():
    events = [(1, "trying", 5, "success", "2021-01-01T00:00:00Z")]
    assert reconstruct_staging_batches(events) == []


def test_infer_culprits_finds_dropped_pr_from_later_successful_subset():
    batches = [
        Batch(job_id=1, prs=[10, 20], outcome="failure", timestamps=["2021-01-01T00:00:00Z"]),
        Batch(job_id=2, prs=[10], outcome="success", timestamps=["2021-01-01T01:00:00Z"]),
    ]
    inferences = infer_culprits(batches)
    assert len(inferences) == 1
    assert inferences[0].confidence == "clear"
    assert inferences[0].likely_culprits == {20}


def test_infer_culprits_reports_unresolved_when_no_clean_evidence():
    batches = [
        Batch(job_id=1, prs=[10, 20], outcome="failure", timestamps=["2021-01-01T00:00:00Z"]),
    ]
    inferences = infer_culprits(batches)
    assert inferences[0].confidence == "unresolved"
    assert inferences[0].likely_culprits == set()


def test_real_replay_is_fully_correct_and_matches_audited_numbers():
    """The core Milestone 5 result, locked in as a regression test: replay
    every 'clear'-confidence real failed batch from the real openebs/mayastor
    dataset through the actual engine (Milestones 1-3) and confirm it
    decodes every one exactly right, at the measured, audited round counts
    (see docs/build-plan.md Milestone 5 for the full write-up, including
    the manual audit that found some of these 'clear' inferences are
    murkier in reality than the automated heuristic suggests)."""
    results = run_full_replay()
    assert len(results) == 16
    assert all(r.engine_correct for r in results)

    avg_engine_rounds = sum(r.engine_rounds for r in results) / len(results)
    avg_isolate = sum(r.rounds_to_isolate for r in results) / len(results)
    assert avg_engine_rounds < 1.2
    assert avg_isolate > avg_engine_rounds  # engine still wins the fair comparison


def test_replay_batch_ci_runs_are_positive():
    events = load_distilled_events()
    batches = reconstruct_staging_batches(events)
    inferences = [inf for inf in infer_culprits(batches) if inf.confidence == "clear"]
    result = replay_batch(inferences[0], batches)
    assert result.engine_ci_runs > 0
    assert result.rounds_to_isolate >= 1
    assert result.rounds_to_full_resolution >= result.rounds_to_isolate
