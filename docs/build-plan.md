# Tryage — Proof of Concept Build Plan

*Working name: Tryage. A merge-queue failure resolution engine that identifies every faulty PR in a failed batch in a single parallel round, instead of the sequential one-at-a-time bisection used by every existing tool today.*

## 1. Product Overview

### 1.1 The problem

Companies with large, high-velocity monorepos batch multiple pull requests together into a single CI run before merging, to save compute (this is what a "merge queue" does — tools like Mergify, Aviator, and GitHub's native merge queue all offer this). When a batch fails, the system must figure out which PR(s) caused the failure. Every current tool does this by **bisection**: split the batch in half, retest, repeat, until one bad PR is isolated. This reliably finds exactly **one** faulty PR per full bisection pass.

The gap: if **two or more** PRs in the same batch are simultaneously broken, today's tools need to repeat the entire bisection cycle multiple times — batch again, bisect again — to catch each one. This is slow (each round waits on the previous one to finish) and, in the one case where a competitor (Graphite, now owned by Cursor) has addressed multi-fault detection in a single pass, it does so by brute force — testing every PR in the batch separately in parallel, which is fast but expensive at scale.

### 1.2 The solution

Use techniques from **combinatorial group testing** (the same mathematical idea behind pooled medical testing, e.g. pooled COVID sample testing) to design a small number of cleverly overlapping test batches. Run them all in parallel in a single round. From the pattern of which pooled batches pass and which fail, mathematically decode exactly which PR(s) are at fault — without needing to test every PR individually, and without needing multiple sequential rounds.

**In one sentence:** identify every faulty PR in a batch, in one parallel round, using meaningfully fewer CI runs than testing everyone separately.

### 1.3 Why this, why now

- Confirmed via direct prior work experience (Shopify's CI/merge-queue team) that this was a known, unresolved backlog item even at Graphite, the most sophisticated commercial player in this space.
- Confirmed via research: no academic paper, patent, or shipped product combines non-adaptive combinatorial group testing with CI batch bisection. Google's and Uber's most sophisticated public culprit-finding systems are still single-culprit sequential search under the hood.
- Real, live pain: GitHub had a production incident (April 2026) specifically because their merge-queue bisection logic wasn't built to handle more than one bad PR in a batch.
- Graphite's brute-force "parallel isolation" mode is gated in private beta, not GA, not marketed — confirming the market hasn't standardized on an efficient answer yet.

### 1.4 Target customer

Engineering organizations running merge queues at real scale — large monorepos, high PR velocity. This is a narrow, high-value niche (hundreds to low thousands of companies globally), not a mass market. Customers are already paying for Mergify, Aviator, or tolerating GitHub's native queue, and have real budget because CI compute cost and engineer-blocking time are expensive at their scale.

### 1.5 Positioning

Sits **alongside** the customer's existing merge queue rather than replacing it. Listens for batch-failure webhooks, resolves the failure using its own CI orchestration, and hands the result back via the existing queue tool's API (dequeue bad PRs, release good ones). No need to convince a company to rip out tooling they've already adopted — this dramatically lowers sales friction versus building a full competing merge queue.

### 1.6 Long-term shape of the company

This PoC is the wedge, not the whole company. The intended growth path (for context, not for PoC scope):

1. **Wedge:** multi-fault bisection engine (this document)
2. **Expand surface:** general CI failure resolution layer (flaky-vs-real classification, auto-suggested fixes)
3. **Go org-wide:** aggregate resolution history into org-wide failure intelligence (risky code owners, flaky hotspots)
4. **Become the platform:** bundle in predictive test selection, build cost optimization — competing at the level of Gradle Develocity / Datadog CI Visibility / Depot
5. **Outcome:** either an independent CI-intelligence platform, or a natural acquisition by a player already circling this space (Cursor, GitHub, Datadog)

## 2. High-Level Architecture

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  EXISTING MERGE QUEUE        │     │  CI PROVIDER                 │
│  Mergify / Aviator / GitHub  │     │  GitHub Actions / CircleCI /  │
│  (source of batch-failure    │     │  Buildkite                    │
│   events)                    │     │  (runs the actual test jobs)  │
└──────────────┬────────────────┘     └──────────────┬────────────────┘
               │                                       │
               ▼                                       │
        ┌─────────────────┐                            │
        │ 1. Webhook       │                            │
        │    Listener      │                            │
        └────────┬─────────┘                            │
                 ▼                                       │
        ┌─────────────────┐                              │
        │ 2. Batch Context │                              │
        │    Builder       │                              │
        └────────┬─────────┘                              │
                 ▼                                        │
        ┌─────────────────┐                               │
        │ 3. Combinatorial │  ◄── CORE DIFFERENTIATOR      │
        │    Design Engine │                               │
        └────────┬─────────┘                               │
                 ▼                                         │
        ┌─────────────────┐                                │
        │ 4. CI Orchestrator│──── triggers pooled runs ─────┘
        └────────┬─────────┘
                 ▼
        ┌─────────────────┐
        │ 5. Result Decoder │  ◄── CORE DIFFERENTIATOR
        │  (probabilistic,  │
        │   flake-aware)    │
        └────────┬─────────┘
              ┌───┴────┐
              ▼         ▼
     ┌────────────┐  ┌────────────┐
     │ 6a. Root-   │  │ 6b. Action  │──── dequeues/releases via
     │ Cause LLM   │  │ Layer       │     merge-queue API
     └──────┬──────┘  └──────┬──────┘
            └────────┬───────┘
                     ▼
             ┌────────────────┐
             │ 7. Resolution   │  ◄── THE MOAT (history compounds)
             │    Store        │
             └────────┬────────┘
                      ▼
             ┌────────────────┐
             │ 8. Dashboard     │──── feedback loop back to
             │  (confirm/       │     Design Engine (recalibration)
             │   override)      │
             └────────────────┘
```

Blue-tier / differentiated components: **3 (Design Engine)** and **5 (Result Decoder)**. Everything else is necessary plumbing that gets you to the point where the differentiated core can do its job.

## 3. Tech Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Combinatorial design + decoding engine | **Python** (numpy, scipy) | This is where the real IP lives. Python's numerical ecosystem is the path of least resistance for matrix/pooling-design math and probabilistic decoding. Build as a standalone library, independently testable, with zero dependency on the integration layer. |
| Integration / orchestration layer | **TypeScript/Node**, using **Probot** | Probot is purpose-built for GitHub Apps (handles auth, webhook verification, installation management). This layer touches the messiest real-world surface (GitHub, CircleCI, Buildkite APIs), so use the ecosystem with the least boilerplate. |
| Workflow orchestration | **Temporal** (or **Inngest** if Temporal feels heavy to self-host initially) | The core workflow shape is "fire off N parallel CI runs, wait on async webhook callbacks, handle timeouts/retries, then decide" — a durable-workflow problem, not a request/response one. Don't hand-roll this state machine. |
| Primary datastore | **Postgres** | System of record for PRs, batches, resolutions, confidence scores, org config. This schema **is** the long-term resolution-history moat — treat it as a first-class asset from day one, not an afterthought. |
| Ephemeral/fast state | **Redis** | In-flight job state, rate limiting. |
| Root-cause explanation | **Claude API**, direct call | Thin, well-scoped LLM use — feed it the failing logs + culprit PR diff, get back a plain-English explanation and optional fix suggestion. No agent framework needed at this stage. |
| Dashboard | **Next.js + Tailwind + shadcn/ui** | Fastest path for a solo dev to ship a credible-looking UI without weeks of frontend plumbing. |
| Hosting (PoC/early pilots) | **Fly.io** or **Railway** for services, **Vercel** for dashboard | Avoid AWS complexity until a real customer needs VPC/compliance guarantees. Migrate later. |
| Error tracking | **Sentry** | Standard, low-effort to wire in early. |

## 4. PoC Definition & Success Criteria

The PoC's job is to prove the **core algorithmic claim** — not to sign a customer, not to build the full dashboard. Everything in Milestones 0–4 can be built and validated with **zero customers and zero live integrations**. Only after the algorithm is proven does it make sense to build the GitHub integration and pursue a design partner.

**The PoC is successful if, in simulation across realistic batch sizes (4–32 PRs) and realistic fault counts (0–3 simultaneous faults):**

1. The design engine resolves all faulty PRs in **1 round** (occasionally 2, via the adaptive fallback) in the overwhelming majority of cases — versus the ~k × log₂(N) sequential rounds a standard bisector needs for k faults.
2. The design engine uses **meaningfully fewer total CI runs** than brute-force "test every PR separately" (Graphite's current approach) — target: roughly 40–60% fewer runs at typical batch sizes with 1–2 faults.
3. The decoder maintains **>95% correct-culprit identification** even when realistic flaky-test noise is injected (industry flake rates run roughly 10–25% of test suites).
4. Historical replay on real open-source CI data shows a consistent, positive result (not just synthetic data) — this is the number that goes in a pitch deck.

If any of these fail — especially #3 (noise robustness) — that's critical information to have **before** building any integration work, not after.

## 5. Phased Feature Build Plan

Each milestone lists: what to build, what tech is involved, the definition of done, and — critically — **what to test immediately after finishing it**, before moving to the next milestone. Do not proceed to the next milestone until the current one's tests pass. This is meant to be worked through sequentially with Claude Code, one milestone at a time.

### Milestone 0 — Project scaffolding

**Build:** Repo structure as above. Python package skeleton for engine/ with a test runner (pytest) configured. Node/TypeScript skeleton for app/ with a test runner (vitest or jest) configured. CI (GitHub Actions) running both test suites on every push.

**Definition of done:** pytest and npm test both run green on an empty/skeleton codebase, CI pipeline is green.

**Test immediately after:** Confirm CI actually fails on a deliberately broken test in each package (sanity-check that your own test infra works before building anything real on top of it).

### Milestone 1 — Core combinatorial design engine (clean case)

**Build:** Given N (batch size) and k (max simultaneous faults to tolerate), generate a non-adaptive pooling matrix — a set of overlapping groups such that any combination of up to k faulty items among N produces a unique failure pattern. Start with the simplest working construction (binary/Sperner-style encoding for k=1, extend to small k via a cover-free/disjunct-matrix construction — a random or structured construction is fine for PoC, doesn't need to be asymptotically optimal). Pair it with a **decoder** that takes a clean (noiseless) pass/fail pattern and returns the exact set of faulty PRs.

**Tech:** Python, numpy for matrix operations.

**Definition of done:** For N in {8, 16, 32} and k in {1, 2, 3}, the design+decode round-trip correctly identifies every possible combination of faulty PRs, with zero noise, 100% of the time.

**Test immediately after:**

- Unit tests: exhaustively test every possible fault combination (not just random samples) for small N (e.g., all C(8,2) combinations for N=8, k=2) — this is small enough to brute-force-verify completely.
- Property test: verify the number of groups used is meaningfully smaller than N (i.e., meaningfully better than brute force) for each (N, k) pair — log this as a benchmark table, this is your first real evidence for the pitch.
- **Do not proceed to Milestone 2 until this is 100% correct on the clean case.** Noise-handling on top of a shaky clean-case foundation will hide bugs, not reveal them.

### Milestone 2 — Noisy/probabilistic decoding

**Build:** Replace the clean boolean decoder with a probabilistic one. Each pooled test result is treated as a noisy signal (a "fail" might be a flaky test, not a real fault). Implement a simple Bayesian confidence update: given the observed pass/fail pattern and a per-repo prior flake rate, compute a posterior probability that each PR is genuinely faulty, rather than a hard yes/no. (Don't reach for a full probabilistic programming framework like PyMC here — a hand-rolled Bayesian update is simpler to reason about and sufficient for PoC.)

**Tech:** Python, numpy/scipy for the probability math.

**Definition of done:** Given a batch with injected synthetic faults **and** injected synthetic flakiness (configurable flake rate), the decoder outputs a confidence score per PR, and the highest-confidence PR(s) match the true injected faults above a target accuracy threshold.

**Test immediately after:**

- Build the **simulation harness** now if not already started (see Milestone 4 — you may want to pull it forward here since you need it to properly test this milestone anyway): inject faults + flakiness at varying rates (0%, 5%, 15%, 25%) across many randomized trials, measure decode accuracy at each flake rate.
- This is the single most important test in the whole PoC: **if accuracy collapses at realistic flake rates (~15-25%), that's the moment to stop and rethink the decoding approach**, before any further building. Don't skip past a bad result here.

**Result (2026-07-29):** Measured, not projected. Milestone 1's default construction (merely *separable* — every fault combo produces a distinct outcome, with no minimum-Hamming-distance margin between outcomes) collapses well below the 95% target at 15–25% flake (typically 25–65% exact-match at 15%, 25–55% at 25%, across N∈{8,16,32}, k∈{1,2,3}). Root cause confirmed by direct measurement: minimum pairwise Hamming distance across enumerated fault-combo outcomes was only 1–2. Fix applied: `build_pooling_matrix` now takes a `min_distance` parameter that rejects candidates too close to any already-placed outcome, recovering real accuracy — but at a steep, batch-size-dependent row cost (e.g. at N=32 it takes *more* rows than brute force to clear 95% at 15% flake; at N=64 there's a genuine window that beats brute force at 15% flake, but 25% flake still requires far more rows than brute force even after aggressive tuning). Full measured table and reasoning in `engine/tests/test_milestone2.py`. This surfaced an open calibration question (see Milestone 2.5) and a promising follow-up direction (paying the redundancy cost only on ambiguous decodes via an adaptive round, folded into Milestone 3's scope).

### Milestone 2.5 — Calibrate flake-rate semantics against reality

**Build:** Milestone 2's noise model treats `flake_rate` as the probability any single pooled-test *outcome* is corrupted (flipped) — a worst-case, per-observation corruption rate. The build plan's "10–25% of test suites are flaky" figure (Section 1.3, Section 4 #3) almost certainly means something different: the fraction of individual *tests* that are flaky, which only turns into a pooled-outcome corruption rate when one of those specific flaky tests actually misfires on that specific run — plausibly a much smaller effective number in practice, especially for pools containing many tests. Before spending further design effort chasing accuracy at `flake_rate`=25% under the current (likely too pessimistic) model, derive a defensible mapping from "X% of the test suite is flaky" (plus assumptions about per-test flake probability per run — from published incident data, or reasoned estimates if none are available) to the effective per-pooled-outcome corruption rate that the harness should actually be driven with.

**Tech:** Python, reasoning/analysis — no new engine code required unless the calibration changes the harness's inputs (e.g. accepting "fraction of tests flaky" + tests-per-pool instead of a single raw `flake_rate`).

**Definition of done:** A documented, justified mapping from industry flake-rate figures to the `flake_rate` parameter actually used across the simulation harness (Milestones 2 and 4), with the reasoning shown, not just an assumed number.

**Test immediately after:**

- Rerun the Milestone 2 flake-rate sweep (and any Milestone 4 comparisons already run) using the calibrated rate(s) instead of the raw 0/5/15/25% figures, and note whether the accuracy/row-cost picture changes materially.
- Flag explicitly if the calibrated realistic flake rate still lands in the range where Milestone 2 found the tradeoff unfavorable — this step doesn't get to quietly resolve that finding, only to make sure it's being judged against the right number. If real data from Milestone 5's historical replay becomes available later, revisit this calibration against it and correct if the estimate was off.

*(Placed here, not later: this is a direct, unresolved fallout of Milestone 2's central finding, and it determines whether that finding's accuracy/row-cost tradeoff is a real blocker or an artifact of testing against an unrealistically harsh noise level. It's sequenced before Milestone 3 so the adaptive-fallback work there gets designed against a well-calibrated noise target instead of a placeholder one.)*

**Result (2026-07-29):** Derived, not assumed - grounded in Google's widely-cited flaky-test figures (Micco, GTAC 2016): ~1.5% of all individual test *runs* are flaky, aggregated across the whole test population, concentrated in ~16% of distinct tests (so the conditional per-run misfire rate among flaky tests alone is ~9.4%). `engine/calibration.py`'s `calibrate_flake_rate(tests_per_run, retries)` turns that into the effective per-pooled-outcome corruption probability Milestone 2's `flake_rate` needs, as a function of (a) how many tests run per pooled CI job (typically the whole suite) and (b) how many times the customer's CI already retries a failing test before accepting the result — a flaky test only corrupts the final pooled outcome if it misfires on the original attempt *and every retry*.

**This does not rescue Milestone 2's finding - if anything it sharpens it.** For the actual target customer profile (Section 1.4: large monorepos), a *modest* retry policy (2 retries) on a 2,000-test suite calibrates to ~23% - squarely inside the 15-25% range Milestone 2 already tested, and decode accuracy there measures 38-39% exact-match (N=16/32, k=2), consistent with the original collapse. Smaller suites or more aggressive retry policies do calibrate much lower (S=100/2 retries → 1.3% → 95-98% accuracy; S=2000/3 retries → 2.4% → 91-93% accuracy) - so retry depth, not suite size, is the lever that actually matters, and a customer's existing CI hygiene turns out to be a real, material factor in whether Tryage's core value proposition holds for them. Full worked table in `engine/tests/test_milestone2_5.py`. Caveat carried forward honestly: the model assumes retry-independence, which understates risk for "systemic" flakiness (a broken shared fixture failing deterministically within one run) - a real phenomenon per the flaky-test literature, not accounted for here.

### Milestone 3 — Adaptive fallback for exceeded fault tolerance

**Build:** Handle the case where more PRs are simultaneously faulty than the design was built to tolerate (e.g., design was built for k=2 but 3 PRs are actually bad). Detect decode ambiguity, trigger one small, targeted follow-up round to resolve it, rather than silently returning a wrong or incomplete answer.

**Tech:** Python, extends the Milestone 1/2 engine.

**Definition of done:** When fault count exceeds the design's tolerance, the system correctly detects ambiguity (does not confidently return a wrong answer) and the fallback round successfully resolves the remaining uncertainty.

**Test immediately after:**

- Simulate fault counts deliberately exceeding k (e.g., design for k=2, inject 3–4 faults) across many trials — verify ambiguity is detected, not silently misreported, and the fallback round converges to the correct answer.
- Regression test: rerun all Milestone 1 and 2 tests to confirm nothing broke.

**Result (2026-07-29):** The first implementation's assumption - that an exceeded-k fault set's outcome coincidentally matching some ≤k combo's outcome would be astronomically rare (~2^-T) and safe to ignore - was wrong, and caught by this milestone's own correctness sweep, not by inspection: a naive "trust any exact-table match" decoder was confidently wrong on roughly half of exceeded-k trials in initial testing. Root cause: T is chosen close to the *minimum* needed to separate combos up to size k, so the outcome space is packed tightly enough that larger fault sets collide with a smaller combo's outcome routinely, not rarely.

Fix: `engine/adaptive.py`'s `AdaptiveDecoder` verifies every exact-table match with one extra pooled test (the complement pool - everyone NOT in the candidate set) before trusting it; a fault hiding outside the candidate makes that pool fail, which directly catches both an incomplete candidate and a coincidental match against a completely unrelated fault set. If verification fails, or the table found no match at all, the decoder falls back fully: `compute_cleared_and_suspects` uses the "any item in a passing pool is definitely innocent" rule - true regardless of the actual fault count - to narrow to a suspects set that's provably a superset of the true faults, then tests each suspect individually as one parallel round.

Measured: 100% correct across every (n, k, exceeded-fault-count) combination tested, including a broader stress sweep (4,500+ trials across multiple design seeds, N∈{16,32,64}, k∈{1,2,3}, excess fault counts 0–5). Overhead is honest and two-tiered: within-tolerance decodes always pay exactly one extra verification test (cheap, fixed cost - confirmed empirically, not assumed); exceeded-tolerance decodes pay for testing the suspects set, which is real but meaningfully smaller than full brute force in the cases measured (e.g., N=32,k=3,5 faults: ~15.6% of N; N=16,k=2,4 faults: ~70.6% of N - suspects-set size clearly grows with how far the fault count exceeds k relative to N, so "small" is not guaranteed in every regime and should be read from `engine/tests/test_milestone3.py`'s table, not assumed).

### Milestone 4 — Formal simulation & benchmark harness

**Build:** A proper CLI/script tool (if not already pulled forward from Milestone 2) that runs configurable large-scale simulations: batch size distribution, fault-count distribution, flake rate, and CI parallelism ceiling as inputs; outputs comparison metrics (rounds needed, total CI runs, wall-clock estimate) for three approaches side by side: (a) sequential bisection [today's standard], (b) brute-force parallel isolation [Graphite's current approach], (c) your combinatorial design.

**Tech:** Python, plus a simple plotting/reporting layer (matplotlib or just tabular output) to generate shareable results.

**Definition of done:** Running the harness across a realistic range of parameters produces a clear, defensible comparison table/chart showing time and cost deltas between all three approaches. This is the artifact that goes in front of design partners.

**Test immediately after:**

- Sanity-check the harness itself against hand-computed small cases (don't trust a benchmarking tool you haven't verified).
- Run the full battery across the parameter ranges defined in the PoC success criteria (Section 4) and check the results against the target thresholds. If targets aren't met, this is a decision point: adjust the design construction, or revisit assumptions, before moving to real-world data.

**Result (2026-07-29):** `engine/benchmark.py` runs all three approaches side by side on the same injected fault/flake scenario, tracking three distinct metrics: `logical_rounds` (the true sequential-dependency chain length - the number behind "1 round vs k×log2(N)"), `ci_runs` (total jobs, the cost metric), and `waves` (ci_runs chunked by a parallelism ceiling - the wall-clock proxy, so the Section 7 risk about limited CI parallelism is modeled explicitly, not assumed away). Sanity-checked against hand computation first (N=8 single-fault bisection = exactly 5 sequential CI runs; combinatorial row counts cross-checked against Milestone 1's already-verified numbers) before trusting it for the full sweep.

This sweep caught a real bug in itself before it caught anything about the product: the first cut of the combinatorial simulator used only Milestone 2's confidence-threshold check for every case, which can't detect an exceeded-k situation - it measured 18-20% accuracy on **clean** (0% flake) N=16/32 trials with 3 faults against a k=2 design, when Milestone 3's `AdaptiveDecoder` already proved that exact scenario is 100% correct. Fixed by delegating flake_rate=0 cases to `AdaptiveDecoder` directly rather than re-deriving weaker logic; regression test added (`test_combinatorial_clean_exceeded_k_is_always_exact`).

**Checked against all four PoC success criteria (Section 4), full sweep N∈{4,8,16,32}, faults∈{0,1,2,3}, flake∈{0%,5%,15%,25%}, k=2:**

1. **Rounds (1, occasionally 2) vs. bisection's k×log2(N):** clearly met, dramatically. The design never exceeds 2 logical rounds in any scenario tested. Bisection scales as expected - up to ~19-22 rounds at N=32 with 2-3 faults (clean), vs. the design's flat 1-2.
2. **40-60% fewer CI runs than brute force at 1-2 faults:** met at N=32 (the top of the "typical" 4-32 range) - 15 rows vs. 32 brute-force, a 53% reduction, holding up even under moderate noise (15.2 rows at 5% flake, still ~53% fewer). Only partially met at smaller N: ~31% fewer at N=16, and *no* savings at all at N=8 (matches Milestone 1's already-established k=2 threshold - N=8 just isn't large enough).
3. **>95% accuracy at realistic (10-25%) flake rates:** **not met**, confirming Milestone 2/2.5's finding rather than resolving it - accuracy at 15-25% flake with 1+ real faults ranges roughly 6-70% depending on N and fault count, nowhere near 95% except in the specific "zero actual faults, is this just flakiness" case at N≥16 (94.7-95.3%, right at the line).
4. **Historical replay:** not yet done (Milestone 5).

**The one genuinely new, positive finding this milestone adds:** criterion #3's failure looked worse in isolation (Milestone 2/2.5) than it does side by side with the alternatives it's actually competing against. At every flake rate tested, brute force is also badly degraded (sometimes to 0% accuracy at 25% flake, since it has zero redundancy - a single flaky test misfire on any of N individual jobs corrupts the whole verdict) and bisection degrades similarly (a single corrupted read anywhere in its dependency chain can send the whole search down the wrong path). The combinatorial design is the best of the three at nearly every (N, fault_count, flake_rate) combination tested, often by a wide margin (e.g. N=16, 1 fault, 5% flake: 96.0% vs. bisection's 71.3% vs. brute force's 43.3%) - it just doesn't clear the absolute 95% bar on its own. None of the three approaches would be a safe, unmonitored autonomous system at 15-25% flake without more work; the design is consistently the least-bad option available today.

**Net verdict:** criteria #1 and #2 hold, #2 with a caveat about needing N toward the higher end of "typical." Criterion #3 does not hold in absolute terms - this remains the central open problem (tracked in Milestone 2.5, and a candidate for the adaptive-round idea noted there: pay redundancy only on low-confidence decodes rather than universally). Per Section 4's own instruction, this is exactly the situation to have clear before further integration work, and it's now measured with real, defensible, three-way comparative numbers rather than assumed.

### Milestone 5 — Historical replay validation

**Build:** A module that ingests real historical CI/merge-queue data from public sources (open-source projects using bors/homu-style merge queues, or public GitHub Actions history with merge groups) and replays failed batches through your engine to see what it would have done versus what actually happened.

**Tech:** Python, GitHub API (read-only, public data) or downloaded CI logs, reusing the engine/ package from Milestones 1–3.

**Definition of done:** A report showing, across a meaningful sample of real historical batch failures, how often multiple simultaneous faults actually occurred (this also tells you how common the problem is — valuable for the pitch) and what your engine's resolution time/cost would have been versus the sequential approach that was actually used.

**Test immediately after:**

- Manually audit a sample of the replayed results by hand (pick 10–15 real historical failures, verify your engine's conclusion against what you can independently confirm from the commit history) — this catches subtle bugs that synthetic simulation can't, because real data is messier than synthetic data.

**Result (2026-07-29):** Found a real, working data source after ruling out two others: GitHub's native merge queue exposes no queryable historical data once a merge group is consumed (confirmed via GitHub's own community discussions - even GitHub staff acknowledge users can't see historical merge-queue groupings); bors-ng's own dogfooded repo (bors-ng/bors-ng) turned out too low-velocity for genuine multi-PR batches. **openebs/mayastor** (a real, public CNCF-adjacent storage project) uses bors-ng properly and has 831 PRs with real bors bot activity spanning 2020-2023 - a legitimate historical dataset, not synthetic.

Real bors-ng batches are reconstructed from bors[bot]'s PR comments (each comment links a Jenkins "staging" job URL; PRs sharing the same job were tested together as one batch) - `engine/replay.py`, data snapshot in `engine/replay_data/`. This caught a real data-quality bug before it caught anything about the product: Jenkins job IDs get **reused/reset over time** - one job ID matched a PR from 2020 and a PR from 2023, three years apart. Naive grouping by job ID alone would silently fabricate a false batch; fixed by also requiring comment timestamps to fall within a bounded window (regression test: `test_job_id_reuse_across_years_is_not_grouped_as_one_batch`).

**How common are multi-fault batches, really?** Out of 1,169 reconstructed real batches, only 26 (2.2%) were genuinely multi-PR (bors-ng mostly merges one PR at a time even when configured for batching - PR review/approval velocity at this project rarely produced 2+ simultaneously-ready PRs). Of those 26, 18 failed (69%) - and among the 16 with clean-enough evidence to assign ground truth, real fault counts of 1, 2, and 3 simultaneous culprits all actually occurred. This is the honest answer to Section 7's flagged risk ("if simultaneous multi-fault batches turn out to be rare... the value prop shifts from 'we solve a common problem' to 'we solve an expensive, occasional problem'"): at this project, multi-fault batches are genuinely uncommon in absolute terms, but real, and not vanishingly rare when they do happen (both 2- and 3-fault incidents are present in fewer than 900 merged PRs' worth of history).

**Manual audit (16 cases, all with "clear" automated ground truth) surfaced real nuance the automation couldn't catch on its own** - exactly what this test step is for: one case (job 387, PRs 641+642) had PR 641 fail *alone* 13 more times over 4 days before finally passing, meaning the "clean" bisection story (642 was the culprit, 641 was always fine) undersells what actually happened - 641 pretty clearly had real issues too, just resolved through code iteration rather than a clean revert. Another case showed PR 673 closed *without merging*, with a later PR 693 sharing its exact title merging successfully - a fix-and-resubmit pattern, not a simple pass/fail. These don't invalidate the automated inference (it's still consistent with the observed batch behavior), but they're a reminder that real history is messier than the "PR X is the one true culprit" framing assumes.

**Replay result, run through the actual engine (Milestones 1-3), not a synthetic re-derivation:** all 16 clear cases decoded **exactly correctly** (`engine_correct` 16/16). Two round-count metrics, kept deliberately separate rather than conflated into one flattering number:
- Engine: avg **1.12 rounds** (matches "1, occasionally 2" almost exactly).
- Real bors, fair comparison (`rounds_to_isolate` - rounds until bors stops testing the original full batch together): avg **2.12 rounds**.
- Real bors, full incident resolution (`rounds_to_full_resolution` - includes the job-387-style human fix-iteration time): avg **5.06 rounds** - real, but not fairly attributable to bisection overhead, since neither bors's process nor this engine's decoding can shortcut a human's actual debugging time.

**Honest caveat this data surfaces that Milestones 1/4 already implied:** real batch sizes here are tiny (2-4 PRs), well below the N≥16 threshold where Milestone 1 found k=2 designs start saving meaningful CI runs. So this real dataset mainly validates success criterion #1 (rounds) clearly; it doesn't demonstrate criterion #2's CI-run savings, because these real batches are smaller than the regime where that savings shows up - consistent with, not contradicting, what was already found synthetically.

### Milestone 5.5 — Strict/deterministic decode mode (no flake tolerance)

**Build:** Some customers won't want probabilistic flake-tolerance at all — their policy is "any observed failure in the merge queue is a deal-breaker," full stop, with no attempt to hedge or exonerate a PR because a failure *might* have been flaky. For that segment, the Milestone 2 Bayesian layer is unnecessary complexity: build a configurable "strict mode" that decodes using only the exact Milestone 1 table (deterministic, no confidence scores) and, on any observed pattern that doesn't match a known fault-combo entry — which will still happen, since a flaky test can corrupt a pooled result regardless of the customer's tolerance policy — fails safe by treating every PR in the affected pool as held and triggering Milestone 3's adaptive fallback round, rather than raising an unhandled error or guessing.

Also measure the risk this mode trades away: pooling groups multiple PRs per test, so a single flaky result can force strict mode to hold *every* PR in that pool, whereas sequential bisection isolates down to one PR before implicating anyone. Quantify "PRs held per flaky incident" for strict-mode pooling versus bisection, using real flake-incident rates from Milestone 5's historical replay data (not synthetic assumptions) now that it's available.

**Tech:** Python, extends the Milestone 1/3 engine (no Milestone 2 Bayesian dependency for the decode path itself).

**Definition of done:** A working strict mode that never produces a confidence score or hedges — every decode either resolves to an exact fault set or deterministically triggers the fallback round — plus a measured, real-data-grounded comparison of "PRs held per flaky incident" between strict-mode pooling and sequential bisection.

**Test immediately after:**

- Confirm strict mode is fully deterministic: same observed pattern always produces the same outcome (decoded set or fallback trigger), never a probability.
- Run the blast-radius comparison across the batch sizes already tested and report it plainly, including if it comes out unfavorably (e.g. if strict-mode pooling ends up holding *more* PRs per flaky incident than bisection would have) — that's a real, segment-specific finding to report honestly, the same way Milestone 2's flake-collapse finding was, not something to soften or bury.

**This is the end of the "no integration needed" phase of the PoC.** Milestones 0–5.5 can be fully completed solo, with no customer, no live GitHub App, no design partner. If the results here don't hold up, stop and revisit the algorithm before building any integration work.

### Milestone 6 — GitHub webhook listener + batch context builder

**Build:** A minimal Probot-based GitHub App that listens for merge-queue / check-suite events, normalizes them into your internal batch-failure event format, and fetches the relevant PR diffs and metadata for a failed batch.

**Tech:** Node/TypeScript, Probot, Octokit.

**Definition of done:** Deployed against a sandbox GitHub repo you control, the app correctly detects a batch failure event and produces a structured internal representation of the batch (which PRs, what they changed).

**Test immediately after:**

- Set up a sandbox repo with a real merge queue configured. Manually push a batch of PRs, some deliberately broken, and confirm the webhook listener correctly captures the failure event and batch composition.
- Test edge cases: batch of 1 PR, batch where all PRs are fine (false trigger check), webhook delivery retries/duplicates.

### Milestone 7 — CI orchestrator

**Build:** Given the pooling design from the engine/ package, dynamically create the necessary branches/PRs combining the designed PR groups, and trigger CI runs for each pooled group via the CI provider's API (start with GitHub Actions only for PoC).

**Tech:** Node/TypeScript, GitHub Actions API, Temporal (or Inngest) to manage the "wait for N parallel runs" workflow.

**Definition of done:** For a real failed batch on the sandbox repo, the orchestrator correctly creates and triggers the pooled test runs designed by the engine, and collects their results.

**Test immediately after:**

- End-to-end test on the sandbox repo: inject synthetic faults into real PRs (deliberately break 1–2 of them), let the full pipeline (webhook → context builder → design engine → orchestrator) run, and manually verify the correct pooled runs were triggered.
- Test failure handling: what happens if one of the pooled CI runs itself times out or errors (not fails, but infrastructure-errors) — this needs to be distinguished from a genuine test failure.

### Milestone 8 — Result decoder integration

**Build:** Wire real CI results from Milestone 7 into the probabilistic decoder from Milestone 2/3. Produce a final, confidence-scored list of faulty PRs from a real batch failure.

**Tech:** Node/TypeScript orchestration layer calling into the Python engine/ package (via HTTP API or subprocess — pick whichever is simpler to wire up, HTTP via a small FastAPI wrapper around the engine is probably cleanest).

**Definition of done:** A full real end-to-end run — real batch failure on the sandbox repo, through webhook, context building, design, orchestration, and decoding — produces a correct, confidence-scored culprit list.

**Test immediately after:**

- Full end-to-end test on the sandbox repo with **realistic flaky test injection** (deliberately make one test in the suite intermittently fail, unrelated to the actual injected faults) — verify the system doesn't get confused by it.
- This is your first fully real, integrated test of the entire core value proposition. Run it multiple times across different fault/flake combinations before moving on.

### Milestone 9 — Root-cause LLM layer

**Build:** Once a culprit PR is identified, pull its diff and the relevant failure logs, and call the Claude API to generate a plain-English explanation of why it likely broke the batch, with an optional suggested fix.

**Tech:** Node/TypeScript, Claude API.

**Definition of done:** For a resolved batch failure, the system produces a readable, accurate-sounding explanation referencing the actual diff and failure logs (not a generic/hallucinated one).

**Test immediately after:**

- Qualitative review: run this against 15–20 of the historical replay cases from Milestone 5 (you already know the ground truth for those) and manually grade the explanations for accuracy and usefulness.
- Check for hallucination: does the explanation ever reference code or behavior that isn't actually in the diff? This needs a human eye, not an automated test, at this stage.

### Milestone 10 — Action layer + dashboard (shadow mode only)

**Build:** The action layer posts a comment on the culprit PR and, in **shadow mode**, does *not* actually dequeue anything — it only records what it *would* have done. The dashboard shows the resolution history, confidence scores, and the LLM explanation, and lets a human confirm or override.

**Tech:** Node/TypeScript for the action layer (posting comments, recording decisions — but not yet calling dequeue APIs), Next.js dashboard reading from Postgres.

**Definition of done:** A working, deployed dashboard showing real (sandbox) resolution history, and the system correctly identifies itself as running in shadow/recommend-only mode — no autonomous queue actions yet.

**Test immediately after:**

- Full dry run on the sandbox repo end-to-end, confirm the dashboard accurately reflects what happened, confidence scores are visible and sensible, and the "would have acted" log is being recorded correctly.
- **This is the point where the PoC is complete and ready to take to a design partner** for the shadow-mode pilot described in the growth plan — read-only access to a real company's batch-failure webhooks, comparing your system's conclusions against their real, existing bisection outcomes, without ever touching their production merge queue.

## 6. What Comes After the PoC (not in scope for this build, for context only)

1. **Shadow-mode pilot** with 1–2 real design partners (via warm intro network) — passive observation, comparing your system's conclusions against their real outcomes, no production actions.
2. **Active mode** — once shadow-mode shows a consistent, real win, allow the action layer to actually dequeue PRs, initially gated by a high confidence threshold, with easy one-click undo.
3. **Multi-provider support** — extend beyond GitHub Actions to CircleCI, Buildkite.
4. **Resolution store intelligence layer** — org-wide dashboards on repeat offenders, flaky hotspots, risky code owners, once enough history has accumulated.
5. Everything in the longer growth map (Section 1.6).

## 7. Key Risks to Monitor Throughout (revisit at every milestone)

- **Flakiness overwhelming the signal.** If Milestone 2's accuracy collapses at realistic flake rates, this is the single biggest threat to the whole premise. Don't paper over a bad result here.
- **Compute tradeoff not panning out at realistic CI parallelism limits.** The cost advantage over brute force assumes you can actually run several pooled groups in parallel — if a customer's CI infrastructure has a low parallelism ceiling, the advantage shrinks. Model this explicitly in Milestone 4's harness, don't assume unlimited parallelism.
- **Multi-fault batches being rarer in practice than assumed.** Milestone 5's historical replay is what answers this honestly — if simultaneous multi-fault batches turn out to be very rare in real data, that changes how the product should be pitched (still valuable, but the value prop shifts from "we solve a common problem" to "we solve an expensive, occasional problem").
- **Engineering complexity vs. payoff.** Sequential bisection is dumb and simple, which is why every existing vendor uses it. Everything in Milestones 1–3 is real algorithmic complexity — make sure each milestone's tests are genuinely passed, not just "close enough," before adding more complexity on top.

## 8. Glossary (for shared vocabulary with Claude Code / future collaborators)

- **Merge queue:** a system that batches multiple PRs together, tests them as one unit before merging, to save CI compute.
- **Bisection:** the standard method of finding a faulty commit/PR by repeatedly splitting a batch in half and retesting.
- **Combinatorial group testing:** a mathematical technique for identifying multiple "bad" items among many by testing cleverly overlapping pools rather than each item individually — the same principle used in pooled medical testing.
- **Non-adaptive design:** a testing plan where all test groups are decided up front and run in parallel, as opposed to adaptive/sequential testing where each round depends on the previous one's result.
- **Cover-free / disjunct matrix:** the mathematical structure used to guarantee that any combination of up to k simultaneous faults produces a uniquely decodable failure pattern.
- **Flaky test:** a test that passes or fails inconsistently without any underlying code change, purely from noise/nondeterminism.
- **Shadow mode:** running the system passively, observing and recording what action it *would* take, without actually taking that action — used to build trust and validate accuracy before going autonomous.
