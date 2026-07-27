# Tryage

A merge-queue failure resolution engine that identifies every faulty PR in a failed CI batch in a single parallel round, using combinatorial group testing instead of sequential bisection.

See [`docs/build-plan.md`](docs/build-plan.md) for the full PoC build plan, architecture, tech stack, and milestone breakdown.

## Repo structure

- `engine/` — Python: combinatorial design + decoder (the core IP)
- `app/` — Node/TypeScript: Probot GitHub App, webhook listener, CI orchestrator, action layer
- `dashboard/` — Next.js dashboard
- `replay/` — historical replay validation tooling
- `docs/` — project documentation
