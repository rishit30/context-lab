# Python/LangChain MVP completion gate — 2026-10-10

## Built

- `contextlab/session.py`: public real-Claude session, shared spend reservations,
  actual usage settlement, Jev fees, durable call limits, exact recall, and final
  summaries after cleanup. Supports baseline, ContextLab and genuine Headroom.
- `contextlab/core/duplicates.py`: byte/provenance proofs and persisted full-copy
  anchors. Pins/recent outputs are protected; invalid dependencies restore originals.
- `contextlab/providers/jev_gate.py`: complete retained evidence, unique-information-loss
  questions, durable candidate coverage, bounded requests and conservative failures.
- `harness/runner.py`: up to two shared phase recovery continuations. Public checks
  gate progression; private grading remains separate. All attempts consume original caps.
- `examples/file_agent.py`: real read-only agent, reviewed plans, immutable input
  snapshots, three questions, three arms, JSON answers and a portable HTML report.
- `harness/audit.py`: reusable read-only receipt/accounting/archive reconciliation.
- `Dockerfile`, `requirements.runtime.lock`, `compose.yaml`: runnable software image,
  distinct from the existing task executor. `docs/python-integration.md` is the runbook.
- `.github/workflows/checks.yml`: offline tests/lint/type checks and no-key plan generation.
  This workflow has not yet run on GitHub.

## Verified vs assumed

Existing pinned LangChain/Anthropic/Headroom contracts and the official Jev transport
remain unchanged. Public-provider modules no longer import the evaluation harness.
The wheel builds and imports outside the checkout. The software image runs as UID
10001 with no embedded `.env`, Git directory or provider credentials. Its CLI,
imports, plan generation and complete offline demo passed with networking disabled.

The installed Docker CLI lacks the Compose plugin. The equivalent `docker run`
workflow passed; Compose configuration is supplied but was not executed locally.
The separate task image and live benchmark preflight are ready.

## Tests

459 tests passed; repository-wide Ruff and strict core mypy passed. `git diff --check`
passed. Tests include actual LangChain graphs with test-only fake providers,
duplicate anchor failures, recall, restarts, public-phase recovery, real example
orchestration, malformed receipts, spend exposure, report escaping and missing costs.
No new paid inference was performed during this implementation/verification stage.

## Results

The host and Docker offline mechanics runs each completed nine reference-fixture
checks. These are scripted simulations, not AI task-performance results.

The new generic auditor reconciled every saved provider receipt from the historical
runs below. This validates saved accounting, not the effectiveness of the new policy.

| Historical run | Provider calls | Usage-derived cost | Recorded task outcomes |
|---|---:|---:|---|
| `live-001` | 103 | $0.53146455 | 9/9 passed |
| `comparison-001` | 85 | $9.119886 | 3/6 passed |
| `jev-pilot-001` | 13 | $0.777890412 | 0/1 passed |
| `jev-validation-001` | 62 | $5.303748552 | 3/3 passed |

These runs used earlier source snapshots. The last Jev comparison made no context
changes, so its lower ContextLab bill was not established compression savings.

## What surprised us

The earlier Jev question asked whether old information was useful without showing
what information remained visible. It repeatedly assessed the same oldest candidates.
The revised question and selection address these mechanisms; improvement remains
a live-evaluation hypothesis. Small examples also need a request after a completed
re-read phase for a subsequent-boundary compaction to affect model input.

## Failures and boundaries

Incomplete phases previously advanced despite missing output. Recovery now stops
after bounded public feedback. Ledger summary failures remain unknown instead of
being presented as zero cost. Three old comparison fixtures needed valid public
artifacts to exercise the strengthened runner; their tampering checks still pass.

Exact duplicate anchors deliberately remain full. Changed historical values still
need conservative semantic decisions/recall. Jev probabilities are uncalibrated.
The example does not independently grade natural-language answers. The session API
is synchronous and single-instance per session; callers own canonical graph history
and custom-tool execution safety. Routing, learned memory and an HTTP gateway are
not part of the completed Python MVP.

## Cost

New provider spend in the implementation/verification stage: **$0**. Docker and
wheel dependency installation incurred no model inference. Historical failed work
remains included in the table above. New paid plans require separate review.

## Go/no-go

Go for bounded live verification of the Python integration: **459 passing offline
tests**, installable wheel and an executed software container. No-go for a claim of
repeatable savings or production readiness until representative paid evaluation
and outcome evidence support it.

## Live verification deferred by user

The user selected **“No paid runs now”** after reviewing the $3 real-file-agent
and $9 graded-audit plans. Neither plan was executed; new provider spend remains
**$0**. The local Python MVP is implemented and verified. Fresh live evaluation
and any savings claims remain unverified for this revision. No API spending or
automation is scheduled.
