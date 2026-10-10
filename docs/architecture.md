# ContextLab implementation architecture — v0.2

This document governs the implemented M0. `architecture-original.md` and `implementation-brief-original.md` preserve the supplied proposal unchanged. Their statements about superiority, cache guarantees, forever-retention, and minimum sample sizes are hypotheses or have been corrected below.

## Product and scope

The product combines an embeddable context controller with an evidence console. A developer running an API-billed tool-using agent can inspect outputs, see proposed lifecycle decisions, activate an intervention, recover historical evidence, and compare independently checked task outcomes and all-in usage cost.

M0 implements compaction and instrumentation in one synchronous LangChain/Anthropic adapter. Routing by bounded delegation remains M3. The local offline demonstration exercises the actual mechanics but deliberately does not claim real model savings or quality.

## Completed architecture decisions

1. **Canonical history and projection remain distinct.** The controller changes only eligible text tool-result content. It does not reorder messages into artificial layers or mutate agent history. System instructions/tools use the same provider caching middleware in every arm.
2. **Lifecycle evidence is conservative.** An edit alone does not supersede a read. A later read must cover the old region, or be a full read. A matching fail-to-pass transition is evidence about historical tests, not proof that all current requirements are met. Missing references do not prove irrelevance.
3. **Independent correctness checks include original public assertions.** Held-out checks and originals live outside the agent workspace and are passed to the isolated evaluator. An agent cannot earn success merely by deleting its public tests. Grading is not claimed resistant to actively malicious Python.
4. **No assumed prefix cache hit.** Shared bytes do not automatically correspond to a provider cache breakpoint. The default forecast gives zero preserved-prefix credit inside rewritten messages. The controller can accept a verified stable-token bound from a future adapter. The current adapter does not provide one.
5. **Cold cache is priced, not an automatic rebuild trigger.** Both branches include their next-call write and future read assumptions, plus policy/recall overhead. A cold cache can still favor keeping context. TTL warmth is a forecast; actual provider receipts determine measured cache use.
6. **Transactional storage.** Original chunk text is immutable SQLite TEXT with a content hash and virtual reference. This keeps originals, metadata, and recall pins atomic. A future large-object store can implement the same contract. There is no need to materialize arbitrary host paths.
7. **Bounded retrieval.** Recall is scoped to the session, supports line and long-line continuation, records history, and pins recalled chunks. Returning archived content appends an ordinary tool result. Absence of a recall does not establish that nothing useful was omitted.
8. **One budget ledger per live experiment.** Session records share a WAL database so experiment-wide reservations are atomic. Offline demo sessions use separate databases because they incur no model spend. This deliberately corrects the per-run-file proposal's cross-session budget gap.
9. **Reserve before dispatch.** A remote token count enforces input admission. Reserve the configured maximum input cost plus bounded output before inference, settle from original usage receipts, include failed calls, and block on unknown usage. Count-endpoint failures prevent inference rather than inventing zero-cost model responses.
10. **Freeze the reviewed plan.** Approval binds model, pricing, matrix, compaction config, source, lockfiles and Dockerfile. Runtime uses the frozen policy. Mutable image tags and account/provider changes remain external provenance limitations; capture/pin the image digest before production experiments.
11. **Explicit available arms.** `baseline`, installed LangChain `threshold_edit`, and `contextlab`. Headroom/Deep Agents comparators are unavailable until verified in this environment; they are never silently substituted.
12. **All modes have provenance.** Simulation records have `mode=simulation` and synthetic pricing; live records have API usage and rate snapshots. Reports group them separately. Repeat count alone does not demonstrate statistical significance or non-inferiority.

## Data flow

1. The shared agent receives a task phase and exposes identical tools in each arm.
2. The observer ingests immutable messages and tool results, preserving provenance and deduplicating after restart.
3. At a matching test transition, the controller proposes visibility changes while preserving recent outputs and recall pins.
4. The economic model compares keep/rebuild forecasts; the deterministic renderer applies an accepted plan.
5. Protocol and original-content checks validate the projection. A controller failure sends the original request, logs the failure where possible, and never retries an already dispatched provider request.
6. The same Anthropic caching middleware configures all arms. Inner metering counts the final request, reserves budget, records the real usage receipt, and settles.
7. Agent tool execution is confined to a fixture workspace. Generated Python executes in disposable Docker containers without network, credentials, host home, or Docker socket access. Time/output limits stop containers explicitly.
8. The evaluator executes fixed public assertions plus held-out checks. Reports include failed runs in total cost and expose local traces.

## Data contracts

`core/models.py` contains typed Event, Chunk, Signal, Pricing, Usage, CompactionConfig, CostInputs, CostDecision, ViewState, ProjectionResult and RunResult schemas. Money uses Decimal. Forecast token estimates are distinct from API usage.

The ledger includes the original proposal's sessions, events, chunks, refs, signals, visibility, epochs, decisions, recalls, model_calls, delegations and facts tables. Additional tables store budgets, reservations and latest projection state. Delegations/facts are schema-only at M0.

## Gaps to resolve in later milestones

| Milestone | Required design/completion work |
| --- | --- |
| M0 live gate | Review pricing/spend; provision Docker + API access; run actual trajectories; verify cache usage and billed receipts; verify a genuine Headroom comparator or explicitly accept the threshold baseline. |
| M1 | Add representative long tasks, a short negative control, and a historical-evidence challenge. Call X-ray output an estimated candidate-context share; it is neither proven waste nor a savings ceiling. |
| M2 | Calibrate remaining-turn/recall priors on held-out runs, add real competitors, control cold/warm cache conditions and cross-arm sharing, and report task-stratified uncertainty. Three repeats are only an initial pilot. |
| M3 | Delegate an explicit subtask with typed acceptance criteria. Run write-capable delegates on isolated snapshots; merge a validated patch atomically or discard it. Maintain a single main model and charge failed delegates, retries and escalation. Never treat model self-approval as sufficient. |
| M4 | Make segment extraction verbatim and test omission failures. Avoid double offloading when composing middleware. Learn priors without training/evaluating on the same trajectories. |
| M5 | Version facts by entity, scope and effective time; enforce access before retrieval; carry pending recent events; compute exhaustive aggregations in code; bound evidence retrieval. |
| M6 | Require explicit authenticated tenant/session IDs. A hash of system prompt plus first message can collide across users and is not an authorization boundary. Validate streaming, cancellation, retries and idempotency. Replayed requests test protocol/determinism, not provider cache hits. |

## Product evidence gate

Primary metric: total cost of all attempted runs divided by successful runs, alongside pass rate, latency, sample counts and task-level breakdowns. Undefined when no run succeeds. No fixed savings target has been fabricated. The current simulation validates mechanics only; product value remains unproven until representative live evaluation.
