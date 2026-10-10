# ContextLab implementation architecture — v0.7

This document governs the implemented Python MVP and partial M1 analysis tooling. `architecture-original.md` and `implementation-brief-original.md` preserve the supplied proposal unchanged. Their statements about superiority, cache guarantees, forever-retention, and minimum sample sizes are hypotheses or have been corrected below. `requirements-status.md` maps the original scope to implementation and acceptance gaps.

## Product and scope

The product combines an embeddable context controller with an evidence console. A developer running an API-billed tool-using agent can inspect outputs, see proposed lifecycle decisions, activate an intervention, recover historical evidence, and compare independently checked task outcomes and all-in usage cost.

The current completion target is the Python/LangChain MVP, as requested by the user. `contextlab/session.py` is the public budgeted integration, `examples/file_agent.py` is a real read-only Claude agent in three middleware configurations, and `Dockerfile` packages the runnable software. `harness audit` verifies saved accounting without inference. The HTTP gateway is deferred by explicit preference. See `python-integration.md` for the complete operator/developer workflow.

M0 implements compaction and instrumentation in one synchronous LangChain/Anthropic adapter. A partial M3 now adds opt-in, bounded read-only evidence delegation; see `routing.md`. The local offline demonstrations exercise actual mechanics but deliberately do not claim real model savings or quality.

An optional Jev relevance gate now extends the context controller. This is a semantic veto over rules-eligible reductions, not task-model routing. Live inference and accounting passed in the first Jev pilot, which stopped on an invalid recall call. After shared tool-input recovery fixes, a fresh baseline/ContextLab+Jev/Headroom short-task comparison passed all three arms. Both middlewares made zero context changes, so different observed costs do not establish optimization savings. Rules remain the default for historical configurations.

The latest revision adds code-proven exact duplicate retirement, retained-evidence-aware Jev prompts, persistent candidate coverage and public phase-completion recovery. Its tests are offline; the previous paid runs do not validate this updated policy. Exact duplicates require equal bytes, path, file version and read range. Their full replacement is protected by persisted `ViewState.duplicate_sources`; every projection revalidates the relationship and restores unsafe/pinned originals. The duplicate path remains subject to the cost gate, recent-output protections and protocol validation.

Jev now receives a conservative subset of complete chunks guaranteed to remain full after all proposed reductions. It evaluates unique information lost, considering historical constraints. Candidate and support packing is atomic under request limits. Least-assessed candidates are considered first at later boundaries, based on durable valid assessment receipts; no previous judgment is reused as a fresh approval. Assessment coverage distinguishes eligible, submitted, assessed and unassessed candidates.

Benchmark phase completion uses public observable checks before advancing a fixture or recording a boundary. Truncated output or missing public artifacts receives at most two shared continuation attempts, under the original call and spend limits. Private grader verdicts never become recovery instructions. Exhaustion is a recorded failure. The generic read-only example instead stops on an unfinished response and explicitly does not grade answer correctness.

Offline `harness xray` reconstructs saved baseline requests causally. It reports rules-evidenced candidate content separately from boundary eligibility, costs and semantic approval. The primary byte share uses normalized message views, which omit system instructions and tool schemas. Provider receipts and saved rates support separate input-cost reconciliation; candidate token and price allocations remain heuristic. Missing traces/receipts stay unknown, failed sessions remain included, and simulation/live modes are separated. Source runs are opened read-only and optional reports use a new directory. No provider calls or tool execution occur during analysis.

Future benchmark traces record public-completion markers with a versioned source, phase index and canonical message-prefix hash, copied into each subsequent request across all arms. Recovery attempts and incomplete phases cannot create these markers. Historical final phase records do not retroactively supply this evidence. Offline acceptance now includes a paired nine-phase evidence-consumption simulation and omission/recovery cases; these tests validate mechanics, not real-model effectiveness.

## Completed architecture decisions

1. **Canonical history and projection remain distinct.** The controller changes only eligible text tool-result content. It does not reorder messages into artificial layers or mutate agent history. System instructions/tools use the same provider caching policy in every arm. The new comparison disables caching in every arm to prevent cross-arm cache carryover.
2. **Lifecycle evidence is conservative.** An edit alone does not supersede a read. A later read must cover the old region, or be a full read. A matching fail-to-pass transition is evidence about historical tests, not proof that all current requirements are met. Missing references do not prove irrelevance.
3. **Independent correctness checks include original public assertions.** Held-out checks and originals live outside the agent workspace and are passed to the isolated evaluator. An agent cannot earn success merely by deleting its public tests. Grading is not claimed resistant to actively malicious Python.
4. **No assumed prefix cache hit.** Shared bytes do not automatically correspond to a provider cache breakpoint. The default forecast gives zero preserved-prefix credit inside rewritten messages. The controller can accept a verified stable-token bound from a future adapter. The current adapter does not provide one.
5. **Cold cache is priced, not an automatic rebuild trigger.** Both branches include their next-call write and future read assumptions, plus policy/recall overhead. A cold cache can still favor keeping context. TTL warmth is a forecast; actual provider receipts determine measured cache use.
6. **Transactional storage.** Original chunk text is immutable SQLite TEXT with a content hash and virtual reference. This keeps originals, metadata, and recall pins atomic. A future large-object store can implement the same contract. There is no need to materialize arbitrary host paths.
7. **Bounded retrieval.** Recall is scoped to the session, supports line and long-line continuation, records history, and pins recalled chunks. Returning archived content appends an ordinary tool result. Absence of a recall does not establish that nothing useful was omitted.
8. **One budget ledger per live experiment.** Session records share a WAL database so experiment-wide reservations are atomic. Offline demo sessions use separate databases because they incur no model spend. This deliberately corrects the per-run-file proposal's cross-session budget gap.
9. **Reserve before dispatch.** Anthropic's remote token count enforces main-model input admission. Reserve the configured maximum input cost plus bounded output before inference, settle from original usage receipts, include failed calls, and block on unknown usage. Count-endpoint failures prevent inference rather than inventing zero-cost model responses. Jev has no verified count endpoint; its bounded request reserves a conservative 65,536 input tokens against the same ledger and stops further calls on unknown usage or a reservation overrun.
10. **Freeze the reviewed plan.** Approval binds model, pricing, matrix, compaction config, source, lockfiles and Dockerfile. Runtime uses the frozen policy. Mutable image tags and account/provider changes remain external provenance limitations; capture/pin the image digest before production experiments.
11. **Explicit available arms.** `baseline`, installed LangChain `threshold_edit`, `contextlab`, and optional pinned Headroom. The baseline runner constructs no ContextLab middleware. Headroom uses its public compression API with a documented lossless agent profile, without ML/CCR or hidden model calls. Deep Agents remains unavailable; competitors are never silently substituted.
12. **All modes have provenance.** Simulation records have `mode=simulation` and synthetic pricing; live records have API usage and rate snapshots. Reports group them separately. Repeat count alone does not demonstrate statistical significance or non-inferiority.
13. **Trusted workflow boundaries.** A harness may call `mark_boundary(label)` after a completed workflow phase. This records an idempotent internal event, never a fabricated test success. Tool/user text cannot create it. Existing covering-read or matching-test evidence is still required to retire content. With caching disabled, all projected future input is priced at the normal input rate.
14. **Optional model judgment has limited authority.** `core/semantic.py` defines a framework-neutral gate. After rules propose sufficient reductions, Jev evaluates complete candidate tool results against the current task and earlier constraints. It may only veto existing candidates; Python retains pins, eligibility, protocol and spend authority. A probability of need at or above the configured threshold, missing evidence, or any invalid decision keeps context. Considered boundaries persist to avoid repeating a decision request at every model turn.
15. **Decision fees are real ledger entries.** `harness/jev_decision.py` reserves and settles TypeSafe requests as `purpose=decision`. The economic forecast adds the actual fee and uses the larger of the configured recall prior and Jev's uncalibrated probability for each retained proposal. Assessment records are explanatory, never extra financial charges. A pending decision settlement prevents another main-model call. Total reports include both providers and break out decision overhead.

## Data flow

1. The shared agent receives a task phase and exposes identical tools in each arm.
2. The observer ingests immutable messages and tool results, preserving provenance and deduplicating after restart.
3. At a matching test transition or explicit trusted workflow boundary, the controller proposes visibility changes while preserving recent outputs and recall pins.
4. If enabled, the budgeted Jev gate vetoes reductions when context may still be needed (only clearly unnecessary candidates may be reduced). The economic model then compares keep/rebuild forecasts, including decision fees; the deterministic renderer applies an accepted plan.
5. Protocol and original-content checks validate the projection. A controller failure sends the original request, logs the failure where possible, and never retries an already dispatched provider request.
6. The same cache policy configures all arms. When disabled, no caching middleware or cache tags are added. Inner metering counts the final request, reserves budget, records the real usage receipt, and settles.
7. Agent tool execution is confined to a fixture workspace. Generated Python executes in disposable Docker containers without network, credentials, host home, or Docker socket access. Time/output limits stop containers explicitly.
8. The evaluator executes fixed public assertions plus held-out checks. Reports include failed runs in total cost and expose local traces.

## Data contracts

`core/models.py` contains typed Event, Chunk, Signal, Pricing, Usage, CompactionConfig, CostInputs, CostDecision, ViewState, ProjectionResult and RunResult schemas. Money uses Decimal. Forecast token estimates are distinct from API usage.

The ledger includes the original proposal's sessions, events, chunks, refs, signals, visibility, epochs, decisions, recalls, model_calls, delegations and facts tables. Additional tables store budgets, reservations, latest projection state and immutable evidence-catalog contracts. Delegations now record durable request claims and receipt-linked outcomes; facts remain schema-only.

## Gaps to resolve in later milestones

| Milestone | Required design/completion work |
| --- | --- |
| M0 live gate | Historical pilots verified integration and receipt accounting; the current revision still needs fresh approved live verification. |
| M1 | X-ray and offline historical-evidence/long-trajectory scenarios are implemented. Richer references/signals, six representative tasks and repeated live acceptance remain incomplete. Candidate-context share is neither proven waste nor a savings ceiling. |
| M2 | Calibrate remaining-turn/recall priors on held-out runs, add real competitors, control cold/warm cache conditions and cross-arm sharing, and report task-stratified uncertainty. Three repeats are only an initial pilot. |
| M3 | Read-only evidence extraction is implemented with a single main model, bounded worker, exact-source validation and charged failure/fallback. Validate real task quality and all-in cost next. Write-capable delegates still require isolated snapshots and independently validated atomic patch merging. Never treat model self-approval as sufficient. |
| M4 | Make segment extraction verbatim and test omission failures. Avoid double offloading when composing middleware. Learn priors without training/evaluating on the same trajectories. |
| M5 | Version facts by entity, scope and effective time; enforce access before retrieval; carry pending recent events; compute exhaustive aggregations in code; bound evidence retrieval. |
| M6 | Require explicit authenticated tenant/session IDs. A hash of system prompt plus first message can collide across users and is not an authorization boundary. Validate streaming, cancellation, retries and idempotency. Replayed requests test protocol/determinism, not provider cache hits. |

The earlier M1 comparison used four-phase file audits, a historical-evidence challenge, intervention telemetry, genuine Headroom and cache-off controls: 85 calls, $9.119886, all short cases passed and all long cases failed. It demonstrated real rules-only ContextLab compaction and correct retrieval on the short case. A later failed Jev pilot cost $0.777890412. The fresh three-arm Jev comparison then passed all short arms: 60 Claude calls plus two Jev calls, $5.303748552. It made no context changes, so repeatable optimization savings and long-task reliability remain pending. See `gates/M1-comparison.md` and `real-comparison.md`.

## Product evidence gate

Primary metric: total cost of all attempted runs divided by successful runs, alongside pass rate, latency, sample counts and task-level breakdowns. Undefined when no run succeeds. No fixed savings target has been fabricated. The current simulation validates mechanics only; product value remains unproven until representative live evaluation.
