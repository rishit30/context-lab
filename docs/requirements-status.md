# ContextLab requirements and acceptance status

The original Routing + Context Compaction roadmap is not complete. The implemented
Python/LangChain MVP is a subset. Choosing the Python integration over an HTTP
gateway did not cancel the routing requirement.

This checklist separates implemented software from evidence that it improves real
agent outcomes. Original specifications remain in `implementation-brief-original.md`
and `architecture-original.md`. Current corrections are in `architecture.md`.

| Area | Implementation | Acceptance still needed |
| --- | --- | --- |
| M0: archive, tool protocol, budgets, accounting | Implemented; immutable originals, recall, provider receipts, reservations and failure accounting | Current revision needs fresh approved live verification; old runs retain their original source snapshots |
| Python/LangChain integration | Public `ContextLabSession`, real file-reading example, baseline/ContextLab/Headroom configurations | Representative real workflows; synchronous, one active instance per session remains the supported scope |
| Packaging | Installable Python package, software Docker image, isolated task image, CI configuration | CI must run on GitHub; Compose was supplied but its plugin was unavailable locally |
| M1: signals and context analysis | Covering reads, resolved test logs, visibility levels, recall, timelines; offline candidate-context analyzer added | Richer symbol/error/test references, additional safe signals, six representative tasks and the original repeated live matrix remain incomplete |
| M2: economics and evaluation | Fixed-prior cost forecasts, cache-price handling, cost per solved task, all attempted-work cost | Learned turn/recall estimates, additional boundaries, DeepAgents comparator, calibrated uncertainty and repeated evaluation |
| M3: routing by delegation | Not implemented; routing remains disabled | Brief construction, cheaper delegate, typed acceptance, isolated changes, escalation, complete cost accounting and component ablations |
| Jev decisions | Real transport and metering; a conservative semantic veto on eligible context reductions | Calibration and live validation of the revised information-loss question; this is not task-model routing |
| M4: triage and recall learning | Shadow mode and optional semantic assessment | Verbatim segment extraction, omission tests, learned recall priors and combined Headroom/ContextLab evaluation |
| M5: conversation memory | Schema placeholders only | Versioned facts, hybrid retrieval, query routing, evidence checks and independently graded conversation suite |
| M6: HTTP gateway | Deferred by user preference | Authenticated session identity, protocol/streaming/cancellation/retry tests and provider cache evidence if resumed |

## Current offline milestone

The context analyzer replays saved **baseline** request histories without executing
tools or contacting providers. It identifies candidates using only evidence already
present at each request. Reports distinguish candidate message bytes, heuristic
token/price estimates, and verified workflow-boundary coverage.

The offline acceptance scenarios use deterministic test providers and real local
middleware/tool/archive code. They exercise observable evidence preservation,
actual duplicate retirement, and recovery. Passing them demonstrates mechanics,
not Claude answer quality or measured savings.

## Release and product claims

- A lower bill from one successful run does not establish compaction savings.
- A lifecycle signal does not prove that historical content is unnecessary.
- Candidate message share excludes missing system/tool definitions and cannot be
  used as an exact whole-request spend fraction or savings ceiling.
- Routing must account for failed delegates and escalation, and must preserve
  task correctness. Adding a cheap model alone does not satisfy M3.
- Paid verification remains deferred under the user's **“No paid runs now”** instruction.
- Code changes invalidate earlier live-plan digests. Historical plan files and
  results must remain unchanged; a future approved run needs a newly printed plan.

Next: use the offline evidence to improve task selection and compaction safety,
then evaluate current behavior on representative approved live runs before making
an optimization claim or treating the compaction evidence gate as passed.
