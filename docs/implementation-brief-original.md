# Codex Prompt: Build ContextLab (Routing + Context Compaction)

## How to use this file

1. Create an empty repo and put `architecture.md` at `docs/architecture.md`.
2. Paste **Part A** into Codex as the first message. Codex saves it as `AGENTS.md` so every later session reads it automatically.
3. For each milestone, paste the matching section from **Part B** as a new task.
4. At the end of every milestone, paste **Part C** (gate review). Do not start the next milestone until you have read the gate report.

---

# Part A: Standing brief (save as AGENTS.md)

You are building **ContextLab**, a decision layer between an AI agent and its model provider. It makes two kinds of decisions:

1. **Compaction**: what past content the model sees, at what size, and when the prompt is reorganized.
2. **Routing**: whether a subtask runs on a cheaper model, with what brief, and when to fall back.

The full design is in `docs/architecture.md`. Read it before writing code. When this brief and the architecture disagree, stop and ask.

## Non-negotiable rules

1. **Never invent APIs.** For LangChain, Deep Agents, Headroom, the Anthropic SDK and TypeSafe Jev, read the installed package source or official docs before using an API. If you cannot verify an API, write a stub that raises `NotImplementedError("verify: <what>")` and list it in the gate report.
2. **Pin versions.** Record every dependency version in `pyproject.toml` and in `docs/verified_apis.md` with what you checked and how.
3. **Fair arms.** All experiment arms share the same agent code, model, system prompt, tool definitions, caching settings, task set and seeds. The only difference is the layer under test. A test must assert byte-identical system prompt and tool definitions across arms.
4. **Fail closed.** If ContextLab's rewriting throws or produces an invalid request, send the baseline-equivalent request, log the error, and continue. The agent must never crash because of ContextLab.
5. **Tool-call pairing.** Never remove a `tool_use` or `tool_result` block. Hiding means replacing the result's content with a signpost. Validate every rendered request before sending.
6. **Code owns exact work.** Arithmetic, counting, dates, token counts, permissions and pairing are always code. The decision engine is only for narrow fuzzy judgments listed in the architecture's decision register.
7. **Charge every cost.** Decision-engine calls, triage calls, delegation calls and escalations are added to the ContextLab arm's total cost. Never report them separately as ignorable overhead.
8. **Spend control.** Before any batch of real API runs, print an estimated cost and the run matrix, then stop and wait for my explicit "go". Hard caps per session and per experiment live in `config/experiment.yaml`; exceeding a cap aborts the batch.
9. **Secrets.** API keys come only from environment variables. Never log them, print them, or write them into run artifacts.
10. **Honest results.** Fewer than 3 repeats per task per arm means results are labelled "indicative only". Never describe an indicative difference as a win.
11. **Stop at gates.** At the end of each milestone, produce the gate report (Part C format) and stop.

## Tech stack

- Python 3.11+, `uv` for environment management, `pytest`, `ruff`, `mypy --strict` on `contextlab/core`.
- LangChain v1 `create_agent` with middleware for the agent loop.
- `langchain-anthropic` for the model; prompt caching enabled identically in every arm.
- Deep Agents (`deepagents`) filesystem backend for storing hidden content and for `read_file`-style recall, if its API fits; otherwise a minimal local equivalent with the same semantics.
- SQLite (WAL) for stores, one database per run directory. SQLite FTS5 for keyword search; vector search only from M5.
- Pydantic for all config and record schemas.
- Plain HTML (Jinja2) for reports. No web framework until the gateway milestone.

## Repo layout

```text
contextlab/
  core/            # framework-agnostic engine: events, chunks, signals, cost model, planner, view builder
    events.py
    chunks.py
    refs.py
    signals.py
    cost_model.py
    planner.py
    view_builder.py
    validator.py
    router.py       # M3
    brief.py        # M3
    triage.py       # M4
    memory/         # M5: facts, retrieval, evidence gate
  decisions/
    base.py         # DecisionEngine protocol, Distribution type
    rules.py
    llm.py          # cheap model with structured output
    jev.py          # stub until verified
  adapters/
    langchain_mw.py
    gateway/        # M6
  ledger/
    store.py        # SQLite schema + writes
    pricing.py
  tools/
    recall.py
harness/
  arms/             # baseline.py, headroom.py, contextlab.py, ... one file per arm
  agent.py          # the single shared agent definition used by all arms
  runner.py         # interleaved run matrix, caps, cost estimate
  tasks/            # task specs + fixture repos
  checks.py         # task success checks
  scoreboard.py
  xray.py           # waste X-ray from baseline runs
  report/           # HTML templates
config/
  pricing.yaml      # verified prices, cache multipliers, TTL; date verified
  models.yaml       # model ids per role: main, cheap, decision
  compaction.yaml   # thresholds K, M, theta, N prior, recall priors
  routing.yaml      # delegable kinds, p_fail priors, brief budget
  experiment.yaml   # arms, tasks, repeats, caps
docs/
  architecture.md
  verified_apis.md
  gates/            # one gate report per milestone
tests/
runs/               # gitignored; one directory per experiment
```

## Coding standards

- `contextlab/core` has no imports from LangChain, Anthropic or any network library. Adapters translate.
- Every pure function in `core` has unit tests with hand-built event sequences. No API calls in unit tests.
- Rendering is deterministic: the same events, visibility and epoch produce identical bytes. Test it.
- Every decision writes a row to `decisions` with inputs, output, engine, confidence, latency and cost.
- Small, reviewable commits with clear messages. One milestone per branch.

---

# Part B: Milestones

Each milestone lists what to build, required tests, and acceptance criteria. Do not pull work forward from later milestones.

## M0: Foundations and three live arms

Goal: three arms (`baseline`, `headroom`, `contextlab`) running the same tasks from day one, with trustworthy cost accounting.

Build:

1. **Verification first.** Before writing feature code, check and record in `docs/verified_apis.md`:
   - LangChain middleware hook names and signatures in the installed version (`wrap_model_call`, `wrap_tool_call`, or whatever exists).
   - How `langchain-anthropic` enables prompt caching and how it reports cache-write and cache-read tokens in usage metadata.
   - Current Anthropic prices per model, cache-write multipliers and cache TTL. Ask me to confirm the numbers before they go into `config/pricing.yaml`.
   - Headroom's open-source package, its LangChain integration, and its licence. If no usable LangChain integration exists, tell me and propose a substitute; do not fake the arm.
   - Deep Agents filesystem backend API.
2. **Ledger.** SQLite schema from architecture section 5 (all tables; later milestones fill them). Pricing module computes cost from the four token types.
3. **Shared agent.** `harness/agent.py` defines the one agent: model from `config/models.yaml`, a fixed system prompt, tools `read_file`, `write_file`, `edit_file`, `search` (ripgrep), `run_tests`, `shell` (restricted to the task workspace), plus `recall` (present in every arm so tool definitions stay identical; in non-ContextLab arms it returns "nothing hidden").
4. **Arms.** `baseline` (no layer), `headroom` (Headroom wrapper), `contextlab` (middleware with the M0 slice below).
5. **ContextLab M0 slice.** Ingestor + chunk store + refs for paths only + one signal (`superseded_read`) + one boundary (tests go from failing to passing) + the move-or-stay rule from architecture 6.6 with a fixed `N` from config + view builder with the 4-layer layout + validator + recall tool.
6. **Tasks.** Three coding tasks in `harness/tasks/`, each a small fixture repo with a seeded bug, a test suite, and at least two phases (fix bug, then a follow-up change). Success check is code (tests pass + task-specific assertion).
7. **Runner.** Interleaved run matrix (round-robin over arm × task × repeat), per-session and per-experiment caps, cost estimate and wait for "go".
8. **Scoreboard v0.** Table per arm: runs, solved, pass rate, total cost, cost per solved task, four token types, cache hit ratio, turns.

Tests:

- Arm invariance: system prompt and tool definitions byte-identical across all arms.
- Pricing: hand-computed cost for a fake usage record matches the module.
- Signal: a read → edit → read sequence marks the first read `superseded_read`.
- Validator: a rendered request with a hidden chunk keeps every `tool_use`/`tool_result` pair.
- Determinism: rendering the same state twice yields identical bytes.
- Fail-closed: an injected exception in the planner yields a baseline-equivalent request.
- Move-or-stay: table-driven tests including the cold-cache case.

Acceptance:

- One real run per task per arm (9 sessions) completes after my "go".
- Ledger totals match the provider's reported usage for every call.
- Scoreboard prints, labelled "indicative only".
- Gate report lists every unverified API.

## M1: Staleness signals, waste X-ray, recall tracking

Goal: know how much of baseline spend is stale. This is the first go/no-go.

Build:

1. Refs: add symbols (Python AST for `.py`, regex fallback), error signatures (normalized exception type + location), test ids.
2. Signals: `resolved_test_log`, `unreferenced_search`, `moved_on` (thresholds from config).
3. Visibility ladder `full/long/short/hidden` with deterministic code renderings for `long` and `short` per chunk kind.
4. Recall logging: every recall row records which signal hid the chunk.
5. **Waste X-ray** (`harness/xray.py`): replay signals offline over baseline sessions and compute the share of input spend (weighted by token price type) that went to content stale at the time it was read. Report per task and overall.
6. Session timeline HTML: turns, epochs, signals, rebuilds, recalls, cost per turn.
7. Grow tasks to 6, including **one adversarial task** where a detail from phase 1 is needed in phase 2 after it would look stale.

Tests: one unit test per signal with positive and negative sequences; X-ray on a synthetic session with a known answer.

Acceptance:

- 6 tasks × 3 arms × 2 repeats after my "go".
- X-ray result reported prominently in the gate report. **If baseline stale share is below about 10%, flag it as a likely no-go for compaction and recommend whether to continue.**

## M2: Cost-aware rebuilds and a real scoreboard

Goal: the compaction core, measured with enough repeats to mean something.

Build:

1. Boundary detection: all candidates from architecture 6.5, including cache-expiry.
2. Full move-or-stay with `N` estimated from completed sessions (lower quantile), falling back to config prior.
3. Hide-or-keep per chunk (architecture 6.7), with recall cost including the extra-turn reread. `p_recall` from history, then config prior. No model calls yet.
4. Promotion: hidden chunks whose refs reappear are promoted at the next rebuild.
5. New arm `deepagents_evict` (size-based eviction to files, Deep Agents default threshold).
6. Scoreboard v1: bootstrap confidence intervals on cost per solved task and pass rate; p50/p90 cost per run; rebuilds, hides, recalls per task.
7. Grow tasks to 8.

Tests: rebuild never happens mid-epoch; prefix bytes stable between rebuilds; hide decision flips when `p_recall` crosses the break-even point.

Acceptance:

- 8 tasks × 4 arms × 3 repeats after my "go" (show the estimate first).
- Gate report answers plainly: does ContextLab beat `baseline`, `headroom` and `deepagents_evict` on cost per solved task without lowering pass rate, and with what confidence? What happened on the adversarial task?

## M3: Routing by delegation, plus the decision engine

Goal: add routing and find out which component pays: compaction, routing, or both.

Build:

1. `DecisionEngine` protocol with `RulesEngine`, `LLMEngine` (cheap model, structured output), and `JevEngine` stub. Engine selectable per decision in config. Every call logged.
2. D7 subtask classification against the delegable-kinds allowlist in `config/routing.yaml`.
3. Brief builder: goal + chunks selected by ref overlap, capped at a token budget.
4. Delegation runner: cheap model with its own tool budget, acceptance checks (code first, then D8), escalation to the frontier model with the same brief plus a failure note.
5. Delegation cost check (architecture 7.4), `p_fail` from history with config prior.
6. Arms: `naive_router` (per-request classification, full history, model switches allowed), `cl_compact_only`, `cl_route_only`; `contextlab` now includes both.
7. Report: cache hit ratio per model, cost of escalations, delegation success by kind.

Tests: brief never exceeds budget; escalation path runs when acceptance fails; main-loop model never changes in ContextLab arms; decision cost appears in arm totals.

Acceptance:

- Ablation table: `baseline`, `naive_router`, `cl_compact_only`, `cl_route_only`, `contextlab`, 3 repeats.
- Gate report states which component drives savings, and whether `naive_router` lost money to cache misses as predicted (or not; report what the data says).

## M4: Goal-aware triage and recall learning

Build:

1. Triage for outputs above threshold (architecture 6.9): cheap model returns segment indices; code extracts verbatim; full output stored and recallable.
2. Recall learning: per-signal `p_recall` priors updated from recall history across sessions.
3. Optional D6 score from the decision engine for chunks with no history.
4. Arm `headroom_plus_cl`.
5. Shadow mode flag: ContextLab logs decisions and estimated savings but sends the baseline request.

Acceptance: triage cost included; shadow-mode estimated savings compared with the real `contextlab` arm on the same tasks.

## M5: Memory layer and query routing for long conversations

Goal: the conversation workload from the earlier design (J1–J8), built only after the coding workload has a verdict.

Build:

1. Fact store with versioning and supersession (architecture section 9).
2. D12 (memory value) and D13 (support, revision vs conflict) running asynchronously after each turn; unprocessed events always included in retrieval.
3. Hybrid retrieval: FTS5 + vectors (pick a local embedding option and record it), neighbour expansion.
4. D10 query routing with flags; aggregation route computes in code by enumerating records.
5. Evidence gate (D14) with bounded loop: max rounds, time, spend in code.
6. Conversation suite: scripted long chats with superseded facts, no-keyword questions, multi-hop, exhaustive counts, one-turn-old corrections.
7. Arms for this suite: `full_history`, `summary_only`, `naive_rag` (top-k similarity), `contextlab_memory`.

Acceptance: per-category accuracy (not just overall), cost per correct answer, and a list of every wrong answer with the route and evidence that produced it.

## M6: Gateway adapter

Goal: make ContextLab usable by Claude Code and other agents through `ANTHROPIC_BASE_URL`.

Build:

1. Anthropic Messages-compatible proxy (FastAPI or Starlette), streaming supported.
2. Session identification (configurable: header, or hash of system prompt + first user message), with stored rendered prefix per session so rewriting is byte-stable across requests.
3. Shadow mode on by default.
4. Replay test: record a real Claude Code session's requests, replay through the gateway, and verify cache hits are preserved after a rebuild.

Acceptance: replay shows correct tool pairing on every request and cache-read tokens consistent with the in-harness results.

---

# Part C: Gate review (paste at the end of every milestone)

Produce `docs/gates/M<n>.md` with exactly these sections, then stop:

1. **Built**: what exists now, with file paths.
2. **Verified vs assumed**: every external API used, how it was verified, and anything still unverified.
3. **Tests**: count, pass/fail, and anything not tested that should be.
4. **Results**: scoreboard table with run counts and confidence intervals where available; "indicative only" label if fewer than 3 repeats.
5. **What surprised you**: results that contradict the architecture's predictions.
6. **Failures**: every failed run, recall miss, fail-closed event, escalation, and the adversarial task outcome.
7. **Cost**: total spend this milestone, split by arm and by purpose (main, decision, triage, delegate).
8. **Go/no-go recommendation** for the next milestone, with the single number that most supports it.
9. **Questions for me**: anything you need decided before continuing.

Self-check before submitting the report:

- Did any arm get a different system prompt, tool set, model, or caching setting?
- Is any decision-engine or delegation cost missing from an arm's total?
- Did any claim of improvement rely on fewer than 3 repeats?
- Did any rendered request remove a tool block instead of replacing its content?
- Did you use any API you did not verify?
