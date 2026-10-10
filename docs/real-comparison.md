# Real Claude agent comparison

This implements the requested three versions of one LangChain agent. All three call Claude directly through the same `ChatAnthropic` configuration, tools and task prompts. Local unit tests use test doubles; the `run` command never supplies scripted model answers.

The first approved six-session comparison is now complete: 85 real calls, $9.119886 in receipt-derived list-price cost. All short cases passed and all long cases failed. See [the measured results and failure analysis](gates/M1-comparison.md). Commands below remain the run procedure; `runs/comparison-001` already exists on the development machine, and a new paid batch requires a fresh reviewed plan and output directory.

| Mode | Context processing | Provider billing |
| --- | --- | --- |
| `baseline` | Full canonical conversation; no ContextLab middleware | Original Claude usage receipts |
| `contextlab` | Eligible historical tool outputs become archived pointers after trusted workflow boundaries; exact recall remains available | Original Claude usage receipts, including recall continuations |
| `headroom` | Genuine pinned Headroom public compression API, documented lossless agent profile | Original Claude usage receipts after compression |

Shared request path: LangChain agent → selected middleware (or none) → metering/admission/reservation → Claude → usage settlement → shared tools → next agent turn. Docker runs generated tests without API credentials or network access. This is an embeddable middleware benchmark, not an HTTP proxy server.

## Workload

The agent audits synthetic deployment inventories with real-looking replicas, image digests, probes, ingress rules, resource limits and rollback records. Data are deterministic local fixtures, not production customer records. Runtime answers must come from Claude.

1. Read the policy and all four initial deployment files; produce an initial risk audit.
2. The harness replaces those files with an updated snapshot. Reread all four and produce the current audit.
3. Recover historical image digests, change tickets and rollback windows, alongside current digests.
4. Produce a handoff counting resolved, introduced and persistent findings and deployment readiness by region.

The long case contains 48 services (approximately 99 KB of initial JSON); the short case contains eight (approximately 17 KB). Prompts deliberately require full file reads to test a file-reading workload. The agent may save notes or write helpers. These are constrained diagnostics, not a representative sample of all agent workloads. The short case can still compact; it is not guaranteed to be a no-intervention control.

Public tests check JSON structure. Independent graders, withheld from the agent, check exact complete outputs against deterministic answers. Their standard-library imports run with `python -I` so workspace modules cannot replace them. Each phase records an independent diagnostic; the final grader determines output correctness, and snapshot-integrity checks prevent input edits from earning a pass. Intermediate verdicts are not sent to the model. This evaluates a cooperative agent; it is not a general hostile-code grading service.

## Fairness and Headroom profile

`config/comparison.yaml` disables prompt caching in every arm. That avoids the previous pilot's cache carryover across identical prefixes. ContextLab's forecast also prices every future input at the normal input rate. A later cache-enabled study must be a separate experiment; cache-off findings do not establish savings under caching.

The Headroom dependency is `headroom-ai==0.40.0`, calling public `headroom.compress` with `mode='agent'`, `verify_lossless=True`, and `kompress_model='disabled'`. It compresses eligible text tool outputs. System/user content, tool protocol and canonical history remain unchanged. Offline estimation and telemetry/update opt-outs avoid unmetered auxiliary inference or model downloads. Exact settings are recorded in each approved plan.

**Headroom 0.40 protects `read_file` outputs by default**, including old ones. That default is retained. Its structured test-output compression is verified locally, but it may make no changes during this file-audit workload. CCR/retrieval and ML compression are disabled in this selected profile; this is not a comparison against all of Headroom's configurations or capabilities. No intervention is reported as zero, never hidden or represented as a successful optimization.

ContextLab only retires content supported by existing lifecycle evidence (for example, a complete later read covering an earlier read). A completed harness phase creates a trusted decision boundary; it does not declare old data irrelevant or fabricate test success. Original outputs remain recoverable from SQLite.

## Install and inspect without spending

Run from the repository root:

```sh
uv sync --extra headroom --frozen
docker build -f Dockerfile.task -t contextlab-task:local .
.venv/bin/python -m harness check --config config/comparison.yaml \
  --pricing config/pricing.live.example.yaml
.venv/bin/python -m harness plan --config config/comparison.yaml \
  --pricing config/pricing.live.example.yaml
```

Keep `ANTHROPIC_API_KEY` in the existing project-root `.env`, or export it. It is never copied into fixtures, Docker, Git or reports. No need to send it in chat. `check` verifies local key presence and Docker readiness; it does not authenticate or spend provider tokens. `plan` does not load the key or contact Claude. If using `uv run`, include `--extra headroom` to retain the optional dependency.

## Proposed paid matrix and prices

The default comparison has **six sessions**: two tasks × three modes × one repeat. Order rotates by task:

| Task | Order |
| --- | --- |
| `file_audit_long` | baseline → ContextLab → Headroom |
| `file_audit_short` | ContextLab → Headroom → baseline |

Model: `claude-sonnet-4-6`, temperature 0, no SDK retries, 4,096 maximum output tokens per call, 48 calls per session, 100,000 input-token admission limit. Temperature 0 does not guarantee identical trajectories.

Reviewed list prices on 2026-10-10: **$3 per million uncached input tokens, $15 per million output tokens**. The snapshot also records $3.75/$6 per million for 5-minute/1-hour cache writes and $0.30 for cache reads, although this experiment requests no caching. Confirm any account-specific pricing separately. Source: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).

Proposed ceiling: **$3 per session, $18 for all six sessions**. These are limits, not a forecast. For scale only, 18 long-task calls averaging 35,000 input and 1,000 output tokens cost $2.16 per arm; 12 short-task calls averaging 12,000 input and 800 output tokens cost $0.576 per arm. Together those assumed trajectories cost $8.208. Real call counts and tokens may differ substantially; errors, recalls and retries by the agent count too.

Before each provider invocation the runner counts the final request, enforces the input ceiling, and reserves **$0.66144** using the most expensive configured input bucket plus the output ceiling. Therefore a session can stop while it still has less than $0.66144 available; completion is not guaranteed by a $3 limit. Unknown usage retains a conservative charge and blocks the experiment. Provider receipts determine measured cost; the provider invoice remains authoritative. Caps cannot retroactively undo an unexpected provider charge.

## Run after approval

The repository's `AGENTS.md` requires explicit approval of the printed plan, prices, matrix and ceiling before an assistant launches a paid batch. The previous pilot's approval does not authorize this new matrix. Once approved:

```sh
.venv/bin/python -m harness run --config config/comparison.yaml \
  --pricing config/pricing.live.example.yaml \
  --approve-plan PASTE_REVIEWED_DIGEST --output runs/comparison-001
```

Do not generate a digest and run it blindly. Source/config/lockfile changes invalidate the reviewed digest. Use a fresh output directory. Phase/session progress prints to the terminal; use the final report to inspect results. To regenerate it without API calls:

```sh
.venv/bin/python -m harness report runs/comparison-001
```

Artifacts: `report.html`, `summary.json`, task-level `summary.csv`, frozen `plan.json`, shared `ledger.sqlite`, and per-session `result.json`, `trace.json`, input hashes and workspace. Raw traces contain task material; all runs stay local and gitignored. No LLM API is called by opening a report.

## What to inspect

- Exact answer pass rate and integrity checks; passing public shape checks alone is insufficient.
- Real uncached input, cache writes/reads and output tokens, model calls and total cost including failures.
- Cost per successful task, undefined if no task passes; compare task-level rows as well as totals.
- Actual modified requests, ContextLab rebuilds/recalls, middleware processing time and fallback errors.
- End-to-end latency includes model calls, token counting, tools and grading. Provider-call latency is also in the ledger.
- Any missing sessions or budget stops; unequal completed workloads cannot support a pooled winner.

One repeat is an integration pilot. Even if ContextLab costs less, verify that it actually changed inputs, preserved answers, and included recall overhead before attributing a benefit. Do not call token estimates billed savings or present this narrow file-audit comparison as general superiority over Headroom.

## Main implementation files

`harness/agent.py` defines the shared agent/tools; `harness/runner.py` executes phases, budgets and checks. `contextlab/adapters/langchain_mw.py` implements ContextLab integration; `contextlab/adapters/headroom_mw.py` implements Headroom projection. `harness/tasks/file_audit.py` defines fixtures and independent answers. `harness/report/` and `harness/scoreboard.py` produce evidence exports.
