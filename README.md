# ContextLab

ContextLab is a context lifecycle controller and evaluation harness for tool-using AI agents. It archives original tool outputs, selectively shortens historical evidence after workflow transitions, supports exact partial recall, and accounts for the cost of attempted work.

**Start here: [Python/LangChain end-to-end guide](docs/python-integration.md).** The reusable API is `ContextLabSession`; the new real, read-only file agent supports baseline, ContextLab + Jev, and Headroom. Print its plan with `.venv/bin/python -m examples.file_agent plan`. It requires review of that plan before paid execution. The software image is built with `docker build -t contextlab:local .`; `Dockerfile.task` remains the separate generated-code executor.

**Current implementation:** verified exact duplicate reads can be retired while a full copy remains visible; Jev evaluates information loss against retained evidence and rotates candidate coverage; incomplete/truncated benchmark phases receive bounded, metered recovery; `python -m harness audit RUN_DIRECTORY` reconciles stored receipts without inference. These changes have their own offline validation. Historical live results below belong to earlier source snapshots and do not establish savings for this revision.

**Offline context analysis:** `python -m harness xray RUN_DIRECTORY --output NEW_REPORT_DIRECTORY` replays saved baseline requests and exports a portable evidence report without inference. It separates rule-candidate message share, receipt-based input costs, conditional estimates and boundary coverage. Candidates are not proven waste or savings. See [offline validation](docs/offline-validation.md) and the [full requirements checklist](docs/requirements-status.md).

**First live pilot:** all **9/9 sessions passed**, using **103 successful model calls** and **$0.53146455** in usage-derived list-price cost. No ContextLab compaction or recall occurred in that pilot: the short coding tasks never produced the required fail-to-pass boundary. It verified live integration. Learned recall, long-conversation memory and the HTTP gateway remain later milestones. See [the historical M0 gate](docs/gates/M0.md) and `runs/live-001/report.html`.

**Read-only routing:** opt-in bounded evidence delegation now supplements compaction.
The [five-configuration demonstration](docs/routing.md) compares baseline,
compaction-only, routing-only, combined ContextLab and Headroom using the same
inputs and tools. It runs offline with explicit simulation providers, exercises
quote validation and fallback, and makes no real savings or quality claim.

**Docker validation:** `contextlab-task:local` passed 17 real-container checks on the development machine. Start Docker and build the image on your own machine before live runs. See [local Docker readiness](docs/local-docker.md) for the validation record and commands. The readiness check below verifies local configuration; it does not authenticate with Anthropic or make model requests.

**Real three-mode comparison completed:** baseline, ContextLab and genuine `headroom-ai==0.40.0` made **85 real Claude calls for $9.119886**. All three short audits passed; all three long audits failed. ContextLab actively compacted context and correctly recovered historical facts on the short task, costing $2.071371 versus baseline's $2.118492. One pair does not establish savings; Headroom made no context changes in this file-reading workload. See [the comparison gate and failure analysis](docs/gates/M1-comparison.md), [run guide](docs/real-comparison.md), and local `runs/comparison-001/report.html`.

**Latest Jev comparison: all three arms passed.** After shared tool-input recovery fixes, fresh short runs cost baseline **$1.730220**, ContextLab + Jev **$1.531584552**, and Headroom **$2.041944**. Total **$5.303748552**, including two Jev calls costing $0.000552552. ContextLab's observed bill was 24.99% below Headroom, but neither middleware changed context: Claude used different call sequences. This is not evidence of compression savings or general superiority. See [the fresh comparison audit](docs/gates/M1-comparison.md#fresh-three-arm-jev-comparison-all-passed) and local `runs/jev-validation-001/report.html`.

**Optional Jev decision layer:** Claude remains the task agent; Jev can veto context reductions, while Python owns eligibility, protocol and cost arithmetic. Rules remain the default. The earlier failed Jev pilot cost $0.777890412 and remains in the record, bringing these two Jev evaluation batches to **$6.081638964**. See [the failed-pilot diagnosis](docs/gates/M1-comparison.md#jev-follow-up-live-integration-verified-task-failed).

## Run the offline demo

Clone the repository and run from its root. This prototype uses repository-relative configuration files; the supported setup is a source checkout with its local environment.

```sh
git clone https://github.com/rishit30/context-lab.git
cd context-lab
uv sync --frozen
uv run python -m harness demo
```

The command prints the path of a self-contained `report.html` in `runs/`. Open it in a browser. Generated reports, raw traces and SQLite ledgers are local and excluded from Git. Paths such as `runs/live-001/` in the validation documents refer to the original development-machine artifacts, not files included in this checkout.

The demo uses **scripted synthetic stress traces**, the real controller, real installed LangChain threshold middleware, durable SQLite storage, partial recall, and independently checked bundled reference solutions. It makes **zero API calls**. Its token counts and costs are simulations; fixture checks are not evidence that an AI agent solved the tasks. Recall probes are standalone functionality checks, not costed model continuations.

## What exists

- Framework-independent Pydantic records and pure policy code.
- Covering-read and matching-test lifecycle signals, recent-output guards, explicit boundaries, deterministic visibility rendering.
- Warm/cold economic forecasts that count policy and expected recall costs. Unknown cache-prefix eligibility gets no credit.
- Immutable session-scoped SQLite snapshots, WAL, durable view state, logged partial recalls, and recall pins.
- Content-only tool-result rewriting; original history, tool IDs/order, and opaque blocks remain intact. On interception failure the original request continues without a provider retry.
- Verified synchronous LangChain middleware and Anthropic caching/usage integration.
- Optional TypeSafe Jev relevance assessment, with conservative keep-on-uncertainty behavior and separately reported decision costs included in the shared budget.
- Shared agent with read/write/edit/search/test tools; `shell` only lists files or returns workspace status. Generated Python runs only in an explicitly configured Docker executor.
- Available experiment arms: `baseline`, `threshold_edit`, `contextlab`, and optional `headroom`.
- Interleaved runs, frozen approved configuration, atomic spend reservations, actual request-change telemetry, failed-attempt inclusion, and HTML/JSON/CSV reports.

The Headroom arm calls the pinned project's public compression API with its lossless agent profile and ML compression disabled. It preserves file-read outputs under its upstream protection rules; that limitation is visible in the benchmark. It is separate from `threshold_edit`, which remains available for the historical coding suite. Deep Agents is deferred.

## Validate

```sh
uv run pytest -q
uv run ruff check .
uv run mypy --strict contextlab/core
```

For Headroom integration tests, first run `uv sync --extra headroom --frozen`, and use `uv run --extra headroom ...` when running checks through uv. Tests requiring the optional package skip when it is absent.

Tests perform no real inference. They include actual LangChain graph execution with a fake tool-capable model, persistence/restart, malformed tool protocols, cost math, partial reads, simulated provider failures, and spend-cap integration.

## Prepare a live experiment

1. Install/start Docker and build the isolated task image:

   ```sh
   docker build -f Dockerfile.task -t contextlab-task:local .
   ```

2. Set `ANTHROPIC_API_KEY` in the project-root `.env` file, using `.env.example` as a template, or export it in your shell. The `check` and `run` commands load this key and, only for a Jev configuration, `TYPESAFE_API_KEY`; an exported value takes precedence. Quoted values and an optional `export` prefix are supported; shell commands and variable expansion are never evaluated. Other `.env` entries are ignored. The file is gitignored, excluded from Docker builds, and never copied into run artifacts. Keep it readable only by your account (`chmod 600 .env`). Do not put credentials in source, traces, or prompts. Disable external LangSmith tracing in your shell for the local evaluation.
3. Review `config/pricing.live.example.yaml`, the exact model ID, and your account's current rates. The default `config/pricing.yaml` is synthetic and cannot launch live inference.
4. Inspect readiness and the exact plan:

   ```sh
   uv run python -m harness check --pricing config/pricing.live.example.yaml
   uv run python -m harness plan --pricing config/pricing.live.example.yaml
   ```

   Planning and readiness checks do not contact the model provider. A successful check confirms local key presence and Docker readiness, not key validity, model access or account credit. Offline `plan`, `demo` and `report` commands do not load `.env`. The default matrix is 3 tasks × 3 arms × 1 repeat. The configured cap is **$2 per session, $10 per experiment**. These are ceilings, not predicted spend. The digest binds the model, prices, matrix, policy, source files, lockfiles, and task container definition.

5. After explicitly reviewing and approving that plan, launch it using the printed digest:

   ```sh
   uv run python -m harness run --pricing config/pricing.live.example.yaml \
     --approve-plan PASTE_APPROVED_DIGEST --output runs/live-001
   ```

The project's supplied brief requires explicit approval before paid batches; an assistant should not auto-generate and execute the approval step. Changes to the bound configuration/source require a new plan. Live results remain an indicative pilot even after three repeats; that threshold alone does not establish statistical reliability.

For a one-session ContextLab smoke test, add `--config config/manual.yaml` to `check`, `plan` and `run`. That configuration has a $2 total ceiling. Use a new output directory for every run. To launch with the existing environment directly, use `.venv/bin/python -m harness` from the repository root.

The provider token-count endpoint gates each inference request. The budget reserves a conservative configured input ceiling plus bounded output at the most expensive applicable input rate. Unknown usage charges the outstanding reservation and blocks further experiments pending reconciliation. Hidden SDK retries are disabled. The provider remains the billing authority; actual over-reservation charges are recorded and stop subsequent calls.

## Use Jev for context decisions

Set `TYPESAFE_API_KEY` alongside the Claude key. From the repository root:

```sh
# Authenticated model listing only: no model inference or task content.
.venv/bin/python -m harness jev-models

# Local readiness and a reviewable plan; neither command performs inference.
.venv/bin/python -m harness check --config config/jev-comparison.yaml --pricing config/pricing.live.example.yaml
.venv/bin/python -m harness plan --config config/jev-comparison.yaml --pricing config/pricing.live.example.yaml
```

The prepared pilot is **one ContextLab short file-audit session, one repeat, $3 total ceiling shared by Claude and Jev**. It permits at most 48 Claude calls and 8 Jev requests. Run only after reviewing and approving the printed digest:

```sh
.venv/bin/python -m harness run --config config/jev-comparison.yaml \
  --pricing config/pricing.live.example.yaml --approve-plan PASTE_APPROVED_DIGEST \
  --output runs/jev-pilot-001
```

Jev is pinned to `jev-1.13.0`, priced at $0.042 per million input tokens and zero for output, verified against [TypeSafe's model reference](https://docs.typesafe.ai/models) on 2026-10-10. A new eligible workflow boundary can trigger one batched request. The current user request, earlier user constraints and complete candidate tool results go to TypeSafe, bounded to 24,000 state bytes and eight candidates. Content that does not fit stays visible. This is additional provider data sharing when Jev is enabled.

Each `noul` answer estimates whether removing a candidate would lose needed unique information, given the complete evidence that will remain visible. Candidates and their selected replacement evidence are packed together; previously unassessed candidates get priority at later boundaries. Only values below 0.15 permit the existing reduction; other values keep it. Proven exact duplicates bypass Jev while their full retained copies remain protected. This threshold and the recall-risk prior are **uncalibrated**. Jev cannot remove additional chunks or override recent-output/recall pins. Python recalculates the economic forecast including the actual Jev fee. An unusable assessment keeps the current view. Original content remains archived for recall.

The transport uses the official HTTPS endpoint with no redirects or retries. Each request reserves $0.002752512 against the shared cap (65,536 input tokens, conservatively interpreting the documented 64k limit). TypeSafe provides no verified token-count endpoint or output-limit parameter; byte caps are admission limits, not exact token counts. Actual usage settles the reservation. Missing usage blocks subsequent paid calls; malformed decisions with valid usage still incur their real cost. Reports show decision calls, tokens, fees and assessments; totals already include those fees.

The first approved pilot is preserved at `runs/jev-pilot-001/`. Its failed result remains part of attempted-work cost. The approved follow-up in `runs/jev-validation-001/` completed all three modes successfully for $5.303748552. Its configuration is `config/jev-validation.yaml`: baseline, ContextLab + Jev, and Headroom, one run each, **$3 per session and $9 total ceiling**. Future repetitions require a new output directory and explicit approval of their printed plan. The common agent fixes changed prompts/tool schemas, so this follow-up used fresh comparator runs. No claim that ContextLab is cheaper in every case is supported.

## Evidence and limitations

- Read `docs/architecture.md` for corrections to the supplied design and the remaining roadmap.
- Read `docs/verified_apis.md` for inspected SDK versions, source paths, and external references.
- Read `docs/gates/M0.md` for the milestone gate and live validation gaps.
- Task originals and model/tool traces are retained locally in gitignored `runs/`. Provider requests send selected contents to Anthropic and, when enabled, TypeSafe. Local retention lasts until the operator removes an experiment, not a promise of permanent storage.
- The token forecast is a byte heuristic. Actual API usage determines live costs. Stable bytes do not prove a cache hit.
- The first suite has three small coding tasks. The second live suite tested synthetic deployment audits and historical recall; all short cases passed after real interventions, but long cases failed. Broader representative tasks, robust execution and repeats remain necessary before product claims.
- Docker limits execution, files, networking, runtime and output; the grading procedure is for cooperative coding agents and is not a malicious-code-proof grading sandbox.
- The current middleware is synchronous. The local archive is scoped per session, but this release is not an authenticated multi-tenant service.
- Full raw history is retained, so host-side history/storage still grows. Compaction reduces selected model input, not local storage requirements.
