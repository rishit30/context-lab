# ContextLab

ContextLab is a context lifecycle controller and evaluation harness for tool-using AI agents. It archives original tool outputs, selectively shortens historical evidence after workflow transitions, supports exact partial recall, and accounts for the cost of attempted work.

**Status:** M0 implementation with a completed first live pilot. All **9/9 sessions passed**, using **103 successful model calls** and **$0.53146455** in usage-derived list-price cost. No ContextLab compaction or recall occurred: the short tasks never produced the required fail-to-pass boundary. This verifies live integration, not compaction savings or quality equivalence. Routing, learned recall, long-conversation memory and the HTTP gateway remain later milestones. See [the current gate](docs/gates/M0.md) and `runs/live-001/report.html`.

**Docker validation:** `contextlab-task:local` passed 17 real-container checks on the development machine. Start Docker and build the image on your own machine before live runs. See [local Docker readiness](docs/local-docker.md) for the validation record and commands. The readiness check below verifies local configuration; it does not authenticate with Anthropic or make model requests.

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
- Shared agent with read/write/edit/search/test tools; `shell` only lists files or returns workspace status. Generated Python runs only in an explicitly configured Docker executor.
- Three available experiment arms: `baseline`, `threshold_edit`, and `contextlab`.
- Interleaved runs, frozen approved configuration, atomic spend reservations, all-purpose cost records, failed-attempt inclusion, and static HTML reports.

Headroom is **not** impersonated by another compressor. Its comparator is unavailable until its optional package integration is installed and tested. Deep Agents is likewise deferred. The implemented third arm is explicitly named `threshold_edit` and uses LangChain's installed `ContextEditingMiddleware`.

## Validate

```sh
uv run pytest -q
uv run ruff check .
uv run mypy --strict contextlab/core
```

Tests perform no real inference. They include actual LangChain graph execution with a fake tool-capable model, persistence/restart, malformed tool protocols, cost math, partial reads, simulated provider failures, and spend-cap integration.

## Prepare a live experiment

1. Install/start Docker and build the isolated task image:

   ```sh
   docker build -f Dockerfile.task -t contextlab-task:local .
   ```

2. Set `ANTHROPIC_API_KEY` in the project-root `.env` file, using `.env.example` as a template, or export it in your shell. The `check` and `run` commands load only this key from `.env`; an exported value takes precedence. Quoted values and an optional `export` prefix are supported; shell commands and variable expansion are never evaluated. Other `.env` entries are ignored. The file is gitignored, excluded from Docker builds, and never copied into run artifacts. Keep it readable only by your account (`chmod 600 .env`). Do not put credentials in source, traces, or prompts. Disable external LangSmith tracing in your shell for the local evaluation.
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

## Evidence and limitations

- Read `docs/architecture.md` for corrections to the supplied design and the remaining roadmap.
- Read `docs/verified_apis.md` for inspected SDK versions, source paths, and external references.
- Read `docs/gates/M0.md` for the milestone gate and live validation gaps.
- Task originals and all model/tool content are local in `runs/`; this directory is gitignored. Retention is until the operator removes an experiment, not a promise of permanent storage.
- The token forecast is a byte heuristic. Actual API usage determines live costs. Stable bytes do not prove a cache hit.
- The first suite has three small diagnostic coding tasks. Representative long tasks, historical-recall challenges, short controls, and broader competitors are still required before product claims.
- Docker limits execution, files, networking, runtime and output; the grading procedure is for cooperative coding agents and is not a malicious-code-proof grading sandbox.
- The current middleware is synchronous. The local archive is scoped per session, but this release is not an authenticated multi-tenant service.
- Full raw history is retained, so host-side history/storage still grows. Compaction reduces selected model input, not local storage requirements.
