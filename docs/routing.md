# Bounded read-only evidence delegation

ContextLab can now delegate a narrow evidence-selection subtask to a separately
priced Claude model. The main agent keeps its model, tools and canonical history.
Routing is opt-in and off by default. This is an initial read-only part of M3;
it does not implement arbitrary agent routing or write-capable delegates.

## What the worker does

The application supplies immutable `EvidenceSource` records containing relative
paths and complete UTF-8 file snapshots. The main agent calls `analyze_files`
with a goal, question-ID map and source IDs. All five comparison configurations
expose the same tool schema. Baseline, compaction-only and Headroom return the
original requested evidence. Routing-only and combined ContextLab may delegate.

1. Validate the request, source IDs, content hashes and complete brief. Oversized
   input is rejected before dispatch; files and questions are never silently cut.
2. Apply the explicit read-only task allowlist, persistent call limits, budget
   admission and an economic forecast. Optional Jev can veto suitability; it
   cannot change permissions, override limits or approve its own spending.
3. Count the worker's actual input using the provider endpoint and reserve its
   configured maximum charge. Call it once with bounded output and SDK retries
   disabled. It receives only the brief: no tools, filesystem or shell access.
4. Settle its usage receipt, including failed or invalid responses. Validate JSON,
   question IDs, source IDs, exact line ranges and byte-for-byte source quotes.
5. Return grounded candidate quotes to the existing main agent. Rejected routing
   or invalid worker output returns complete original evidence instead. The next
   main-agent call follows normal metering; no hidden fallback model is invoked.

Exact quotes prove provenance and question-ID coverage. They do **not** prove
relevance, completeness or correctness. The main agent must assess the evidence
and can use the same read-only file tools to inspect originals. Independent
task-level evaluation is still needed before making quality or savings claims.

The economic gate is a forecast with an uncalibrated failure prior. It includes
worker cost, decision fees, main-model evidence overhead and expected fallback.
It credits only estimated original-source input avoided at the main model;
the main agent's final output remains payable on either path. Small bundled
incident files may correctly bypass delegation because worker overhead is larger
than the potential input reduction.
It is not a measured saving, a guarantee of beating Headroom, or a calibrated
prediction for warm-cache workloads.

## Run the offline demonstration

From the repository root, after installing the pinned Headroom extra:

```sh
.venv/bin/python -m examples.routing_agent demo --output runs/routing-demo-001
.venv/bin/python -m harness audit runs/routing-demo-001
```

Open `runs/routing-demo-001/report.html`. The five configurations are `baseline`,
`cl_compact_only`, `cl_route_only`, `contextlab` (combined), and `headroom`.
The main model, public questions, tool schemas and frozen files are shared.

This demo uses deterministic local providers, synthetic prices and real LangChain
graphs, tools, middleware, validators and ledgers. It exercises duplicate reads,
accepted exact quotes and an intentionally invalid quote that returns the full
evidence. Mechanical answers are checked independently. Provider spend is zero;
the displayed synthetic cost units are not measured dollars or LLM quality.

## Add routing to an application

```python
from pathlib import Path
import yaml
from contextlab.core.models import Pricing
from contextlab.core.routing import EvidenceSource, RoutingConfig
from contextlab.session import ContextLabSession, SessionConfig

main = Pricing.model_validate(yaml.safe_load(Path("config/pricing.live.example.yaml").read_text()))
worker = Pricing.model_validate(yaml.safe_load(Path("config/pricing.worker.example.yaml").read_text()))
settings = SessionConfig(
    pricing=main,
    routing=RoutingConfig(enabled=True, worker_pricing=worker),
)
sources = [EvidenceSource(id="incident.txt", path="incident.txt", content="owner: platform-team\n")]
with ContextLabSession(Path("runs/application/ledger.sqlite"), "incident", settings) as session:
    graph = session.build_agent(
        tools=[],  # Supply your application's read-only original-file tools here.
        system_prompt="Use analyze_files for evidence. Assess quote relevance and completeness.",
        evidence_sources=sources,
    )
    # graph.invoke(...) performs real paid inference; review prices and caps first.
```

`SessionConfig.routing.engine="jev"` enables a separate routing suitability
decision. `SessionConfig.decision_engine="jev"` controls context-compaction
assessments; these are different decisions. Both share the same budget ledger.
Provider keys belong in the process environment. The reusable session does not
load `.env` itself.

The catalog is bound to the session by a durable hash. Reopening a session with
different sources is rejected; use a new session ID for new snapshots. Keep one
active synchronous session/agent per ID. Persist the full conversation yourself.
The injected LangChain tool-call ID makes a repeated tool execution idempotent:
the saved result is returned without another worker charge. Changed input under
the same ID is rejected. An interrupted or unknown receipt cannot silently retry.

## Prepare a future real comparison

This prints a plan and makes no provider requests:

```sh
.venv/bin/python -m examples.routing_agent plan \
  --worker-pricing config/pricing.worker.example.yaml --routing-engine jev
```

The plan binds code, lockfiles, both model/pricing snapshots, the five settings,
Headroom profile, immutable inputs, questions and spend caps. The worker example
uses a pinned legacy model for SDK compatibility; verify its current availability
and account rates before any real run. Replacing it invalidates the plan.

Only after explicit approval of a newly printed plan:

```sh
.venv/bin/python -m examples.routing_agent run \
  --worker-pricing config/pricing.worker.example.yaml --routing-engine jev \
  --approve-plan PASTE_REVIEWED_DIGEST --output runs/routing-live-001
```

Live execution always uses real providers. It cannot accept simulated providers
or fixture answers. Completing the file example is not independent grading.
Current live quality, savings and Jev calibration remain unverified. No paid run
is authorized by the offline implementation or these command examples.
