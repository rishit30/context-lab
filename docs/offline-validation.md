# Inspect ContextLab without API spending

These commands run from the repository root using the existing development
environment. They do not load provider keys or execute model inference.

## Inspect an existing baseline run

```sh
.venv/bin/python -m harness xray runs/jev-validation-001 \
  --output runs/context-xray-001
```

Use any saved benchmark directory that contains baseline sessions. The analyzer
supports a shared root ledger and the offline demo's per-session ledgers. Historical
development-machine runs are gitignored and are not distributed with the repository.
For a fresh checkout, generate the labeled simulation below instead.

The command prints JSON. With `--output`, it also creates `xray.json` and a portable
`report.html` in a **new directory outside the source run**. Existing directories
are refused to prevent replacing prior evidence. Without `--output`, it writes
nothing. An invalid or missing trace produces an explicit error and exit status 2;
missing evidence is not silently converted into zero opportunities.

## Run the no-key demonstration on a fresh checkout

Install the pinned development environment with `uv sync --extra headroom --frozen`
if it is not installed already. Then:

```sh
.venv/bin/python -m harness demo --output runs/offline-demo-001
.venv/bin/python -m harness xray runs/offline-demo-001 \
  --output runs/offline-demo-xray-001
```

Choose new output names when repeating. Open the generated `report.html` in a
browser. The demo uses scripted traces and synthetic usage. Its financial X-ray
fields remain unknown because there are no provider receipts to attribute.
It demonstrates the software, not model performance.

## Run the acceptance scenarios

```sh
.venv/bin/python -m pytest -q tests/test_offline_acceptance.py \
  --junitxml=runs/offline-acceptance.xml -o junit_family=legacy
```

The deterministic test provider derives answers only from the messages delivered
to it. Real LangChain graphs, local Workspace tools, context projection, SQLite
storage and recall run underneath it. Network access is forbidden by these tests.

The scenarios check:

- A long paired inspection with repeated reads: exact independently computed
  answers must match, requests must actually shrink, and originals must remain intact.
- Unique old and current facts: a test-only conservative semantic veto preserves
  both versions. This does not measure Jev accuracy.
- A later historical question: the graph must recover retired facts through the
  real recall tool rather than derive answers from a hidden oracle.
- Incomplete work: reduction must wait for a trusted workflow boundary.
- Paginated recall and restart: every original byte must be recoverable and the pin
  must survive reopening the archive.
- Corrupt duplicate dependency state: evidence must be restored before the answer
  consumer sees the next request.

JUnit properties expose changed-request counts and byte measurements. They are
mechanical observations; do not label them token savings or provider cost savings.

## What the X-ray numbers mean

The primary metric is **estimated candidate-context share**: tool-result payload
bytes identified by supported rules divided by normalized serialized message-view
bytes, summed across saved baseline requests. JSON escaping and protocol fields
remain in that denominator. System instructions and tool schemas are absent from
the saved message views, so this is not the fraction of total provider input.

The analyzer uses current candidate logic and the saved plan's compaction settings,
or explicitly reported defaults when no settings exist. It checks each request's
own visible prefix. Later events, final workspace contents, mutable final pins and
private grader outcomes cannot make earlier content into a candidate.

Candidates respect recent-output and pin protections and retain full duplicate
anchors. They have **not** passed the economic gate or a Jev assessment. Changed
historical values can be candidates while still being necessary for correctness.
The analyzer does not simulate their deletion or subsequent agent behavior.

Boundary evidence is separate. Historical phase indices or final `completed` flags
alone cannot establish a validated boundary. Future harness traces carry copied
public-completion markers tied to a message-prefix hash; matching fail-to-pass
tool observations can also supply evidence. Missing timing stays unknown.

For real saved receipts, the report reconciles usage and saved prices. Candidate
token counts use a byte heuristic. Allocation bounds show the least and most
expensive assignment of those estimated tokens across observed input-price
buckets, because individual chunks' cache placement is unknown. Even equal bounds
are conditional estimates, not a measured invoice item or achievable savings.
Decision, recall, rebuild, output and quality effects are outside that estimate.

Reports separate live and simulated observations, expose missing coverage, and
include failed sessions. A low candidate share is a reason to examine workload
fit; the original proposed 10% threshold is not a calibrated product decision.

See `requirements-status.md` for the remaining full-project requirements. Routing
and learned memory remain incomplete, and paid verification is still deferred.
