# M1 comparison — live compaction and recall verified; efficiency unproven

Date: 2026-10-10, Asia/Kolkata. The user approved the six-session plan with **“go run”**. The batch ran from 09:44:08 to 10:00:42 UTC. The pre-live gate is preserved in `runs/comparison-001/m1-before-live.md`.

## Measured results

All six sessions ran. **All three short tasks passed; all three long tasks failed.** There were **85 successful, usage-accounted Claude calls**, costing **$9.119886** at the approved list prices. Successful API responses do not imply successful agent tasks. No additional paid batch was launched.

| Task | Mode | Result | Calls | Input tokens | Output tokens | Cost USD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Long | Baseline | Failed: ValueError | 4 | 136,760 | 4,705 | 0.480855 |
| Long | ContextLab | Failed: BudgetExceeded | 8 | 446,945 | 9,786 | 1.487625 |
| Long | Headroom | Failed: FileNotFoundError | 6 | 321,051 | 9,048 | 1.098873 |
| Short | Baseline | Passed | 23 | 639,529 | 13,327 | 2.118492 |
| Short | ContextLab | Passed | 24 | 622,187 | 13,654 | 2.071371 |
| Short | Headroom | Passed | 20 | 553,395 | 13,499 | 1.862670 |

Total: **2,719,867 input tokens and 64,019 output tokens**. All cache buckets are zero. These are receipt-derived list-price costs, not a reconciled provider invoice.

Including failures, arm totals are baseline **$2.599347**, ContextLab **$3.558996**, and Headroom **$2.961543**. Each arm passed one task, so these also equal this batch's all-attempt cost per passed task. Long tasks stopped at different points; a cheaper failure is not an efficiency win.

## Approval and accounting

- Model `claude-sonnet-4-6`, temperature 0, SDK retries disabled; identical shared system/tool/model configuration and paired initial input manifests.
- Two synthetic audit tasks, three modes, one repeat; caching disabled throughout.
- Headroom 0.40.0 used its documented lossless agent profile with ML/CCR disabled and default file-read protection retained.
- Limits: $3/session, $18 total, 48 model calls/session, 100,000 input tokens, 4,096 output tokens/call.
- Approved digest: `1bd30f5aac7378a15b3f9b6b5b796950b35ea1c20e9e54cd6869f36b5920fb2e`.
- Source/configuration/lockfiles still matched approval after execution. All 85 original receipts and Decimal price calculations reconciled. All reservations settled, with **$0 outstanding**, no unknown usage, no blocked budgets, no monetary cap violations and no adapter failures.
- Recorded input counts matched actual usage. Largest admitted input: 94,296 tokens. SQLite integrity passed, with no foreign-key violations.
- Observed Docker image: `sha256:3e8596ffb608968407fce52fed084a4610a52a3602bc80a208b844f65afb44f6`. The runner used a mutable local tag, not a per-call digest pin. No task containers remained running afterward.

Evidence is local and gitignored under `runs/comparison-001/`: `audit.json`, `intervention-audit.json`, `plan.json`, `approval.json`, `ledger.sqlite`, `report.html`, `summary.json`, `summary.csv`, and per-session traces/results.

## Actual interventions and correctness

The successful short ContextLab run made **one accepted rebuild, changed 12 admitted requests, and performed four genuine recalls**. All private phase checks, final public tests and exact held-out answers passed; input snapshots remained intact. One full recall page recovered each old shard. A later proposal was rejected by the cost safety margin. Accepted/rejected decisions and correct historical retrieval now have real API evidence.

The long ContextLab run rebuilt once and changed two admitted requests. It retired four initial file observations at the historical-review phase, even though the new question required their historical details. Claude immediately recalled every shard, then requested continuation pages. Eight pages executed, but only the first four reached another paid request. The ninth projection attempt was rejected before admission and is excluded from modified-request counts.

Its recorded error is `BudgetExceeded`. Spending was only $1.487625, with enough funds for another $0.66144 reservation and no blocked budget. The runner's checks and ledger point to the **100,000-input-token admission ceiling**, not dollar-budget exhaustion. The rejected token count was not saved; no exact value is claimed.

The fixed 0.1 recall prior did not reflect a phase explicitly requesting old evidence. Large raw JSON-envelope recall pages also amplified context. These policy and retrieval weaknesses remain despite the successful short result.

Headroom made **zero accepted tool-text changes across all 26 paid requests**. Its default protects file reads; some attempted transforms were reverted by lossless verification. Transform labels alone are not accepted compression evidence.

## Interpretation and next fixes

ContextLab's successful short run cost **$0.047121 (2.22%) less than baseline**. This single pair does not establish repeatable savings. Headroom was cheaper still but performed no compression and used fewer model calls; its cost difference cannot be attributed to compression. Temperature 0 did not produce identical trajectories. Aggregate cost per passed task was higher for ContextLab when failed attempts are included.

All long runs reached the 4,096-token output ceiling during verbose analysis and failed to create the initial audit artifact. The harness advanced after the graph returned anyway. Baseline later recorded ValueError and Headroom FileNotFoundError. Their final response and exact failure stage were not persisted, so the precise tool arguments/path cannot be reconstructed.

Before another paid run:

1. Persist stop reasons, final response/tool-call metadata and sanitized failure stages. Detect output truncation and missing deliverables before advancing; support bounded, accounted recovery.
2. Encode file/recall argument bounds in tool schemas and make ordinary invalid-argument/missing-file errors recoverable.
3. Consider the active question before retiring historical evidence; retrieve specific records or source spans rather than large JSON-envelope pages.
4. Review the revised plan, then evaluate the same families plus independent tasks, repeats and separately controlled cache-enabled conditions. Count failed runs and recovery work.

**Gate:** pass for real three-mode integration, audited usage, active compaction, exact recall and correct short-task output. Robust long-task completion and repeatable efficiency remain unproven. No production-savings claim is justified.

## Jev follow-up: live integration verified, task failed

On 2026-10-10, an optional TypeSafe `jev-1.13.0` relevance gate was implemented for item 3.
Claude remains the task agent. The gate can veto existing lifecycle reductions, preserves
context on uncertainty or malformed responses, and includes actual decision fees in the
existing ledger and economic forecast. It does not implement the other long-task repairs above.

- **307 offline tests passed**, Ruff passed, strict mypy passed across eight core modules.
- A real controller/gate/SQLite/budget test with fake provider receipts verifies historical
  context retention, no repeated decision at a persisted boundary, later eligible compaction,
  exact decision fees, and immutable originals. This is not live model-quality evidence.
- Failure tests cover invalid probabilities/usage, unknown-cost blocking, failed settlement,
  durable pending requests after gate recreation, cross-session rejection and no redirects/retries.
- Authenticated `GET /v1/models` succeeded with the supplied key. No inference or task text was
  sent by that check. Local Docker/credential preflight reported ready.
- `config/jev-comparison.yaml` prepared one `file_audit_short` ContextLab session with a shared
  $3 cap, at most 48 Claude requests and eight Jev requests. The frozen review artifact is
  local `runs/jev-pilot-plan.json`. The user subsequently approved execution with "now lets test".

The live pilot is recorded in `runs/jev-pilot-001/`. Approved digest:
`b0caccfc9eed4e39f80544cb0bf9607507363127eb28f208e4a718c0c7426769`.
Independent audits verified the frozen source hash before any repair edits.

| Provider | Calls | Input tokens | Output tokens | Cost USD |
| --- | ---: | ---: | ---: | ---: |
| Claude Sonnet 4.6 | 12 | 215,461 | 8,749 | 0.777618 |
| Jev 1.13.0 | 1 | 6,486 | 58 | 0.000272412 |
| Total | 13 | 221,947 | 8,807 | 0.777890412 |

All receipts/prices reconciled; all 13 reservations settled. There were no unknown charges,
outstanding reservations, blocked budgets, cache tokens or integrity failures. Both initial
and current snapshot audit phases passed exact held-out checks and preserved source inputs.

At the historical question, Jev returned probabilities of need 0.89, 0.94 and 0.95 for three
old shard reads. It vetoed all assessed retirements. The fourth shard did not fit the state
byte limit and stayed visible without assessment. All four originals remained in context;
there were zero rebuilds and zero modified main-model requests.

Claude then called `recall` with four invented IDs, such as
`read_file|deployments/shard-a.json|0`. Archived assistant event `e000037` preserves these
exact calls. Replaying the inputs against `Workspace.recall` reproduces `ValueError("Invalid
chunk ID")`: the slash is rejected before accessing the recall reader. The tool boundary did
not convert this input error into a recoverable tool result. The task stopped in phase three;
neither `history.json` nor the final handoff was produced, and no valid recall completed.

This failed attempt must stay in total-cost accounting. Its lower spend than successful
historical Headroom/baseline runs is not an efficiency win. Jev's threshold remains
uncalibrated; one context-retention decision does not establish broad effectiveness.

The follow-up uses shared tool-error recovery and explicit argument bounds for every arm,
plus a prompt that requires observed archive IDs. Because these alter the common agent,
`config/jev-validation.yaml` prepares fresh baseline/ContextLab+Jev/Headroom short runs,
one repeat, $3/session and $9 total. No follow-up inference is authorized by the earlier
$3 pilot approval. Long-task truncation and representative repeated evaluation remain open.

The repair gate passed **354 offline tests**, Ruff, strict core mypy and local Docker/Headroom
preflight. Tests replay the invalid-ID behavior through a real LangChain graph, recover via an
error tool result, and complete the subsequent write. Expected argument/file errors recover;
budget, permission, ledger and executor failures still propagate. Shared read/recall schemas
publish bounds, and the shared prompt requires observed archive IDs. New bounded response
summaries and sanitized failure-stage/phase diagnostics preserve evidence without provider
headers, opaque thinking blocks or credential values.

The follow-up review artifact is `runs/jev-validation-plan.json`, digest
`c13ce8d33b48af46c8471e677d46f207de6b10d2818828b017cc4fbd7acd2e24`.
It was subsequently explicitly approved with "run all three" and executed as recorded below.
The failed pilot remains in the cost record; no automatic paid retry was performed.

## Fresh three-arm Jev comparison: all passed

The approved `config/jev-validation.yaml` matrix completed in `runs/jev-validation-001/`:
one short file-audit session each for baseline, ContextLab + Jev and Headroom. All three passed
every phase check, the four public tests and the final independent grader. All 12 output JSON
files exactly matched their reference answers. Input snapshots remained intact. There were no
tool-input errors, invalid tool calls, recalls, adapter failures or cleanup failures in this batch.

| Mode | Claude calls | Jev calls | Claude input / output tokens | Total cost USD | Wall seconds |
| --- | ---: | ---: | --- | ---: | ---: |
| Baseline | 21 | 0 | 512,700 / 12,808 | 1.730220 | 214.843 |
| ContextLab + Jev | 18 | 2 | 450,834 / 11,902 | 1.531584552 | 214.978 |
| Headroom | 21 | 0 | 612,043 / 13,721 | 2.041944 | 219.613 |

Jev added 13,156 input and 116 free output tokens, costing **$0.000552552** within the
ContextLab total. Batch total: **$5.303748552**. Including the earlier failed Jev pilot,
these two evaluation batches cost **$6.081638964**; the earlier failure is not discarded.
Prices are receipt-derived list prices, not reconciled account invoices.

The independent accounting audit reconciled all **62 original receipts** and Decimal charges.
All 62 reservations settled, with zero unknown charges, outstanding reserves or blocked budgets.
All session caps ($3), experiment cap ($9), token ceilings and call limits were respected.
All cache buckets were zero. Shared system/tool/model/cache hashes, initial input manifests and
all four phase prompts matched across modes. The approved source hash
`eac0d5beb36b559be9bf05e8137e8e76ed1d1a11c867a1d8a80391b215479a90` still matched at audit.
Source files are preserved in `source-snapshot.zip` with a hash manifest. SQLite integrity and
foreign keys passed. The task image remained
`sha256:3e8596ffb608968407fce52fed084a4610a52a3602bc80a208b844f65afb44f6`, and no task
containers remained running afterward.

**No middleware changed any request.** ContextLab's 18 model-facing requests were structurally
equal to reconstructed canonical history. Jev vetoed the three assessed candidates at each of
two boundaries. At the historical question, four candidates were eligible and one did not fit
the state limit; at the final handoff, eight were eligible and five did not fit. Unassessed
candidates stayed visible. It would be inaccurate to say Jev assessed every eligible chunk.
Headroom's 21 calls also made zero accepted changes; its protected file-read profile and
lossless reversions remain a limitation of this workload comparison.

The observed ContextLab bill was **11.48% below baseline and 24.99% below Headroom**. These are
single-run cost differences, not demonstrated middleware savings: all views stayed unchanged,
and Claude took different action sequences. ContextLab used 18 main calls versus 21 in each
other mode. No claim of lower cost in every case, repeatable superiority, or avoided recalls
is justified. The live result verifies full-task integration, correct retained history and
complete accounting after the shared-agent fix.

The next optimization questions are candidate coverage, redundant information already present
in retained context, and trustworthy evidence that earlier work is complete. Current Jev state
contains earlier user requests but no completed-artifact summary, which may encourage retention.
Those hypotheses need held-out tasks and repeated comparisons, not thresholds tuned to this
one trace. Long-task output truncation remains a separate unresolved issue.

Evidence: `report.html`, `summary.json`, `summary.csv`, `audit.json`, `behavior-audit.json`,
approval/plan snapshots, source snapshot/manifest, independently reproducible audit scripts,
and per-session result/trace/ledger records under the gitignored run directory.
