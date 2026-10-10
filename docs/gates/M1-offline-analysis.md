# M1 offline context analysis — 2026-10-10

## Built

- `harness/xray.py`: read-only, causal analysis of saved baseline requests. Supports
  shared live ledgers and per-session simulation ledgers, candidate protection,
  separate boundary evidence, conditional price allocations and explicit unknowns.
- `harness/cli.py`: `xray RUN_DIRECTORY [--output NEW_REPORT_DIRECTORY]`.
- `harness/report/xray.py` and `xray.html`: portable HTML/JSON reports that preserve
  the source run, escape labels and omit original prompts/tool contents.
- `harness/runner.py`: all arms record public phase completions against hashed
  conversation prefixes on subsequent calls. Future records cannot validate earlier requests.
- `tests/test_offline_acceptance.py`: six independent mechanical acceptance
  scenarios using real local integration code and an explicitly deterministic test provider.
- `docs/offline-validation.md` and `docs/requirements-status.md`: run instructions
  and the remaining original requirements. CI includes the offline demo-to-X-ray flow.

## Verified vs assumed

No new provider contract or model was introduced. Analysis uses frozen historical
prices and saved provider receipts; these are not assertions about today's rates.
The message-view metric excludes system instructions and tool definitions.
Candidate token counts use a byte heuristic, not a provider tokenizer.

Saved source ledgers, plans and traces were hashed before and after analysis;
all remained unchanged. Independently recomputed input costs matched all 83 real
baseline receipts. The analyzer only considers content visible at each call;
future reads, final mutable pins and private graders are excluded.

The wheel built offline and the analyzer/report imported and executed outside the
checkout. The updated `contextlab:local` software image ran as UID 10001, with
networking disabled, a read-only saved-run mount, and no embedded `.env` or `.git`.
It reconstructed 21 calls and generated HTML/JSON. Its analysis source hash matched
the host and wheel: `9fc753e1782e4e94caec3fda2e3d801aee8460089123c0a451664e5ed005af03`.
The image manifest is `sha256:b23c0db74e03f5f6cee52879cdc55bea7d853304b8493a7e8da2da1ff2a18398`.

## Tests

**496 tests passed**; repository Ruff, strict core mypy and whitespace checks passed.
The new tests cover causal replay, missing receipts/traces, price-bucket allocation,
historical values, failed sessions, unpaired records, malformed/future boundary
markers, source-path containment, unknown report fields and report escaping.

The public-completion tests exercise all three configurations using fake providers
and executors. Private grading failures cannot become public completion evidence.
The fresh offline demonstration completed 9/9 reference-fixture checks; its 15
baseline requests were analyzed through both the API and CLI. No result here is a
real-model outcome or a new live comparison.

## Results

Historical baselines analyzed with current candidate logic and saved policy settings:

| Saved batch | Baseline calls reconstructed | Candidate payload / message-view bytes | Historical input cost |
| --- | ---: | ---: | ---: |
| `live-001` | 35/35 | 6.45% | $0.08120760 |
| `comparison-001` | 27/27 | 26.64% | $2.328867 |
| `jev-validation-001` | 21/21 | 26.75% | $1.538100 |

These percentages are **candidate message shares**, not total-input token shares,
verified waste, or monetary savings. Historical input costs include the saved
input/cache buckets and exclude output charges. Failure attempts remain included.
The reports show conditional price bounds separately.

The comparison batch's short task alone had 32.65% candidate message share. Its
failed long task had zero in only four recorded requests; that early termination
does not characterize a completed long workflow. Baseline boundary coverage is
insufficient for a whole-run boundary-eligible share in all three historical batches,
so that aggregate correctly remains unknown.

In the new paired **simulation**, a nine-phase audit made 24 file reads and 17
model invocations per arm. Both arms produced the exact independently computed
answers. ContextLab changed 13 requests; final tool-result text was 387,378 bytes
versus 769,104 baseline bytes, about a 49.6% reduction. Canonical originals remained
intact and no semantic provider was needed for the identical copies. This validates
duplicate mechanics, not a Claude token, quality or cost result.

Other acceptance cases preserved old/current values, recovered a historical answer
through a real recall call, deferred changes until a trusted boundary, recovered
32,048 bytes over three recall pages with a durable pin, and restored evidence
after a saved duplicate anchor was corrupted.

Local artifacts: `runs/offline-analysis-001/` contains reports, source-input hashes,
JUnit properties and the wheel. These artifacts are ignored by Git and contain no
new provider receipts.

## What surprised us

The short file-audit workload has materially more candidate exposure than the
small coding pilot. However, old file versions are deliberately needed later in
the audit. A superseding read therefore cannot establish that removal is safe.

Most historical baselines have no trustworthy call-time phase-boundary metadata.
The old long-task progression bug makes final `completed` flags especially unsafe
as a substitute. The new telemetry addresses future observability only.

## Failures

Review caught temporal eligibility errors in the first analyzer draft: current
recent-output guards could incorrectly describe an earlier boundary, and a later
reread could mask an earlier qualifying trigger. Boundary candidate computation
now uses the boundary's own prefix before intersecting current originals.
Malformed records, future-labeled markers and unknown receipts have regression tests.

The original M1 six-task repeated live matrix has not run. Richer reference
extraction and additional lifecycle signals remain incomplete. M2 statistics and
learning, M3 routing, M4 triage/learning and M5 memory are not completed by this work.
The HTTP gateway remains deferred. The new CI steps have not yet run on GitHub.

## Cost

New provider spend: **$0**. Saved traces were reused; no paid inference, key
validation or model-list requests were made. Package/image preparation used no
model inference. The user's **“No paid runs now”** instruction remains in effect.

## Go/no-go recommendation

Go for better workload selection and further offline compaction-safety work.
The 26.75% candidate message share in the latest short audit is a useful opportunity
signal, with the explicit historical-evidence caveat above. It is not the original
proposed 10% stale-spend gate and does not establish achievable benefit.

No-go for a savings claim or marking the full compaction/routing acceptance gate
complete. Current real-model quality and cost effects still require representative,
explicitly approved live evaluation. Routing remains disabled.

## Questions for the user

None required for this offline milestone. Paid evaluation remains deferred; no
approval is inferred from continuing implementation.
