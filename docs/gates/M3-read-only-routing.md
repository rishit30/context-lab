# M3 read-only delegation: offline acceptance — 2026-10-10

## Implemented scope

The public Python session supports opt-in evidence-selection delegation while
retaining a single main model. Immutable source snapshots form complete bounded
briefs. A worker has no tools, shell, filesystem or write capability. Optional
Jev suitability decisions can veto a request; deterministic policy retains budget,
permission and validation authority.

Worker responses must satisfy strict JSON, known question/source IDs, exact line
ranges and verbatim quotations. Failed validation returns the complete originals
to the existing main agent. This proves provenance and question-ID coverage,
not semantic relevance, completeness or task quality.

The runtime reserves and settles worker/decision calls in the same ledger as
main calls. Known failed work is charged from usage; unknown usage retains
conservative exposure and blocks later dispatch. Stable tool-call IDs prevent
redispatch after restart. Source-catalog changes under an existing session ID
are rejected. Full audit metadata stays in SQLite; model-facing tool output
contains evidence rather than financial/internal control details.

## Validation

**661 tests passed**, with repository Ruff, strict core mypy and whitespace checks
passing. New tests cover brief bounds and nested mutation, exact source hashes,
Unicode/CRLF quotes, duplicate JSON keys, question coverage, known and unknown
provider failures, settlement write failures, call limits, restart idempotency,
Jev veto/fees, receipt reconciliation and comparison-plan invariants.

The five-configuration demo ran real LangChain graphs and local middleware with
explicit deterministic providers: baseline, compaction-only, routing-only,
combined ContextLab and genuine Headroom. The same main model, prompt, complete
tool schemas and frozen files were used throughout. All **15/15 mechanical phase
checks passed** across **34 simulated invocations**. Four were worker invocations:
each routing configuration accepted one exact selection and rejected one corrupt
quote, charged its synthetic usage, and continued with original evidence.
Only compaction configurations modified projected requests. All reservations
settled, and the independent read-only auditor reconciled the five result files,
per-arm price snapshots, caps, failed calls and aggregate costs without errors.

Audits explicitly warn that these are simulations. Application sessions remain
reopenable rather than marking a harness task as independently solved, so the
generic lifecycle audit reports them as unfinished. This is separate from the
successful mechanical checks and absence of outstanding reservations.

The wheel was built offline using the pinned build backend, unpacked outside the
checkout, imported independently and compared byte-for-byte with the core,
runtime, session, ledger, auditor and report sources. A saved-run audit passed
through the wheel. No new dependencies or provider contracts were introduced.

The final `contextlab:routing` Docker image also passed all 15 phases and the
34-call accounting audit with networking disabled, a read-only root filesystem,
UID 10001, all capabilities dropped and no API keys. Core/runtime/session/example
hashes matched the host. Image digest:
`sha256:3cdb18c4acf7906eb3759e72ae9f74ae61d616beaea9ae7e052ffa515e27f82f`.
Temporary validation containers and their output volume were removed after the
reports and ledger were copied into the ignored local evidence directory.

Local artifacts are under `runs/routing-offline-001/`: JUnit results, no-request
live plan, host demo/report, read-only audit, wheel verification and Docker evidence.
They remain gitignored. GitHub CI now exercises the routing plan, offline demo
and saved-result audit; remote CI status belongs to the corresponding PR checks.

## Corrections found during review

- The initial forecast incorrectly credited avoiding main-model output, even
  though the main agent still answers on both paths. It now credits only
  estimated original-source input, excluding worker instructions, and includes
  selected-evidence envelope overhead, decision fees and a failure prior.
- Rebinding the same catalog now preserves the existing router, so an in-memory
  unsettled receipt cannot be lost and replaced by a larger unknown reservation.
- Live `provider_spend_usd` remains unknown rather than presenting list-price
  estimates or conservative reservations as an invoice. Separate fields identify
  unknown usage and reservation exposure; simulations explicitly report zero.
- Even a self-consistent approved plan cannot give comparison arms different
  main models, prices, limits or policies and still claim a controlled ablation.

## Acceptance boundary and next evidence

This is a **partial M3 implementation with passing offline mechanics**. Routing
stays off by default. No new model inference, token counting, key validation or
authenticated provider request was made. New provider spend: **$0**.

The worker cost forecast, byte-to-token estimates, failure prior and Jev threshold
are uncalibrated. Small workloads may correctly bypass delegation. Neither this
demo nor earlier historical runs establishes real cost savings, a win over
Headroom, semantic quality or general task routing. A fresh explicitly approved
live matrix with representative tasks and independent quality checks remains
necessary. Write-capable workers, atomic patch merging, learned policies,
conversation memory and the deferred HTTP gateway remain outside this change.
