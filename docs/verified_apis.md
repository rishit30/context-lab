# Verified integration APIs

Checked on 2026-10-10. Installed sources take precedence over floating documentation examples.
No provider completion or token-count request was made during the initial implementation. Automated
tests use local objects and fake model responses. A subsequent approved pilot completed 103 successful
model calls and nine passing sessions; see `docs/gates/M0.md`. This file records inspected SDK contracts,
while the gates distinguish integration from effectiveness. A second approved 85-call comparison
verified active compaction and correct retrieval on the short file-audit task; general efficiency
remains unproven. See `docs/gates/M1-comparison.md`.

## Installed and used

| Distribution | Installed version | Verified use |
|---|---|---|
| `langchain` | 1.4.4 | `create_agent`, `AgentMiddleware`, `ModelRequest.override`, `ModelResponse.result`, `ContextEditingMiddleware`, `ClearToolUsesEdit` |
| `langchain-core` | 1.6.9 | Typed messages, `BaseTool`, `@tool`, fake model for offline graph tests |
| `langchain-anthropic` | 1.7.5 | `ChatAnthropic`, `AnthropicPromptCachingMiddleware`, usage conversion and token counting source |
| `anthropic` | 1.13.0 | Transitive provider client; model calls are made through LangChain |
| `headroom-ai` (optional) | 0.40.0 | Public `compress` / `CompressConfig`, verified lossless agent profile, offline tokenizer registration |

Versions are pinned in `pyproject.toml`; the environment lock records transitive dependencies.

### TypeSafe Jev decision service

Verified on 2026-10-10 against the official [HTTP contract](https://docs.typesafe.ai/api),
[OpenAPI schema](https://api.typesafe.ai/openapi.json), and [model reference](https://docs.typesafe.ai/models).
The optional adapter uses Python's standard HTTPS client directly, with no redirects or retries;
no additional SDK dependency is required. Authenticated `GET https://api.typesafe.ai/v1/models`
succeeded and listed `jev-latest` and `jev-preview`. Subsequently, the explicitly approved
pilot `runs/jev-pilot-001/` completed one real inference with pinned `jev-1.13.0`: 6,486 input
tokens, 58 zero-priced output tokens, $0.000272412. The full agent task failed later because
Claude invented invalid recall IDs. After the shared agent fix, an approved three-arm comparison
in `runs/jev-validation-001/` passed all arms. ContextLab made two valid Jev requests with 13,156
input tokens and 116 zero-priced output tokens, costing $0.000552552. Both assessments retained
all evaluated candidates; no context was removed. API compatibility and successful full-task
integration are verified; optimization benefit remains unproven.

Inference is `POST https://api.typesafe.ai/v1/systemone`, with bearer `TYPESAFE_API_KEY` and
JSON `{model, state, questions}`. The configured version is `jev-1.13.0`; the official reference
states that versioned IDs work even when model listing shows only aliases. Each candidate has
a `noul` question with explicit instructions and true/false criteria. Its answer is a probability
in `[0,1]`, not a separate confidence field. The adapter requires the pinned response model,
exact question IDs, finite probabilities, and nonnegative integer input/output usage.

Price: $0.042/M billable input tokens; output is free. Shared state is billed once for a batch.
Published limits are 64k total request tokens and 32k for state plus the longest question.
The provider exposes no verified count endpoint or tokenizer. The local gate uses complete
candidate text, a 24,000-byte state limit, at most eight questions and a 32,000-byte JSON body
limit. It reserves 65,536 tokens per request against the existing experiment cap. These are
conservative admission/reservation policies, not measured token counts. Real receipts settle
actual fees; unknown usage retains the reservation charge and blocks the experiment.

Jev requests include current/prior user instructions and selected tool results. Full request
and receipt records remain in the local gitignored ledger for audit. Keys and provider error
bodies are excluded. Probability thresholds have not been calibrated on held-out tasks.
This addition does not retroactively change any result from the earlier rules-only pilots.

Other direct dependency pins verified from the installed environment: Pydantic 2.14.0
(record/config validation), PyYAML 6.0.3 (safe configuration loading), Jinja2 3.1.6
(autoescaped static reports), pytest 9.1.1 (local tests), Ruff 0.17.0 (lint), mypy 2.4.0
(strict core checks), and types-PyYAML 6.0.12.20260906. The build backend is pinned
to setuptools 80.9.0. `uv.lock` and `requirements.lock` capture the full resolved environment.

### LangChain middleware

Inspected `.venv/lib/python3.13/site-packages/langchain/agents/middleware/types.py`
and the installed `create_agent` signature using Python `inspect`.

The synchronous hook is `wrap_model_call(request: ModelRequest, handler)`. It returns a
`ModelResponse`, whose `result` holds messages. A new request is produced with
`request.override(messages=...)`. The first middleware in the list is outermost.
This release deliberately supports synchronous `invoke`; async/streaming adapters are deferred.

ContextLab observes canonical messages at model boundaries, instead of assuming a tool-hook result
has a final stable message ID. It archives the original tool text verbatim, and copies only the
`ToolMessage.content` field when applying a validated projection. Tool IDs, artifacts, assistant
content, signed thinking blocks, and all other provider metadata are preserved. The normalized
Anthropic-shaped view is for validation and policy evaluation, not a replacement provider payload.

Reference: [custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom).

### Common baseline and threshold editing

Installed signatures verified:

```python
ContextEditingMiddleware(edits=[ClearToolUsesEdit(
    trigger=12000, keep=3, clear_tool_inputs=False, exclude_tools=["recall"]
)])
```

The actual threshold arm uses that library implementation. It is named `threshold_edit`, never
`headroom` or `deepagents_evict`. Threshold and keep count are experiment configuration, not a
claim that these are LangChain defaults. Every arm has identical tool schemas, instructions,
model configuration and cache settings; tests assert schema bytes and cache tags. An optional
observe-only ContextLab middleware records canonical events for comparison without rewriting them.

Reference: [built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in).

### Anthropic prompt caching

Inspected `langchain_anthropic/middleware/prompt_caching.py`, specifically
`AnthropicPromptCachingMiddleware`, `_tag_tools`, and `_tag_system_message`.
The verified middleware tags the last system content block and last tool definition, and sets
`model_settings['cache_control']={'type':'ephemeral','ttl':'5m'}`. The chat model forwards this
automatic caching setting for the direct API transport. These settings are identical in all arms.

This MVP does **not** add a separately managed explicit epoch breakpoint. It preserves stable
rendering between policy transitions and uses the provider integration's automatic trailing caching.
Cache eligibility, prefix matches, minimum token requirements, expiration, and actual hit rate still
depend on the provider. A stable prefix is not proof of a cache hit. The policy's warm/cold input is
a clock-based forecast since the preceding adapter call, not observed provider cache state; it must
not be reported as measured savings. Restart resets this forecast to cold.

Reference: [prompt-caching middleware API](https://reference.langchain.com/python/langchain-anthropic/middleware/prompt_caching/AnthropicPromptCachingMiddleware).

### Usage and failed calls

Inspected `_create_usage_metadata` and response construction in
`langchain_anthropic/chat_models.py`. Anthropic raw `input_tokens` excludes cached input;
LangChain `AIMessage.usage_metadata['input_tokens']` includes it. ContextLab subtracts the
`cache_read` and cache-write details from the LangChain total to recover uncached input.
Specific `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens` replace the aggregate
`cache_creation` bucket; they must not be charged twice. Output reasoning tokens are already
within output tokens, not an extra billable bucket.

`MeteringMiddleware` invokes its callback once per attempted handler call, including provider
failures. Unknown or missing usage is `None`, not zero. The runner retains its conservative cost
reservation when usage is unknown. The shared agent rejects Anthropic clients with SDK retries
enabled; hidden retries would otherwise escape per-invocation accounting. Adapter failures never
retry a provider request. API-key values and exception messages are not sent to usage callbacks.
The sixth callback argument retains only the original LangChain usage metadata and the provider's
usage dictionary, when present, so normalization can be audited without storing full response metadata.

### Token counting

Verified `ChatAnthropic.get_num_tokens_from_messages(messages, tools=None, **kwargs)` in installed
source. It calls Anthropic's remote `messages.count_tokens`, formats tool definitions, and includes
the system message when passed in the message list. It is **not an offline tokenizer**. The live
runner can use it only as part of a separately authorized API execution. Offline context estimates
are byte heuristics and are never substituted for measured token usage in result claims.

Reference: [Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting).

### Headroom

The optional `headroom` extra pins `headroom-ai==0.40.0`. The adapter calls the genuine public
`headroom.compress` API before shared LangChain metering; it does not replace the provider model.
Its explicit `agent_lossless_offline` profile uses `CompressConfig(mode='agent',
kompress_model='disabled', verify_lossless=True)`, retaining upstream tool protections.
System/user compression and CCR are disabled. No auxiliary inference occurs; Headroom recall is
not applicable to this profile. `configuration()` records every option in the approved plan.

Installed source and socket-blocked tests verified: `ModelRequest.override` carries only copied
tool-result text; canonical history, tool IDs, opaque blocks, system prompt and tool schemas remain
unchanged; shared metering receives the transformed request and original provider usage. Projection
errors forward the original request once and are recorded. Environment controls disable telemetry,
update checks, ML downloads and persistent CCR/TOIN stores. The official estimating tokenizer is
registered explicitly to avoid vocabulary downloads; its counts are estimates, never billing data.

An offline smoke test compressed structured `run_tests` JSON from 2,467 to 352 estimated tokens,
but genuine `Workspace.read_file` envelopes remained unchanged. Upstream 0.40 defaults protect all
`read_file` results (`protect_recent_reads_fraction=0.0`), including older reads. Tests exercise that
behavior rather than renaming tools or weakening upstream protection to produce favorable numbers.
This benchmark evaluates this exact profile on file audits, not every Headroom capability. Neither
the smoke test nor integration tests establish provider cost savings.

The cache-off comparison's prepared ChatAnthropic payload has zero `cache_control` fields,
`context_management=None`, and empty extra model kwargs. Installed LangChain composition verifies
projection → optional caching → metering → model. Usage conversion preserves all cache buckets
without double counting. These are local SDK checks, not new provider calls.

Sources: [upstream metadata](https://github.com/headroomlabs-ai/headroom/blob/main/pyproject.toml),
[integration guide](https://github.com/headroomlabs-ai/headroom/blob/main/wiki/integration-guide.md).

### Read-only routing contracts inspected in this revision

The installed `langchain_core/tools/base.py` exposes `InjectedToolCallId`. It is
excluded from the model-facing tool schema and populated from the enclosing
tool-call envelope. ContextLab uses that ID for durable delegation idempotency.
The same installed ChatAnthropic token-count and receipt contracts described
above apply to the bounded worker, with retries disabled and no tools bound.

Worker compatibility example pricing was checked against the official
[Haiku 4.5 reference](https://platform.claude.com/docs/en/models/haiku-4-5/overview)
on 2026-10-10: pinned `claude-haiku-4-5-20251001`, $1/M input and $5/M output,
$1.25/M five-minute cache writes, $2/M one-hour writes and $0.10/M reads. This is
a legacy model; availability must be rechecked before a live plan is approved.
The configuration is never selected automatically. The main model retains its
separate explicit pricing snapshot.

Routing optionally reuses the verified TypeSafe HTTP transport for a distinct
bounded suitability question. Public model documentation was rechecked, but no
authenticated requests, token counts or inference were made for this revision.
No live routing quality or savings acceptance is implied by SDK inspection.

## Deferred integrations

### Deep Agents filesystem

Official docs expose `FilesystemBackend(root_dir=..., virtual_mode=True)` and partial
`read(file_path, offset, limit)`, along with composite backends. A filesystem backend alone is not
a process sandbox, and the default non-virtual mode does not confine file access. ContextLab uses
the architecture's permitted minimal local equivalent: session-scoped immutable SQLite originals,
read-only `/context/<chunk_id>.txt` access, bounded line/character continuation and durable recall
records. It does not claim to implement or benchmark Deep Agents' eviction middleware.

Source: [Deep Agents backends](https://docs.langchain.com/oss/python/deepagents/backends).

## Intentional limitations and verification still needed

- The first pilot verified model access, caching and cost accounting. The second comparison verified
  active ContextLab compaction and correct short-task output after recall, but all long tasks failed.
  Repeatable savings, broad post-compaction quality and invoice reconciliation remain outstanding.
- System/tool cache formatting is checked offline; live receipts confirmed cache use for the selected
  model and account. Availability and cache behavior for other configurations remain unverified.
- Archive and projection state survive adapter restart when canonical message history is supplied
  again. Resuming the entire agent process needs a separate graph checkpointing integration.
- Workspace path checks reject traversal, reserved paths and symlinks. They do not sandbox generated
  Python. The default workspace refuses test execution without an injected isolated executor.
- The `shell` compatibility tool accepts only `list_files` and `workspace_status`; it never starts
  a shell. Literal search uses a fixed `rg` argument vector, with a local fallback when unavailable.
- Jev, LLM policy calls, delegation, a gateway, async execution, and streaming are not implemented
  in this foundation. They must not appear as working features in a demo.
