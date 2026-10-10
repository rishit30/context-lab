# Verified integration APIs

Checked on 2026-10-10. Installed sources take precedence over floating documentation examples.
No provider completion or token-count request was made during the initial implementation. Automated
tests use local objects and fake model responses. A subsequent approved pilot completed 103 successful
model calls and nine passing sessions; see `docs/gates/M0.md`. This file records inspected SDK contracts,
while the gate distinguishes live validation from compaction effectiveness, which remains unproven.

## Installed and used

| Distribution | Installed version | Verified use |
|---|---|---|
| `langchain` | 1.4.4 | `create_agent`, `AgentMiddleware`, `ModelRequest.override`, `ModelResponse.result`, `ContextEditingMiddleware`, `ClearToolUsesEdit` |
| `langchain-core` | 1.6.9 | Typed messages, `BaseTool`, `@tool`, fake model for offline graph tests |
| `langchain-anthropic` | 1.7.5 | `ChatAnthropic`, `AnthropicPromptCachingMiddleware`, usage conversion and token counting source |
| `anthropic` | 1.13.0 | Transitive provider client; model calls are made through LangChain |

Versions are pinned in `pyproject.toml`; the environment lock records transitive dependencies.

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

## Evaluated but not installed as experiment arms

### Headroom

The upstream project metadata identifies `headroom-ai` and Apache-2.0. Current main metadata
advertises version 0.40.0 and its integration guide documents `HeadroomChatModel` wrapping a
LangChain model. This is a real competitor with compression and recoverability; it is not merely
a threshold-clearing implementation.

Its current core dependency tree includes LiteLLM and a native AST utility; the LangChain extra
also adds another provider integration. We did not install or validate that wrapper against the
pinned Anthropic stack. `build_agent(arm='headroom')` raises a clear unavailable error. No performance
claim against Headroom is justified by this implementation. Add a pinned optional dependency and
verify bind-tools, Anthropic cache controls, usage propagation, recall, and request fidelity before
enabling a genuine arm.

Sources: [upstream metadata](https://github.com/headroomlabs-ai/headroom/blob/main/pyproject.toml),
[integration guide](https://github.com/headroomlabs-ai/headroom/blob/main/wiki/integration-guide.md).

### Deep Agents filesystem

Official docs expose `FilesystemBackend(root_dir=..., virtual_mode=True)` and partial
`read(file_path, offset, limit)`, along with composite backends. A filesystem backend alone is not
a process sandbox, and the default non-virtual mode does not confine file access. ContextLab uses
the architecture's permitted minimal local equivalent: session-scoped immutable SQLite originals,
read-only `/context/<chunk_id>.txt` access, bounded line/character continuation and durable recall
records. It does not claim to implement or benchmark Deep Agents' eviction middleware.

Source: [Deep Agents backends](https://docs.langchain.com/oss/python/deepagents/backends).

## Intentional limitations and verification still needed

- The first live pilot verified model access, real cache usage and usage-derived cost accounting.
  No active compaction occurred, so cost savings and post-compaction quality remain unverified.
  Provider invoice reconciliation is still outstanding.
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
