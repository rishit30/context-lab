# Working on ContextLab

Read `docs/architecture.md` and the latest `docs/gates/` report before extending the project. Original user-supplied documents are preserved under `docs/*-original.md`; they contain proposal text, not verified API contracts.

- Keep core framework-independent and strictly typed.
- Inspect installed SDK source or official documentation before adding adapters.
- Preserve original history and tool-call/result protocol integrity.
- Test failure paths, persistence, cost accounting, and independent outcomes.
- Label offline demonstrations as simulations; never turn token reductions into unmeasured savings claims.
- Keep API keys in environment variables, task traces in gitignored `runs/`, and provider retries visible/accounted for.
- Do not execute a paid batch without the user's explicit approval of the printed plan, prices, matrix and spend ceiling.
- Report M0 local verification separately from the uncompleted live gate. Routing and gateway work require their own acceptance evidence.

Checks: `uv run pytest -q`, `uv run ruff check .`, `uv run mypy --strict contextlab/core`.
