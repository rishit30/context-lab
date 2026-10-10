"""Bind historical snapshot recall to one session; never accept host paths."""
from __future__ import annotations

import json
from collections.abc import Callable

from contextlab.ledger.store import LedgerStore


def make_recall(store: LedgerStore, session_id: str, get_seq: Callable[[], int]):
    """Return a plain callable suitable for wrapping as a framework tool.

    offset/limit use zero-based original line indices. A long single line may be
    continued with the returned next_offset and next_char_offset values.
    """
    def recall(chunk_id: str, offset: int = 0, limit: int = 80, char_offset: int = 0) -> str:
        """Read a historical tool output in this session, at most 200 lines/12K chars."""
        result = store.recall(session_id, chunk_id, get_seq(), offset=offset, limit=limit,
                              char_offset=char_offset)
        return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    return recall
