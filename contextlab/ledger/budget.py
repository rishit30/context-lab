"""Atomic reservation-based spend caps shared by all model-call purposes."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from contextlab.core.models import Record
from contextlab.ledger.pricing import money
from contextlab.ledger.store import LedgerConflict, LedgerStore, ModelCall, now


class BudgetExceeded(RuntimeError):
    pass


class Reservation(Record):
    call_id: str
    session_id: str
    amount_usd: Decimal
    model: str
    purpose: str
    seq: int
    status: str = "reserved"


class BudgetManager:
    """Reservations survive restarts. Unknown provider usage is not treated as free.

    Limits are fixed for this experiment. Every call, including a retry, receives
    a unique call_id and reservation before dispatch. The caller must bound input
    and output so its reservation is a true upper bound.
    """
    def __init__(self, store: LedgerStore, session_limit_usd: Any,
                 experiment_limit_usd: Any, experiment_id: str = "default"):
        self.store = store
        self.session_limit = money(session_limit_usd)
        self.experiment_limit = money(experiment_limit_usd)
        self.experiment_id = experiment_id
        with store.transaction() as db:
            self._configure(db, "experiment", experiment_id, self.experiment_limit)

    @staticmethod
    def _configure(db: Any, scope: str, scope_id: str, cap: Decimal) -> None:
        row = db.execute("SELECT cap_usd FROM budgets WHERE scope=? AND scope_id=?", (scope, scope_id)).fetchone()
        if row and money(row[0]) != cap:
            raise LedgerConflict("A persisted spend cap cannot be changed implicitly")
        db.execute("INSERT OR IGNORE INTO budgets(scope,scope_id,cap_usd) VALUES(?,?,?)", (scope, scope_id, str(cap)))

    @staticmethod
    def _spent(db: Any, scope: str, scope_id: str) -> Decimal:
        if scope == "session":
            rows = db.execute("SELECT cost_usd FROM model_calls WHERE session_id=?", (scope_id,))
        else:
            rows = db.execute("SELECT m.cost_usd FROM model_calls m JOIN sessions s ON s.id=m.session_id WHERE s.experiment_id=?", (scope_id,))
        return sum((money(row[0]) for row in rows), Decimal("0"))

    @staticmethod
    def _reserved(db: Any, scope: str, scope_id: str) -> Decimal:
        column = "session_id" if scope == "session" else "experiment_id"
        rows = db.execute(f"SELECT amount_usd FROM reservations WHERE {column}=? AND status='reserved'", (scope_id,))
        return sum((money(row[0]) for row in rows), Decimal("0"))

    def reserve(self, session_id: str, call_id: str, model: str, purpose: str,
                amount_usd: Any, seq: int = 0) -> Reservation:
        amount = money(amount_usd)
        if purpose not in {"main", "delegate", "decision", "triage", "summary"}:
            raise ValueError("Unknown model-call purpose")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ValueError("seq must be a nonnegative integer")
        with self.store.transaction() as db:
            session = db.execute("SELECT experiment_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not session or session[0] != self.experiment_id:
                raise ValueError("Session does not belong to this budget experiment")
            if db.execute("SELECT 1 FROM reservations WHERE call_id=?", (call_id,)).fetchone():
                raise LedgerConflict("Call already reserved; do not dispatch it again")
            if db.execute("SELECT 1 FROM model_calls WHERE id=?", (call_id,)).fetchone():
                raise LedgerConflict("Call already charged")
            self._configure(db, "session", session_id, self.session_limit)
            for scope, scope_id in (("session", session_id), ("experiment", self.experiment_id)):
                row = db.execute("SELECT cap_usd,blocked FROM budgets WHERE scope=? AND scope_id=?", (scope, scope_id)).fetchone()
                if row[1]:
                    raise BudgetExceeded(f"{scope} is blocked after uncertain usage or an underestimated reservation")
                projected = self._spent(db, scope, scope_id) + self._reserved(db, scope, scope_id) + amount
                if projected > money(row[0]):
                    raise BudgetExceeded(f"{scope} spend cap cannot cover this worst-case reservation")
            db.execute("INSERT INTO reservations(call_id,session_id,experiment_id,model,purpose,seq,amount_usd,created_at) VALUES(?,?,?,?,?,?,?,?)",
                       (call_id, session_id, self.experiment_id, model, purpose, seq, str(amount), now()))
        return Reservation(call_id=call_id, session_id=session_id, amount_usd=amount, model=model, purpose=purpose, seq=seq)

    def settle(self, call_id: str, call: ModelCall | None = None,
               error: str = "Provider usage unavailable; reservation charged conservatively") -> ModelCall:
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM reservations WHERE call_id=?", (call_id,)).fetchone()
            if row is None or row["experiment_id"] != self.experiment_id:
                raise KeyError("Unknown reservation in this experiment")
            if row["status"] == "settled":
                saved = db.execute("SELECT record_json FROM model_calls WHERE id=?", (call_id,)).fetchone()
                previous = ModelCall.model_validate_json(saved[0])
                if call is not None and call != previous:
                    raise LedgerConflict("Reservation already settled with different usage")
                return previous
            if row["status"] != "reserved":
                raise LedgerConflict("Cancelled reservation cannot be settled")
            amount = money(row["amount_usd"])
            if call is None:
                call = ModelCall(id=call_id, session_id=row["session_id"], seq=row["seq"],
                                 model=row["model"], purpose=row["purpose"], cost_usd=amount,
                                 status="unknown_usage", cost_basis="reservation", error=error)
            for field in ("session_id", "seq", "model", "purpose"):
                if getattr(call, field) != row[field]:
                    raise ValueError("Actual call does not match its reservation")
            if call.id != call_id:
                raise ValueError("Actual call id does not match its reservation")
            self.store._insert_call(db, call)
            db.execute("UPDATE reservations SET status='settled',settled_at=? WHERE call_id=?", (now(), call_id))
            if call.status == "unknown_usage" or call.cost_usd > amount:
                # Stop additional spend until an operator reconciles the uncertainty.
                db.execute("UPDATE budgets SET blocked=1 WHERE (scope='session' AND scope_id=?) OR (scope='experiment' AND scope_id=?)",
                           (row["session_id"], self.experiment_id))
            return call

    def cancel_before_send(self, call_id: str) -> None:
        """Release only when the caller knows no request reached the provider."""
        with self.store.transaction() as db:
            cursor = db.execute("UPDATE reservations SET status='cancelled',settled_at=? WHERE call_id=? AND experiment_id=? AND status='reserved'",
                                (now(), call_id, self.experiment_id))
            if cursor.rowcount != 1:
                raise LedgerConflict("No active reservation to cancel")

    def status(self, session_id: str) -> dict[str, Any]:
        with self.store.transaction() as db:
            result: dict[str, Any] = {}
            for scope, scope_id in (("session", session_id), ("experiment", self.experiment_id)):
                row = db.execute("SELECT cap_usd,blocked FROM budgets WHERE scope=? AND scope_id=?", (scope, scope_id)).fetchone()
                if row:
                    spent = self._spent(db, scope, scope_id)
                    reserved = self._reserved(db, scope, scope_id)
                    result[scope] = {"cap_usd": money(row[0]), "spent_usd": spent,
                                     "reserved_usd": reserved, "remaining_usd": money(row[0]) - spent - reserved,
                                     "blocked": bool(row[1])}
            return result
