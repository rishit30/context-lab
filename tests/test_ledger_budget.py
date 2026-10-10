import sqlite3
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from contextlab.ledger import BudgetExceeded, BudgetManager, LedgerConflict, LedgerStore, ModelCall


def session(store, name):
    store.create_session(name, "baseline", "task")


def test_both_caps_include_outstanding_reservations(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        session(store, "s1")
        session(store, "s2")
        budget = BudgetManager(store, "1", "1.5")
        budget.reserve("s1", "c1", "model", "main", "0.8")
        with pytest.raises(BudgetExceeded, match="session"):
            budget.reserve("s1", "c2", "model", "decision", "0.3")
        with pytest.raises(BudgetExceeded, match="experiment"):
            budget.reserve("s2", "c2", "model", "main", "0.8")
        budget.settle("c1", ModelCall(id="c1", session_id="s1", seq=0, model="model", cost_usd=Decimal("0.1")))
        budget.reserve("s2", "c2", "model", "main", "0.8")
        assert budget.status("s2")["experiment"]["remaining_usd"] == Decimal("0.6")


def test_failed_and_unknown_usage_charged_and_stop(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        session(store, "s1")
        session(store, "s2")
        budget = BudgetManager(store, "2", "3")
        budget.reserve("s1", "known_failure", "model", "delegate", "0.5")
        failed = ModelCall(id="known_failure", session_id="s1", seq=0, model="model", purpose="delegate",
                           cost_usd=Decimal("0.3"), status="failed")
        budget.settle("known_failure", failed)
        budget.reserve("s1", "unknown", "model", "main", "0.5")
        actual = budget.settle("unknown")
        assert actual.cost_usd == Decimal("0.5")
        assert actual.cost_basis == "reservation"
        assert budget.settle("unknown") == actual
        assert store.session_summary("s1")["cost_usd"] == Decimal("0.8")
        with pytest.raises(BudgetExceeded, match="blocked"):
            budget.reserve("s2", "later", "model", "main", "0.1")


def test_reservation_restart_and_duplicate_dispatch_protection(tmp_path):
    path = tmp_path / "db"
    with LedgerStore(path) as store:
        session(store, "s1")
        budget = BudgetManager(store, "1", "1")
        budget.reserve("s1", "c1", "model", "main", "0.8")
    with LedgerStore(path) as store:
        budget = BudgetManager(store, "1", "1")
        with pytest.raises(BudgetExceeded):
            budget.reserve("s1", "c2", "model", "main", "0.3")
        with pytest.raises(LedgerConflict):
            budget.reserve("s1", "c1", "model", "main", "0.8")
        budget.cancel_before_send("c1")
        budget.reserve("s1", "c2", "model", "main", "0.3")


def test_concurrent_reservations_cannot_overspend(tmp_path):
    path = tmp_path / "db"
    with LedgerStore(path) as store:
        session(store, "s1")
    def attempt(number):
        with LedgerStore(path) as store:
            budget = BudgetManager(store, "1", "1")
            try:
                budget.reserve("s1", str(number), "model", "main", "0.6")
                return True
            except BudgetExceeded:
                return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(attempt, (1, 2))) == 1


def test_underestimated_bound_still_records_actual_and_blocks(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        session(store, "s1")
        budget = BudgetManager(store, "1", "1")
        budget.reserve("s1", "c1", "model", "main", "0.4")
        budget.settle("c1", ModelCall(id="c1", session_id="s1", seq=0, model="model", cost_usd=Decimal("0.6")))
        assert budget.status("s1")["session"]["spent_usd"] == Decimal("0.6")
        with pytest.raises(BudgetExceeded):
            budget.reserve("s1", "c2", "model", "main", "0.1")


def test_settlement_write_failure_keeps_reservation_and_rolls_back_charge(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        session(store, "s1")
        budget = BudgetManager(store, "1", "1")
        budget.reserve("s1", "c1", "model", "main", "0.5")
        store.connection.executescript("""
            CREATE TRIGGER fail_settle BEFORE UPDATE OF status ON reservations BEGIN
            SELECT RAISE(ABORT, 'injected settlement failure'); END;
        """)
        with pytest.raises(sqlite3.IntegrityError):
            budget.settle("c1", ModelCall(id="c1", session_id="s1", seq=0, model="model", cost_usd=Decimal("0.2")))
        status = budget.status("s1")["session"]
        assert status["spent_usd"] == Decimal("0")
        assert status["reserved_usd"] == Decimal("0.5")
