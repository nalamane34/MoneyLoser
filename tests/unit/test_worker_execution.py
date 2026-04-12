from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from moneygone.exchange.types import MarketResult, Settlement


def _load_worker_execution_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "worker_execution.py"
    spec = importlib.util.spec_from_file_location("worker_execution", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("worker_execution", None)
    spec.loader.exec_module(module)
    return module


class _FakeStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.views_created = False
        self.attached = False

    def attach_readonly(self, name: str, db_path: Path) -> None:
        assert name in {"collector", "market_data"}
        self.attached = True

    def create_attached_views(self, mapping) -> None:
        assert "collector" in mapping
        self.views_created = True

    def insert_settlements(self, rows):
        self.rows.extend(rows)


class _FakeRiskManager:
    def __init__(self) -> None:
        self.settlements: list[Settlement] = []

    def post_settlement_update(self, settlement: Settlement) -> None:
        self.settlements.append(settlement)


class _FakeRest:
    def __init__(self, settlements: list[Settlement]) -> None:
        self._settlements = settlements
        self.calls: list[dict[str, object]] = []

    async def get_settlements(self, **filters):
        self.calls.append(dict(filters))
        return list(self._settlements)


def _settlement(
    ticker: str,
    result: MarketResult = MarketResult.YES,
    revenue: str = "1.00",
) -> Settlement:
    return Settlement(
        ticker=ticker,
        market_result=result,
        revenue=Decimal(revenue),
        settled_time=datetime(2026, 4, 11, 2, 0, tzinfo=timezone.utc),
    )


def test_attach_collector_views_builds_expected_views(tmp_path: Path) -> None:
    module = _load_worker_execution_module()
    store = _FakeStore()

    attached = module._attach_collector_views(store, tmp_path / "collector.duckdb")

    assert attached is True
    assert store.attached is True
    assert store.views_created is True


def test_attach_market_data_tables_returns_shared_table_refs(tmp_path: Path) -> None:
    module = _load_worker_execution_module()
    store = _FakeStore()

    market_table, orderbook_table = module._attach_market_data_tables(
        store,
        tmp_path / "market_data.duckdb",
    )

    assert store.attached is True
    assert market_table == "market_data.market_states"
    assert orderbook_table == "market_data.orderbook_snapshots"


@pytest.mark.asyncio
async def test_sync_new_settlements_persists_and_updates_risk_once() -> None:
    module = _load_worker_execution_module()
    settlement = _settlement("KXTEST-1")
    store = _FakeStore()
    risk = _FakeRiskManager()
    rest = _FakeRest([settlement])
    seen: set[str] = set()

    first = await module._sync_new_settlements(
        rest_client=rest,
        store=store,
        risk_manager=risk,
        seen_keys=seen,
        limit=50,
    )
    second = await module._sync_new_settlements(
        rest_client=rest,
        store=store,
        risk_manager=risk,
        seen_keys=seen,
        limit=50,
    )

    assert first == 1
    assert second == 0
    assert len(store.rows) == 1
    assert len(risk.settlements) == 1
    assert rest.calls == [{"limit": 50, "paginate": True}, {"limit": 50, "paginate": True}]
