from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moneygone.exchange.types import Action, Order, OrderStatus, Side
from moneygone.risk.capital_governor import CapitalGovernor


def _order(
    *,
    order_id: str,
    ticker: str = "KXTEST",
    side: Side = Side.YES,
    remaining_count: int = 4,
    price: str = "0.40",
    no_price: str = "0.60",
    status: OrderStatus = OrderStatus.RESTING,
) -> Order:
    return Order(
        order_id=order_id,
        ticker=ticker,
        side=side,
        action=Action.BUY,
        status=status,
        count=remaining_count,
        remaining_count=remaining_count,
        price=Decimal(price),
        no_price=Decimal(no_price),
        taker_fees=Decimal("0"),
        maker_fees=Decimal("0"),
        created_time=datetime(2026, 4, 11, tzinfo=timezone.utc),
    )


def test_sync_open_orders_rebuilds_reserved_capital() -> None:
    governor = CapitalGovernor()
    governor.reserve_intent(
        "intent-1",
        owner="engine",
        ticker="KXINTENT",
        category="sports",
        contracts=1,
        price=Decimal("0.50"),
        available_cash=Decimal("100"),
    )

    governor.sync_open_orders(
        [
            _order(order_id="order-1", ticker="KXYES", side=Side.YES, remaining_count=5, price="0.40"),
            _order(order_id="order-2", ticker="KXNO", side=Side.NO, remaining_count=2, price="0.70", no_price="0.30"),
        ],
        category_lookup={"KXYES": "sports", "KXNO": "weather"},
    )

    snapshot = governor.snapshot()

    assert snapshot.total_reserved == Decimal("3.10")
    assert snapshot.reserved_by_category == {
        "sports": Decimal("2.50"),
        "weather": Decimal("0.60"),
    }
    assert snapshot.reserved_contracts_by_ticker == {
        "KXINTENT": 1,
        "KXYES": 5,
        "KXNO": 2,
    }


def test_pause_and_resume_updates_snapshot() -> None:
    governor = CapitalGovernor()

    governor.pause_trading("closer_kill_switch", "loss streak")
    paused = governor.snapshot()
    governor.resume_trading("closer_kill_switch")
    resumed = governor.snapshot()

    assert paused.paused is True
    assert paused.pause_reasons == {"closer_kill_switch": "loss streak"}
    assert resumed.paused is False
    assert resumed.pause_reasons == {}
