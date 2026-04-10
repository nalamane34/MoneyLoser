#!/usr/bin/env python3
"""Emergency: cancel ALL resting orders on the live account.

Usage:
    python scripts/emergency_cancel_all.py --config config/default.yaml --overlay config/live.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.exchange.rest_client import KalshiRestClient


async def main() -> None:
    parser = argparse.ArgumentParser(description="Emergency cancel all resting orders")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/live.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Just list orders, don't cancel")
    args = parser.parse_args()

    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )

    print(f"Connecting to: {config.exchange.base_url}")
    print(f"Demo mode: {config.exchange.demo_mode}")

    client = KalshiRestClient(config.exchange)

    try:
        # Get all resting orders
        orders = await client.get_orders(status="resting")
        print(f"\nFound {len(orders)} resting orders")

        if not orders:
            print("No resting orders to cancel.")
            return

        # Group by market ticker for display
        by_ticker: dict[str, list] = {}
        for o in orders:
            ticker = getattr(o, "ticker", "unknown")
            by_ticker.setdefault(ticker, []).append(o)

        print(f"\nOrders across {len(by_ticker)} markets:")
        for ticker, ticker_orders in sorted(by_ticker.items()):
            total_contracts = sum(getattr(o, "remaining_count", 0) for o in ticker_orders)
            print(f"  {ticker}: {len(ticker_orders)} orders, {total_contracts} contracts remaining")

        if args.dry_run:
            print("\n--dry-run specified, not canceling.")
            return

        print(f"\nCanceling all {len(orders)} orders...")

        # Cancel in small batches to avoid rate limits
        cancelled = 0
        failed = 0
        batch_size = 5
        order_ids = [getattr(o, "order_id", None) for o in orders]
        order_ids = [oid for oid in order_ids if oid]

        for i in range(0, len(order_ids), batch_size):
            batch = order_ids[i : i + batch_size]
            for oid in batch:
                try:
                    await client.cancel_order(oid)
                    cancelled += 1
                    print(f"  Cancelled {cancelled}/{len(order_ids)}: {oid}")
                except Exception as e:
                    failed += 1
                    print(f"  FAILED {oid}: {e}")
            # Pace between batches
            if i + batch_size < len(order_ids):
                await asyncio.sleep(1.1)

        print(f"\nDone: {cancelled} cancelled, {failed} failed out of {len(order_ids)} total")

        # Verify
        remaining = await client.get_orders(status="resting")
        print(f"Remaining resting orders: {len(remaining)}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
