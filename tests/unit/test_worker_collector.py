from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_worker_collector_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "worker_collector.py"
    spec = importlib.util.spec_from_file_location("worker_collector", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("worker_collector", None)
    spec.loader.exec_module(module)
    return module


class _FakeOddsFeed:
    def __init__(self) -> None:
        self.requests_remaining = 400
        self.fetch_calls: list[str] = []

    async def get_upcoming_games(self, league: str, *, bookmakers, markets):
        self.fetch_calls.append(league)
        return []

    def build_line_history_rows(self, league: str, games, *, captured_at):
        return []


class _FakeStatsFeed:
    pass


@pytest.mark.asyncio
async def test_collect_once_runs_espn_gate_even_on_first_pass(data_store, tmp_path, monkeypatch):
    module = _load_worker_collector_module()
    checked: list[str] = []

    async def _fake_has_games(stats_feed, league: str, lookahead_hours: int, *, reference_time=None):
        checked.append(league)
        return False

    monkeypatch.setattr(module, "_league_has_games_within_window", _fake_has_games)

    config = SimpleNamespace(
        sportsbook=SimpleNamespace(
            fetch_interval_minutes=30,
            leagues=["nba"],
            bookmakers=["pinnacle"],
            markets=["h2h"],
            lookahead_hours=24,
            min_requests_remaining=250,
        )
    )
    odds_feed = _FakeOddsFeed()

    await module.collect_once(
        config,
        data_store,
        tmp_path,
        odds_feed=odds_feed,
        stats_feed=_FakeStatsFeed(),
    )

    assert checked == ["nba"]
    assert odds_feed.fetch_calls == []
