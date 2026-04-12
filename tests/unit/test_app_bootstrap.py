"""Smoke tests for live app bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from moneygone.app import build_app
from moneygone.config import AppConfig, CryptoConfig, ExchangeConfig, SportsbookConfig


class _FakeRestClient:
    def __init__(self, config) -> None:
        self.config = config


class _FakeWebSocket:
    def __init__(self, config) -> None:
        self.config = config


def test_build_app_wires_sports_execution_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("moneygone.app.KalshiRestClient", _FakeRestClient)
    monkeypatch.setattr("moneygone.app.KalshiWebSocket", _FakeWebSocket)
    config = AppConfig(
        data_dir=tmp_path,
        exchange=ExchangeConfig(private_key_path=tmp_path / "key.pem"),
        crypto=CryptoConfig(enabled=False),
        sportsbook=SportsbookConfig(
            enabled=True,
            leagues=["nba"],
            bookmakers=["pinnacle"],
            markets=["h2h"],
        ),
    )

    app = build_app(config)

    try:
        assert app.execution_engine is not None
        assert app.pipeline is not None
        assert app.model is not None
        assert app.model.name == "sharp_sportsbook"
        assert app.resolution_sniper is None
        assert app.live_event_edge is None
        assert app.cross_market_arb is None
        assert app.market_maker is None
    finally:
        app.store.close()


def test_build_app_non_sports_bootstrap_does_not_reference_fill_tracker_before_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("moneygone.app.KalshiRestClient", _FakeRestClient)
    monkeypatch.setattr("moneygone.app.KalshiWebSocket", _FakeWebSocket)
    monkeypatch.setattr(
        "moneygone.app._load_latest_model",
        lambda _config: type("Model", (), {"name": "baseline"})(),
    )
    config = AppConfig(
        data_dir=tmp_path,
        exchange=ExchangeConfig(private_key_path=tmp_path / "key.pem"),
        crypto=CryptoConfig(enabled=False),
        sportsbook=SportsbookConfig(enabled=False, leagues=[]),
    )

    app = build_app(config)

    try:
        assert app.execution_engine is None
        assert app.resolution_sniper is not None
        assert app.market_maker is not None
    finally:
        app.store.close()
