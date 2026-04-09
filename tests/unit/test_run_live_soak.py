"""Tests for conservative paper-soak boot behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from moneygone.config import load_config


def _load_run_live_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "run_live.py"
    spec = importlib.util.spec_from_file_location("run_live", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("run_live", None)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_paper_soak_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_run_live_module()
    monkeypatch.setattr(sys, "argv", ["run_live.py"])

    args = module.parse_args()

    assert args.overlay == "config/paper-soak.yaml"


def test_paper_soak_overlay_is_conservative() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(
        base_path=repo_root / "config" / "default.yaml",
        overlay_path=repo_root / "config" / "paper-soak.yaml",
    )

    assert config.exchange.demo_mode is True
    assert config.weather.enabled is False
    assert config.crypto.enabled is False
    assert config.sportsbook.enabled is True
    assert config.sportsbook.leagues == ["nba"]
    assert config.sportsbook.bookmakers == ["pinnacle"]
    assert config.sportsbook.markets == ["h2h"]
    assert config.sportsbook.fetch_interval_minutes >= 15  # Must poll often enough for line movement
    assert config.sportsbook.lookahead_hours == 12
    assert config.sportsbook.min_requests_remaining >= 250
    assert config.execution.evaluation_interval_seconds >= 15.0
    assert config.risk.kelly_fraction <= 0.05


def test_validate_paper_soak_config_rejects_broader_scope() -> None:
    module = _load_run_live_module()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(
        base_path=repo_root / "config" / "default.yaml",
        overlay_path=repo_root / "config" / "paper-soak.yaml",
    )

    config.sportsbook.fetch_interval_minutes = 5

    with pytest.raises(ValueError, match="too aggressive"):
        module._validate_paper_soak_config(config)
