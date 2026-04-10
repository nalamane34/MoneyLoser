"""Configuration loading with YAML files and environment variable overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ExchangeConfig(BaseModel):
    base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    ws_url: str = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    api_key_id: str = ""
    private_key_path: Path = Path("key.pem")
    demo_mode: bool = True
    rate_limit_rps: float = 10.0


class WeatherConfig(BaseModel):
    enabled: bool = True
    noaa_enabled: bool = True
    ecmwf_enabled: bool = True
    locations: list[dict[str, Any]] = Field(default_factory=list)
    fetch_interval_minutes: int = 60


class CryptoConfig(BaseModel):
    enabled: bool = True
    exchanges: list[str] = Field(default_factory=lambda: ["binanceus"])
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    fetch_interval_seconds: int = 30
    coinalyze_api_key: str = ""


class SportsbookConfig(BaseModel):
    enabled: bool = False
    leagues: list[str] = Field(default_factory=lambda: ["nba"])
    bookmakers: list[str] = Field(default_factory=lambda: ["pinnacle"])
    markets: list[str] = Field(default_factory=lambda: ["h2h"])
    fetch_interval_minutes: int = 240
    lookahead_hours: int = 18
    min_requests_remaining: int = 75


class ModelConfig(BaseModel):
    model_dir: Path = Path("models/")
    calibration_method: str = "isotonic"
    ensemble_method: str = "inverse_variance"
    retrain_interval_hours: int = 24


class ExecutionConfig(BaseModel):
    min_edge_threshold: float = 0.02
    max_edge_sanity: float = 0.30
    prefer_maker: bool = True
    max_order_staleness_seconds: int = 60
    evaluation_interval_seconds: float = 5.0
    max_model_market_disagreement: float = 0.08


class RiskConfig(BaseModel):
    kelly_fraction: float = 0.25
    max_position_per_market: int = 100
    max_position_per_category_pct: float = 0.20
    max_total_exposure_pct: float = 0.50
    daily_loss_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    min_contract_price: float = 0.10
    max_contract_price: float = 0.90
    max_tail_exposure_pct: float = 0.05


class BacktestConfig(BaseModel):
    initial_bankroll: float = 10000.0
    fill_model: str = "realistic"
    slippage_bps: float = 0.0


class MonitoringConfig(BaseModel):
    drift_window: int = 100
    psi_threshold: float = 0.2
    ece_threshold: float = 0.05
    alert_webhook_url: str | None = None


class AppConfig(BaseSettings):
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    crypto: CryptoConfig = Field(default_factory=CryptoConfig)
    sportsbook: SportsbookConfig = Field(default_factory=SportsbookConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    data_dir: Path = Path("data/")
    log_level: str = "INFO"

    model_config = {"env_prefix": "MONEYGONE_", "env_nested_delimiter": "__"}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base, overlay wins on conflicts."""
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    base_path: Path = Path("config/default.yaml"),
    overlay_path: Path | None = None,
) -> AppConfig:
    """Load config from YAML with optional overlay, then apply env var overrides.

    Resolution order: default.yaml -> overlay (paper.yaml/live.yaml) -> env vars.
    """
    base_data: dict[str, Any] = {}
    if base_path.exists():
        with open(base_path) as f:
            base_data = yaml.safe_load(f) or {}

    if overlay_path and overlay_path.exists():
        with open(overlay_path) as f:
            overlay_data = yaml.safe_load(f) or {}
        base_data = _deep_merge(base_data, overlay_data)

    return AppConfig(**base_data)
