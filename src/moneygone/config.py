"""Configuration loading with YAML files and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


DEFAULT_WEATHER_LOCATIONS: list[dict[str, Any]] = [
    {"name": "New York City", "lat": 40.7668, "lon": -73.9829},
    {"name": "Chicago", "lat": 41.7856, "lon": -87.7527},
    {"name": "Los Angeles", "lat": 33.9425, "lon": -118.4080},
    {"name": "Miami", "lat": 25.7954, "lon": -80.2901},
    {"name": "Dallas", "lat": 32.8972, "lon": -97.0377},
    {"name": "Denver", "lat": 39.8617, "lon": -104.6732},
    {"name": "Seattle", "lat": 47.4499, "lon": -122.3118},
    {"name": "Atlanta", "lat": 33.6367, "lon": -84.4279},
    {"name": "Houston", "lat": 29.9844, "lon": -95.3414},
    {"name": "Phoenix", "lat": 33.4343, "lon": -112.0116},
    {"name": "Philadelphia", "lat": 39.8744, "lon": -75.2424},
    {"name": "Boston", "lat": 42.3656, "lon": -71.0096},
    {"name": "Washington DC", "lat": 38.8512, "lon": -77.0402},
    {"name": "Las Vegas", "lat": 36.0801, "lon": -115.1522},
    {"name": "San Francisco", "lat": 37.6213, "lon": -122.3790},
    {"name": "San Antonio", "lat": 29.5337, "lon": -98.4698},
    {"name": "Oklahoma City", "lat": 35.3931, "lon": -97.6007},
    {"name": "Austin", "lat": 30.1945, "lon": -97.6699},
    {"name": "New Orleans", "lat": 29.9934, "lon": -90.2580},
    {"name": "Minneapolis", "lat": 44.8848, "lon": -93.2223},
]


def default_weather_locations() -> list[dict[str, Any]]:
    """Return a fresh copy of the default weather location list."""
    return [dict(location) for location in DEFAULT_WEATHER_LOCATIONS]


class ExchangeConfig(BaseModel):
    base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    ws_url: str = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    api_key_id: str = ""
    private_key_path: Path = Path("key.pem")
    subaccount: int = 0
    demo_mode: bool = True
    rate_limit_rps: float = 10.0

    @model_validator(mode="after")
    def _load_api_key_id_from_env(self) -> "ExchangeConfig":
        """Prefer KALSHI_API_KEY_ID env var over config file value."""
        env_val = os.environ.get("KALSHI_API_KEY_ID")
        if env_val:
            self.api_key_id = env_val
        return self


class WeatherConfig(BaseModel):
    enabled: bool = True
    noaa_enabled: bool = True
    ecmwf_enabled: bool = True
    open_meteo_api_key: str = ""
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
    sportsgameodds_api_key: str = ""
    prefer_sportsgameodds: bool = True  # Use SportsGameOdds when key available


class ModelConfig(BaseModel):
    model_dir: Path = Path("models/")
    calibration_method: str = "isotonic"
    ensemble_method: str = "inverse_variance"
    retrain_interval_hours: int = 24


class ExecutionConfig(BaseModel):
    min_edge_threshold: float = 0.02
    min_conviction_score: float = 0.02
    max_edge_sanity: float = 0.30
    prefer_maker: bool = True
    max_order_staleness_seconds: int = 30
    evaluation_interval_seconds: float = 5.0
    max_model_market_disagreement: float = 0.08
    max_watched_markets: int = 800
    max_markets_per_cycle: int = 200
    max_core_markets_per_category: int = 150
    max_fallback_markets_per_category: int = 60
    max_core_market_horizon_hours: float = 120.0
    max_fallback_market_horizon_hours: float = 48.0
    min_market_volume: int = 25
    min_market_open_interest: int = 25
    max_market_spread: float = 0.20


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

    # Strip keys not in the Pydantic model (e.g. closer strategy config)
    # to avoid 'extra inputs not permitted' errors.  These are read
    # directly from the YAML by the worker scripts that need them.
    _extra_keys = {"closer"}
    filtered_data = {k: v for k, v in base_data.items() if k not in _extra_keys}

    config = AppConfig(**filtered_data)

    # Safety: refuse to start if demo_mode is set but production URLs are used.
    # This catches misconfigurations where someone thinks they're in demo mode
    # but orders would go to the live exchange.
    _PROD_HOSTS = ("api.elections.kalshi.com", "trading-api.kalshi.com")
    if config.exchange.demo_mode:
        base_url = (config.exchange.base_url or "").lower()
        ws_url = (config.exchange.ws_url or "").lower()
        for host in _PROD_HOSTS:
            if host in base_url or host in ws_url:
                raise ValueError(
                    f"SAFETY: demo_mode=true but config uses production URL "
                    f"({host}). Either set demo_mode=false or use demo-api.kalshi.co. "
                    f"Refusing to start to prevent accidental live trading."
                )

    return config
