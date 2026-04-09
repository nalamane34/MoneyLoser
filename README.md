# MoneyLoser

Automated trading system for [Kalshi](https://kalshi.com) prediction markets. Discovers statistical edge in binary event contracts, calibrates probability models, and executes trades with Kelly-based position sizing and rigorous risk management.

---

## What This Is

Kalshi sells binary contracts that resolve to $1 (yes wins) or $0 (no wins) on real-world events — weather, sports outcomes, economic releases, crypto prices, and more. Each contract trades at an implied probability between $0.01 and $0.99.

This system:
1. **Ingests** data from Kalshi's API, weather models, crypto exchanges, sportsbooks, and economic feeds
2. **Computes features** that are predictive of contract outcomes
3. **Trains calibrated models** that produce better probability estimates than the market price
4. **Calculates edge** — the gap between model probability and market price, after Kalshi's fees
5. **Sizes positions** using fractional Kelly criterion
6. **Executes orders** via Kalshi's REST and WebSocket APIs
7. **Monitors** for model drift, calibration degradation, and drawdown

---

## Backtested Edge (83,453 traded markets)

| Category | Model Brier | Market Brier | Edge (lower = better) |
|---|---|---|---|
| **Sports Props** | 0.2234 | 0.2500 | **+0.0266** |
| Crypto Range | 0.2436 | 0.2500 | +0.0064 |
| Sports Spread | 0.2486 | 0.2500 | +0.0014 |
| Crypto Daily | 0.2490 | 0.2500 | +0.0010 |
| Game Winners | 0.2498 | 0.2500 | +0.0002 |

Sports props show the most durable edge. Crypto range and spread markets are the next priority. Game winner edge is currently minimal but the feature set is in place.

---

## System Architecture

```
Kalshi WS/REST + NOAA/ECMWF + Kraken + ESPN + Odds API + FRED
                         |
              [ DataStore (DuckDB) ]
                         |
              [ FeaturePipeline ] ── point-in-time feature vectors
                         |
       [ ProbabilityModel (LightGBM + isotonic calibration) ]
                         |
         [ EdgeCalculator ] ── fee-adjusted edge vs market price
                         |
            [ KellySizer ] ── 0.25 fractional Kelly
                         |
          [ RiskManager ] ── pre-trade checks (limits, drawdown, exposure)
                         |
          [ OrderManager ] ── submit to Kalshi
                         |
             [ Monitoring ] ── drift, calibration, P&L, regime
```

---

## Project Structure

```
src/moneygone/
├── exchange/               # Kalshi API layer
│   ├── auth.py             # RSA-PSS SHA-256 authentication
│   ├── rest_client.py      # 41-endpoint async REST client
│   ├── ws_client.py        # WebSocket with orderbook reconstruction
│   ├── types.py            # All exchange data types (typed dataclasses)
│   └── rate_limiter.py     # Token-bucket rate limiter
│
├── data/                   # Data ingestion & storage
│   ├── store.py            # DuckDB with point-in-time queries
│   ├── crypto/
│   │   ├── ccxt_feed.py    # Kraken OHLCV via CCXT
│   │   └── volatility.py   # Realized vol, Deribit DVOL, ATR, BRTI proxy
│   ├── sports/
│   │   ├── stats.py        # ESPN player stats, game logs, injuries
│   │   └── odds.py         # The Odds API — sportsbook lines, props, moneylines
│   ├── weather/
│   │   ├── noaa.py         # NOAA NWS hourly/gridded forecasts
│   │   └── ecmwf.py        # ECMWF ensemble forecasts
│   └── economic/
│       └── releases.py     # FRED API — CPI, unemployment, GDP releases
│
├── features/               # Feature engineering
│   ├── market_features.py  # Orderbook microstructure (spread, depth, imbalance)
│   ├── crypto_features.py  # Funding rate, OI, vol regime, ATR, IV, trend
│   ├── sports_features.py  # Player mean/variance, usage, game script, matchup
│   ├── game_winner_features.py  # Sportsbook consensus, sharp money, power ratings
│   ├── weather_features.py # Ensemble spread, forecast revision, disagreement
│   └── temporal.py         # Time-to-expiry, price velocity, momentum
│
├── models/                 # Probabilistic models
│   ├── trainers/
│   │   ├── gbm.py          # LightGBM (primary)
│   │   ├── logistic.py     # Logistic regression (baseline)
│   │   └── bayesian.py     # Bayesian inference
│   ├── calibration.py      # Isotonic, Platt, beta calibration
│   ├── ensemble.py         # Inverse-variance weighted combiner
│   └── evaluation.py       # Brier score, ECE, log loss, reliability diagrams
│
├── signals/
│   ├── fees.py             # Kalshi fee formula: 0.07×c×p×(1-p), max $0.02
│   └── edge.py             # Fee-adjusted edge calculator
│
├── sizing/
│   └── kelly.py            # Fractional Kelly (default 0.25×full Kelly)
│
├── execution/
│   ├── engine.py           # Event-driven trading loop
│   ├── order_manager.py    # Order lifecycle (submit, track, cancel, amend)
│   └── strategies.py       # Passive (post-only), aggressive, TWAP-like
│
├── strategies/             # Trading strategies
│   ├── resolution_sniper.py    # Bet near resolution when model is confident
│   ├── live_event_edge.py      # Trade live events as new data arrives
│   ├── cross_market_arb.py     # Exploit pricing gaps across correlated markets
│   └── market_maker.py         # Post passive quotes, earn maker rebate
│
├── risk/
│   ├── manager.py          # Pre/post-trade risk orchestration
│   ├── drawdown.py         # Drawdown monitoring + circuit breakers
│   └── exposure.py         # Per-market/category exposure limits
│
├── monitoring/
│   ├── drift.py            # PSI, KS test on prediction distributions
│   ├── calibration_monitor.py  # Rolling Brier/ECE/log loss
│   └── pnl.py              # P&L tracking + attribution
│
└── backtest/
    ├── engine.py           # Replays historical events through live pipeline
    ├── sim_exchange.py     # Simulated exchange (fills, fees, orderbook)
    └── guards.py           # Lookahead bias + leakage prevention
```

---

## Features

### Market Microstructure
Computed from Kalshi's live orderbook for every market:
- Bid-ask spread, mid price, depth within 5¢ of best bid
- Order book imbalance (bid depth vs ask depth)
- Price velocity and momentum (rolling windows)
- Time-to-expiry (log-scaled)
- Volume and open interest

### Crypto Markets
From Kraken (OHLCV) and Deribit (options):
| Feature | Source | Description |
|---|---|---|
| `realized_vol_24h` / `7d` / `30d` | Kraken OHLCV | Annualized log-return vol |
| `implied_vol` | Deribit DVOL index | 30-day BTC implied vol |
| `vol_spread` | Derived | IV − RV (fear premium) |
| `atr_14` / `atr_24` | Kraken OHLCV | Normalized Average True Range |
| `trend_regime` | Kraken OHLCV | Multi-timeframe: 8h/24h/72h momentum |
| `brti_price` | Kraken mid | Bitcoin Reference Rate proxy |
| `funding_rate` | Exchange | Perp funding (bullish/bearish bias) |
| `open_interest_change` | Exchange | OI change (new longs vs shorts) |

### Sports Props
From ESPN (free public API) and The Odds API:
| Feature | Description |
|---|---|
| `player_mean` | Season average for the stat |
| `player_variance` | Game-to-game standard deviation |
| `player_recent_form` | Last-5 average ÷ season average |
| `usage_rate` | NBA: % of team possessions used |
| `game_script` | Absolute spread (blowout vs close game) |
| `matchup_effect` | Opponent def rank vs league avg (normalized) |
| `injury_impact` | Key teammate injury count |
| `minutes_expected` | Season avg minutes |
| `prop_line_vs_market` | Sportsbook implied over-prob − Kalshi price |
| `sharp_money_indicator` | Opening vs current line movement |

### Game Winner Markets
| Feature | Description |
|---|---|
| `sportsbook_win_prob` | Consensus moneyline probability (overround-removed) |
| `kalshi_vs_sportsbook_edge` | Sportsbook − Kalshi (lag/arbitrage signal) |
| `moneyline_movement` | Opening vs current line shift (sharp money proxy) |
| `sharp_vs_public_bias` | Sportsbook prob − public betting % |
| `power_rating_edge` | Team Elo/rating differential |
| `home_field_advantage` | Home (+1.0) vs Away (−1.0) |
| `team_injury_impact` | Opponent injuries − own injuries (severity-weighted) |
| `spread_implied_win_prob` | Normal approximation P(win) from spread |

### Weather Markets
From NOAA NWS and ECMWF ensemble:
- Ensemble mean/spread/disagreement (model uncertainty)
- Forecast revision from prior run
- Temperature deviation from climatology
- Precipitation probability

---

## Kalshi API Coverage

The REST client covers all 41 Kalshi API endpoints:

**Orders:** create, cancel, amend, decrease, batch create/cancel, get order(s), queue position  
**Order Groups:** create, trigger, delete, reset, update limit — bundle orders with fill caps  
**Portfolio:** balance, positions, fills, settlements, total resting order value  
**Market Data:** markets, orderbooks (single + bulk), trades, candlesticks, series  
**Historical:** markets, candlesticks, trades, fills, orders, cutoff timestamps  
**Exchange:** status, schedule, announcements  
**Events:** get event, forecast percentile history  
**Search:** tags by category  

---

## Fee Model

Kalshi charges takers only:
```
fee = 0.07 × contracts × price × (1 - price)
max fee = $0.02 per contract
```

At $0.50 (maximum fee price), fee = $0.0175/contract. Maker orders (resting limit orders) pay **zero fees**. The system defaults to passive/maker execution wherever possible.

Edge is only positive if:
```
model_probability - market_price > fee_per_dollar
```

---

## Setup

```bash
# Install
pip install -e ".[dev]"

# Configure credentials
cp .env.example .env
# Add your Kalshi API key ID and private key path

# Record live market data
python scripts/record_data.py

# Ingest historical markets (500K+ markets)
python scripts/ingest_historical.py

# Train models on all categories
python scripts/train_full_models.py

# Run backtests
python scripts/run_backtest.py

# Paper trade (Kalshi demo API)
python scripts/run_live.py --config config/paper.yaml

# Live trade
python scripts/run_live.py --config config/live.yaml
```

---

## API Keys Required

| Service | Use | Cost |
|---|---|---|
| Kalshi | Exchange access | Free with account |
| The Odds API | Sportsbook lines + props | 500 credits/month free |
| FRED | Economic data (CPI, unemployment) | Free |
| NOAA NWS | US weather forecasts | Free |
| ECMWF | Ensemble weather | Free tier |

Crypto data (Kraken OHLCV, Deribit DVOL) requires no API key.

---

## Risk Controls

- **Fractional Kelly:** 0.25× full Kelly by default — accounts for model uncertainty
- **Minimum edge threshold:** Only trade when fee-adjusted edge > configurable floor
- **Per-market exposure cap:** Max contracts per single market
- **Category exposure cap:** Max total exposure per category (sports, crypto, weather)
- **Drawdown circuit breaker:** Halt trading if portfolio drawdown exceeds limit
- **Total resting order value check:** Pre-trade buying power validation
- **Order groups:** Cap total fills across correlated markets
- **Model drift detection:** PSI + KS test monitoring for distribution shift

---

## Design Principles

1. **Fee-first** — edge is never evaluated without subtracting fees
2. **Shared pipeline** — backtest uses identical feature/model/sizing code as live trading
3. **Point-in-time fencing** — all data queries use `as_of` timestamps; leakage guards validate this in backtesting
4. **Maker-first execution** — passive limit orders wherever possible to avoid taker fees
5. **Calibration over prediction** — models are evaluated on Brier score, not accuracy; only deployed if they beat the market baseline
