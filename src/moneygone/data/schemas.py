"""DuckDB table definitions for the MoneyGone data layer.

All tables are append-only with ``ingested_at`` timestamps so that
point-in-time queries can reconstruct state as of any historical moment.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Market data tables
# ---------------------------------------------------------------------------

CREATE_MARKET_STATES = """
CREATE TABLE IF NOT EXISTS market_states (
    ticker        VARCHAR NOT NULL,
    event_ticker  VARCHAR NOT NULL,
    title         VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    yes_bid       DOUBLE,
    yes_ask       DOUBLE,
    last_price    DOUBLE,
    volume        BIGINT,
    open_interest BIGINT,
    close_time    TIMESTAMP NOT NULL,
    snapshot_time TIMESTAMP,
    result        VARCHAR,
    category      VARCHAR,
    ingested_at   TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

ALTER_MARKET_STATES_ADD_SNAPSHOT_TIME = """
ALTER TABLE market_states ADD COLUMN IF NOT EXISTS snapshot_time TIMESTAMP;
"""

CREATE_ORDERBOOK_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    ticker        VARCHAR NOT NULL,
    yes_levels    JSON NOT NULL,
    no_levels     JSON NOT NULL,
    seq           BIGINT,
    snapshot_time TIMESTAMP NOT NULL,
    ingested_at   TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id   VARCHAR NOT NULL,
    ticker     VARCHAR NOT NULL,
    count      INTEGER NOT NULL,
    yes_price  DOUBLE NOT NULL,
    taker_side VARCHAR NOT NULL,
    trade_time TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

# ---------------------------------------------------------------------------
# Weather / forecast tables
# ---------------------------------------------------------------------------

CREATE_FORECAST_ENSEMBLES = """
CREATE TABLE IF NOT EXISTS forecast_ensembles (
    location_name VARCHAR NOT NULL,
    lat           DOUBLE NOT NULL,
    lon           DOUBLE NOT NULL,
    variable      VARCHAR NOT NULL,
    init_time     TIMESTAMP NOT NULL,
    valid_time    TIMESTAMP NOT NULL,
    member_values JSON NOT NULL,
    ensemble_mean DOUBLE NOT NULL,
    ensemble_std  DOUBLE NOT NULL,
    ingested_at   TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

# ---------------------------------------------------------------------------
# Crypto tables
# ---------------------------------------------------------------------------

CREATE_FUNDING_RATES = """
CREATE TABLE IF NOT EXISTS funding_rates (
    exchange    VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    rate        DOUBLE NOT NULL,
    timestamp   TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

CREATE_OPEN_INTEREST = """
CREATE TABLE IF NOT EXISTS open_interest (
    exchange    VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    value       DOUBLE NOT NULL,
    timestamp   TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

# ---------------------------------------------------------------------------
# Sportsbook tables
# ---------------------------------------------------------------------------

CREATE_SPORTSBOOK_GAME_LINES = """
CREATE TABLE IF NOT EXISTS sportsbook_game_lines (
    event_id       VARCHAR NOT NULL,
    sport          VARCHAR NOT NULL,
    home_team      VARCHAR NOT NULL,
    away_team      VARCHAR NOT NULL,
    bookmaker      VARCHAR NOT NULL,
    commence_time  TIMESTAMP,
    home_price     DOUBLE NOT NULL,
    away_price     DOUBLE NOT NULL,
    draw_price     DOUBLE,
    spread_home    DOUBLE,
    total          DOUBLE,
    captured_at    TIMESTAMP NOT NULL,
    ingested_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

# ---------------------------------------------------------------------------
# Sports mapping + training tables
# ---------------------------------------------------------------------------

CREATE_MARKET_EVENT_MAPPINGS = """
CREATE TABLE IF NOT EXISTS market_event_mappings (
    kalshi_ticker       VARCHAR NOT NULL,
    kalshi_event_ticker VARCHAR,
    event_id            VARCHAR NOT NULL,
    sport               VARCHAR NOT NULL,
    home_team           VARCHAR NOT NULL,
    away_team           VARCHAR NOT NULL,
    priced_team         VARCHAR NOT NULL,
    is_home_team        BOOLEAN NOT NULL,
    match_score         INTEGER NOT NULL,
    commence_time       TIMESTAMP,
    created_at          TIMESTAMP NOT NULL DEFAULT current_timestamp,
    verified_at         TIMESTAMP
);
"""

CREATE_SPORTS_OUTCOMES = """
CREATE TABLE IF NOT EXISTS sports_outcomes (
    kalshi_ticker         VARCHAR NOT NULL,
    event_id              VARCHAR,
    sport                 VARCHAR NOT NULL,
    home_team             VARCHAR NOT NULL,
    away_team             VARCHAR NOT NULL,
    priced_team           VARCHAR NOT NULL,
    is_home_team          BOOLEAN NOT NULL,
    market_result         VARCHAR NOT NULL,
    kalshi_last_price     DOUBLE,
    kalshi_volume         BIGINT,
    close_time            TIMESTAMP,
    commence_time         TIMESTAMP,
    pinnacle_home_price   DOUBLE,
    pinnacle_away_price   DOUBLE,
    pinnacle_draw_price   DOUBLE,
    pinnacle_home_prob    DOUBLE,
    consensus_home_price  DOUBLE,
    consensus_away_price  DOUBLE,
    consensus_draw_price  DOUBLE,
    consensus_home_prob   DOUBLE,
    opening_home_price    DOUBLE,
    opening_away_price    DOUBLE,
    spread_home           DOUBLE,
    total                 DOUBLE,
    home_rating           DOUBLE,
    away_rating           DOUBLE,
    line_age_hours        DOUBLE,
    is_consensus_fallback BOOLEAN,
    created_at            TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

# ---------------------------------------------------------------------------
# Feature / model tables
# ---------------------------------------------------------------------------

CREATE_FEATURES = """
CREATE TABLE IF NOT EXISTS features (
    ticker           VARCHAR NOT NULL,
    observation_time  TIMESTAMP NOT NULL,
    feature_name      VARCHAR NOT NULL,
    feature_value     DOUBLE NOT NULL,
    ingested_at       TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    ticker           VARCHAR NOT NULL,
    model_name       VARCHAR NOT NULL,
    model_version    VARCHAR NOT NULL,
    probability      DOUBLE NOT NULL,
    raw_probability  DOUBLE NOT NULL,
    confidence       DOUBLE NOT NULL,
    prediction_time  TIMESTAMP NOT NULL,
    ingested_at      TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

# ---------------------------------------------------------------------------
# Execution / accounting tables
# ---------------------------------------------------------------------------

CREATE_FILLS_LOG = """
CREATE TABLE IF NOT EXISTS fills_log (
    trade_id    VARCHAR NOT NULL,
    ticker      VARCHAR NOT NULL,
    side        VARCHAR NOT NULL,
    action      VARCHAR NOT NULL,
    count       INTEGER NOT NULL,
    price       DOUBLE NOT NULL,
    is_taker    BOOLEAN NOT NULL,
    fee_paid    DOUBLE,
    category    VARCHAR,
    model_name  VARCHAR,
    predicted_prob DOUBLE,
    predicted_confidence DOUBLE,
    raw_edge    DOUBLE,
    fee_adjusted_edge DOUBLE,
    strategy    VARCHAR,
    signal_source VARCHAR,
    expected_profit DOUBLE,
    fill_time   TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

CREATE_SETTLEMENTS_LOG = """
CREATE TABLE IF NOT EXISTS settlements_log (
    ticker        VARCHAR NOT NULL,
    market_result VARCHAR NOT NULL,
    revenue       DOUBLE NOT NULL,
    payout        DOUBLE NOT NULL,
    settled_time  TIMESTAMP NOT NULL,
    ingested_at   TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

# ---------------------------------------------------------------------------
# Ordered list for schema initialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-worker table partitions (for multi-process architecture)
# ---------------------------------------------------------------------------

COLLECTOR_TABLES: list[str] = [
    CREATE_SPORTSBOOK_GAME_LINES,
    CREATE_FORECAST_ENSEMBLES,
    CREATE_FUNDING_RATES,
    CREATE_OPEN_INTEREST,
]

MARKET_DATA_TABLES: list[str] = [
    CREATE_MARKET_STATES,
    ALTER_MARKET_STATES_ADD_SNAPSHOT_TIME,
    CREATE_ORDERBOOK_SNAPSHOTS,
    CREATE_TRADES,
]

EXECUTION_TABLES: list[str] = [
    CREATE_SPORTSBOOK_GAME_LINES,
    CREATE_MARKET_EVENT_MAPPINGS,
    CREATE_SPORTS_OUTCOMES,
    CREATE_FEATURES,
    CREATE_PREDICTIONS,
    CREATE_FILLS_LOG,
    CREATE_SETTLEMENTS_LOG,
]

ALL_TABLES: list[str] = [
    CREATE_MARKET_STATES,
    ALTER_MARKET_STATES_ADD_SNAPSHOT_TIME,
    CREATE_ORDERBOOK_SNAPSHOTS,
    CREATE_TRADES,
    CREATE_FORECAST_ENSEMBLES,
    CREATE_FUNDING_RATES,
    CREATE_OPEN_INTEREST,
    CREATE_SPORTSBOOK_GAME_LINES,
    CREATE_MARKET_EVENT_MAPPINGS,
    CREATE_SPORTS_OUTCOMES,
    CREATE_FEATURES,
    CREATE_PREDICTIONS,
    CREATE_FILLS_LOG,
    CREATE_SETTLEMENTS_LOG,
]
