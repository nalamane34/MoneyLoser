# Sports Model: Road to Live Trading

## The Edge
We don't beat sportsbooks at picking winners. We catch when **Kalshi misprices
vs sportsbooks** and bet the sharp side. This is a pure arbitrage-like strategy:

- Sharp consensus says 65% → Kalshi prices 55¢ → buy YES (10% edge)
- The model is just: `edge = sharp_prob - kalshi_price`
- We only trade when edge > 8% (after fees)

## What's Done
- [x] Sharp sportsbook model v2 (Pinnacle-anchored, data-quality confidence)
- [x] SBR historical odds scraper (30 days backfilled)
- [x] Training pipeline (build-dataset → train → evaluate)
- [x] Team name matching (Kalshi ↔ sportsbook events)
- [x] Data quality features (line staleness, consensus fallback, match quality)
- [x] Mapping table schema
- [x] Odds API collector on EC2 (live sportsbook data)

## What's Needed for Live Trading

### Phase 1: Coverage Expansion (immediate)
1. Add all 28 league series tickers to the pipeline
2. Add SBR slugs for all Tier 1 leagues
3. Expand Odds API collector to cover soccer leagues
4. Build forward prediction system (like weather: predict → verify)

### Phase 2: Forward Validation (1-2 weeks of data)
1. Every day: scan all active sports markets on Kalshi
2. Compare Kalshi price vs sharp sportsbook consensus
3. Log predicted edge for every market
4. After settlement: verify edge calls were correct
5. Track: edge accuracy by league, edge bucket, time-to-game

### Phase 3: Paper Trading (after validation confirms edge)
1. Run the bot in paper mode with sports enabled
2. Verify order placement, sizing, risk limits
3. Track simulated P&L for 1 week minimum

### Phase 4: Live (after paper validates)
1. Start with Tier 1 leagues only
2. Very tight position limits (max 3 contracts per market)
3. Min edge threshold 8% (after fees)
4. Expand to Tier 2/3 as confidence grows
