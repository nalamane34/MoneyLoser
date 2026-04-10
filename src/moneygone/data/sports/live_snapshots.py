"""Store-backed sports snapshots for live game-winner evaluation.

This module avoids burning additional Odds API quota in the execution loop by
building live ``sports_snapshot`` payloads from previously recorded sportsbook
lines in DuckDB, then enriching them with free ESPN ratings and injury data.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from moneygone.data.store import DataStore
from moneygone.data.sports.power_ratings import ESPNPowerRatings
from moneygone.data.sports.stats import PlayerStatsFeed, TeamInjurySummary
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import Market

logger = structlog.get_logger(__name__)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_LEAGUE_MARKERS = {
    "nba": ("nba", "basketball"),
    "nfl": ("nfl", "football"),
    "nhl": ("nhl", "hockey"),
    "mlb": ("mlb", "baseball"),
    "ncaab": ("ncaab", "ncaa", "college basketball"),
    "ncaaf": ("ncaaf", "ncaa", "college football"),
    "soccer_epl": ("epl", "premier league"),
    "soccer_usa_mls": ("mls", "major league soccer"),
    "soccer_spain_la_liga": ("la liga", "laliga"),
    "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_italy_serie_a": ("serie a", "seriea"),
    "soccer_france_ligue_one": ("ligue 1", "ligue1"),
}


def _to_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_team_name(value: str) -> str:
    cleaned = value.lower().replace("&", " and ")
    replacements = {
        "los angeles": "la",
        "new york": "ny",
        "st ": "saint ",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return _NON_ALNUM_RE.sub("", cleaned)


def _team_aliases(value: str) -> set[str]:
    cleaned = value.lower().replace("&", " and ").replace(".", "")
    replacements = {
        "los angeles": "la",
        "new york": "ny",
        "st ": "saint ",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    words = [word for word in re.split(r"[^a-z0-9]+", cleaned) if word]
    aliases = {_normalize_team_name(value)}
    if len(words) >= 2:
        city = "".join(words[:-1])
        mascot = words[-1]
        aliases.add(mascot)
        aliases.add(city + mascot[:1])
        if len(city) >= 3:
            aliases.add(city)
    elif words:
        aliases.add(words[0])
    return {alias for alias in aliases if len(alias) >= 3}


def _matches_alias(value: str, aliases: set[str]) -> bool:
    if not value:
        return False
    return any(value == alias or (len(value) >= 4 and value in alias) for alias in aliases)


def _contains_alias(text: str, aliases: set[str]) -> bool:
    return any(alias in text for alias in aliases)


def _extract_priced_team(
    title: str,
    *,
    yes_sub_title: str = "",
) -> str:
    if yes_sub_title.strip():
        return yes_sub_title.strip()

    patterns = (
        r"-\s*(.+?)\s+to\s+win",
        r"[Ww]ill\s+(?:the\s+)?(.+?)\s+win",
        r"[Ww]ill\s+(?:the\s+)?(.+?)\s+(?:beat|defeat|defeat\w*|overcome)",
        r"(.+?)\s+to\s+win",
        r"(.+?)\s+moneyline",
        r"(.+?)\s+vs\.?\s+",
    )
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            return match.group(1).strip()
    return ""


def _implied_prob(home_price: float | None, away_price: float | None) -> float | None:
    if home_price is None or away_price is None or home_price <= 1.0 or away_price <= 1.0:
        return None
    raw_home = 1.0 / home_price
    raw_away = 1.0 / away_price
    total = raw_home + raw_away
    if total <= 0:
        return None
    return raw_home / total


class StoreBackedSportsSnapshotProvider:
    """Build live sports snapshots from stored sportsbook lines plus ESPN data."""

    def __init__(
        self,
        store: DataStore,
        *,
        leagues: list[str],
        rest_client: KalshiRestClient | None = None,
        stats_feed: PlayerStatsFeed | None = None,
        power_ratings: ESPNPowerRatings | None = None,
        max_line_age: timedelta | None = None,
        key_minutes_threshold: float = 20.0,
    ) -> None:
        self._store = store
        self._leagues = tuple(sorted({league.lower() for league in leagues}))
        self._league_markers = tuple(
            sorted(
                {
                    marker
                    for league in self._leagues
                    for marker in _LEAGUE_MARKERS.get(league, (league,))
                }
            )
        )
        self._rest = rest_client
        self._stats_feed = stats_feed or PlayerStatsFeed()
        self._power_ratings = power_ratings or ESPNPowerRatings()
        self._owns_stats_feed = stats_feed is None
        self._owns_power_ratings = power_ratings is None
        self._max_line_age = max_line_age or timedelta(hours=12)
        self._key_minutes_threshold = key_minutes_threshold
        self._snapshot_cache: dict[str, dict[str, Any]] = {}
        self._matched_tickers: set[str] = set()
        self._event_title_cache: dict[str, str] = {}

    async def close(self) -> None:
        if self._owns_stats_feed:
            await self._stats_feed.close()
        if self._owns_power_ratings:
            await self._power_ratings.close()

    async def refresh(self, markets: list[Market]) -> list[Market]:
        """Refresh the ticker -> sports snapshot cache and return matched markets."""
        latest_rows: list[dict[str, Any]] = []
        opening_by_event: dict[str, dict[str, Any]] = {}
        ratings_by_league: dict[str, dict[str, Any]] = {}

        consensus_by_event: dict[str, dict[str, Any]] = {}

        for league in self._leagues:
            try:
                latest_by_event = self._store.get_latest_sportsbook_lines(
                    bookmaker="pinnacle",
                    sport=league,
                )
            except Exception:
                # Table may not exist if using execution store instead of collector
                logger.debug("sports_snapshots.no_sportsbook_table", league=league)
                continue
            if not latest_by_event:
                continue
            latest_rows.extend(latest_by_event.values())
            try:
                opening_by_event.update(
                    self._store.get_opening_sportsbook_lines(
                        bookmaker="pinnacle",
                        sport=league,
                        event_ids=list(latest_by_event.keys()),
                    )
                )
            except Exception:
                pass
            # Fetch consensus (non-Pinnacle) lines for sportsbook_home_win_prob
            for bk in ("fanduel", "draftkings", "betmgm"):
                try:
                    consensus = self._store.get_latest_sportsbook_lines(
                        bookmaker=bk,
                        sport=league,
                    )
                except Exception:
                    continue
                if consensus:
                    for eid, row in consensus.items():
                        if eid not in consensus_by_event:
                            consensus_by_event[eid] = row
                    break  # Use first available consensus bookmaker

            ratings_by_league[league] = await self._power_ratings.get_ratings(league)

        snapshots: dict[str, dict[str, Any]] = {}
        matched: list[Market] = []
        injury_cache: dict[tuple[str, str], TeamInjurySummary] = {}
        rejected = {"no_match": 0, "unoriented": 0, "stale_line": 0}

        for market in markets:
            built, reason = await self._build_snapshot(
                market,
                latest_rows=latest_rows,
                opening_by_event=opening_by_event,
                consensus_by_event=consensus_by_event,
                ratings_by_league=ratings_by_league,
                injury_cache=injury_cache,
            )
            if built is None:
                if reason in rejected:
                    rejected[reason] += 1
                continue
            snapshots[market.ticker] = built
            matched.append(market)

        self._snapshot_cache = snapshots
        self._matched_tickers = {market.ticker for market in matched}
        logger.info(
            "sports_snapshots.refreshed",
            leagues=list(self._leagues),
            matched=len(matched),
            no_match=rejected["no_match"],
            unoriented=rejected["unoriented"],
            stale_line=rejected["stale_line"],
        )
        return matched

    async def get_snapshot(self, market: Market) -> dict[str, Any] | None:
        """Return the cached sports snapshot for a market, updated with live price."""
        cached = self._snapshot_cache.get(market.ticker)
        if cached is None:
            return None
        snapshot = dict(cached)
        snapshot["kalshi_implied_prob"] = self._kalshi_implied_prob(market)
        return snapshot

    def watched_tickers(self) -> list[str]:
        return sorted(self._matched_tickers)

    async def _build_snapshot(
        self,
        market: Market,
        *,
        latest_rows: list[dict[str, Any]],
        opening_by_event: dict[str, dict[str, Any]],
        consensus_by_event: dict[str, dict[str, Any]],
        ratings_by_league: dict[str, dict[str, Any]],
        injury_cache: dict[tuple[str, str], TeamInjurySummary],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not self._looks_like_game_winner_market(market):
            return None, "no_match"

        match = await self._match_market_to_event(market, latest_rows)
        if match is None:
            return None, "no_match"

        league = str(match.get("sport", "")).lower()
        captured_at = _to_utc(match.get("captured_at"))
        if captured_at is None or datetime.now(timezone.utc) - captured_at > self._max_line_age:
            logger.debug("sports_snapshots.stale_line", ticker=market.ticker, event_id=match.get("event_id"))
            return None, "stale_line"

        home_team = str(match.get("home_team", ""))
        away_team = str(match.get("away_team", ""))
        event_title = await self._get_event_title(market.event_ticker)
        priced_team = _extract_priced_team(
            market.title,
            yes_sub_title=market.yes_sub_title,
        )
        priced_norm = _normalize_team_name(priced_team)
        market_text = self._market_text(market, event_title=event_title)
        home_aliases = _team_aliases(home_team)
        away_aliases = _team_aliases(away_team)

        if _matches_alias(priced_norm, home_aliases):
            is_home_team = True
        elif _matches_alias(priced_norm, away_aliases):
            is_home_team = False
        elif _contains_alias(market_text, home_aliases) and not _contains_alias(market_text, away_aliases):
            is_home_team = True
        elif _contains_alias(market_text, away_aliases) and not _contains_alias(market_text, home_aliases):
            is_home_team = False
        else:
            logger.debug(
                "sports_snapshots.unoriented_market",
                ticker=market.ticker,
                title=market.title,
                home_team=home_team,
                away_team=away_team,
            )
            return None, "unoriented"

        opening = opening_by_event.get(str(match.get("event_id", "")), {})
        ratings = ratings_by_league.get(league, {})
        home_rating = self._power_ratings.lookup(home_team, ratings) if ratings else None
        away_rating = self._power_ratings.lookup(away_team, ratings) if ratings else None
        home_injury = await self._get_injury_summary(
            league,
            home_rating.team_id if home_rating is not None else "",
            injury_cache,
        )
        away_injury = await self._get_injury_summary(
            league,
            away_rating.team_id if away_rating is not None else "",
            injury_cache,
        )

        current_home = self._float_or_none(match.get("home_price"))
        current_away = self._float_or_none(match.get("away_price"))
        opening_home = self._float_or_none(opening.get("home_price"))
        opening_away = self._float_or_none(opening.get("away_price"))
        pinnacle_prob = _implied_prob(current_home, current_away)

        # Consensus (non-Pinnacle) probability for sportsbook_home_win_prob
        event_id_str = str(match.get("event_id", ""))
        consensus_row = consensus_by_event.get(event_id_str)
        if consensus_row is not None:
            cons_home = self._float_or_none(consensus_row.get("home_price"))
            cons_away = self._float_or_none(consensus_row.get("away_price"))
            consensus_prob = _implied_prob(cons_home, cons_away)
        else:
            consensus_prob = None

        return {
            "event_id": event_id_str,
            "sport": league,
            "home_team": home_team,
            "away_team": away_team,
            "is_home_team": int(is_home_team),
            "kalshi_implied_prob": self._kalshi_implied_prob(market),
            "sportsbook_home_win_prob": consensus_prob,
            "pinnacle_home_win_prob": pinnacle_prob,
            "pinnacle_moneyline_home": current_home,
            "pinnacle_moneyline_away": current_away,
            "current_moneyline_home": current_home,
            "current_moneyline_away": current_away,
            "opening_moneyline_home": opening_home,
            "opening_moneyline_away": opening_away,
            "spread": self._float_or_none(match.get("spread_home")),
            "total": self._float_or_none(match.get("total")),
            "home_team_rating": home_rating.rating if home_rating is not None else None,
            "away_team_rating": away_rating.rating if away_rating is not None else None,
            "home_key_injuries": home_injury.key_injuries if home_injury is not None else None,
            "away_key_injuries": away_injury.key_injuries if away_injury is not None else None,
            "home_injury_severity": home_injury.injury_severity if home_injury is not None else None,
            "away_injury_severity": away_injury.injury_severity if away_injury is not None else None,
            "line_captured_at": captured_at.isoformat(),
            "line_source": "store:pinnacle",
        }, None

    async def _get_injury_summary(
        self,
        league: str,
        team_id: str,
        injury_cache: dict[tuple[str, str], TeamInjurySummary],
    ) -> TeamInjurySummary | None:
        if not team_id:
            return None
        cache_key = (league, team_id)
        if cache_key not in injury_cache:
            injury_cache[cache_key] = await self._stats_feed.get_team_injury_summary(
                league,
                league,
                team_id,
                key_minutes_threshold=self._key_minutes_threshold,
            )
        return injury_cache[cache_key]

    async def _match_market_to_event(
        self,
        market: Market,
        latest_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        market_text = self._market_text(market)
        best_row, best_score, best_time_delta = self._score_rows(
            market,
            market_text=market_text,
            latest_rows=latest_rows,
        )

        if not self._looks_like_game_winner_market(market) or not self._matches_league_scope(market):
            return best_row if best_score >= 3 else None
        if best_score >= 4:
            return best_row
        if not market.event_ticker or self._rest is None:
            return best_row if best_score >= 3 else None

        event_title = await self._get_event_title(market.event_ticker)
        if not event_title:
            return best_row if best_score >= 3 else None

        fallback_text = self._market_text(market, event_title=event_title)
        event_row, event_score, event_time_delta = self._score_rows(
            market,
            market_text=fallback_text,
            latest_rows=latest_rows,
        )
        if event_score > best_score or (event_score == best_score and event_time_delta < best_time_delta):
            best_row = event_row
            best_score = event_score

        if best_score < 3:
            return None
        return best_row

    def _score_rows(
        self,
        market: Market,
        *,
        market_text: str,
        latest_rows: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, int, float]:
        priced_team_norm = _normalize_team_name(
            _extract_priced_team(
                market.title,
                yes_sub_title=market.yes_sub_title,
            )
        )
        best_row: dict[str, Any] | None = None
        best_score = -1
        best_time_delta = float("inf")

        for row in latest_rows:
            home_team = str(row.get("home_team", ""))
            away_team = str(row.get("away_team", ""))
            home_aliases = _team_aliases(home_team)
            away_aliases = _team_aliases(away_team)
            title_has_home = _contains_alias(market_text, home_aliases)
            title_has_away = _contains_alias(market_text, away_aliases)
            priced_matches_home = _matches_alias(priced_team_norm, home_aliases)
            priced_matches_away = _matches_alias(priced_team_norm, away_aliases)

            score = 0
            if title_has_home and title_has_away and (priced_matches_home or priced_matches_away):
                score = 4
            elif title_has_home and title_has_away:
                score = 3
            elif priced_matches_home or priced_matches_away:
                score = 2
            elif title_has_home or title_has_away:
                score = 1

            if score <= 0:
                continue

            commence_time = _to_utc(row.get("commence_time"))
            if commence_time is not None:
                time_delta = abs((market.close_time.astimezone(timezone.utc) - commence_time).total_seconds())
            else:
                time_delta = float("inf")

            if score > best_score or (score == best_score and time_delta < best_time_delta):
                best_row = row
                best_score = score
                best_time_delta = time_delta

        return best_row, best_score, best_time_delta

    def _market_text(self, market: Market, *, event_title: str = "") -> str:
        return _normalize_team_name(
            " ".join(
                value
                for value in [
                    market.title,
                    market.subtitle,
                    market.yes_sub_title,
                    market.no_sub_title,
                    event_title,
                ]
                if value
            )
        )

    def _looks_like_game_winner_market(self, market: Market) -> bool:
        text = " ".join(
            value.lower()
            for value in [
                market.ticker,
                market.event_ticker,
                market.series_ticker,
                market.title,
                market.subtitle,
                market.yes_sub_title,
            ]
            if value
        )
        excluded = (
            "1h",
            "2h",
            "first half",
            "second half",
            "spread",
            "total",
            "teamtotal",
            "team total",
            "score over",
            "points",
            "mention",
            "announcers say",
        )
        if any(marker in text for marker in excluded):
            return False
        return any(
            marker in text
            for marker in ("game", "winner", "moneyline", " to win", " beat ", " defeat ", " win?")
        )

    def _matches_league_scope(self, market: Market) -> bool:
        scope_text = " ".join(
            value.lower()
            for value in [
                market.ticker,
                market.event_ticker,
                market.series_ticker,
                market.title,
                market.subtitle,
                market.yes_sub_title,
                market.no_sub_title,
            ]
            if value
        )
        return any(marker in scope_text for marker in self._league_markers)

    async def _get_event_title(self, event_ticker: str) -> str:
        if not event_ticker or self._rest is None:
            return ""
        if event_ticker in self._event_title_cache:
            return self._event_title_cache[event_ticker]
        try:
            event = await self._rest.get_event(event_ticker)
        except Exception:
            logger.debug("sports_snapshots.event_lookup_failed", event_ticker=event_ticker, exc_info=True)
            self._event_title_cache[event_ticker] = ""
            return ""
        title = str(event.get("title", ""))
        self._event_title_cache[event_ticker] = title
        return title

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _kalshi_implied_prob(market: Market) -> float | None:
        if market.yes_ask > 0:
            return float(market.yes_ask)
        if market.last_price > 0:
            return float(market.last_price)
        if market.yes_bid > 0 and market.yes_ask > 0:
            return float((market.yes_bid + market.yes_ask) / 2)
        return None
