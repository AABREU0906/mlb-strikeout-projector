"""
News/injury provider.

Honest limitation: there is no free, ToS-compliant, structured API for MLB
beat-reporter injury news. Automated classification of scraped sports-media
HTML into "confirmed/reported/inferred/speculative" would require either
(a) a licensed news API key, or (b) HTML scraping of specific sites, which
this project's rules restrict to cases with no reliable structured
alternative and require robots.txt/ToS compliance per source.

This module therefore provides:
1. `NewsApiProvider` -- works only if NEWS_API_KEY is set, pointed at a
   generic licensed headline-search API (NewsAPI.org-compatible shape).
   Returns raw headlines only; it does NOT auto-classify confidence, because
   doing so reliably needs either an LLM classification step (out of scope
   for a deterministic data provider) or a specialized injury feed.
2. `ManualWarningEntry` / `WarningLog` -- the always-available path: the
   user (who reads beat-reporter Twitter/X, MLB.com, etc. before running a
   projection) enters warnings directly, and the system stores and displays
   them with full required metadata (player, issue, source, pub date,
   confidence, effect). This satisfies the project's hard requirement of
   "never present rumors as confirmed" by making confidence an explicit,
   required field rather than an inferred one.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.data_sources.base import NewsProvider, SourcedPayload, utc_now_iso
from app.utilities.http_client import http_client

logger = get_logger(__name__)

SOURCE_NAME = "news_api"

CONFIDENCE_LEVELS = ("confirmed", "reported", "inferred", "speculative")


class Warning(BaseModel):
    player: str
    issue: str
    source: str
    published_date: Optional[str] = None
    published_time: Optional[str] = None
    confidence: str  # one of CONFIDENCE_LEVELS
    expected_effect: str
    source_reference: Optional[str] = None

    def validate_confidence(self) -> None:
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"confidence must be one of {CONFIDENCE_LEVELS}")


class NewsApiProvider(NewsProvider):
    """Optional: raw headline search only, no auto-classification."""

    def __init__(self):
        self.api_key = settings.news_api_key

    def search_player_news(self, player_name: str, since_days: int = 7) -> SourcedPayload:
        payload = SourcedPayload(source=SOURCE_NAME, retrieved_at=utc_now_iso())
        if not self.api_key:
            return payload._replace_data(
                {
                    "available": False,
                    "reason": "NEWS_API_KEY not configured -- use manual warning entry.",
                    "headlines": [],
                }
            )

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{player_name}" AND (injury OR IL OR "pitch count" OR rehab OR limited)',
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.api_key,
        }
        resp = http_client.get_json(
            url, params=params, cache_category="news",
            cache_ttl_seconds=settings.cache_ttl_news_minutes * 60,
        )
        if resp is None:
            return payload._replace_data({"available": False, "reason": "fetch_failed", "headlines": []})

        articles = resp.json_body.get("articles", [])
        headlines = [
            {
                "title": a.get("title"),
                "source": (a.get("source", {}) or {}).get("name"),
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
            }
            for a in articles[:15]
        ]
        return payload.model_copy(
            update={"retrieved_at": resp.retrieved_at, "from_cache": resp.from_cache}
        )._replace_data({"available": True, "headlines": headlines})


class WarningLog:
    """In-memory + serializable collection of warnings for one projection run."""

    def __init__(self):
        self._warnings: list[Warning] = []

    def add(self, warning: Warning) -> None:
        warning.validate_confidence()
        self._warnings.append(warning)

    def add_raw(
        self,
        player: str,
        issue: str,
        source: str,
        confidence: str,
        expected_effect: str,
        published_date: Optional[str] = None,
        published_time: Optional[str] = None,
        source_reference: Optional[str] = None,
    ) -> None:
        self.add(
            Warning(
                player=player,
                issue=issue,
                source=source,
                published_date=published_date,
                published_time=published_time,
                confidence=confidence,
                expected_effect=expected_effect,
                source_reference=source_reference,
            )
        )

    def all(self) -> list[Warning]:
        return list(self._warnings)

    def to_json(self) -> list[dict]:
        return [w.model_dump() for w in self._warnings]

    def confidence_penalty(self) -> float:
        """Aggregate, documented confidence-score deduction (0.0-1.0 scale,
        used by the confidence-rating module). Confirmed issues touching the
        starting pitcher's availability/workload matter most."""
        weights = {"confirmed": 0.35, "reported": 0.20, "inferred": 0.10, "speculative": 0.04}
        penalty = 0.0
        for w in self._warnings:
            penalty += weights.get(w.confidence, 0.05)
        return min(penalty, 0.9)
