"""
Shared HTTP client used by every data provider.

Responsibilities (per project data-source rules):
- Descriptive User-Agent on every request.
- Respect robots.txt for any host we scrape HTML from (not required for
  JSON APIs we're authorized to call, but we check anyway for HTML fetches).
- File-based response caching with per-category TTLs.
- Bounded retries with backoff; never raises on exhausted retries -- callers
  get None and must handle missing data explicitly (no silent fabrication).
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.logging_config import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

_robots_cache: dict[str, robotparser.RobotFileParser] = {}


def _cache_key(url: str, params: Optional[dict]) -> str:
    raw = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(category: str, key: str) -> Path:
    d = settings.cache_dir_path / category
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def is_allowed_by_robots(url: str, user_agent: str) -> bool:
    """Check robots.txt for the given URL's host. Fails OPEN only for JSON
    API hosts we already have explicit authorization to call would be wrong
    to assume, so on any robots.txt fetch failure we fail CLOSED (deny) for
    HTML scraping call sites, and the caller should treat False as "do not
    scrape this page"."""
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in _robots_cache:
        rp = robotparser.RobotFileParser()
        rp.set_url(root + "/robots.txt")
        try:
            rp.read()
        except Exception:
            logger.warning("Could not read robots.txt for %s; denying scrape.", root)
            return False
        _robots_cache[root] = rp
    return _robots_cache[root].can_fetch(user_agent, url)


@dataclass
class CachedResponse:
    status_code: int
    json_body: Any
    from_cache: bool
    retrieved_at: str


class HttpClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.http_user_agent})

    @retry(
        stop=stop_after_attempt(settings.http_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get(self, url: str, params: Optional[dict], headers: Optional[dict]) -> requests.Response:
        return self.session.get(
            url, params=params, headers=headers, timeout=settings.http_timeout_seconds
        )

    def get_json(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        cache_category: str = "misc",
        cache_ttl_seconds: int = 0,
        require_robots_check: bool = False,
    ) -> Optional[CachedResponse]:
        """GET a JSON endpoint with caching. Returns None on any failure
        (network error, non-200, invalid JSON) rather than raising, so
        calling code can apply documented fallback behavior."""

        if require_robots_check and not is_allowed_by_robots(url, settings.http_user_agent):
            logger.warning("robots.txt disallows fetching %s -- skipping.", url)
            return None

        key = _cache_key(url, params)
        path = _cache_path(cache_category, key)

        if cache_ttl_seconds > 0 and path.exists():
            age = time.time() - path.stat().st_mtime
            if age < cache_ttl_seconds:
                try:
                    payload = json.loads(path.read_text())
                    return CachedResponse(
                        status_code=200,
                        json_body=payload["body"],
                        from_cache=True,
                        retrieved_at=payload["retrieved_at"],
                    )
                except Exception:
                    pass  # fall through to a live fetch on any cache corruption

        try:
            resp = self._get(url, params, headers)
        except Exception as exc:
            logger.warning("HTTP GET failed for %s: %s", url, exc)
            return None

        if resp.status_code != 200:
            logger.warning("HTTP GET %s returned status %s", url, resp.status_code)
            return None

        try:
            body = resp.json()
        except Exception as exc:
            logger.warning("Failed to parse JSON from %s: %s", url, exc)
            return None

        retrieved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if cache_ttl_seconds > 0:
            try:
                path.write_text(json.dumps({"body": body, "retrieved_at": retrieved_at}))
            except Exception as exc:
                logger.debug("Failed to write cache file %s: %s", path, exc)

        return CachedResponse(
            status_code=200, json_body=body, from_cache=False, retrieved_at=retrieved_at
        )


http_client = HttpClient()
