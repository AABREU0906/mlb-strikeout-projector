"""
Central application configuration.

All environment-driven configuration lives here. Nothing else in the
application should call os.environ / os.getenv directly -- import `settings`
from this module instead. This keeps every tunable value discoverable in one
place and makes it possible to override settings in tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_path: str = "data/database/strikeout_projector.db"

    # --- Timezone ---
    default_timezone: str = "America/New_York"

    # --- Odds API ---
    odds_api_key: Optional[str] = None
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_api_region: str = "us"
    odds_api_bookmakers: str = "draftkings,fanduel,betmgm,caesars"

    # --- Weather ---
    weather_api_base_url: str = "https://api.open-meteo.com/v1/forecast"

    # --- MLB Stats API ---
    mlb_stats_api_base_url: str = "https://statsapi.mlb.com/api/v1"
    mlb_stats_api_base_url_v1_1: str = "https://statsapi.mlb.com/api/v1.1"

    # --- News ---
    news_api_key: Optional[str] = None

    # --- Caching ---
    cache_dir: str = "data/cache"
    cache_ttl_schedule_minutes: int = 60
    cache_ttl_probable_pitchers_minutes: int = 30
    cache_ttl_confirmed_lineup_minutes: int = 5
    cache_ttl_player_stats_hours: int = 12
    cache_ttl_weather_minutes: int = 20
    cache_ttl_odds_minutes: int = 5
    cache_ttl_news_minutes: int = 15

    # --- HTTP ---
    http_timeout_seconds: int = 15
    http_max_retries: int = 3
    http_user_agent: str = (
        "mlb-strikeout-projector/0.1 (personal, non-commercial research tool)"
    )

    # --- Simulation ---
    default_monte_carlo_iterations: int = 25000
    default_random_seed: Optional[int] = None

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "logs"

    # --- Modeling ---
    min_projections_for_ml_retrain: int = 150

    @property
    def database_full_path(self) -> Path:
        p = Path(self.database_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def cache_dir_path(self) -> Path:
        p = Path(self.cache_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def log_dir_path(self) -> Path:
        p = Path(self.log_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def odds_api_bookmakers_list(self) -> list[str]:
        return [b.strip() for b in self.odds_api_bookmakers.split(",") if b.strip()]


settings = Settings()

# Ensure required directories exist as soon as settings are imported.
settings.database_full_path.parent.mkdir(parents=True, exist_ok=True)
settings.cache_dir_path.mkdir(parents=True, exist_ok=True)
settings.log_dir_path.mkdir(parents=True, exist_ok=True)
