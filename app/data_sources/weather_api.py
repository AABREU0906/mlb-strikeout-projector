"""
Weather provider using Open-Meteo (https://open-meteo.com) - free, keyless,
no rate-limit auth required for non-commercial use. Also holds the static
ballpark reference table (location, altitude, roof type, historical K/run
factors) since weather adjustments need the park's coordinates and factors
need to be looked up from *somewhere* documented rather than invented ad hoc.

Ballpark factors below are intentionally modest and documented as a fixed
seed table (see BALLPARK_FACTOR_NOTES). They are indexed off MLB venue_id so
the schedule/game data (which already carries venue_id) can join directly.
"""
from __future__ import annotations

from typing import Optional

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.data_sources.base import SourcedPayload, WeatherProvider, utc_now_iso
from app.utilities.http_client import http_client

logger = get_logger(__name__)

SOURCE_NAME = "open_meteo"

BALLPARK_FACTOR_NOTES = (
    "Ballpark K/run factors are seeded from publicly reported multi-year "
    "park-factor summaries and should be refreshed periodically. They are "
    "deliberately compressed toward 1.00 (modest effect) rather than taken "
    "at face value, per project rules against exaggerating environmental "
    "effects on strikeouts."
)

# venue_id -> reference data. Populate/extend as needed; unknown venues fall
# back to neutral factors (1.00) rather than a guessed value.
BALLPARK_REFERENCE: dict[int, dict] = {
    19: {
        "name": "Coors Field", "latitude": 39.7559, "longitude": -104.9942,
        "altitude_ft": 5200, "roof_type": "open", "k_factor": 0.94, "run_factor": 1.12,
    },
    15: {
        "name": "Chase Field", "latitude": 33.4455, "longitude": -112.0667,
        "altitude_ft": 1100, "roof_type": "retractable", "k_factor": 1.00, "run_factor": 1.01,
    },
    2394: {
        "name": "Oracle Park", "latitude": 37.7786, "longitude": -122.3893,
        "altitude_ft": 0, "roof_type": "open", "k_factor": 1.03, "run_factor": 0.94,
    },
    2392: {
        "name": "Daikin Park", "latitude": 29.7573, "longitude": -95.3555,
        "altitude_ft": 22, "roof_type": "retractable", "k_factor": 1.01, "run_factor": 1.02,
    },
    12: {
        "name": "Tropicana Field", "latitude": 27.7683, "longitude": -82.6534,
        "altitude_ft": 10, "roof_type": "fixed_dome", "k_factor": 1.04, "run_factor": 0.97,
    },
}

NEUTRAL_BALLPARK = {
    "name": "Unknown/Unmapped Park", "latitude": None, "longitude": None,
    "altitude_ft": None, "roof_type": "unknown", "k_factor": 1.00, "run_factor": 1.00,
}


def get_ballpark_reference(venue_id: Optional[int]) -> dict:
    if venue_id is None:
        return dict(NEUTRAL_BALLPARK)
    ref = BALLPARK_REFERENCE.get(venue_id)
    if ref is None:
        logger.info("No ballpark reference row for venue_id=%s; using neutral factors.", venue_id)
        return dict(NEUTRAL_BALLPARK)
    return dict(ref)


class OpenMeteoWeatherProvider(WeatherProvider):
    def __init__(self):
        self.base_url = settings.weather_api_base_url

    def get_game_weather(self, latitude: float, longitude: float, game_datetime_utc: str) -> SourcedPayload:
        if latitude is None or longitude is None:
            return SourcedPayload(source=SOURCE_NAME, retrieved_at=utc_now_iso())._replace_data(
                {"available": False, "reason": "unknown_ballpark_coordinates"}
            )

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(
                [
                    "temperature_2m", "relative_humidity_2m", "precipitation_probability",
                    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
                ]
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
            "forecast_days": 3,
        }
        resp = http_client.get_json(
            self.base_url, params=params, cache_category="weather",
            cache_ttl_seconds=settings.cache_ttl_weather_minutes * 60,
        )
        payload = SourcedPayload(
            source=SOURCE_NAME,
            retrieved_at=utc_now_iso() if resp is None else resp.retrieved_at,
            from_cache=False if resp is None else resp.from_cache,
        )
        if resp is None:
            return payload._replace_data({"available": False, "reason": "fetch_failed"})

        hourly = resp.json_body.get("hourly", {})
        times = hourly.get("time", [])
        target = game_datetime_utc[:13]
        idx = None
        for i, t in enumerate(times):
            if t.startswith(target):
                idx = i
                break
        if idx is None and times:
            idx = min(range(len(times)), key=lambda i: abs(i - len(times) // 2))
        if idx is None:
            return payload._replace_data({"available": False, "reason": "no_matching_hour"})

        def _at(key):
            arr = hourly.get(key, [])
            return arr[idx] if idx < len(arr) else None

        return payload._replace_data(
            {
                "available": True,
                "temperature_f": _at("temperature_2m"),
                "humidity_pct": _at("relative_humidity_2m"),
                "precipitation_probability_pct": _at("precipitation_probability"),
                "wind_speed_mph": _at("wind_speed_10m"),
                "wind_direction_deg": _at("wind_direction_10m"),
                "surface_pressure_hpa": _at("surface_pressure"),
                "matched_hour_utc": times[idx] if idx < len(times) else None,
            }
        )
