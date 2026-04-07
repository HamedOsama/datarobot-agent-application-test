# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Research tools: weather, country info, and flight search with real API + LLM fallbacks."""

import json
import os
from datetime import date as _date
from typing import Optional

import requests
from langchain_core.tools import tool
from langchain_litellm.chat_models import ChatLiteLLM
from pydantic import BaseModel, Field

from agent.config import Config

# ---------------------------------------------------------------------------
# Structured output model for LLM-generated weather fallback
# ---------------------------------------------------------------------------


class _DailyWeather(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format.")
    temperature_c: float = Field(description="Expected daytime temperature in Celsius.")
    feels_like_c: float = Field(description="Feels-like temperature in Celsius.")
    condition: str = Field(
        description="Short weather condition label, e.g. 'Sunny', 'Rainy'."
    )
    description: str = Field(
        description="One-sentence weather description for the day."
    )
    humidity_pct: int = Field(
        description="Expected relative humidity percentage (0-100)."
    )
    wind_kph: float = Field(description="Expected wind speed in km/h.")


class _WeatherForecast(BaseModel):
    summary: str = Field(
        description=(
            "A 3-5 sentence overview of the weather across the full date range: "
            "dominant conditions, temperature range, any notable changes day to day, "
            "and practical advice for the traveller (e.g. pack a rain jacket, "
            "expect afternoon heat)."
        )
    )
    forecast: list[_DailyWeather] = Field(
        description="One entry per day covering the requested date range."
    )


def _llm_weather_fallback(
    city: str,
    country_code: Optional[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Ask the LLM for plausible weather data when the real API is unavailable.

    Returns a dict with ``summary`` (str) and ``forecast`` (list of daily dicts).
    """
    cfg = Config()
    api_base_url = os.environ.get("LITELLM_API_BASE", "")
    api_key = os.environ.get(
        "DATAROBOT_API_TOKEN", os.environ.get("OPENAI_API_KEY", "")
    )
    model = cfg.llm_default_model

    location = f"{city}, {country_code.upper()}" if country_code else city

    if date_from and date_to:
        date_context = f"from {date_from} to {date_to} (one entry per day)"
    elif date_from:
        date_context = f"for {date_from}"
    else:
        date_context = "for today"

    prompt = (
        f"Provide a plausible day-by-day weather forecast for {location} {date_context}. "
        "Base your answer on the typical climate for that location and time of year. "
        "Return only the structured fields with no additional commentary."
    )

    _fallback_day = {
        "temperature_c": 22.0,
        "feels_like_c": 21.0,
        "condition": "Partly Cloudy",
        "description": "partly cloudy with light breeze",
        "humidity_pct": 52,
        "wind_kph": 14.0,
    }

    try:
        llm = ChatLiteLLM(
            model=model,
            api_base=api_base_url or None,
            api_key=api_key or None,
            timeout=15,
            streaming=False,
            max_retries=2,
        )
        structured_llm = llm.with_structured_output(_WeatherForecast)
        result: _WeatherForecast = structured_llm.invoke(prompt)  # type: ignore[assignment]
        return {
            "summary": result.summary,
            "forecast": [day.model_dump() for day in result.forecast],
        }
    except Exception:
        date_label = date_from or "today"
        return {
            "summary": f"Weather information for {city} is currently unavailable.",
            "forecast": [{**_fallback_day, "date": date_label}],
        }


# ---------------------------------------------------------------------------
# Mock fallback data
# ---------------------------------------------------------------------------

_MOCK_COUNTRY_BASE = {
    "capital": "Rome",
    "region": "Europe",
    "subregion": "Southern Europe",
    "population": 60_317_000,
    "languages": ["Italian"],
    "currencies": ["Euro (€)"],
    "timezones": ["UTC+01:00"],
}

_MOCK_FLIGHTS = [
    {
        "airline": "Lufthansa",
        "flight_number": "LH3401",
        "departure": "07:45",
        "arrival": "11:10",
        "duration_h": 3.4,
        "price_usd": 318,
        "stops": 0,
    },
    {
        "airline": "easyJet",
        "flight_number": "U24502",
        "departure": "13:20",
        "arrival": "17:50",
        "duration_h": 4.5,
        "price_usd": 189,
        "stops": 1,
    },
    {
        "airline": "Ryanair",
        "flight_number": "FR8871",
        "departure": "18:05",
        "arrival": "21:30",
        "duration_h": 3.4,
        "price_usd": 142,
        "stops": 0,
    },
]


# ---------------------------------------------------------------------------
# Amadeus token cache (module-level, not persisted across restarts)
# ---------------------------------------------------------------------------

_amadeus_token_cache: dict[str, str] = {}


def _get_amadeus_token(api_key: str, api_secret: str) -> str:
    cache_key = f"{api_key}:{api_secret}"
    if cache_key in _amadeus_token_cache:
        return _amadeus_token_cache[cache_key]
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    form_data = (
        f"grant_type=client_credentials&client_id={api_key}&client_secret={api_secret}"
    )
    resp = requests.post(
        "https://test.api.amadeus.com/v1/security/oauth2/token",
        data=form_data,
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    _amadeus_token_cache[cache_key] = token
    return token


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_destination_weather(
    city: str,
    country_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Fetch weather for a destination city, optionally over a date range.

    When a date range is provided the response contains a day-by-day forecast
    instead of a single current-conditions snapshot.

    Args:
        city: City name, e.g. 'Rome' or 'Tokyo'.
        country_code: Optional ISO 3166-1 alpha-2 country code, e.g. 'IT', 'JP'.
        date_from: Optional start date in YYYY-MM-DD format, e.g. '2025-06-10'.
        date_to: Optional end date in YYYY-MM-DD format, e.g. '2025-06-15'.

    Returns:
        JSON string. With no date range: a single weather snapshot. With a date
        range: an object with a ``summary`` (3-5 sentence overview of the period)
        and a ``forecast`` list with one entry per day.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    meta = {"city": city}
    if country_code:
        meta["country"] = country_code.upper()

    # ------------------------------------------------------------------ #
    # No API key → LLM fallback for the full date range                   #
    # ------------------------------------------------------------------ #
    if not api_key:
        llm_data = _llm_weather_fallback(city, country_code, date_from, date_to)
        if date_from or date_to:
            return json.dumps(
                {
                    **meta,
                    "summary": llm_data["summary"],
                    "forecast": llm_data["forecast"],
                },
                indent=2,
            )
        return json.dumps({**meta, **llm_data["forecast"][0]}, indent=2)

    query = f"{city},{country_code}" if country_code else city

    # ------------------------------------------------------------------ #
    # Date range → OWM 5-day/3-hour forecast endpoint, filtered by dates  #
    # ------------------------------------------------------------------ #
    if date_from or date_to:
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"q": query, "appid": api_key, "units": "metric", "cnt": 40},
                timeout=10,
            )
            resp.raise_for_status()
            raw_list = resp.json().get("list", [])

            start = _date.fromisoformat(date_from) if date_from else None
            end = _date.fromisoformat(date_to) if date_to else None

            # Collapse 3-hour slots into one representative entry per day
            # (pick the midday slot closest to 12:00 for each day).
            by_day: dict[str, list[dict]] = {}
            for slot in raw_list:
                slot_date = slot["dt_txt"].split(" ")[0]
                by_day.setdefault(slot_date, []).append(slot)

            forecast = []
            for day_str, slots in sorted(by_day.items()):
                day = _date.fromisoformat(day_str)
                if start and day < start:
                    continue
                if end and day > end:
                    continue
                # prefer the slot nearest to noon
                best = min(
                    slots, key=lambda s: abs(int(s["dt_txt"].split(" ")[1][:2]) - 12)
                )
                forecast.append(
                    {
                        "date": day_str,
                        "temperature_c": best["main"]["temp"],
                        "feels_like_c": best["main"]["feels_like"],
                        "condition": best["weather"][0]["main"],
                        "description": best["weather"][0]["description"],
                        "humidity_pct": best["main"]["humidity"],
                        "wind_kph": round(best["wind"]["speed"] * 3.6, 1),
                    }
                )

            if forecast:
                return json.dumps({**meta, "forecast": forecast}, indent=2)
            # OWM free tier only covers ~5 days; fall back to LLM for longer ranges
            raise ValueError("no matching forecast slots")
        except Exception:
            llm_data = _llm_weather_fallback(city, country_code, date_from, date_to)
            return json.dumps(
                {
                    **meta,
                    "summary": llm_data["summary"],
                    "forecast": llm_data["forecast"],
                },
                indent=2,
            )

    # ------------------------------------------------------------------ #
    # No date range → OWM current-conditions endpoint                     #
    # ------------------------------------------------------------------ #
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": query, "appid": api_key, "units": "metric"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        result = {
            "city": data.get("name", city),
            "country": data.get("sys", {}).get("country", ""),
            "temperature_c": data["main"]["temp"],
            "feels_like_c": data["main"]["feels_like"],
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "humidity_pct": data["main"]["humidity"],
            "wind_kph": round(data["wind"]["speed"] * 3.6, 1),
        }
        return json.dumps(result, indent=2)
    except Exception:
        llm_data = _llm_weather_fallback(city, country_code, date_from, date_to)
        return json.dumps({**meta, **llm_data["forecast"][0]}, indent=2)


@tool
def get_country_info(country_name: str) -> str:
    """Fetch general information about a country: capital, region, languages, currencies, timezone.

    Args:
        country_name: Full or partial country name, e.g. 'Italy', 'Japan'.

    Returns:
        JSON string with country facts useful for travel planning.
    """
    try:
        resp = requests.get(
            f"https://restcountries.com/v3.1/name/{country_name}",
            params={
                "fields": "name,capital,region,subregion,population,languages,currencies,timezones"
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("empty response")
        entry = data[0]
        languages = list(entry.get("languages", {}).values())
        currencies = [
            f"{v.get('name', k)} ({v.get('symbol', '')})"
            for k, v in entry.get("currencies", {}).items()
        ]
        result = {
            "name": entry["name"]["common"],
            "capital": entry.get("capital", ["Unknown"])[0],
            "region": entry.get("region", ""),
            "subregion": entry.get("subregion", ""),
            "population": entry.get("population", 0),
            "languages": languages,
            "currencies": currencies,
            "timezones": entry.get("timezones", ["UTC"]),
        }
        return json.dumps(result, indent=2)
    except Exception:
        result = {"name": country_name, **_MOCK_COUNTRY_BASE}
        return json.dumps(result, indent=2)


@tool
def search_flights(
    origin: str, destination: str, departure_date: str, adults: int = 1
) -> str:
    """Search for available flights between two airports or cities.

    Args:
        origin: IATA airport code or city name for departure, e.g. 'LHR' or 'London'.
        destination: IATA airport code or city name for arrival, e.g. 'FCO' or 'Rome'.
        departure_date: Date in YYYY-MM-DD format, e.g. '2025-06-10'.
        adults: Number of adult passengers (default 1).

    Returns:
        JSON string listing up to 3 flight options with price, airline, duration and stops.
    """
    api_key = os.environ.get("AMADEUS_API_KEY", "")
    api_secret = os.environ.get("AMADEUS_API_SECRET", "")

    if not api_key or not api_secret:
        flights = [
            {
                **f,
                "origin": origin.upper(),
                "destination": destination.upper(),
                "departure_date": departure_date,
            }
            for f in _MOCK_FLIGHTS
        ]
        return json.dumps({"flights": flights}, indent=2)

    try:
        token = _get_amadeus_token(api_key, api_secret)
        resp = requests.get(
            "https://test.api.amadeus.com/v2/shopping/flight-offers",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "originLocationCode": origin.upper(),
                "destinationLocationCode": destination.upper(),
                "departureDate": departure_date,
                "adults": adults,
                "max": 3,
                "currencyCode": "USD",
            },
            timeout=12,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        flights = []
        for offer in raw:
            seg = offer["itineraries"][0]["segments"][0]
            price = float(offer["price"]["grandTotal"])
            flights.append(
                {
                    "airline": seg["carrierCode"],
                    "flight_number": seg["carrierCode"] + seg["number"],
                    "origin": seg["departure"]["iataCode"],
                    "destination": seg["arrival"]["iataCode"],
                    "departure": seg["departure"]["at"],
                    "arrival": seg["arrival"]["at"],
                    "stops": len(offer["itineraries"][0]["segments"]) - 1,
                    "price_usd": price,
                }
            )
        return json.dumps({"flights": flights}, indent=2)
    except Exception:
        flights = [
            {
                **f,
                "origin": origin.upper(),
                "destination": destination.upper(),
                "departure_date": departure_date,
            }
            for f in _MOCK_FLIGHTS
        ]
        return json.dumps({"flights": flights}, indent=2)
