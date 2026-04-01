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
"""Research tools: weather, country info, and flight search with real API + mock fallbacks."""

import json
import os
from typing import Optional

import requests
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Mock fallback data
# ---------------------------------------------------------------------------

_MOCK_WEATHER_BASE = {
    "temperature_c": 22,
    "feels_like_c": 21,
    "condition": "Partly Cloudy",
    "description": "partly cloudy with light breeze",
    "humidity_pct": 52,
    "wind_kph": 14,
}

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
def get_destination_weather(city: str, country_code: Optional[str] = None) -> str:
    """Fetch current weather for a destination city.

    Args:
        city: City name, e.g. 'Rome' or 'Tokyo'.
        country_code: Optional ISO 3166-1 alpha-2 country code, e.g. 'IT', 'JP'.

    Returns:
        JSON string with temperature, condition, humidity and wind information.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        result = {"city": city, **_MOCK_WEATHER_BASE}
        if country_code:
            result["country"] = country_code.upper()
        return json.dumps(result, indent=2)

    query = f"{city},{country_code}" if country_code else city
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
        result = {"city": city, **_MOCK_WEATHER_BASE}
        if country_code:
            result["country"] = country_code.upper()
        return json.dumps(result, indent=2)


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
