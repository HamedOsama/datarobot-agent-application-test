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
"""Budget tools: currency conversion (ExchangeRate-API) and cost breakdown (pandas)."""

import json
import os

import pandas as pd
import requests
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Fallback rates relative to USD (approximate mid-market, 2025)
# ---------------------------------------------------------------------------

_MOCK_RATES_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "CAD": 0.74,
    "AUD": 0.65,
    "CHF": 1.12,
    "CNY": 0.138,
    "INR": 0.012,
    "MXN": 0.058,
    "BRL": 0.19,
    "KRW": 0.00075,
    "SGD": 0.74,
    "THB": 0.028,
    "TRY": 0.031,
    "AED": 0.272,
}


def _convert_mock(amount: float, from_currency: str, to_currency: str) -> float:
    from_usd = _MOCK_RATES_TO_USD.get(from_currency.upper(), 1.0)
    to_usd = _MOCK_RATES_TO_USD.get(to_currency.upper(), 1.0)
    return round(amount * from_usd / to_usd, 2)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert a monetary amount from one currency to another.

    Uses the ExchangeRate-API when EXCHANGERATE_API_KEY is set.
    Falls back to approximate hardcoded rates if the API is unavailable.

    Args:
        amount: The numeric amount to convert, e.g. 1200.0.
        from_currency: ISO 4217 source currency code, e.g. 'USD'.
        to_currency: ISO 4217 target currency code, e.g. 'EUR'.

    Returns:
        JSON string with original amount, converted amount, exchange rate and source note.
    """
    api_key = os.environ.get("EXCHANGERATE_API_KEY", "")
    from_c = from_currency.upper()
    to_c = to_currency.upper()

    if api_key:
        try:
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_c}/{to_c}/{amount}"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            if data.get("result") == "success":
                rate = data["conversion_rate"]
                converted = data["conversion_result"]
                return json.dumps(
                    {
                        "from": f"{amount} {from_c}",
                        "to": f"{converted} {to_c}",
                        "rate": rate,
                        "source": "ExchangeRate-API (live)",
                    },
                    indent=2,
                )
        except Exception:
            pass

    converted = _convert_mock(amount, from_c, to_c)
    rate = round(converted / amount, 6) if amount else 0
    return json.dumps(
        {
            "from": f"{amount} {from_c}",
            "to": f"{converted} {to_c}",
            "rate": rate,
            "source": "mock rates (approximate mid-market 2025)",
        },
        indent=2,
    )


@tool
def calculate_budget_breakdown(
    total_budget: float,
    flight_cost: float,
    hotel_cost_per_night: float,
    num_days: int,
    activities_budget: float,
    currency: str = "USD",
) -> str:
    """Calculate a detailed travel budget breakdown across major cost categories.

    Computes per-category costs, percentage of total budget, and remaining
    contingency. Uses pandas for structured tabular output.

    Args:
        total_budget: Total trip budget in the given currency.
        flight_cost: Total round-trip flight cost.
        hotel_cost_per_night: Nightly hotel rate.
        num_days: Number of nights / travel days.
        activities_budget: Estimated total for activities, tours, and admission fees.
        currency: Currency code for display, e.g. 'USD'. Default 'USD'.

    Returns:
        JSON string with itemised breakdown table, totals, and a per-day average.
    """
    hotel_total = round(hotel_cost_per_night * num_days, 2)
    food_estimate = round(num_days * 60, 2)
    transport_local = round(num_days * 20, 2)
    misc = round(total_budget * 0.05, 2)

    categories = {
        "Flights (round-trip)": flight_cost,
        f"Hotel ({num_days} nights × {hotel_cost_per_night} {currency})": hotel_total,
        "Activities & Admissions": activities_budget,
        f"Food & Dining (~60 {currency}/day)": food_estimate,
        f"Local Transport (~20 {currency}/day)": transport_local,
        "Miscellaneous (5% buffer)": misc,
    }

    total_estimated = round(sum(categories.values()), 2)
    remaining = round(total_budget - total_estimated, 2)

    rows = [
        {
            "Category": cat,
            f"Cost ({currency})": cost,
            "% of Budget": round(cost / total_budget * 100, 1) if total_budget else 0,
        }
        for cat, cost in categories.items()
    ]
    rows.append(
        {
            "Category": "--- TOTAL ESTIMATED ---",
            f"Cost ({currency})": total_estimated,
            "% of Budget": round(total_estimated / total_budget * 100, 1)
            if total_budget
            else 0,
        }
    )
    rows.append(
        {
            "Category": "Remaining / Savings",
            f"Cost ({currency})": remaining,
            "% of Budget": round(remaining / total_budget * 100, 1)
            if total_budget
            else 0,
        }
    )

    df = pd.DataFrame(rows)
    result = {
        "currency": currency,
        "total_budget": total_budget,
        "total_estimated": total_estimated,
        "remaining": remaining,
        "per_day_average": round(total_estimated / num_days, 2) if num_days else 0,
        "within_budget": remaining >= 0,
        "breakdown": df.to_dict(orient="records"),
    }
    return json.dumps(result, indent=2)
