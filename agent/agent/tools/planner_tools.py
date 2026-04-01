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
"""Planner tools: datetime, itinerary builder (pandas), and PII remover."""

import json
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from langchain_core.tools import tool


@tool
def get_current_datetime(timezone: Optional[str] = "UTC") -> str:
    """Return the current date and time, optionally in a specific timezone.

    Args:
        timezone: IANA timezone string, e.g. 'Europe/Rome', 'Asia/Tokyo', 'America/New_York'.
                  Defaults to 'UTC'.

    Returns:
        Current datetime as a formatted string including timezone name.
    """
    try:
        tz = ZoneInfo(timezone or "UTC")
        now = datetime.now(tz)
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({timezone})")
    except ZoneInfoNotFoundError:
        now = datetime.utcnow()
        return now.strftime(
            "%Y-%m-%d %H:%M:%S (UTC) [invalid timezone provided, defaulted to UTC]"
        )


@tool
def build_itinerary(
    destination: str,
    num_days: int,
    activities: list[str],
    hotel_name: str,
    start_date: Optional[str] = None,
) -> str:
    """Build a structured day-by-day travel itinerary as a JSON table.

    Each day gets a morning, afternoon, and evening slot. Activities are distributed
    evenly across the days. If fewer activities are provided than needed, generic
    exploration activities are added automatically.

    Args:
        destination: Travel destination, e.g. 'Rome, Italy'.
        num_days: Number of travel days, e.g. 5.
        activities: List of activity names or descriptions to include.
        hotel_name: Name of the accommodation.
        start_date: Optional start date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        JSON string with the full day-by-day itinerary table.
    """
    from datetime import timedelta  # noqa: PLC0415

    if num_days < 1:
        return json.dumps({"error": "num_days must be at least 1"})

    try:
        base_date = (
            datetime.strptime(start_date, "%Y-%m-%d")
            if start_date
            else datetime.today()
        )
    except ValueError:
        base_date = datetime.today()

    fillers = [
        f"Explore local neighborhoods in {destination}",
        f"Visit a local market in {destination}",
        f"Enjoy cuisine and dining in {destination}",
        f"Museum or gallery visit in {destination}",
        f"Scenic walk / photography in {destination}",
        f"Day trip to nearby attraction from {destination}",
    ]

    slots_needed = num_days * 3
    padded = list(activities)
    while len(padded) < slots_needed:
        padded.extend(fillers)
    padded = padded[:slots_needed]

    rows = []
    for day in range(num_days):
        day_date = base_date + timedelta(days=day)
        rows.append(
            {
                "Day": day + 1,
                "Date": day_date.strftime("%Y-%m-%d"),
                "Hotel": hotel_name,
                "Morning": padded[day * 3],
                "Afternoon": padded[day * 3 + 1],
                "Evening": padded[day * 3 + 2],
            }
        )

    df = pd.DataFrame(rows)
    result = {
        "destination": destination,
        "total_days": num_days,
        "hotel": hotel_name,
        "itinerary": df.to_dict(orient="records"),
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# PII patterns to strip
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    # Email addresses
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "[EMAIL REMOVED]",
    ),
    # International phone numbers (+1 555 123 4567 / (555) 123-4567 / 555-123-4567)
    (re.compile(r"(\+?\d[\d\s\-().]{7,}\d)"), "[PHONE REMOVED]"),
    # Credit card numbers (16 digits, optionally separated by spaces or dashes)
    (re.compile(r"\b(?:\d[ \-]?){13,16}\b"), "[CARD REMOVED]"),
    # Passport-style codes: letter(s) + 6-9 digits
    (re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"), "[PASSPORT REMOVED]"),
    # Social security numbers (US): 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REMOVED]"),
]


@tool
def remove_pii(text: str) -> str:
    """Scan the given text and remove personally identifiable information (PII).

    Removes: email addresses, phone numbers, credit card numbers, passport codes,
    and US Social Security Numbers.

    Args:
        text: Raw text that may contain PII.

    Returns:
        Cleaned text with PII replaced by placeholder tokens.
    """
    cleaned = text
    for pattern, replacement in _PII_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned
