"""Deterministic parsing and deep links for common flight-search requests."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, Optional
from urllib.parse import urlencode

MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"),
        start=1,
    )
}

CITY_CODES = {
    "mumbai": ("Mumbai", "bom"),
    "bombay": ("Mumbai", "bom"),
    "bengaluru": ("Bengaluru", "blr"),
    "bengulru": ("Bengaluru", "blr"),
    "bengluru": ("Bengaluru", "blr"),
    "bangalore": ("Bengaluru", "blr"),
    "delhi": ("Delhi", "del"),
    "new delhi": ("Delhi", "del"),
    "chennai": ("Chennai", "maa"),
    "madras": ("Chennai", "maa"),
    "hyderabad": ("Hyderabad", "hyd"),
    "kolkata": ("Kolkata", "ccu"),
    "calcutta": ("Kolkata", "ccu"),
    "pune": ("Pune", "pnq"),
    "ahmedabad": ("Ahmedabad", "amd"),
    "goa": ("Goa", "goi"),
    "kochi": ("Kochi", "cok"),
    "cochin": ("Kochi", "cok"),
    "jaipur": ("Jaipur", "jai"),
    "lucknow": ("Lucknow", "lko"),
    "varanasi": ("Varanasi", "vns"),
    "amritsar": ("Amritsar", "atq"),
    "indore": ("Indore", "idr"),
    "bhopal": ("Bhopal", "bho"),
    "patna": ("Patna", "pat"),
    "chandigarh": ("Chandigarh", "ixc"),
    "surat": ("Surat", "stv"),
    "nagpur": ("Nagpur", "nag"),
    "visakhapatnam": ("Visakhapatnam", "vtz"),
    "vizag": ("Visakhapatnam", "vtz"),
}

MONTH_PATTERN = "|".join(MONTHS)
DATE_PATTERNS = (
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
    re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PATTERN})\s*,?\s*(\d{{4}})?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})?\b",
        re.IGNORECASE,
    ),
)
ROUTE_RE = re.compile(
    r"\bfrom\s+(.+?)\s+to\s+(.+?)"
    r"(?=\s+(?:for|on|depart(?:ing)?|leav(?:e|ing)|and\s+(?:book|buy|reserve|find|search)|"
    rf"round[ -]?trip|return)\b|\s+\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH_PATTERN})\b|$)",
    re.IGNORECASE,
)
SITE_PATTERNS = (
    ("skyscanner", re.compile(r"\bskyscanner\b", re.IGNORECASE)),
    ("google_flights", re.compile(r"\bgoogle\s+flights?\b", re.IGNORECASE)),
    ("kayak", re.compile(r"\bkayak\b", re.IGNORECASE)),
    ("expedia", re.compile(r"\bexpedia\b", re.IGNORECASE)),
    ("makemytrip", re.compile(r"\bmake\s*my\s*trip\b", re.IGNORECASE)),
    ("cleartrip", re.compile(r"\bcleartrip\b", re.IGNORECASE)),
    ("ixigo", re.compile(r"\bixigo\b", re.IGNORECASE)),
)
SITE_URLS = {
    "skyscanner": "https://www.skyscanner.co.in/flights",
    "google_flights": "https://www.google.com/travel/flights",
    "kayak": "https://www.kayak.co.in/flights",
    "expedia": "https://www.expedia.co.in/Flights",
    "makemytrip": "https://www.makemytrip.com/flights/",
    "cleartrip": "https://www.cleartrip.com/flights",
    "ixigo": "https://www.ixigo.com/flights",
}


def _date_from_request(text: str) -> Optional[date]:
    for index, pattern in enumerate(DATE_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        try:
            if index == 0:
                year, month, day = (int(value) for value in match.groups())
            elif index == 1:
                day = int(match.group(1))
                month = MONTHS[match.group(2).lower()]
                year = int(match.group(3)) if match.group(3) else None
            else:
                month = MONTHS[match.group(1).lower()]
                day = int(match.group(2))
                year = int(match.group(3)) if match.group(3) else None

            today = date.today()
            year = year or today.year
            resolved = date(year, month, day)
            has_explicit_year = bool(match.group(3)) if index in (1, 2) else True
            if not has_explicit_year:
                if resolved < today:
                    resolved = date(year + 1, month, day)
            return resolved
        except (ValueError, TypeError):
            return None
    return None


def _return_date_from_request(text: str) -> Optional[date]:
    marker = re.search(r"\b(?:return(?:ing)?|back on|return date)\b", text, re.IGNORECASE)
    if not marker:
        return None
    return _date_from_request(text[marker.end():])


def _city(value: str) -> tuple[str, str]:
    cleaned = re.sub(r"\b(?:airport|city)\b", "", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    alias = CITY_CODES.get(cleaned.lower())
    if alias:
        return alias
    if re.fullmatch(r"[A-Za-z]{3}", cleaned):
        return cleaned.upper(), cleaned.lower()
    return cleaned, ""


def parse_flight_request(
    text: str, site_override: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Extract an itinerary and build an official Skyscanner day-view URL."""
    lowered = (text or "").lower()
    site = site_override or next(
        (name for name, pattern in SITE_PATTERNS if pattern.search(text or "")),
        "google_flights",
    )
    route = ROUTE_RE.search(text or "")
    if not route:
        return None

    origin, origin_code = _city(route.group(1))
    destination, destination_code = _city(route.group(2))
    flight_date = _date_from_request(text or "")
    return_date = _return_date_from_request(text or "")
    if not origin or not destination:
        return None

    cheapest = bool(re.search(r"\b(?:cheapest|lowest|least expensive)\b", lowered))
    round_trip = bool(return_date) or bool(
        re.search(r"\b(?:round[ -]?trip|return flight|and back)\b", lowered)
    )
    plan: Dict[str, Any] = {
        "platform": site,
        "origin": origin,
        "origin_code": origin_code,
        "destination": destination,
        "destination_code": destination_code,
        "travel_date": flight_date.isoformat() if flight_date else "",
        "return_date": return_date.isoformat() if return_date else "",
        "trip_type": "round_trip" if round_trip else "one_way",
        "sort": "cheapest" if cheapest else "best",
        "action": "travel",
    }
    if site == "skyscanner" and origin_code and destination_code and flight_date:
        params = {
            "origin": origin_code,
            "destination": destination_code,
            "outboundDate": flight_date.isoformat(),
            "adultsv2": "1",
            "cabinclass": "economy",
            "rtn": "1" if round_trip else "0",
            "sortby": plan["sort"],
            "market": "IN",
            "locale": "en-IN",
            "currency": "INR",
        }
        if return_date:
            params["inboundDate"] = return_date.isoformat()
        plan["url"] = (
            "https://www.skyscanner.co.in/g/referrals/v1/flights/day-view/?"
            + urlencode(params)
        )
    else:
        plan["url"] = SITE_URLS.get(site, f"https://www.{site}.com")
    plan["query"] = f"{origin} to {destination}" + (
        f" on {plan['travel_date']}" if plan["travel_date"] else ""
    )
    return plan
