"""Location check shared by the registry importers.

A name match alone does not tie a registry record to a business: chains and common
names repeat across a state, so a name-only match can attach another location's
contact. Name matches must also pass same_location().
"""

from __future__ import annotations

import re
from typing import Optional


def zip5(value: Optional[str]) -> str:
    """Return the 5-digit ZIP, or '' when the value has none."""
    digits = re.sub(r"\D", "", value or "")
    return digits[:5] if len(digits) >= 5 else ""


def city_key(value: Optional[str]) -> str:
    """Upper-case a city name and collapse punctuation and spacing."""
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", (value or "").upper()).split())


def same_location(
    zip_a: Optional[str],
    city_a: Optional[str],
    zip_b: Optional[str],
    city_b: Optional[str],
) -> bool:
    """True when two addresses are confirmed to be the same place.

    Compares 5-digit ZIPs when both sides have one, otherwise the city.
    A missing location never confirms a match.
    """
    za, zb = zip5(zip_a), zip5(zip_b)
    if za and zb:
        return za == zb
    ca, cb = city_key(city_a), city_key(city_b)
    return bool(ca) and ca == cb
