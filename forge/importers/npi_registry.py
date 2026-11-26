"""NPI Registry importer. Enriches healthcare businesses with NPI numbers and specialties.

Queries the free NPPES API (no key required), matches by phone or name+location.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from forge.importers.location import same_location

logger = logging.getLogger("forge.importers.npi_registry")

NPI_API_URL = "https://npiregistry.cms.hhs.gov/api/"

# Map NPI taxonomy to our industry categories
TAXONOMY_TO_INDUSTRY = {
    "dentist": "dentist",
    "dental": "dentist",
    "orthodont": "dentist",
    "periodon": "dentist",
    "endodont": "dentist",
    "chiropract": "chiropractor",
    "veterinar": "veterinarian",
    "optometr": None,  # not in our whitelist
    "physical therap": "personal-trainer",  # closest match
    "massage": None,
    "salon": "salon",
    "barber": "barber",
    "cosmetol": "salon",
}


def normalize_phone(phone: str) -> Optional[str]:
    """Strip phone to 10 digits."""
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def classify_taxonomy(taxonomy_desc: Optional[str]) -> Optional[str]:
    """Map NPI taxonomy description to our industry category."""
    if not taxonomy_desc:
        return None
    desc_lower = taxonomy_desc.lower()
    for keyword, industry in TAXONOMY_TO_INDUSTRY.items():
        if keyword in desc_lower:
            return industry
    return None


def _official_part(basic: Dict, field: str) -> str:
    """Read one authorized_official_* field; NPPES uses '--' for blank."""
    value = (basic.get(f"authorized_official_{field}") or "").strip()
    return "" if value == "--" else value


def extract_authorized_official(basic: Dict) -> Optional[Dict[str, Optional[str]]]:
    """Return the named person NPPES lists as an organization's authorized official.

    Every NPI-2 (organization) record carries one: name, title and phone.
    Returns None when the record has no first and last name.
    """
    first = _official_part(basic, "first_name")
    last = _official_part(basic, "last_name")
    if not first or not last:
        return None
    middle = _official_part(basic, "middle_name")
    return {
        "name": " ".join(p for p in (first, middle, last) if p),
        "title": _official_part(basic, "title_or_position") or None,
        "phone": normalize_phone(_official_part(basic, "telephone_number")),
    }


def _parse_npi_result(r: Dict) -> Dict:
    """Parse a single NPI API result into a provider dict."""
    basic = r.get("basic", {})
    addresses = r.get("addresses", [])
    taxonomies = r.get("taxonomies", [])

    practice_addr = None
    for addr in addresses:
        if addr.get("address_purpose") == "LOCATION":
            practice_addr = addr
            break
    if not practice_addr and addresses:
        practice_addr = addresses[0]

    primary_tax = ""
    for tax in taxonomies:
        if tax.get("primary"):
            primary_tax = tax.get("desc", "")
            break

    phone = (
        normalize_phone((practice_addr or {}).get("telephone_number", ""))
        if practice_addr
        else None
    )
    official = extract_authorized_official(basic) or {}
    return {
        "npi": str(r.get("number", "")),
        "org_name": basic.get("organization_name", ""),
        "phone": phone,
        "city": (practice_addr or {}).get("city", "").upper(),
        "state": (practice_addr or {}).get("state", "").upper(),
        "zip": (practice_addr or {}).get("postal_code", "")[:5],
        "taxonomy": primary_tax,
        "industry": classify_taxonomy(primary_tax),
        "official_name": official.get("name"),
        "official_title": official.get("title"),
        "official_phone": official.get("phone"),
    }


def query_npi_api(state: str, taxonomy: str = "", skip: int = 0, limit: int = 200) -> List[Dict]:
    """Query the NPI Registry API for providers in a state."""
    params: Dict[str, Any] = {
        "version": "2.1",
        "enumeration_type": "NPI-2",
        "state": state,
        "limit": limit,
        "skip": skip,
    }
    if taxonomy:
        params["taxonomy_description"] = taxonomy

    try:
        resp = httpx.get(NPI_API_URL, params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return [_parse_npi_result(r) for r in data.get("results", [])]
    except Exception as e:
        logger.warning("NPI API error for state=%s, skip=%d: %s", state, skip, e)
        return []


def _get_forgedb(db_path=None):
    """Create a ForgeDB instance from db_path (SQLite) or env vars (PostgreSQL)."""
    import os

    from forge.db import ForgeDB

    if db_path:
        db_config = {"db_path": db_path}
    else:
        db_host = os.environ.get("FORGE_DB_HOST", "")
        db_password = os.environ.get("FORGE_DB_PASSWORD", "")
        if not db_host or not db_password:
            raise ValueError(
                "Database credentials required. Either pass --db-path for SQLite "
                "or set FORGE_DB_HOST and FORGE_DB_PASSWORD environment variables."
            )
        db_config = {
            "db_host": db_host,
            "db_port": int(os.environ.get("FORGE_DB_PORT", "5432")),
            "db_user": os.environ.get("FORGE_DB_USER", ""),
            "db_password": db_password,
            "db_name": os.environ.get("FORGE_DB_NAME", "forge"),
        }

    db = ForgeDB.from_config(db_config)
    db.ensure_schema()
    return db


def _fetch_healthcare_businesses(db, state: str, ph: str) -> List[Dict]:
    """Fetch healthcare businesses needing NPI enrichment for a state."""
    healthcare_keywords = [
        "%dent%",
        "%chiropract%",
        "%veterinar%",
        "%doctor%",
        "%medical%",
        "%health%",
        "%clinic%",
        "%therap%",
        "%optom%",
        "%pharm%",
        "%physical_therap%",
    ]

    if db.is_postgres:
        return db.fetch_dicts(
            f"SELECT id, name, phone, city, state, zip, sub_industry "
            f"FROM businesses "
            f"WHERE state = {ph} "
            f"AND (npi_number IS NULL OR npi_number = '') "
            f"AND sub_industry IS NOT NULL "
            f"AND sub_industry ILIKE ANY(ARRAY[{','.join([ph] * len(healthcare_keywords))}]) "
            f"LIMIT 5000",
            (state, *healthcare_keywords),
        )

    like_clauses = " OR ".join([f"sub_industry LIKE {ph}" for _ in healthcare_keywords])
    return db.fetch_dicts(
        f"SELECT id, name, phone, city, state, zip, sub_industry "
        f"FROM businesses "
        f"WHERE state = {ph} "
        f"AND (npi_number IS NULL OR npi_number = '') "
        f"AND sub_industry IS NOT NULL "
        f"AND ({like_clauses}) "
        f"LIMIT 5000",
        (state, *healthcare_keywords),
    )


def _lookup_npi(name: str, state: str) -> Optional[Dict]:
    """Perform a single NPI API lookup by organization name + state.

    Returns the parsed JSON response or None on failure.
    """
    try:
        resp = httpx.get(
            NPI_API_URL,
            params={
                "version": "2.1",
                "enumeration_type": "NPI-2",
                "organization_name": name,
                "state": state,
                "limit": 5,
            },
            timeout=15.0,
        )
        return resp.json()
    except Exception:
        return None


def _match_npi_results(
    results: List[Dict],
    phone: Optional[str],
    name: str,
    stats: Dict[str, int],
    zip_code: Optional[str] = None,
    city: Optional[str] = None,
) -> tuple:
    """Match NPI API results against a business by phone, or by name at the same location.

    A name match must also share the ZIP (or the city when a ZIP is missing). Chains and
    common names repeat across a state, and a name-only match would save another
    branch's authorized official as this business's contact.

    Returns (matched_npi, matched_industry, authorized_official) or (None, None, None).
    """
    for r in results:
        basic = r.get("basic", {})
        org_name = basic.get("organization_name", "")
        taxonomies = r.get("taxonomies", [])
        location: Dict[str, Any] = next(
            (a for a in r.get("addresses", []) if a.get("address_purpose") == "LOCATION"), {}
        )
        npi_phone = normalize_phone(location.get("telephone_number", ""))

        matched_npi = None
        if phone and npi_phone and phone == npi_phone:
            matched_npi = str(r.get("number", ""))
            stats["phone_matches"] += 1
        elif org_name.upper() == name.upper() and same_location(
            zip_code, city, location.get("postal_code"), location.get("city")
        ):
            matched_npi = str(r.get("number", ""))
            stats["name_matches"] += 1

        if matched_npi:
            matched_industry = None
            for tax in taxonomies:
                if tax.get("primary"):
                    matched_industry = classify_taxonomy(tax.get("desc", ""))
                    break
            return matched_npi, matched_industry, extract_authorized_official(basic)

    return None, None, None


def _write_npi_match(
    db,
    biz_id: str,
    matched_npi: str,
    matched_industry: Optional[str],
    stats: Dict[str, int],
    ph: str,
    official: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """Write matched NPI data to the database."""
    try:
        now_expr = "NOW()" if db.is_postgres else "datetime('now')"
        uuid_cast = f"{ph}::uuid" if db.is_postgres else ph
        updates = [f"npi_number = COALESCE(npi_number, {ph})"]
        params_list: List[Any] = [matched_npi]
        stats["npi_written"] += 1

        if matched_industry:
            updates.append(f"industry = COALESCE(industry, {ph})")
            params_list.append(matched_industry)
            stats["industry_written"] += 1

        contact_written = False
        if official:
            # Write the person as one unit, and only when the row has no contact yet,
            # so an existing contact never ends up with this official's title or phone.
            # SET expressions read the pre-update row, so contact_name here is the old value.
            no_contact = "COALESCE(contact_name, '') = ''"
            for col, value in (
                ("contact_name", official["name"]),
                ("contact_title", official["title"]),
                ("contact_phone", official["phone"]),
            ):
                updates.append(f"{col} = CASE WHEN {no_contact} THEN {ph} ELSE {col} END")
                params_list.append(value)
            contact_written = not (db.get_business(biz_id) or {}).get("contact_name")

        updates.append(f"updated_at = {now_expr}")
        query = f"UPDATE businesses SET {', '.join(updates)} WHERE id = {uuid_cast}"
        params_list.append(biz_id)
        with db.transaction() as tx:
            tx.execute(query, tuple(params_list))
        if contact_written:
            stats["contact_written"] += 1
    except Exception as e:  # Non-critical: count the failure, keep importing
        logger.warning("NPI write failed for business %s: %s", biz_id, e)
        stats["errors"] += 1


def import_npi_for_state(
    state: str,
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    """Import NPI data for a single state by looking up our healthcare businesses."""
    db = _get_forgedb(db_path)
    ph = "%s" if db.is_postgres else "?"

    stats = {
        "our_healthcare_businesses": 0,
        "npi_lookups": 0,
        "phone_matches": 0,
        "name_matches": 0,
        "npi_written": 0,
        "industry_written": 0,
        "contact_written": 0,
        "errors": 0,
    }

    businesses = _fetch_healthcare_businesses(db, state, ph)
    stats["our_healthcare_businesses"] = len(businesses)
    logger.info("NPI %s: found %d healthcare businesses to look up", state, len(businesses))

    for biz in businesses:
        phone = normalize_phone(biz.get("phone", "") or "")
        name = (biz.get("name", "") or "").strip()
        if not name:
            continue

        data = _lookup_npi(name, state)
        if data is None:
            stats["errors"] += 1
            time.sleep(1)
            continue
        stats["npi_lookups"] += 1

        results = data.get("results", [])
        matched_npi, matched_industry, official = _match_npi_results(
            results, phone, name, stats, zip_code=biz.get("zip"), city=biz.get("city")
        )

        if matched_npi:
            _write_npi_match(db, biz["id"], matched_npi, matched_industry, stats, ph, official)

        time.sleep(0.3)

    db.close()
    return stats


def import_npi_all_states(
    states: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    """Import NPI data for all states."""
    if states is None:
        states = [
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
            "DC",
        ]

    totals: Dict[str, int] = {}

    for state in states:
        logger.info("Processing NPI for %s...", state)
        state_stats = import_npi_for_state(state, db_path=db_path)
        for k, v in state_stats.items():
            totals[k] = totals.get(k, 0) + v
        logger.info("  %s done: %s", state, json.dumps(state_stats))

    return totals


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Import NPI Registry data into FORGE")
    parser.add_argument("--state", type=str, help="Single state to import (e.g., CA)")
    parser.add_argument("--all-states", action="store_true", help="Import all 51 states")
    parser.add_argument(
        "--limit-states", type=int, default=None, help="Limit to first N states (for testing)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite database path (default: use PostgreSQL from env vars)",
    )
    args = parser.parse_args()

    if args.state:
        stats = import_npi_for_state(args.state, db_path=args.db_path)
    elif args.all_states or args.limit_states:
        states = None
        if args.limit_states:
            all_states = [
                "CA",
                "TX",
                "FL",
                "NY",
                "PA",
                "IL",
                "OH",
                "GA",
                "NC",
                "MI",
                "NJ",
                "VA",
                "WA",
                "AZ",
                "MA",
                "TN",
                "IN",
                "MO",
                "MD",
                "WI",
            ]
            states = all_states[: args.limit_states]
        stats = import_npi_all_states(states, db_path=args.db_path)
    else:
        print("Specify --state CA, --all-states, or --limit-states 5")
        exit(1)

    print(f"\n{'=' * 50}")
    print("NPI REGISTRY IMPORT RESULTS")
    print(f"{'=' * 50}")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
