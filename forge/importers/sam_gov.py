"""
FORGE SAM.gov Importer — Matches federal registrant data to business records.

Queries the SAM.gov Entity Management API (v3) for registered entities and
enriches our businesses table with contact information extracted from
Points of Contact (POC) records.

SAM.gov is the System for Award Management — every business that wants to
contract with the federal government must register here.  The API exposes
registrant name, physical address, NAICS codes, and POC emails/phones.

API details:
  - Base URL: https://api.sam.gov/entity-information/v3/entities
  - Free API key from api.data.gov
  - Rate limit: 1000 requests/hour (1 req per 3.6 s)
  - Returns paginated JSON with nested registrant + POC structures

Matching strategy:
  Normalized business name + state (same normalize_name() as fcc_uls.py)

Writes matched data using COALESCE pattern:
  - email, contact_name, contact_email  (from POC fields)
  - email_source = 'sam_gov'

Usage:
    python -m forge.importers.sam_gov --api-key YOUR_KEY
    python -m forge.importers.sam_gov --state CA --limit 500
    python -m forge.importers.sam_gov --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("forge.importers.sam_gov")

SAM_API_BASE = "https://api.sam.gov/entity-information/v3/entities"
CHECKPOINT_FILE = "/tmp/sam_gov_resume_checkpoint.json"

# Rate limit: 1000 requests / hour = 1 request per 3.6 seconds
RATE_LIMIT_INTERVAL = 3.6

def _get_forgedb(db_path=None):
    """Create a ForgeDB instance from db_path (SQLite) or env vars (PostgreSQL)."""
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


# ---------------------------------------------------------------------------
# Normalization (same logic as fcc_uls.py)
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Normalize a business name for matching."""
    name = name.upper().strip()
    for suffix in [" LLC", " INC", " INC.", " CORP", " CORP.", " CO.", " CO",
                   " LTD", " LTD.", " LP", " LLP", " PC", " PLLC", " PA",
                   " DBA", " THE", ",", "."]:
        name = name.replace(suffix, "")
    return name.strip()


# ---------------------------------------------------------------------------
# SAM.gov API interaction
# ---------------------------------------------------------------------------

def _build_params(
    page: int,
    page_size: int = 100,
    state_filter: Optional[str] = None,
) -> Dict[str, str]:
    """Build query parameters for the SAM.gov entity search."""
    params: Dict[str, str] = {
        "registrationStatus": "A",       # Active registrations only
        "purposeOfRegistrationCode": "Z2",  # Federal assistance + contracts
        "includeSections": "entityRegistration,coreData,pointsOfContact",
        "page": str(page),
        "size": str(page_size),
    }
    if state_filter:
        params["stateCode"] = state_filter.upper()
    return params


def _fetch_page(
    client: httpx.Client,
    api_key: str,
    page: int,
    page_size: int = 100,
    state_filter: Optional[str] = None,
) -> Tuple[List[Dict], int]:
    """
    Fetch a single page of entities from SAM.gov.

    Returns (entities_list, total_records).
    Raises on non-200 responses after retries.
    """
    params = _build_params(page, page_size, state_filter)
    headers = {"X-Api-Key": api_key}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.get(SAM_API_BASE, params=params, headers=headers, timeout=60.0)

            if resp.status_code == 429:
                # Rate limited — back off aggressively
                wait = 30 * (attempt + 1)
                logger.warning("Rate limited (429). Sleeping %ds before retry.", wait)
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                raise RuntimeError(
                    "SAM.gov returned 403 Forbidden. Check your API key. "
                    "Register at https://api.data.gov to get a free key."
                )

            resp.raise_for_status()
            data = resp.json()

            entities = data.get("entityData", [])
            total = data.get("totalRecords", 0)
            return entities, total

        except httpx.TimeoutException:
            wait = 10 * (attempt + 1)
            logger.warning("Timeout on page %d, attempt %d/%d. Retrying in %ds.",
                           page, attempt + 1, max_retries, wait)
            time.sleep(wait)

        except httpx.HTTPStatusError as e:
            if attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                logger.warning("HTTP %d on page %d, attempt %d/%d. Retrying in %ds.",
                               e.response.status_code, page, attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed to fetch page {page} after {max_retries} retries")


# ---------------------------------------------------------------------------
# Entity parsing
# ---------------------------------------------------------------------------

def _extract_entity(entity: Dict) -> Optional[Dict[str, Any]]:
    """
    Extract relevant fields from a single SAM.gov entity record.

    Returns dict with: org_name, name_normalized, state, city, zip_code,
    poc_email, poc_name, poc_phone, naics_codes.
    Returns None if essential fields are missing.
    """
    registration = entity.get("entityRegistration", {})
    core = entity.get("coreData", {})
    pocs = entity.get("pointsOfContact", {})

    # Organization name
    org_name = registration.get("legalBusinessName", "").strip()
    if not org_name:
        return None

    # Physical address
    phys_addr = core.get("physicalAddress", {})
    state = (phys_addr.get("stateOrProvinceCode") or "").strip().upper()
    city = (phys_addr.get("city") or "").strip().upper()
    zip_code = (phys_addr.get("zipCode") or "").strip()[:5]

    if not state:
        return None

    # Points of contact — prefer government POC, fall back to electronic POC
    poc_email = None
    poc_name = None
    poc_phone = None

    for poc_key in ["governmentBusinessPOC", "electronicBusinessPOC",
                    "governmentBusinessAlternatePOC", "electronicBusinessAlternatePOC"]:
        poc = pocs.get(poc_key, {})
        if not poc:
            continue

        email = (poc.get("email") or "").strip().lower()
        if email and "@" in email and not poc_email:
            poc_email = email

        first = (poc.get("firstName") or "").strip()
        middle = (poc.get("middleInitial") or "").strip()
        last = (poc.get("lastName") or "").strip()
        full_name = " ".join(part for part in [first, middle, last] if part)
        if full_name and not poc_name:
            poc_name = full_name

        phone = (poc.get("USPhone") or "").strip()
        if phone and not poc_phone:
            poc_phone = phone

    if not poc_email:
        return None

    # NAICS codes
    naics_list = core.get("naicsCode", [])
    naics_codes = []
    if isinstance(naics_list, list):
        for entry in naics_list:
            if isinstance(entry, dict):
                code = entry.get("naicsCode", "")
                if code:
                    naics_codes.append(str(code))
            elif isinstance(entry, (str, int)):
                naics_codes.append(str(entry))

    return {
        "org_name": org_name,
        "name_normalized": normalize_name(org_name),
        "state": state,
        "city": city,
        "zip_code": zip_code,
        "poc_email": poc_email,
        "poc_name": poc_name,
        "poc_phone": poc_phone,
        "naics_codes": naics_codes,
    }


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def _build_name_state_index(db, state_filter: Optional[str] = None) -> Dict[str, Dict]:
    """
    Build an in-memory index of businesses by (normalized_name, state) -> {id, name, state}.

    Only includes businesses that currently lack email or contact_email,
    so we only attempt to fill gaps.
    """
    logger.info("Building name+state index from businesses table...")
    ph = "%s" if db.is_postgres else "?"

    if state_filter:
        rows = db.fetch_dicts(
            f"SELECT id, name, state FROM businesses "
            f"WHERE (email IS NULL OR email = '' OR contact_email IS NULL OR contact_email = '') "
            f"AND state = {ph} ORDER BY id",
            (state_filter.upper(),),
        )
    else:
        rows = db.fetch_dicts(
            "SELECT id, name, state FROM businesses "
            "WHERE (email IS NULL OR email = '' OR contact_email IS NULL OR contact_email = '') "
            "ORDER BY id",
        )

    index: Dict[str, Dict] = {}
    count = 0
    for row in rows:
        name = row.get("name") or ""
        state = (row.get("state") or "").upper()
        if not name or not state:
            continue
        key = f"{normalize_name(name)}|{state}"
        if key not in index:
            index[key] = {
                "id": str(row["id"]),
                "name": name,
                "state": state,
            }
        count += 1

    logger.info("Name+state index built: %d unique keys from %d rows", len(index), count)
    return index


def _flush_updates(db, batch: List[Tuple], stats: Dict[str, int]) -> None:
    """
    Flush a batch of matched updates to the DB.

    Each tuple: (email, contact_name, contact_email, business_id)
    Uses COALESCE so existing non-null values are preserved.

    Args:
        db: ForgeDB instance.
    """
    if not batch:
        return

    ph = "%s" if db.is_postgres else "?"
    uuid_cast = f"{ph}::uuid" if db.is_postgres else ph
    now_expr = "NOW()" if db.is_postgres else "datetime('now')"

    try:
        with db.transaction() as tx:
            for email, contact_name, contact_email, biz_id in batch:
                query = (
                    f"UPDATE businesses "
                    f"SET email         = COALESCE(NULLIF(email, ''), {ph}), "
                    f"    contact_name  = COALESCE(NULLIF(contact_name, ''), {ph}), "
                    f"    contact_email = COALESCE(NULLIF(contact_email, ''), {ph}), "
                    f"    email_source  = COALESCE(NULLIF(email_source, ''), 'sam_gov'), "
                    f"    updated_at    = {now_expr} "
                    f"WHERE id = {uuid_cast} "
                    f"  AND (email IS NULL OR email = '' "
                    f"       OR contact_email IS NULL OR contact_email = '')"
                )
                tx.execute(query, (email, contact_name, contact_email, biz_id))
                stats["rows_updated"] += 1
    except Exception as e:
        stats["db_errors"] += 1
        logger.error("Batch flush failed: %s", e)


# ---------------------------------------------------------------------------
# Checkpoint (resume support)
# ---------------------------------------------------------------------------

def _save_checkpoint(page: int, api_calls: int, stats: Dict[str, int]) -> None:
    """Save resume checkpoint to disk."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "page": page,
            "api_calls": api_calls,
            "stats": stats,
            "timestamp": time.time(),
        }, f)


def _load_checkpoint() -> Optional[Dict]:
    """Load resume checkpoint from disk. Expires after 48 hours."""
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
            age_hours = (time.time() - data.get("timestamp", 0)) / 3600
            if age_hours > 48:
                logger.info("Checkpoint is %.1f hours old, starting fresh", age_hours)
                return None
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _clear_checkpoint() -> None:
    """Remove checkpoint file on successful completion."""
    try:
        os.remove(CHECKPOINT_FILE)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Main import logic
# ---------------------------------------------------------------------------

def import_sam_gov(
    api_key: str,
    state_filter: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = False,
    page_size: int = 100,
    flush_every: int = 50,
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    Import SAM.gov entity contact data into the businesses table.

    Uses ForgeDB, supporting both SQLite (via db_path) and PostgreSQL (via env vars).

    Args:
        api_key:      SAM.gov API key (from api.data.gov)
        state_filter:  Two-letter state code to restrict query (e.g. "CA")
        limit:        Maximum number of API pages to fetch (None = all)
        resume:       If True, resume from last checkpoint
        page_size:    Entities per API page (max 100)
        flush_every:  Flush DB writes after this many matched records
        db_path:      SQLite database path (default: use PostgreSQL from env vars)

    Returns:
        Dictionary of import statistics.
    """
    stats: Dict[str, int] = {
        "api_calls": 0,
        "entities_fetched": 0,
        "entities_with_email": 0,
        "matches_found": 0,
        "rows_updated": 0,
        "already_had_data": 0,
        "db_errors": 0,
    }

    # Resume from checkpoint if requested
    start_page = 0
    if resume:
        checkpoint = _load_checkpoint()
        if checkpoint:
            start_page = checkpoint["page"] + 1
            stats.update(checkpoint.get("stats", {}))
            logger.info(
                "Resuming from checkpoint: page=%d, api_calls=%d, rows_updated=%d",
                start_page, stats["api_calls"], stats["rows_updated"],
            )

    # Connect to DB via ForgeDB
    db = _get_forgedb(db_path)

    # Build in-memory index of businesses needing email enrichment
    name_state_index = _build_name_state_index(db, state_filter)
    if not name_state_index:
        logger.warning("No businesses without email found%s. Nothing to do.",
                       f" in {state_filter}" if state_filter else "")
        db.close()
        return stats

    # Track rate limiting
    last_request_time = 0.0
    update_batch: List[Tuple] = []

    client = httpx.Client(
        follow_redirects=True,
        headers={"Accept": "application/json"},
    )

    page = start_page
    total_records = None
    pages_fetched = 0

    try:
        while True:
            # Enforce rate limit: max 1 request per 3.6 seconds
            elapsed = time.time() - last_request_time
            if elapsed < RATE_LIMIT_INTERVAL:
                sleep_time = RATE_LIMIT_INTERVAL - elapsed
                time.sleep(sleep_time)

            # Fetch page from SAM.gov
            last_request_time = time.time()
            try:
                entities, total = _fetch_page(
                    client, api_key, page, page_size, state_filter,
                )
            except Exception as e:
                logger.error("Failed to fetch page %d: %s", page, e)
                _save_checkpoint(page - 1, stats["api_calls"], stats)
                raise

            stats["api_calls"] += 1
            pages_fetched += 1

            if total_records is None:
                total_records = total
                total_pages = (total_records + page_size - 1) // page_size if total_records > 0 else 0
                logger.info(
                    "SAM.gov query: %d total records, ~%d pages (size=%d)%s",
                    total_records, total_pages, page_size,
                    f", state={state_filter}" if state_filter else "",
                )

            if not entities:
                logger.info("No more entities returned at page %d. Done.", page)
                break

            stats["entities_fetched"] += len(entities)

            # Process each entity
            for raw_entity in entities:
                parsed = _extract_entity(raw_entity)
                if not parsed:
                    continue

                stats["entities_with_email"] += 1

                # Match against our businesses index
                key = f"{parsed['name_normalized']}|{parsed['state']}"
                biz = name_state_index.get(key)
                if not biz:
                    continue

                stats["matches_found"] += 1
                update_batch.append((
                    parsed["poc_email"],
                    parsed["poc_name"],
                    parsed["poc_email"],
                    biz["id"],
                ))

                # Remove from index so we don't double-write
                del name_state_index[key]

            # Flush DB writes periodically
            if len(update_batch) >= flush_every:
                _flush_updates(db, update_batch, stats)
                update_batch = []

            # Log progress every 100 API calls
            if stats["api_calls"] % 100 == 0:
                logger.info(
                    "Progress: page=%d, api_calls=%d, fetched=%d, "
                    "with_email=%d, matches=%d, written=%d",
                    page, stats["api_calls"], stats["entities_fetched"],
                    stats["entities_with_email"], stats["matches_found"],
                    stats["rows_updated"],
                )

            # Save checkpoint every 100 API calls
            if stats["api_calls"] % 100 == 0:
                _save_checkpoint(page, stats["api_calls"], stats)

            # Respect --limit flag (pages, not entities)
            if limit is not None and pages_fetched >= limit:
                logger.info("Reached page limit (%d). Stopping.", limit)
                break

            # Check if we've exhausted all pages
            if total_records is not None:
                last_possible_page = (total_records - 1) // page_size
                if page >= last_possible_page:
                    logger.info("Reached last page (%d). Done.", page)
                    break

            page += 1

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving checkpoint at page %d.", page)
        _save_checkpoint(page, stats["api_calls"], stats)
    except Exception as e:
        logger.error("Error during import: %s. Saving checkpoint.", e)
        _save_checkpoint(page, stats["api_calls"], stats)
        raise
    finally:
        # Flush any remaining updates
        if update_batch:
            try:
                _flush_updates(db, update_batch, stats)
            except Exception as e:
                logger.error("Final flush failed: %s", e)

        client.close()

    # Clean up checkpoint on successful completion (no exceptions)
    _clear_checkpoint()
    db.close()

    logger.info("SAM.gov import complete: %s", json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Import SAM.gov entity contact data into FORGE businesses table",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SAM_GOV_API_KEY"),
        help="SAM.gov API key (or set SAM_GOV_API_KEY env var). "
             "Register free at https://api.data.gov",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Filter to a single state (two-letter code, e.g. CA)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of API pages to fetch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite database path (default: use PostgreSQL from env vars)",
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "API key required. Pass --api-key or set SAM_GOV_API_KEY env var. "
            "Register free at https://api.data.gov"
        )

    stats = import_sam_gov(
        api_key=args.api_key,
        state_filter=args.state,
        limit=args.limit,
        resume=args.resume,
        db_path=args.db_path,
    )

    print(f"\n{'=' * 50}")
    print("SAM.GOV IMPORT RESULTS")
    print(f"{'=' * 50}")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
