"""
FORGE FCC ULS Importer — Matches FCC licensee emails to business records.

Downloads EN.dat files from FCC ULS complete dumps, extracts business entities
with email addresses, and matches them to our 10.5M business records by:
  1. Phone number (10-digit exact match — highest confidence)
  2. Business name + state (normalized fuzzy match — medium confidence)

Writes matched emails to Railway PostgreSQL using COALESCE pattern.
Supports resume from checkpoint (--resume flag).

Usage:
    python -m forge.importers.fcc_uls --data-dir forge/data/fcc --resume
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("forge.importers.fcc_uls")

# FCC EN.dat column positions (0-indexed)
COL_ENTITY_TYPE = 5      # L=Licensee, CL=Contact, etc.
COL_ENTITY_NAME = 7      # Business/organization name
COL_PHONE = 12           # 10-digit phone
COL_EMAIL = 14           # Email address
COL_STREET = 15          # Street address
COL_CITY = 16            # City
COL_STATE = 17           # State (2-letter)
COL_ZIP = 18             # ZIP code
COL_APPLICANT_TYPE = 23  # C=Corp, F=LLC, I=Individual, etc.

# Business entity types (exclude individuals and government)
BUSINESS_TYPES = {"C", "D", "E", "F", "J", "L", "N", "O", "P", "T"}

CHECKPOINT_FILE = "/tmp/fcc_resume_checkpoint.txt"


def normalize_phone(phone: str) -> Optional[str]:
    """Strip a phone number to 10 digits."""
    if not phone:
        return None
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    if len(digits) == 10:
        return digits
    return None


def normalize_name(name: str) -> str:
    """Normalize a business name for matching."""
    name = name.upper().strip()
    # Remove common suffixes
    for suffix in [" LLC", " INC", " INC.", " CORP", " CORP.", " CO.", " CO",
                   " LTD", " LTD.", " LP", " LLP", " PC", " PLLC", " PA",
                   " DBA", " THE", ",", "."]:
        name = name.replace(suffix, "")
    return name.strip()


def parse_en_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse an FCC EN.dat pipe-delimited file.

    Returns list of dicts with: name, phone, email, city, state, zip.
    Filters to business entities (not individuals) with email addresses.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            fields = line.strip().split('|')
            if len(fields) < 24:
                continue

            entity_type = fields[COL_ENTITY_TYPE].strip()
            applicant_type = fields[COL_APPLICANT_TYPE].strip()
            email = fields[COL_EMAIL].strip()

            # Only licensee records that are businesses with email
            if entity_type != "L":
                continue
            if applicant_type not in BUSINESS_TYPES:
                continue
            if not email or "@" not in email:
                continue

            phone = normalize_phone(fields[COL_PHONE].strip())
            name = fields[COL_ENTITY_NAME].strip()

            if not name:
                continue

            records.append({
                "name": name,
                "name_normalized": normalize_name(name),
                "phone": phone,
                "email": email.lower(),
                "city": fields[COL_CITY].strip().upper(),
                "state": fields[COL_STATE].strip().upper(),
                "zip": fields[COL_ZIP].strip()[:5],
            })

    return records


def build_phone_index(records: List[Dict]) -> Dict[str, Dict]:
    """Build a phone -> record index for fast matching."""
    index = {}
    for rec in records:
        if rec["phone"]:
            if rec["phone"] not in index:
                index[rec["phone"]] = rec
    return index


def build_name_state_index(records: List[Dict]) -> Dict[str, Dict]:
    """Build a (normalized_name, state) -> record index."""
    index = {}
    for rec in records:
        key = f"{rec['name_normalized']}|{rec['state']}"
        if key not in index:
            index[key] = rec
    return index


def _save_checkpoint(last_id: str, total_checked: int, stats: Dict[str, int]) -> None:
    """Save resume checkpoint to disk."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            "last_id": last_id,
            "total_checked": total_checked,
            "stats": stats,
            "timestamp": time.time(),
        }, f)


def _load_checkpoint() -> Optional[Dict]:
    """Load resume checkpoint from disk."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            age_hours = (time.time() - data.get("timestamp", 0)) / 3600
            if age_hours > 48:
                logger.info("Checkpoint is %.1f hours old, starting fresh", age_hours)
                return None
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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


def _flush_updates(db, update_batch: List[tuple], stats: Dict[str, int]) -> None:
    """
    Flush a batch of updates to the DB in a single transaction.

    Each tuple: (email, source_tag, business_id, match_type)
    Uses COALESCE — won't overwrite existing non-null emails.

    Args:
        db: ForgeDB instance.
    """
    if not update_batch:
        return

    try:
        with db.transaction() as tx:
            ph = tx.placeholder
            uuid_cast = f"{ph}::uuid" if db.is_postgres else ph
            now_expr = "NOW()" if db.is_postgres else "datetime('now')"
            for email, source_tag, biz_id, match_type in update_batch:
                tx.execute(
                    f"UPDATE businesses "
                    f"SET email = COALESCE(email, {ph}), "
                    f"    email_source = COALESCE(email_source, {ph}), "
                    f"    updated_at = {now_expr} "
                    f"WHERE id = {uuid_cast} AND (email IS NULL OR email = '')",
                    (email, source_tag, biz_id),
                )
            # auto-commits on clean exit

        # Count matches after successful commit
        for _email, _source_tag, _biz_id, match_type in update_batch:
            if match_type == "phone":
                stats["phone_matches"] += 1
            else:
                stats["name_matches"] += 1
            stats["emails_written"] += 1

    except Exception as e:
        stats["errors"] += len(update_batch)
        logger.error("Batch flush failed: %s", e)


def _parse_all_en_files(data_dir: str) -> List[Dict]:
    """Parse all FCC EN.dat files from a directory tree."""
    logger.info("Parsing FCC EN.dat files from %s", data_dir)
    all_records: List[Dict] = []
    data_path = Path(data_dir)
    for en_file in data_path.rglob("EN.dat"):
        logger.info("Parsing %s", en_file)
        records = parse_en_file(str(en_file))
        all_records.extend(records)
        logger.info("  -> %d business records with email", len(records))
    logger.info("Total FCC records with email: %d", len(all_records))
    return all_records


def _match_row_to_indexes(
    row: Dict,
    phone_index: Dict[str, Dict],
    name_index: Dict[str, Dict],
    update_batch: List[tuple],
) -> None:
    """Match a single business row against FCC phone and name indexes."""
    biz_phone = normalize_phone(row["phone"] or "")
    biz_name_norm = normalize_name(row["name"] or "")
    biz_state = (row["state"] or "").upper()

    matched_email = None
    match_type = None

    if biz_phone and biz_phone in phone_index:
        matched_email = phone_index[biz_phone]["email"]
        match_type = "phone"

    if not matched_email:
        key = f"{biz_name_norm}|{biz_state}"
        if key in name_index:
            matched_email = name_index[key]["email"]
            match_type = "name_state"

    if matched_email:
        update_batch.append((
            matched_email,
            f"fcc_uls_{match_type}",
            str(row["id"]),
            match_type,
        ))


def _fetch_unenriched_page(db, last_id: Optional[str], fetch_size: int, ph: str, uuid_cast: str) -> list:
    """Fetch a page of businesses without email for FCC matching."""
    if last_id is None:
        return db.fetch_dicts(
            f"SELECT id, name, phone, city, state FROM businesses "
            f"WHERE (email IS NULL OR email = '') ORDER BY id LIMIT {ph}",
            (fetch_size,),
        )
    return db.fetch_dicts(
        f"SELECT id, name, phone, city, state FROM businesses "
        f"WHERE id > {uuid_cast} AND (email IS NULL OR email = '') ORDER BY id LIMIT {ph}",
        (last_id, fetch_size),
    )


def _scan_and_match(
    db, phone_index: Dict[str, Dict], name_index: Dict[str, Dict],
    stats: Dict[str, int], last_id: Optional[str], total_checked: int,
) -> None:
    """Scan businesses via keyset pagination and match against FCC indexes."""
    fetch_size = 5000
    update_batch: List[tuple] = []
    consecutive_errors = 0
    ph = "%s" if db.is_postgres else "?"
    uuid_cast = f"{ph}::uuid" if db.is_postgres else ph

    while True:
        try:
            rows = _fetch_unenriched_page(db, last_id, fetch_size, ph, uuid_cast)
            if not rows:
                break
            consecutive_errors = 0
            last_id = rows[-1]["id"]
            total_checked += len(rows)
            for row in rows:
                _match_row_to_indexes(row, phone_index, name_index, update_batch)
            if len(update_batch) >= 500:
                _flush_updates(db, update_batch, stats)
                update_batch = []
            if total_checked % 50000 == 0:
                _save_checkpoint(str(last_id), total_checked, stats)
                logger.info("Progress: checked %d, matched %d", total_checked, stats["emails_written"])
        except Exception as e:
            consecutive_errors += 1
            logger.error("DB error (attempt %d/10): %s", consecutive_errors, e)
            if consecutive_errors >= 10:
                _save_checkpoint(str(last_id) if last_id else "", total_checked, stats)
                raise
            if last_id:
                _save_checkpoint(str(last_id), total_checked, stats)
            time.sleep(min(2 ** consecutive_errors * 5, 120))

    if update_batch:
        _flush_updates(db, update_batch, stats)


def import_fcc_to_db(
    data_dir: str,
    db_path: Optional[str] = None,
    batch_size: int = 500,
    resume: bool = False,
) -> Dict[str, int]:
    """Import FCC ULS email data into the businesses table.

    Returns dict with match statistics.
    """
    db = _get_forgedb(db_path)

    stats = {
        "fcc_records_parsed": 0, "phone_matches": 0, "name_matches": 0,
        "emails_written": 0, "already_had_email": 0, "errors": 0,
    }

    last_id = None
    total_checked = 0
    if resume:
        checkpoint = _load_checkpoint()
        if checkpoint:
            last_id = checkpoint["last_id"]
            total_checked = checkpoint["total_checked"]
            stats.update(checkpoint.get("stats", {}))
            logger.info("Resuming from checkpoint: last_id=%s, checked=%d, emails=%d",
                        last_id, total_checked, stats["emails_written"])

    all_records = _parse_all_en_files(data_dir)
    stats["fcc_records_parsed"] = len(all_records)
    if not all_records:
        logger.warning("No FCC records found")
        return stats

    phone_index = build_phone_index(all_records)
    name_index = build_name_state_index(all_records)
    logger.info("Phone index: %d entries, Name+State index: %d entries",
                len(phone_index), len(name_index))

    _scan_and_match(db, phone_index, name_index, stats, last_id, total_checked)

    try:
        os.remove(CHECKPOINT_FILE)
    except FileNotFoundError:
        pass

    db.close()
    logger.info("FCC ULS import complete: %s", json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Import FCC ULS emails into FORGE")
    parser.add_argument("--data-dir", default="forge/data/fcc",
                        help="Directory containing extracted FCC EN.dat files")
    parser.add_argument("--db-path", type=str, default=None,
                        help="SQLite database path (default: use PostgreSQL from env vars)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    stats = import_fcc_to_db(args.data_dir, db_path=args.db_path, resume=args.resume)
    print(f"\n{'='*50}")
    print("FCC ULS IMPORT RESULTS")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
