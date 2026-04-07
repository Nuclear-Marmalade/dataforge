"""
FORGE SMTP Email Verifier — Discovers emails for businesses via SMTP RCPT TO verification.

For businesses that have a website_url but no email, this script:
  1. Extracts the domain from website_url
  2. Generates candidate emails (info@, contact@, hello@, office@, sales@)
  3. Verifies each candidate via SMTP (MX lookup → RCPT TO check, no email sent)
  4. Writes the first verified email back with email_source='smtp_verified'

Uses COALESCE pattern (never overwrites existing emails).
Supports --resume via keyset pagination with checkpoint file.

Usage:
    python -m forge.importers.smtp_verifier --resume
    python -m forge.importers.smtp_verifier --limit 5000 --workers 3
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import dns.resolver

logger = logging.getLogger("forge.importers.smtp_verifier")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

CANDIDATE_PREFIXES = ["info", "contact", "hello", "office", "sales"]

CHECKPOINT_FILE = "/tmp/smtp_verify_checkpoint.txt"
BATCH_SIZE = 1000
CHECKPOINT_INTERVAL = 5000  # Save checkpoint every N records processed
SMTP_TIMEOUT = 5  # seconds
RATE_LIMIT_DELAY = 0.5  # 2 verifications per second = 0.5s between each

# Sender identity used in MAIL FROM during RCPT TO probe
PROBE_FROM = os.environ.get("FORGE_SMTP_FROM", "verify@example.com")
PROBE_EHLO = os.environ.get("FORGE_SMTP_EHLO", "localhost")

# ---------------------------------------------------------------------------
# Thread-safe caches
# ---------------------------------------------------------------------------

_mx_cache: Dict[str, Optional[List[str]]] = {}
_mx_cache_lock = threading.Lock()

_catchall_cache: Dict[str, bool] = {}
_catchall_cache_lock = threading.Lock()

# Rate limiter: global timestamp of last SMTP probe
_rate_lock = threading.Lock()
_last_probe_time: float = 0.0


def _rate_limit() -> None:
    """Block until enough time has passed since the last SMTP probe."""
    global _last_probe_time
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_probe_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        _last_probe_time = time.monotonic()


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------

def extract_domain(website_url: str) -> Optional[str]:
    """
    Extract the registrable domain from a website URL.

    Handles URLs with and without scheme, strips www prefix.
    Returns None if the URL is unparseable or clearly not a domain.
    """
    url = website_url.strip()
    if not url:
        return None

    # Ensure scheme so urlparse works
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except Exception:
        return None

    if not host:
        return None

    host = host.lower().strip(".")

    # Strip www prefix
    if host.startswith("www."):
        host = host[4:]

    # Basic sanity: must have at least one dot, no spaces
    if "." not in host or " " in host:
        return None

    # Skip IP addresses
    parts = host.split(".")
    if all(p.isdigit() for p in parts):
        return None

    # Skip common non-business domains
    skip_domains = {
        "facebook.com", "fb.com", "instagram.com", "twitter.com", "x.com",
        "linkedin.com", "youtube.com", "tiktok.com", "yelp.com",
        "google.com", "gmail.com", "yahoo.com", "outlook.com",
        "wix.com", "squarespace.com", "wordpress.com", "godaddy.com",
        "blogspot.com", "tumblr.com", "pinterest.com",
    }
    if host in skip_domains:
        return None

    return host


# ---------------------------------------------------------------------------
# MX resolution
# ---------------------------------------------------------------------------

def get_mx_hosts(domain: str) -> Optional[List[str]]:
    """
    Resolve MX records for a domain. Returns sorted list of MX hostnames
    (lowest priority first) or None if lookup fails.

    Results are cached thread-safely.
    """
    with _mx_cache_lock:
        if domain in _mx_cache:
            return _mx_cache[domain]

    mx_hosts: Optional[List[str]] = None
    try:
        answers = dns.resolver.resolve(domain, "MX", lifetime=5.0)
        records = sorted(answers, key=lambda r: r.preference)  # type: ignore[attr-defined]
        mx_hosts = [str(r.exchange).rstrip(".") for r in records]  # type: ignore[attr-defined]
        if not mx_hosts:
            mx_hosts = None
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
            dns.resolver.NoNameservers, dns.resolver.Timeout,
            dns.exception.DNSException):
        mx_hosts = None

    with _mx_cache_lock:
        _mx_cache[domain] = mx_hosts

    return mx_hosts


# ---------------------------------------------------------------------------
# SMTP RCPT TO verification
# ---------------------------------------------------------------------------

def _smtp_check(email: str, mx_host: str) -> Optional[int]:
    """
    Attempt SMTP RCPT TO check against a single MX host.

    Returns the SMTP response code (250 = accepted, 550 = rejected, etc.)
    or None on connection/timeout failure.
    """
    try:
        smtp = smtplib.SMTP(timeout=SMTP_TIMEOUT)
        smtp.connect(mx_host, 25)
        smtp.ehlo(PROBE_EHLO)

        # Try STARTTLS if available (some servers require it before RCPT TO)
        try:
            smtp.starttls()
            smtp.ehlo(PROBE_EHLO)
        except (smtplib.SMTPException, OSError):
            pass  # Not all servers support STARTTLS; that's fine

        smtp.mail(PROBE_FROM)
        code, _ = smtp.rcpt(email)
        smtp.quit()
        return code
    except smtplib.SMTPServerDisconnected:
        return None
    except smtplib.SMTPResponseException as e:
        return e.smtp_code
    except (smtplib.SMTPException, OSError, socket.timeout, socket.error):
        return None


def verify_email(email: str, mx_hosts: List[str]) -> bool:
    """
    Verify an email address against MX hosts via RCPT TO.

    Tries each MX host in priority order. Returns True if any MX host
    returns 250 for the RCPT TO command.
    """
    _rate_limit()

    for mx_host in mx_hosts[:3]:  # Try top 3 MX hosts at most
        code = _smtp_check(email, mx_host)
        if code is not None:
            if code == 250:
                return True
            elif code in (550, 551, 552, 553, 554):
                # Definitive rejection — no need to try other MX hosts
                return False
            elif code in (450, 451, 452):
                # Greylisting / temporary rejection — skip this MX, try next
                continue
            else:
                # Unknown code — try next MX
                continue
        # Connection failed — try next MX
        continue

    return False


# ---------------------------------------------------------------------------
# Catch-all detection
# ---------------------------------------------------------------------------

def is_catchall_domain(domain: str, mx_hosts: List[str]) -> bool:
    """
    Detect if a domain is a catch-all (accepts any address).

    Sends a probe to a random nonsense address. If the server returns 250,
    it's a catch-all and we can't trust any verification results.

    Results are cached thread-safely.
    """
    with _catchall_cache_lock:
        if domain in _catchall_cache:
            return _catchall_cache[domain]

    # Generate a clearly-fake address
    probe_addr = f"xq7z9k3m2w_{int(time.time())}@{domain}"

    _rate_limit()

    is_catchall = False
    for mx_host in mx_hosts[:2]:
        code = _smtp_check(probe_addr, mx_host)
        if code == 250:
            is_catchall = True
            break
        elif code is not None:
            # Got a definitive response (rejection) — not catch-all
            is_catchall = False
            break

    with _catchall_cache_lock:
        _catchall_cache[domain] = is_catchall

    return is_catchall


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def fetch_batch(db, last_id: str) -> List[Tuple[str, str]]:
    """
    Fetch a batch of businesses with website_url but no email.

    Uses keyset pagination: WHERE id > last_id ORDER BY id LIMIT batch_size.
    Returns list of (id, website_url) tuples.
    """
    ph = "%s" if db.is_postgres else "?"
    uuid_cast = f"{ph}::uuid" if db.is_postgres else ph
    rows = db.fetch_dicts(
        f"SELECT id, website_url FROM businesses "
        f"WHERE website_url IS NOT NULL AND website_url != '' "
        f"AND (email IS NULL OR email = '') "
        f"AND id > {uuid_cast} ORDER BY id LIMIT {ph}",
        (last_id, BATCH_SIZE),
    )
    return [(row["id"], row["website_url"]) for row in rows]


def write_email(db, business_id: str, email: str) -> bool:
    """
    Write a verified email to a business record using COALESCE pattern.

    Only writes if the email column is still NULL/empty (never overwrites).
    Returns True (best-effort; ForgeDB.execute doesn't return rowcount).
    """
    ph = "%s" if db.is_postgres else "?"
    now_expr = "NOW()" if db.is_postgres else "datetime('now')"
    query = (
        f"UPDATE businesses "
        f"SET email = COALESCE(NULLIF(email, ''), {ph}), "
        f"    email_source = COALESCE(NULLIF(email_source, ''), 'smtp_verified'), "
        f"    updated_at = {now_expr} "
        f"WHERE id = {ph} AND (email IS NULL OR email = '')"
    )
    db.execute(query, (email, business_id))
    return True


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> str:
    """Load the last processed business ID from checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            content = f.read().strip()
            if content and content != "0":
                return content
    except (FileNotFoundError, ValueError):
        pass
    return "00000000-0000-0000-0000-000000000000"


def save_checkpoint(last_id: str) -> None:
    """Save the current position to checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(last_id))


# ---------------------------------------------------------------------------
# Worker function for ThreadPoolExecutor
# ---------------------------------------------------------------------------

def verify_business(business_id: str, website_url: str) -> Optional[Tuple[str, str]]:
    """
    Attempt to find a valid email for a single business.

    Returns (business_id, verified_email) on success, or None on failure.
    """
    domain = extract_domain(website_url)
    if not domain:
        return None

    # Get MX records
    mx_hosts = get_mx_hosts(domain)
    if not mx_hosts:
        return None

    # Check for catch-all domain
    if is_catchall_domain(domain, mx_hosts):
        logger.debug("Catch-all domain, skipping: %s (business %d)", domain, business_id)
        return None

    # Try each candidate email prefix
    for prefix in CANDIDATE_PREFIXES:
        candidate = f"{prefix}@{domain}"
        try:
            if verify_email(candidate, mx_hosts):
                logger.debug("Verified: %s for business %d", candidate, business_id)
                return (business_id, candidate)
        except Exception as e:
            logger.debug("Error verifying %s: %s", candidate, e)
            continue

    return None


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(resume: bool = False, limit: Optional[int] = None, workers: int = 5, db_path: Optional[str] = None) -> None:
    """
    Main entry point. Processes businesses in batches with concurrent verification.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    last_id = load_checkpoint() if resume else "00000000-0000-0000-0000-000000000000"
    if resume and last_id != "00000000-0000-0000-0000-000000000000":
        logger.info("Resuming from checkpoint: business id > %s", last_id)
    else:
        logger.info("Starting from beginning")

    if limit:
        logger.info("Processing limit: %d records", limit)
    logger.info("Using %d worker threads", workers)

    db = _get_forgedb(db_path)
    logger.info("Connected to database (%s)", "PostgreSQL" if db.is_postgres else "SQLite")

    total_processed = 0
    total_found = 0
    total_written = 0
    start_time = time.time()
    last_checkpoint_count = 0

    try:
        while True:
            # Check limit
            if limit and total_processed >= limit:
                logger.info("Reached processing limit of %d records", limit)
                break

            # Fetch next batch
            batch = fetch_batch(db, last_id)
            if not batch:
                logger.info("No more records to process")
                break

            # Trim batch to stay within limit
            if limit:
                remaining = limit - total_processed
                if remaining < len(batch):
                    batch = batch[:remaining]

            batch_found = 0
            batch_written = 0

            # Process batch with thread pool
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for business_id, website_url in batch:
                    future = executor.submit(verify_business, business_id, website_url)
                    futures[future] = business_id

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            bid, verified_email = result
                            batch_found += 1
                            total_found += 1

                            # Write to database
                            try:
                                if write_email(db, bid, verified_email):
                                    batch_written += 1
                                    total_written += 1
                                    logger.debug(
                                        "Wrote email %s for business %s",
                                        verified_email, bid,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "DB error writing email for business %s: %s",
                                    bid, e,
                                )
                    except Exception as e:
                        logger.debug("Worker error for business %d: %s", futures[future], e)

            # Update pagination cursor
            last_id = batch[-1][0]  # Last business ID in this batch
            total_processed += len(batch)

            # Save checkpoint periodically
            if total_processed - last_checkpoint_count >= CHECKPOINT_INTERVAL:
                save_checkpoint(last_id)
                last_checkpoint_count = total_processed
                logger.info("Checkpoint saved at id=%s", last_id)

            # Log progress every 1000 records
            if total_processed % 1000 == 0 or total_processed == len(batch):
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d processed | %d emails found | %d written | "
                    "%.1f rec/sec | last_id=%s | MX cache=%d | catch-all cache=%d",
                    total_processed, total_found, total_written,
                    rate, last_id, len(_mx_cache), len(_catchall_cache),
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Always save final checkpoint
        save_checkpoint(last_id)
        logger.info("Final checkpoint saved at id=%s", last_id)

        elapsed = time.time() - start_time
        logger.info(
            "=== SMTP Verification Complete ===\n"
            "  Total processed: %d\n"
            "  Emails found:    %d\n"
            "  Emails written:  %d\n"
            "  MX domains cached: %d\n"
            "  Catch-all domains: %d\n"
            "  Elapsed: %.1f seconds (%.1f rec/sec)",
            total_processed, total_found, total_written,
            len(_mx_cache),
            sum(1 for v in _catchall_cache.values() if v),
            elapsed,
            total_processed / elapsed if elapsed > 0 else 0,
        )

        db.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMTP email verifier for businesses with websites but no email",
        prog="forge.importers.smtp_verifier",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (%s)" % CHECKPOINT_FILE,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent verification threads (default: 5)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite database path (default: use PostgreSQL from env vars)",
    )
    args = parser.parse_args()
    run(resume=args.resume, limit=args.limit, workers=args.workers, db_path=args.db_path)


if __name__ == "__main__":
    main()
