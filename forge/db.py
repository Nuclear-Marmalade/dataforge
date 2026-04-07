"""
FORGE Database Abstraction Layer — The foundation of the entire system.

Two backends, one interface:
  - SQLite: Zero-config default. Ships with Python. No server needed.
  - PostgreSQL: Optional scale backend. Connection pooling via psycopg2.

Auto-detection:
  - Config has 'db_path' → SQLite
  - Config has 'db_host' → PostgreSQL

Every module in FORGE imports from this file. It handles:
  - Schema creation and migration (ensure_schema)
  - COALESCE writes (never overwrite non-null data)
  - Batch operations (single-transaction upserts)
  - CSV import/export with column auto-detection
  - Keyset pagination for enrichment fetching
  - Thread-safe access for both backends
  - Stats and reporting

Dependencies: sqlite3 (stdlib), psycopg2-binary (optional)
Depended on by: every other FORGE module
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger("forge.db")


# ── Column mapping for CSV auto-detection ────────────────────────────────────

COLUMN_ALIASES: Dict[str, str] = {}

_ALIAS_MAP: Dict[str, List[str]] = {
    "name": ["Business Name", "Company", "Name", "company_name", "business_name", "CompanyName"],
    "website_url": ["Website", "URL", "website_url", "Web", "WebsiteURL", "web_url", "website"],
    "email": ["Email", "email_address", "Contact Email", "EmailAddress", "contact_email", "e-mail"],
    "phone": ["Phone", "phone_number", "Telephone", "Phone Number", "Tel", "phone_num"],
    "city": ["City", "city"],
    "state": ["State", "state", "ST", "Province"],
    "zip": ["ZIP", "Zip Code", "zip", "Postal Code", "ZipCode", "zip_code", "postal_code"],
    "address_line1": ["Address", "Street", "address_line1", "Street Address", "StreetAddress", "address"],
    "industry": ["Industry", "Category", "industry", "Business Category", "business_category"],
    "dba_name": ["DBA", "DBA Name", "dba_name", "Doing Business As"],
    "county": ["County", "county"],
    "contact_name": ["Contact Name", "contact_name", "Contact", "ContactName"],
    "contact_email": ["Contact Email", "contact_email", "ContactEmail"],
    "contact_phone": ["Contact Phone", "contact_phone", "ContactPhone"],
    "year_established": ["Year Established", "year_established", "Founded", "YearEstablished"],
    "employee_estimate": ["Employees", "Employee Estimate", "employee_estimate", "EmployeeCount"],
    "npi_number": ["NPI", "NPI Number", "npi_number", "NPI_Number"],
    "sub_industry": ["Sub Industry", "sub_industry", "SubCategory", "Subcategory"],
    "business_type": ["Business Type", "business_type", "Type", "EntityType"],
    "latitude": ["Latitude", "latitude", "lat", "Lat"],
    "longitude": ["Longitude", "longitude", "lon", "lng", "Long"],
}

# Build the flat lookup: lowered alias → canonical column
for canonical, aliases in _ALIAS_MAP.items():
    for alias in aliases:
        COLUMN_ALIASES[alias.lower().strip()] = canonical


# ── Schema definitions ───────────────────────────────────────────────────────

# All columns in the businesses table, with their types per backend.
# Format: (column_name, sqlite_type, pg_type, default_sqlite, default_pg)
BUSINESS_COLUMNS: List[Tuple[str, str, str, str, str]] = [
    ("id",                  "TEXT PRIMARY KEY", "UUID PRIMARY KEY DEFAULT gen_random_uuid()", "", ""),
    ("name",                "TEXT", "TEXT", "", ""),
    ("dba_name",            "TEXT", "TEXT", "", ""),
    ("phone",               "TEXT", "TEXT", "", ""),
    ("email",               "TEXT", "TEXT", "", ""),
    ("website_url",         "TEXT", "TEXT", "", ""),
    ("address_line1",       "TEXT", "TEXT", "", ""),
    ("city",                "TEXT", "TEXT", "", ""),
    ("state",               "TEXT", "TEXT", "", ""),
    ("zip",                 "TEXT", "TEXT", "", ""),
    ("county",              "TEXT", "TEXT", "", ""),
    ("latitude",            "REAL", "FLOAT", "", ""),
    ("longitude",           "REAL", "FLOAT", "", ""),
    ("industry",            "TEXT", "TEXT", "", ""),
    ("sub_industry",        "TEXT", "TEXT", "", ""),
    ("business_type",       "TEXT", "TEXT", "", ""),
    ("employee_estimate",   "TEXT", "TEXT", "", ""),
    ("year_established",    "INTEGER", "INTEGER", "", ""),
    ("ai_summary",          "TEXT", "TEXT", "", ""),
    ("pain_points",         "TEXT", "JSONB", "", ""),
    ("opportunities",       "TEXT", "JSONB", "", ""),
    ("health_score",        "INTEGER", "INTEGER", "", ""),
    ("tech_stack",          "TEXT", "JSONB", "", ""),
    ("cms_detected",        "TEXT", "TEXT", "", ""),
    ("ssl_valid",           "INTEGER", "BOOLEAN", "", ""),  # SQLite has no native BOOLEAN
    ("site_speed_ms",       "INTEGER", "INTEGER", "", ""),
    ("has_booking",         "INTEGER", "BOOLEAN", "", ""),
    ("has_chat",            "INTEGER", "BOOLEAN", "", ""),
    ("npi_number",          "TEXT", "TEXT", "", ""),
    ("email_source",        "TEXT", "TEXT", "", ""),
    ("contact_name",        "TEXT", "TEXT", "", ""),
    ("contact_email",       "TEXT", "TEXT", "", ""),
    ("contact_phone",       "TEXT", "TEXT", "", ""),
    ("all_emails",          "TEXT", "JSONB", "", ""),
    ("last_enriched_at",    "TEXT", "TIMESTAMP WITH TIME ZONE", "", ""),
    ("enrichment_attempts", "INTEGER DEFAULT 0", "INTEGER DEFAULT 0", "", ""),
    ("created_at",          "TEXT DEFAULT (datetime('now'))", "TIMESTAMP WITH TIME ZONE DEFAULT NOW()", "", ""),
    ("updated_at",          "TEXT DEFAULT (datetime('now'))", "TIMESTAMP WITH TIME ZONE DEFAULT NOW()", "", ""),
]

# Indexes to create on the businesses table.
BUSINESS_INDEXES: List[Tuple[str, str]] = [
    ("idx_businesses_email",       "email"),
    ("idx_businesses_website_url", "website_url"),
    ("idx_businesses_state",       "state"),
    ("idx_businesses_industry",    "industry"),
    ("idx_businesses_zip",         "zip"),
    ("idx_businesses_phone",       "phone"),
]

# Fields allowed for enrichment writes (prevents SQL injection via field names).
ENRICHABLE_FIELDS = {
    "name", "dba_name", "phone", "email", "website_url",
    "address_line1", "city", "state", "zip", "county",
    "latitude", "longitude",
    "industry", "sub_industry", "business_type",
    "employee_estimate", "year_established",
    "ai_summary", "pain_points", "opportunities", "health_score",
    "tech_stack", "cms_detected", "ssl_valid", "site_speed_ms",
    "has_booking", "has_chat",
    "npi_number", "email_source",
    "contact_name", "contact_email", "contact_phone", "all_emails",
}

# JSON columns (stored as TEXT in SQLite, JSONB in PostgreSQL).
JSON_COLUMNS = {"pain_points", "opportunities", "tech_stack", "all_emails"}

# Boolean columns (stored as INTEGER 0/1 in SQLite, BOOLEAN in PostgreSQL).
BOOLEAN_COLUMNS = {"ssl_valid", "has_booking", "has_chat"}


# ── Backend: SQLite ──────────────────────────────────────────────────────────

class _SQLiteBackend:
    """
    SQLite backend using a single connection with a threading.Lock for writes.

    Thread-safe for concurrent reads and serialized writes.
    Uses check_same_thread=False so multiple threads can share the connection.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._write_lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info("SQLite backend connected: %s", db_path)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the shared connection (all threads share one connection)."""
        yield self._conn

    @contextmanager
    def write_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the shared connection under the write lock."""
        with self._write_lock:
            yield self._conn

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
        logger.info("SQLite connection closed: %s", self._db_path)

    @property
    def is_postgres(self) -> bool:
        return False

    def placeholder(self, index: int = 0) -> str:
        """Return the parameter placeholder for SQLite."""
        return "?"

    def now_expr(self) -> str:
        """Return the SQL expression for current timestamp."""
        return "datetime('now')"

    def uuid_default(self) -> str:
        """SQLite has no server-side UUID generation; handled in Python."""
        return str(uuid.uuid4())

    def json_cast(self, column: str) -> str:
        """No cast needed for SQLite — JSON columns are TEXT."""
        return column

    def uuid_cast(self, _placeholder: str) -> str:
        """No cast needed for SQLite — UUIDs are TEXT."""
        return _placeholder


# ── Backend: PostgreSQL ──────────────────────────────────────────────────────

class _PostgresBackend:
    """
    PostgreSQL backend using psycopg2's ThreadedConnectionPool.

    Thread-safe via the pool: each thread gets its own connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        dbname: str,
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        import psycopg2
        import psycopg2.pool
        import psycopg2.extras

        self._pool = psycopg2.pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=30,
        )
        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        logger.info(
            "PostgreSQL backend connected: %s:%d/%s (pool %d-%d)",
            host, port, dbname, min_connections, max_connections,
        )

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Get a connection from the pool, return it when done."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def write_connection(self) -> Generator[Any, None, None]:
        """Same as connection() — PostgreSQL pool handles concurrency."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()
        logger.info("PostgreSQL pool closed")

    @property
    def is_postgres(self) -> bool:
        return True

    def placeholder(self, index: int = 0) -> str:
        """Return the parameter placeholder for PostgreSQL."""
        return "%s"

    def now_expr(self) -> str:
        """Return the SQL expression for current timestamp."""
        return "NOW()"

    def uuid_default(self) -> str:
        """PostgreSQL generates UUIDs server-side; return None to let gen_random_uuid() work."""
        return ""

    def json_cast(self, column: str) -> str:
        """Cast to JSONB for PostgreSQL."""
        return f"{column}::jsonb"

    def uuid_cast(self, placeholder: str) -> str:
        """Cast placeholder to UUID for PostgreSQL."""
        return f"{placeholder}::uuid"


# ── Main Interface ───────────────────────────────────────────────────────────

class ForgeDB:
    """
    Unified database interface for FORGE.

    Supports SQLite (default, zero-config) and PostgreSQL (optional, for scale).
    Every other FORGE module uses this class for all database operations.

    Usage:
        # SQLite (default):
        db = ForgeDB.from_config({"db_path": "forge.db"})

        # PostgreSQL:
        db = ForgeDB.from_config({
            "db_host": "localhost",
            "db_port": 5432,
            "db_user": "forge",
            "db_password": "secret",
            "db_name": "forge",
        })

        db.ensure_schema()
        bid = db.upsert_business({"name": "Acme Corp", "city": "Tampa", "state": "FL"})
        db.write_enrichment(bid, {"email": "info@acme.com"}, source="scraper")
        stats = db.get_stats()
        db.close()
    """

    def __init__(self, backend: Any):
        """
        Initialize ForgeDB with a backend instance.

        Use ForgeDB.from_config() instead of calling this directly.

        Args:
            backend: A _SQLiteBackend or _PostgresBackend instance.
        """
        self._backend = backend
        self._in_transaction = threading.local()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ForgeDB":
        """
        Create a ForgeDB instance from a configuration dict.

        Auto-detects the backend:
          - If config has 'db_path' → SQLite
          - If config has 'db_host' → PostgreSQL

        Args:
            config: Dict with connection parameters.
                SQLite: {"db_path": "/path/to/forge.db"}
                PostgreSQL: {"db_host": "...", "db_port": 5432, "db_user": "...",
                             "db_password": "...", "db_name": "forge"}

        Returns:
            ForgeDB instance ready for use.

        Raises:
            ValueError: If config has neither db_path nor db_host.
        """
        if "db_path" in config:
            backend = _SQLiteBackend(db_path=config["db_path"])
            return cls(backend)
        elif "db_host" in config:
            backend = _PostgresBackend(
                host=config["db_host"],
                port=int(config.get("db_port", 5432)),
                user=config.get("db_user", "forge"),
                password=config.get("db_password", ""),
                dbname=config.get("db_name", "forge"),
                min_connections=int(config.get("min_connections", 2)),
                max_connections=int(config.get("max_connections", 10)),
            )
            return cls(backend)
        else:
            raise ValueError(
                "Config must have 'db_path' (SQLite) or 'db_host' (PostgreSQL). "
                f"Got keys: {list(config.keys())}"
            )

    @property
    def is_postgres(self) -> bool:
        """Return True if using PostgreSQL backend."""
        return self._backend.is_postgres

    # Safe WHERE filters that can be used by callers
    SAFE_WHERE_FILTERS = {
        "all": None,
        "with_email": "email IS NOT NULL AND email != ''",
        "with_tech": "tech_stack IS NOT NULL",
        "enriched": "last_enriched_at IS NOT NULL",
        "with_website": "website_url IS NOT NULL AND website_url != ''",
        "with_npi": "npi_number IS NOT NULL",
        "with_ai": "ai_summary IS NOT NULL",
    }

    def _resolve_where(self, where: Optional[str]) -> Optional[str]:
        """Resolve a where filter to safe SQL. Only accepts predefined filter names."""
        if not where:
            return None
        if where in self.SAFE_WHERE_FILTERS:
            return self.SAFE_WHERE_FILTERS[where]
        # Reject unknown filters — no raw SQL allowed
        logger.warning("Rejected unknown WHERE filter: %s", where[:50])
        return None

    # ── Schema Management ────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """
        Create the businesses table and indexes if they don't exist.

        Safe to call repeatedly — uses IF NOT EXISTS for all DDL.
        Handles dialect differences between SQLite and PostgreSQL automatically.
        """
        if self.is_postgres:
            self._ensure_schema_pg()
        else:
            self._ensure_schema_sqlite()
        logger.info("Schema ensured (%s)", "PostgreSQL" if self.is_postgres else "SQLite")

    def _ensure_schema_sqlite(self) -> None:
        """Create SQLite schema."""
        cols = []
        for col_name, sqlite_type, _, _, _ in BUSINESS_COLUMNS:
            cols.append(f"    {col_name} {sqlite_type}")

        ddl = "CREATE TABLE IF NOT EXISTS businesses (\n" + ",\n".join(cols) + "\n)"

        with self._backend.write_connection() as conn:
            conn.execute(ddl)
            for idx_name, idx_col in BUSINESS_INDEXES:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON businesses ({idx_col})"
                )
            conn.commit()

    def _ensure_schema_pg(self) -> None:
        """Create PostgreSQL schema."""
        cols = []
        for col_name, _, pg_type, _, _ in BUSINESS_COLUMNS:
            cols.append(f"    {col_name} {pg_type}")

        ddl = "CREATE TABLE IF NOT EXISTS businesses (\n" + ",\n".join(cols) + "\n)"

        with self._backend.write_connection() as conn:
            cur = conn.cursor()
            # Ensure uuid-ossp extension for gen_random_uuid (pgcrypto provides it too)
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            cur.execute(ddl)
            for idx_name, idx_col in BUSINESS_INDEXES:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON businesses ({idx_col})"
                )
            conn.commit()
            cur.close()

    # ── Single Record Operations ─────────────────────────────────────────────

    def upsert_business(self, data: Dict[str, Any]) -> str:
        """
        Insert or update a business record.

        If data contains an 'id', attempts to update the existing record.
        Otherwise, creates a new record with a generated UUID.

        Uses COALESCE logic on update: new values only fill NULLs, never
        overwrite existing non-null values.

        Args:
            data: Dict of column_name → value. Unknown columns are ignored.

        Returns:
            The business ID (UUID string).
        """
        # Filter to known columns only
        safe_data = {k: v for k, v in data.items() if k in ENRICHABLE_FIELDS or k == "id"}

        business_id = safe_data.pop("id", None) or str(uuid.uuid4())

        if not safe_data:
            logger.warning("upsert_business called with no valid columns")
            return business_id

        if self.is_postgres:
            return self._upsert_business_pg(business_id, safe_data)
        else:
            return self._upsert_business_sqlite(business_id, safe_data)

    def _upsert_business_sqlite(self, business_id: str, data: Dict[str, Any]) -> str:
        """SQLite upsert using INSERT OR REPLACE logic with COALESCE."""
        with self._backend.write_connection() as conn:
            # Check if record exists
            row = conn.execute(
                "SELECT id FROM businesses WHERE id = ?", (business_id,)
            ).fetchone()

            if row:
                # Update with COALESCE — never overwrite non-null
                set_clauses = []
                params: list = []
                for col, val in data.items():
                    if col not in ENRICHABLE_FIELDS:
                        continue
                    processed = self._prepare_value_for_write(col, val)
                    set_clauses.append(f"{col} = COALESCE({col}, ?)")
                    params.append(processed)

                if set_clauses:
                    set_clauses.append(f"updated_at = {self._backend.now_expr()}")
                    query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = ?"
                    params.append(business_id)
                    conn.execute(query, params)
                    conn.commit()
            else:
                # Insert new record
                columns = ["id"]
                placeholders = ["?"]
                params = [business_id]

                for col, val in data.items():
                    if col not in ENRICHABLE_FIELDS:
                        continue
                    columns.append(col)
                    placeholders.append("?")
                    params.append(self._prepare_value_for_write(col, val))

                query = (
                    f"INSERT INTO businesses ({', '.join(columns)}) "
                    f"VALUES ({', '.join(placeholders)})"
                )
                conn.execute(query, params)
                conn.commit()

        return business_id

    def _upsert_business_pg(self, business_id: str, data: Dict[str, Any]) -> str:
        """PostgreSQL upsert using INSERT ... ON CONFLICT with COALESCE."""
        columns = ["id"]
        placeholders = ["%s"]
        params: list = [business_id]

        for col, val in data.items():
            if col not in ENRICHABLE_FIELDS:
                continue
            columns.append(col)
            if col in JSON_COLUMNS:
                placeholders.append("%s::jsonb")
            else:
                placeholders.append("%s")
            params.append(self._prepare_value_for_write(col, val))

        # Build ON CONFLICT DO UPDATE with COALESCE
        conflict_sets = []
        for col in columns[1:]:  # Skip 'id'
            conflict_sets.append(f"{col} = COALESCE(businesses.{col}, EXCLUDED.{col})")
        conflict_sets.append("updated_at = NOW()")

        query = (
            f"INSERT INTO businesses ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON CONFLICT (id) DO UPDATE SET {', '.join(conflict_sets)} "
            f"RETURNING id"
        )

        with self._backend.write_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            result = cur.fetchone()
            conn.commit()
            cur.close()

        return str(result[0]) if result else business_id

    # ── Enrichment Writes (COALESCE pattern) ─────────────────────────────────

    def write_enrichment(
        self,
        business_id: str,
        updates: Dict[str, Any],
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Write enrichment data for a single business using COALESCE pattern.

        COALESCE means: new values only fill NULLs. Existing non-null values
        are never overwritten. This prevents data regression when multiple
        enrichment sources write to the same record.

        Uses self.transaction() internally so connection lifecycle (acquire,
        commit, rollback, release with close=broken) is handled safely.

        Args:
            business_id: UUID of the business to update.
            updates: Dict of column_name → value. Unknown columns are ignored.
            source: Enrichment source identifier for logging (e.g., "scraper", "gemma").

        Returns:
            Dict with status, business_id, and fields_updated.
        """
        if not updates:
            return {"status": "no_updates", "business_id": business_id}

        safe_updates = {k: v for k, v in updates.items() if k in ENRICHABLE_FIELDS}
        if not safe_updates:
            return {
                "status": "no_valid_fields",
                "business_id": business_id,
                "rejected": list(updates.keys()),
            }

        try:
            with self.transaction() as tx:
                ph = tx.placeholder
                set_clauses = []
                params: list = []

                for col, val in safe_updates.items():
                    processed = self._prepare_value_for_write(col, val)
                    if self.is_postgres and col in JSON_COLUMNS:
                        set_clauses.append(f"{col} = COALESCE({col}, %s::jsonb)")
                    else:
                        set_clauses.append(f"{col} = COALESCE({col}, {ph})")
                    params.append(processed)

                # Update timestamps
                set_clauses.append(f"updated_at = {self.now_expr}")
                set_clauses.append(f"last_enriched_at = {self.now_expr}")
                set_clauses.append(f"enrichment_attempts = COALESCE(enrichment_attempts, 0) + 1")

                if self.is_postgres:
                    query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = %s::uuid"
                else:
                    query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = {ph}"
                params.append(business_id)

                tx.execute(query, tuple(params))

            logger.debug(
                "Enrichment written: biz=%s source=%s fields=%s",
                business_id, source, list(safe_updates.keys()),
            )
            return {
                "status": "updated",
                "business_id": business_id,
                "fields_updated": list(safe_updates.keys()),
            }
        except Exception as e:
            logger.error("write_enrichment failed for %s: %s", business_id, e)
            return {"status": "error", "business_id": business_id, "error": str(e)}

    def write_enrichment_batch(
        self,
        batch: List[Tuple[str, Dict[str, Any]]],
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Write enrichment data for multiple businesses in a single transaction.

        Each entry in batch is (business_id, updates_dict). Uses COALESCE
        pattern for each record. All writes are committed in one transaction.

        Uses self.transaction() internally so connection lifecycle (acquire,
        commit, rollback, release with close=broken) is handled safely.

        Args:
            batch: List of (business_id, updates_dict) tuples.
            source: Enrichment source identifier for logging.

        Returns:
            Dict with status, updated count, error count, and total.
        """
        if not batch:
            return {"status": "empty", "updated": 0, "errors": 0, "total": 0}

        updated = 0
        errors = 0

        try:
            with self.transaction() as tx:
                ph = tx.placeholder

                for business_id, updates in batch:
                    safe_updates = {k: v for k, v in updates.items() if k in ENRICHABLE_FIELDS}
                    if not safe_updates:
                        continue

                    set_clauses = []
                    params: list = []

                    for col, val in safe_updates.items():
                        processed = self._prepare_value_for_write(col, val)
                        if self.is_postgres and col in JSON_COLUMNS:
                            set_clauses.append(f"{col} = COALESCE({col}, %s::jsonb)")
                        else:
                            set_clauses.append(f"{col} = COALESCE({col}, {ph})")
                        params.append(processed)

                    set_clauses.append(f"updated_at = {self.now_expr}")
                    set_clauses.append(f"last_enriched_at = {self.now_expr}")
                    set_clauses.append(f"enrichment_attempts = COALESCE(enrichment_attempts, 0) + 1")

                    if self.is_postgres:
                        query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = %s::uuid"
                    else:
                        query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = {ph}"
                    params.append(business_id)

                    try:
                        tx.execute(query, tuple(params))
                        updated += 1
                    except Exception as e:
                        errors += 1
                        logger.warning("Batch write failed for %s: %s", business_id, e)

            logger.debug(
                "Batch enrichment written: %d/%d records, source=%s",
                updated, len(batch), source,
            )

        except Exception as e:
            logger.error("write_enrichment_batch failed: %s", e)
            return {
                "status": "error",
                "updated": updated,
                "errors": errors + (len(batch) - updated - errors),
                "total": len(batch),
                "error": str(e),
            }

        return {
            "status": "completed",
            "updated": updated,
            "errors": errors,
            "total": len(batch),
        }

    def upsert_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert or update multiple business records in a single transaction.

        Each record is a dict of column_name → value. If a record has an 'id'
        field, it is used; otherwise a new UUID is generated.

        Uses COALESCE on update: new values only fill NULLs.

        Args:
            records: List of dicts, each representing a business record.

        Returns:
            Dict with status, inserted count, updated count, and ids list.
        """
        if not records:
            return {"status": "empty", "inserted": 0, "updated": 0, "ids": []}

        inserted = 0
        updated_count = 0
        ids = []

        with self._backend.write_connection() as conn:
            try:
                for record in records:
                    safe_data = {k: v for k, v in record.items() if k in ENRICHABLE_FIELDS or k == "id"}
                    business_id = safe_data.pop("id", None) or str(uuid.uuid4())
                    ids.append(business_id)

                    if not safe_data:
                        continue

                    if self.is_postgres:
                        was_insert = self._upsert_single_pg_in_txn(conn, business_id, safe_data)
                    else:
                        was_insert = self._upsert_single_sqlite_in_txn(conn, business_id, safe_data)

                    if was_insert:
                        inserted += 1
                    else:
                        updated_count += 1

                conn.commit()

            except Exception as e:
                if self.is_postgres:
                    conn.rollback()
                logger.error("upsert_batch failed: %s", e)
                return {
                    "status": "error",
                    "inserted": inserted,
                    "updated": updated_count,
                    "ids": ids,
                    "error": str(e),
                }

        return {
            "status": "completed",
            "inserted": inserted,
            "updated": updated_count,
            "ids": ids,
        }

    def _upsert_single_sqlite_in_txn(
        self, conn: Any, business_id: str, data: Dict[str, Any]
    ) -> bool:
        """Upsert a single record in SQLite within an existing transaction. Returns True if inserted."""
        row = conn.execute(
            "SELECT id FROM businesses WHERE id = ?", (business_id,)
        ).fetchone()

        if row:
            set_clauses = []
            params: list = []
            for col, val in data.items():
                if col not in ENRICHABLE_FIELDS:
                    continue
                processed = self._prepare_value_for_write(col, val)
                set_clauses.append(f"{col} = COALESCE({col}, ?)")
                params.append(processed)

            if set_clauses:
                set_clauses.append(f"updated_at = {self._backend.now_expr()}")
                query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = ?"
                params.append(business_id)
                conn.execute(query, params)
            return False
        else:
            columns = ["id"]
            placeholders = ["?"]
            params = [business_id]

            for col, val in data.items():
                if col not in ENRICHABLE_FIELDS:
                    continue
                columns.append(col)
                placeholders.append("?")
                params.append(self._prepare_value_for_write(col, val))

            query = (
                f"INSERT INTO businesses ({', '.join(columns)}) "
                f"VALUES ({', '.join(placeholders)})"
            )
            conn.execute(query, params)
            return True

    def _upsert_single_pg_in_txn(
        self, conn: Any, business_id: str, data: Dict[str, Any]
    ) -> bool:
        """Upsert a single record in PostgreSQL within an existing transaction. Returns True if inserted."""
        columns = ["id"]
        placeholders = ["%s"]
        params: list = [business_id]

        for col, val in data.items():
            if col not in ENRICHABLE_FIELDS:
                continue
            columns.append(col)
            if col in JSON_COLUMNS:
                placeholders.append("%s::jsonb")
            else:
                placeholders.append("%s")
            params.append(self._prepare_value_for_write(col, val))

        conflict_sets = []
        for col in columns[1:]:
            conflict_sets.append(f"{col} = COALESCE(businesses.{col}, EXCLUDED.{col})")
        conflict_sets.append("updated_at = NOW()")

        query = (
            f"INSERT INTO businesses ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON CONFLICT (id) DO UPDATE SET {', '.join(conflict_sets)} "
            f"RETURNING (xmax = 0) AS was_insert"
        )

        cur = conn.cursor()
        cur.execute(query, params)
        result = cur.fetchone()
        cur.close()
        return bool(result and result[0])

    # ── Fetch for Enrichment (keyset pagination) ─────────────────────────────

    def fetch_for_enrichment(
        self,
        mode: str = "email",
        limit: int = 50,
        resume_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch businesses that need enrichment, with keyset pagination.

        Modes:
          - "email": Businesses with a website but no email.
          - "ai": Businesses without an AI summary.
          - "tech": Businesses with a website but no tech_stack.
          - "all": Businesses missing any enrichment.

        Keyset pagination uses the 'id' column for stable, efficient paging.
        Pass the last returned record's 'id' as resume_id to get the next page.

        Args:
            mode: Enrichment mode ("email", "ai", "tech", "all").
            limit: Maximum records to return per page.
            resume_id: Last record ID from previous page (for pagination).

        Returns:
            List of business dicts ready for enrichment.
        """
        # Build WHERE clause based on mode
        conditions = []

        if mode == "email":
            conditions.append("website_url IS NOT NULL AND website_url != ''")
            conditions.append("(email IS NULL OR email = '')")
        elif mode == "ai":
            conditions.append("(ai_summary IS NULL OR ai_summary = '')")
        elif mode == "tech":
            conditions.append("website_url IS NOT NULL AND website_url != ''")
            if self.is_postgres:
                conditions.append("(tech_stack IS NULL)")
            else:
                conditions.append("(tech_stack IS NULL OR tech_stack = '')")
        elif mode == "all":
            or_parts = [
                "(email IS NULL OR email = '')",
                "(ai_summary IS NULL OR ai_summary = '')",
            ]
            if self.is_postgres:
                or_parts.append("(tech_stack IS NULL)")
            else:
                or_parts.append("(tech_stack IS NULL OR tech_stack = '')")
            conditions.append(f"({' OR '.join(or_parts)})")

        # Only records that haven't been attempted too many times
        conditions.append("(enrichment_attempts < 3 OR enrichment_attempts IS NULL)")

        # Keyset pagination
        if resume_id:
            if self.is_postgres:
                conditions.append(f"id > %s::uuid")
            else:
                conditions.append("id > ?")

        where = " AND ".join(conditions)

        if self.is_postgres:
            query = (
                f"SELECT * FROM businesses WHERE {where} "
                f"ORDER BY id ASC LIMIT %s"
            )
        else:
            query = (
                f"SELECT * FROM businesses WHERE {where} "
                f"ORDER BY id ASC LIMIT ?"
            )

        params: list = []
        if resume_id:
            params.append(resume_id)
        params.append(limit)

        with self._backend.connection() as conn:
            try:
                if self.is_postgres:
                    import psycopg2.extras
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    cur.close()
                    return [dict(r) for r in rows]
                else:
                    cursor = conn.execute(query, params)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error("fetch_for_enrichment failed (mode=%s): %s", mode, e)
                return []

    # ── CSV Import/Export ────────────────────────────────────────────────────

    def import_csv(self, filepath: str, return_details: bool = False) -> Any:
        """
        Import business records from a CSV file.

        Auto-detects column names by matching CSV headers against known aliases
        (e.g., "Business Name" maps to "name", "ZIP" maps to "zip").

        Unknown columns are silently ignored. Records are upserted — existing
        records (matched by name + state + city if no ID) are updated with
        COALESCE; new records are inserted.

        Args:
            filepath: Path to the CSV file.

        Returns:
            Dict with status, total_rows, imported, skipped, and column_mapping.
        """
        if not os.path.exists(filepath):
            if return_details:
                return {"status": "error", "error": f"File not found: {filepath}"}
            raise FileNotFoundError(f"File not found: {filepath}")

        total_rows = 0
        imported = 0
        skipped = 0
        column_mapping: Dict[str, str] = {}

        try:
            with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    return {"status": "error", "error": "CSV has no headers"}

                # Map CSV headers to canonical column names
                for header in reader.fieldnames:
                    canonical = COLUMN_ALIASES.get(header.lower().strip())
                    if canonical:
                        column_mapping[header] = canonical
                    elif header.lower().strip() in ENRICHABLE_FIELDS:
                        column_mapping[header] = header.lower().strip()

                if not column_mapping:
                    return {
                        "status": "error",
                        "error": "No recognizable columns found in CSV headers",
                        "headers": list(reader.fieldnames),
                    }

                logger.info(
                    "CSV import: %s → mapped %d columns: %s",
                    filepath, len(column_mapping), column_mapping,
                )

                records_batch: List[Dict[str, Any]] = []

                for row in reader:
                    total_rows += 1

                    # Map CSV columns to canonical names
                    record: Dict[str, Any] = {}
                    for csv_col, db_col in column_mapping.items():
                        val = row.get(csv_col, "").strip()
                        if val:
                            record[db_col] = val

                    if not record:
                        skipped += 1
                        continue

                    # Validate state is 2-letter if present
                    if "state" in record:
                        state_val = record["state"].upper().strip()
                        if len(state_val) == 2:
                            record["state"] = state_val
                        else:
                            del record["state"]

                    records_batch.append(record)

                    # Flush in batches of 500
                    if len(records_batch) >= 500:
                        result = self.upsert_batch(records_batch)
                        imported += result.get("inserted", 0) + result.get("updated", 0)
                        records_batch = []

                # Flush remaining
                if records_batch:
                    result = self.upsert_batch(records_batch)
                    imported += result.get("inserted", 0) + result.get("updated", 0)

        except Exception as e:
            logger.error("CSV import failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "total_rows": total_rows,
                "imported": imported,
            }

        logger.info(
            "CSV import complete: %d total, %d imported, %d skipped",
            total_rows, imported, skipped,
        )

        details = {
            "status": "completed",
            "total_rows": total_rows,
            "imported": imported,
            "skipped": skipped,
            "new": imported,
            "updated": 0,
            "column_mapping": column_mapping,
        }

        if return_details:
            return details
        return imported

    def export_csv(
        self,
        filepath: str,
        where: Optional[str] = None,
        params: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export business records to a CSV file.

        Args:
            filepath: Output CSV file path.
            where: Optional predefined filter name from SAFE_WHERE_FILTERS
                   (e.g. "with_email", "enriched", "with_tech", "with_website", "with_npi", "with_ai").
                   Raw SQL strings are rejected for security.
            params: Optional list of parameter values for the WHERE clause.

        Returns:
            Dict with status and row_count.
        """
        where = self._resolve_where(where)
        query = "SELECT * FROM businesses"
        query_params: list = params or []

        if where:
            query += f" WHERE {where}"

        query += " ORDER BY created_at DESC"

        exported = 0

        with self._backend.connection() as conn:
            try:
                if self.is_postgres:
                    import psycopg2.extras
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute(query, query_params)
                    rows = cur.fetchall()
                    cur.close()

                    if not rows:
                        return {"status": "completed", "row_count": 0}

                    fieldnames = list(rows[0].keys())

                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in rows:
                            writer.writerow(dict(row))
                            exported += 1

                else:
                    cursor = conn.execute(query, query_params)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=columns)
                        writer.writeheader()
                        for row in rows:
                            writer.writerow(dict(zip(columns, row)))
                            exported += 1

            except Exception as e:
                logger.error("CSV export failed: %s", e)
                return {"status": "error", "error": str(e), "row_count": exported}

        logger.info("CSV export complete: %d rows → %s", exported, filepath)
        return {"status": "completed", "row_count": exported}

    def export_json(
        self,
        filepath: str,
        where: Optional[str] = None,
        params: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export business records to a JSON file.

        Args:
            filepath: Output JSON file path.
            where: Optional predefined filter name from SAFE_WHERE_FILTERS
                   (e.g. "with_email", "enriched", "with_tech", "with_website", "with_npi", "with_ai").
                   Raw SQL strings are rejected for security.
            params: Optional list of parameter values for the WHERE clause.

        Returns:
            Dict with status and row_count.
        """
        where = self._resolve_where(where)
        query = "SELECT * FROM businesses"
        query_params: list = params or []

        if where:
            query += f" WHERE {where}"

        query += " ORDER BY created_at DESC"

        try:
            rows = self.fetch_dicts(query, tuple(query_params))

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, default=str)

            logger.info("JSON export complete: %d rows → %s", len(rows), filepath)
            return {"status": "completed", "row_count": len(rows)}

        except Exception as e:
            logger.error("JSON export failed: %s", e)
            return {"status": "error", "error": str(e), "row_count": 0}

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics for the businesses table.

        Returns:
            Dict with:
              - total_records: Total number of business records.
              - with_email: Records that have an email address.
              - with_tech_stack: Records with tech_stack data.
              - with_npi: Records with NPI numbers.
              - with_website: Records with a website URL.
              - enriched_today: Records enriched in the last 24 hours.
              - last_enriched: Timestamp of the most recent enrichment.
        """
        stats: Dict[str, Any] = {
            "total_records": 0,
            "with_email": 0,
            "with_tech_stack": 0,
            "with_npi": 0,
            "with_website": 0,
            "with_ai_summary": 0,
            "with_health_score": 0,
            "with_industry": 0,
            "enriched_today": 0,
            "last_enriched": None,
        }

        # Build queries per dialect
        if self.is_postgres:
            today_condition = "last_enriched_at >= NOW() - INTERVAL '24 hours'"
            tech_null_check = "tech_stack IS NOT NULL"
        else:
            today_condition = "last_enriched_at >= datetime('now', '-1 day')"
            tech_null_check = "tech_stack IS NOT NULL AND tech_stack != ''"

        queries = {
            "total_records": "SELECT COUNT(*) FROM businesses",
            "with_email": "SELECT COUNT(*) FROM businesses WHERE email IS NOT NULL AND email != ''",
            "with_tech_stack": f"SELECT COUNT(*) FROM businesses WHERE {tech_null_check}",
            "with_npi": "SELECT COUNT(*) FROM businesses WHERE npi_number IS NOT NULL AND npi_number != ''",
            "with_website": "SELECT COUNT(*) FROM businesses WHERE website_url IS NOT NULL AND website_url != ''",
            "with_ai_summary": "SELECT COUNT(*) FROM businesses WHERE ai_summary IS NOT NULL AND ai_summary != ''",
            "with_health_score": "SELECT COUNT(*) FROM businesses WHERE health_score IS NOT NULL",
            "with_industry": "SELECT COUNT(*) FROM businesses WHERE industry IS NOT NULL AND industry != ''",
            "enriched_today": f"SELECT COUNT(*) FROM businesses WHERE {today_condition}",
            "last_enriched": "SELECT MAX(last_enriched_at) FROM businesses",
        }

        with self._backend.connection() as conn:
            try:
                for key, query in queries.items():
                    if self.is_postgres:
                        cur = conn.cursor()
                        cur.execute(query)
                        result = cur.fetchone()
                        cur.close()
                    else:
                        result = conn.execute(query).fetchone()

                    if result:
                        stats[key] = result[0]
            except Exception as e:
                logger.error("get_stats failed: %s", e)

        return stats

    # ── Query Helpers ────────────────────────────────────────────────────────

    def get_business(self, business_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single business record by ID.

        Args:
            business_id: UUID string of the business.

        Returns:
            Dict of column_name → value, or None if not found.
        """
        if self.is_postgres:
            query = "SELECT * FROM businesses WHERE id = %s::uuid"
        else:
            query = "SELECT * FROM businesses WHERE id = ?"

        with self._backend.connection() as conn:
            try:
                if self.is_postgres:
                    import psycopg2.extras
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute(query, (business_id,))
                    row = cur.fetchone()
                    cur.close()
                    return dict(row) if row else None
                else:
                    cursor = conn.execute(query, (business_id,))
                    columns = [desc[0] for desc in cursor.description]
                    row = cursor.fetchone()
                    return dict(zip(columns, row)) if row else None
            except Exception as e:
                logger.error("get_business failed for %s: %s", business_id, e)
                return None

    def count(self, where: Optional[str] = None, params: Optional[List[Any]] = None) -> int:
        """
        Count business records, optionally filtered.

        Args:
            where: Optional predefined filter name from SAFE_WHERE_FILTERS
                   (e.g. "with_email", "enriched", "with_tech", "with_website", "with_npi", "with_ai").
                   Raw SQL strings are rejected for security.
            params: Optional parameter values for the WHERE clause.

        Returns:
            Integer count, or -1 if the query failed.
        """
        where = self._resolve_where(where)
        query = "SELECT COUNT(*) FROM businesses"
        query_params: list = params or []

        if where:
            query += f" WHERE {where}"

        with self._backend.connection() as conn:
            try:
                if self.is_postgres:
                    cur = conn.cursor()
                    cur.execute(query, query_params)
                    result = cur.fetchone()
                    cur.close()
                    return result[0] if result else 0
                else:
                    result = conn.execute(query, query_params).fetchone()
                    return result[0] if result else 0
            except Exception as e:
                logger.error("count() failed: %s", e)
                return -1

    # ── Pool compatibility (for EnrichmentPipeline) ───────────────────────────

    def get_pool(self):
        """Return self as a pool-compatible interface for the enrichment pipeline."""
        return self

    def get_connection(self):
        """
        Return a raw database connection for pipeline compatibility.

        For SQLite: returns the shared connection (thread-safe via WAL).
        For PostgreSQL: gets a connection from the pool.
        """
        if self.is_postgres:
            return self._backend._pool.getconn()
        return self._backend._conn

    def return_connection(self, conn):
        """
        Return a connection to the pool (PostgreSQL) or no-op (SQLite).

        Args:
            conn: The connection to return.
        """
        if self.is_postgres:
            self._backend._pool.putconn(conn)
        # SQLite: no-op, single shared connection

    # ── Backend-agnostic query helpers (for pipeline/tools) ─────────────────

    def fetch_dicts(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as a list of dicts.

        Works on both SQLite and PostgreSQL. Automatically handles
        cursor_factory differences and row-to-dict conversion.

        Args:
            query: SQL SELECT query using backend-appropriate placeholders.
            params: Query parameter values.

        Returns:
            List of dicts, one per row.
        """
        with self._backend.connection() as conn:
            try:
                if self.is_postgres:
                    import psycopg2.extras
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    cur.close()
                    return [dict(r) for r in rows]
                else:
                    cursor = conn.execute(query, params)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error("fetch_dicts failed: %s", e)
                return []

    def execute(self, query: str, params: tuple = ()) -> None:
        """
        Execute and auto-commit a single statement. Same behavior on both backends.

        For multi-statement atomicity, use ``transaction()`` instead.

        When called inside an active ``transaction()`` block on SQLite, the
        auto-commit is skipped so the outer transaction's atomicity is preserved.
        On PostgreSQL this is safe regardless because the pool hands out separate
        connections, so a commit here cannot affect the transaction's connection.

        Args:
            query: SQL query using backend-appropriate placeholders.
            params: Query parameter values.

        Note on in_tx flag:
        - PG uses a separate pool connection per execute(), so auto-commit
          on execute's connection can never affect a transaction's connection.
          The in_tx check is unnecessary on PG but present for symmetry.
        - SQLite shares one connection across the transaction and any nested
          execute(), so the in_tx check prevents auto-commit from leaking
          into the transaction's atomicity.
        """
        in_tx = getattr(self._in_transaction, 'active', False)

        if self.is_postgres:
            # PG: pool gives a different connection, so auto-commit is safe
            # even when another connection holds an open transaction.
            conn = self._backend._pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                conn.commit()
                cur.close()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._backend._pool.putconn(conn)
        else:
            # SQLite: single shared connection.
            with self._backend.write_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, params)
                cur.close()
                if not in_tx:
                    conn.commit()

    def execute_and_commit(self, query: str, params: tuple = ()) -> None:
        """Execute a single statement and commit immediately. Atomic, safe on both backends.

        Convenience wrapper around transaction() for single-statement writes.
        """
        with self.transaction() as tx:
            tx.execute(query, params)

    def executemany(self, query: str, params_list: List[tuple]) -> None:
        """
        Execute a query with multiple parameter sets.

        On SQLite, auto-commits after execution unless called inside an active
        ``transaction()`` block (in which case the transaction handles commit).

        Args:
            query: SQL query using backend-appropriate placeholders.
            params_list: List of parameter tuples.
        """
        in_tx = getattr(self._in_transaction, 'active', False)

        if self.is_postgres:
            conn = self._backend._pool.getconn()
            broken = False
            try:
                cur = conn.cursor()
                cur.executemany(query, params_list)
                conn.commit()
                cur.close()
            except Exception:
                broken = True
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                try:
                    self._backend._pool.putconn(conn, close=broken)
                except Exception:
                    pass
        else:
            with self._backend.write_connection() as conn:
                conn.executemany(query, params_list)
                if not in_tx:
                    conn.commit()

    def commit(self) -> None:
        """No-op — execute() auto-commits. Use transaction() for multi-statement atomicity."""
        pass

    def rollback(self) -> None:
        """No-op — execute() auto-commits. Use transaction() for multi-statement atomicity."""
        pass

    @contextmanager
    def transaction(self):
        """Context manager for a database transaction.

        Acquires a single connection, yields a Transaction object with
        execute/fetch_dicts/commit methods. Auto-commits on clean exit,
        auto-rollbacks on exception, always releases connection.

        Sets a thread-local flag so that ``execute()`` and ``executemany()``
        called on this ForgeDB instance within the block skip their own
        auto-commit on SQLite (where the connection is shared).

        Usage:
            with db.transaction() as tx:
                tx.execute("INSERT INTO ...", params)
                tx.execute("UPDATE ...", params)
                rows = tx.fetch_dicts("SELECT ...", params)
            # auto-commit on exit, auto-rollback on exception
        """
        if self.is_postgres:
            conn = self._backend._pool.getconn()
            broken = False
            self._in_transaction.active = True
            try:
                tx = _Transaction(conn, is_postgres=True, placeholder=self.placeholder)
                yield tx
                conn.commit()
            except Exception:
                broken = True
                try:
                    conn.rollback()
                except Exception:
                    pass  # connection is dead, will be closed below
                raise
            finally:
                self._in_transaction.active = False
                try:
                    self._backend._pool.putconn(conn, close=broken)
                except Exception:
                    pass  # don't mask the original exception
        else:
            # SQLite: use the single shared connection with write lock
            with self._backend._write_lock:
                conn = self._backend._conn
                self._in_transaction.active = True
                try:
                    tx = _Transaction(conn, is_postgres=False, placeholder=self.placeholder)
                    yield tx
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    self._in_transaction.active = False

    @property
    def placeholder(self) -> str:
        """Return the parameter placeholder for the current backend ('?' or '%s')."""
        return self._backend.placeholder()

    @property
    def now_expr(self) -> str:
        """Return the SQL expression for current timestamp."""
        return self._backend.now_expr()

    def interval_ago(self, days: int) -> str:
        """
        Return a SQL expression for 'now minus N days'.

        PostgreSQL: NOW() - INTERVAL '7 days'
        SQLite:     datetime('now', '-7 days')
        """
        if self.is_postgres:
            return f"NOW() - INTERVAL '{days} days'"
        return f"datetime('now', '-{days} days')"

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Close the database connection(s).

        For SQLite, closes the single connection.
        For PostgreSQL, closes all connections in the pool.
        Always call this when done.
        """
        self._backend.close()

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _prepare_value_for_write(self, column: str, value: Any) -> Any:
        """
        Prepare a value for writing to the database.

        Handles JSON serialization, boolean conversion, string truncation,
        and type coercion for both backends.

        Args:
            column: The target column name.
            value: The raw value to prepare.

        Returns:
            The prepared value suitable for the current backend.
        """
        if value is None:
            return None

        # JSON columns: serialize dicts/lists to JSON strings
        if column in JSON_COLUMNS:
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            elif isinstance(value, str):
                # Validate it's valid JSON, or wrap as-is
                try:
                    json.loads(value)
                    return value
                except (json.JSONDecodeError, ValueError):
                    return json.dumps(value)
            return json.dumps(value)

        # Boolean columns: normalize to int for SQLite, bool for PostgreSQL
        if column in BOOLEAN_COLUMNS:
            if isinstance(value, bool):
                return value if self.is_postgres else int(value)
            if isinstance(value, int):
                return bool(value) if self.is_postgres else value
            if isinstance(value, str):
                truthy = value.lower() in ("true", "1", "yes", "t")
                return truthy if self.is_postgres else int(truthy)
            return None

        # String columns: truncate to 1000 chars
        if isinstance(value, str):
            return value[:1000]

        # Numeric columns: pass through
        if isinstance(value, (int, float)):
            return value

        # Fallback: stringify
        return str(value)[:1000]


class _Transaction:
    """A single database transaction with one connection.

    Created by ``ForgeDB.transaction()`` — do not instantiate directly.
    """

    def __init__(self, conn, is_postgres: bool, placeholder: str):
        self._conn = conn
        self._is_postgres = is_postgres
        self._placeholder = placeholder

    @property
    def placeholder(self) -> str:
        return self._placeholder

    def execute(self, query: str, params: tuple = ()) -> None:
        cur = self._conn.cursor()
        cur.execute(query, params)
        cur.close()

    def executemany(self, query: str, params_list) -> None:
        cur = self._conn.cursor()
        cur.executemany(query, params_list)
        cur.close()

    def fetch_dicts(self, query: str, params: tuple = ()) -> list:
        if self._is_postgres:
            import psycopg2.extras
            cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            return rows
        else:
            prev_factory = self._conn.row_factory
            self._conn.row_factory = sqlite3.Row
            try:
                cur = self._conn.cursor()
                cur.execute(query, params)
                rows = [dict(r) for r in cur.fetchall()]
                cur.close()
                return rows
            finally:
                self._conn.row_factory = prev_factory

    def commit(self) -> None:
        """Explicit mid-transaction commit (rare -- usually auto-commits on context exit)."""
        self._conn.commit()

    def rollback(self) -> None:
        """Explicit mid-transaction rollback."""
        self._conn.rollback()
