"""
FORGE Database Schema & Backends — Column mappings, indexes, field sets, and backend classes.

Extracted from db.py to keep that file under 800 lines.

Contains:
  - COLUMN_ALIASES: CSV header -> canonical column name mapping
  - BUSINESS_COLUMNS: table schema per backend (SQLite / PostgreSQL)
  - BUSINESS_INDEXES: indexes to create on the businesses table
  - ENRICHABLE_FIELDS: fields allowed for enrichment writes
  - JSON_COLUMNS, BOOLEAN_COLUMNS: type hints for value preparation
  - _SQLiteBackend, _PostgresBackend: database backend implementations

Depended on by: forge.db
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Set, Tuple

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

# Build the flat lookup: lowered alias -> canonical column
for _canonical, _aliases in _ALIAS_MAP.items():
    for _alias in _aliases:
        COLUMN_ALIASES[_alias.lower().strip()] = _canonical


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
ENRICHABLE_FIELDS: Set[str] = {
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
JSON_COLUMNS: Set[str] = {"pain_points", "opportunities", "tech_stack", "all_emails"}

# Boolean columns (stored as INTEGER 0/1 in SQLite, BOOLEAN in PostgreSQL).
BOOLEAN_COLUMNS: Set[str] = {"ssl_valid", "has_booking", "has_chat"}


# ── Backend: SQLite ──────────────────────────────────────────────────────────

class _SQLiteBackend:
    """SQLite backend using a single connection with a threading.Lock for writes."""

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
        yield self._conn

    @contextmanager
    def write_connection(self) -> Generator[sqlite3.Connection, None, None]:
        with self._write_lock:
            yield self._conn

    def close(self) -> None:
        self._conn.close()
        logger.info("SQLite connection closed: %s", self._db_path)

    @property
    def is_postgres(self) -> bool:
        return False

    def placeholder(self, index: int = 0) -> str:
        return "?"

    def now_expr(self) -> str:
        return "datetime('now')"

    def uuid_default(self) -> str:
        return str(uuid.uuid4())

    def json_cast(self, column: str) -> str:
        return column

    def uuid_cast(self, _placeholder: str) -> str:
        return _placeholder


# ── Backend: PostgreSQL ──────────────────────────────────────────────────────

class _PostgresBackend:
    """PostgreSQL backend using psycopg2's ThreadedConnectionPool."""

    def __init__(self, host: str, port: int, user: str, password: str,
                 dbname: str, min_connections: int = 2, max_connections: int = 10):
        import psycopg2
        import psycopg2.pool
        import psycopg2.extras
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            min_connections, max_connections,
            host=host, port=port, user=user, password=password,
            dbname=dbname, connect_timeout=30,
        )
        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        logger.info("PostgreSQL backend connected: %s:%d/%s (pool %d-%d)", host, port, dbname, min_connections, max_connections)

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def write_connection(self) -> Generator[Any, None, None]:
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()
        logger.info("PostgreSQL pool closed")

    @property
    def is_postgres(self) -> bool:
        return True

    def placeholder(self, index: int = 0) -> str:
        return "%s"

    def now_expr(self) -> str:
        return "NOW()"

    def uuid_default(self) -> str:
        return ""

    def json_cast(self, column: str) -> str:
        return f"{column}::jsonb"

    def uuid_cast(self, placeholder: str) -> str:
        return f"{placeholder}::uuid"


# ── Transaction helper ───────────────────────────────────────────────────────

class _Transaction:
    """A single database transaction with one connection."""

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
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()
