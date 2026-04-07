"""Database abstraction over SQLite and PostgreSQL.

Schema lives in db_schema.py, CSV I/O in db_io.py.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from forge.errors import DatabaseError

logger = logging.getLogger("forge.db")

# Schema definitions and backend classes live in db_schema.py
from forge.db_io import _ForgeDBIOMixin  # noqa: E402
from forge.db_schema import (  # noqa: E402
    BOOLEAN_COLUMNS,
    BUSINESS_COLUMNS,
    BUSINESS_INDEXES,
    ENRICHABLE_FIELDS,
    JSON_COLUMNS,
    _PostgresBackend,
    _SQLiteBackend,
)

# ── Main Interface ───────────────────────────────────────────────────────────


class ForgeDB(_ForgeDBIOMixin):
    """Unified database interface for FORGE (SQLite or PostgreSQL)."""

    def __init__(self, backend: Any):
        """Initialize with a backend instance. Use from_config() instead."""
        self._backend = backend
        self._in_transaction = threading.local()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ForgeDB":
        """Create a ForgeDB from config. Auto-detects SQLite vs PostgreSQL."""
        if "db_path" in config:
            backend: Union[_SQLiteBackend, _PostgresBackend] = _SQLiteBackend(
                db_path=config["db_path"]
            )
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
        # Reject unknown filters, no raw SQL allowed
        logger.warning("Rejected unknown WHERE filter: %s", where[:50])
        return None

    # ── Schema Management ────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """
        Create the businesses table and indexes if they don't exist.

        Safe to call repeatedly. Uses IF NOT EXISTS for all DDL.
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
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON businesses ({idx_col})")
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
                cur.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON businesses ({idx_col})")
            conn.commit()
            cur.close()

    # Single record upsert operations are inherited from _ForgeDBIOMixin (db_io.py)

    # ── Enrichment Writes (COALESCE pattern) ─────────────────────────────────

    def write_enrichment(
        self,
        business_id: str,
        updates: Dict[str, Any],
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Write enrichment data for a single business using COALESCE pattern.

        Args:
            business_id: UUID of the business to update.
            updates: Dict of column_name -> value. Unknown columns are ignored.
            source: Enrichment source identifier for logging.

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
                set_clauses, params = self._build_enrichment_query(safe_updates, ph)
                id_ph = "%s::uuid" if self.is_postgres else ph
                query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = {id_ph}"
                params.append(business_id)
                tx.execute(query, tuple(params))

            logger.debug(
                "Enrichment written: biz=%s source=%s fields=%s",
                business_id,
                source,
                list(safe_updates.keys()),
            )
            return {
                "status": "updated",
                "business_id": business_id,
                "fields_updated": list(safe_updates.keys()),
            }
        except (
            DatabaseError,
            Exception,
        ) as e:  # Non-critical: return error dict instead of crashing caller
            logger.error("write_enrichment failed for %s: %s", business_id, e)
            return {"status": "error", "business_id": business_id, "error": str(e)}

    def _build_enrichment_query(
        self,
        safe_updates: Dict[str, Any],
        ph: str,
    ) -> tuple:
        """Build SET clauses and params for an enrichment UPDATE.

        Returns (set_clauses_list, params_list).
        """
        set_clauses: list = []
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
        set_clauses.append("enrichment_attempts = COALESCE(enrichment_attempts, 0) + 1")
        return set_clauses, params

    def write_enrichment_batch(
        self,
        batch: List[Tuple[str, Dict[str, Any]]],
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Write enrichment data for multiple businesses in a single transaction.

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
                    set_clauses, params = self._build_enrichment_query(safe_updates, ph)
                    id_ph = "%s::uuid" if self.is_postgres else ph
                    query = f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = {id_ph}"
                    params.append(business_id)
                    try:
                        tx.execute(query, tuple(params))
                        updated += 1
                    except Exception as e:  # Non-critical: count error, continue batch
                        errors += 1
                        logger.warning("Batch write failed for %s: %s", business_id, e)
            logger.debug(
                "Batch enrichment written: %d/%d records, source=%s", updated, len(batch), source
            )
        except Exception as e:  # Non-critical: return error dict with partial results
            logger.error("write_enrichment_batch failed: %s", e)
            return {
                "status": "error",
                "updated": updated,
                "errors": errors + (len(batch) - updated - errors),
                "total": len(batch),
                "error": str(e),
            }

        return {"status": "completed", "updated": updated, "errors": errors, "total": len(batch)}

    # Batch upsert operations are inherited from _ForgeDBIOMixin (db_io.py)

    # ── Fetch for Enrichment (keyset pagination) ─────────────────────────────

    def _build_fetch_query(self, mode: str, resume_id: Optional[str]) -> str:
        """Build WHERE conditions and query for fetch_for_enrichment.

        Returns (query_string, needs_resume_param).
        """
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
            or_parts = ["(email IS NULL OR email = '')", "(ai_summary IS NULL OR ai_summary = '')"]
            or_parts.append(
                "(tech_stack IS NULL)"
                if self.is_postgres
                else "(tech_stack IS NULL OR tech_stack = '')"
            )
            conditions.append(f"({' OR '.join(or_parts)})")

        conditions.append("(enrichment_attempts < 3 OR enrichment_attempts IS NULL)")

        if resume_id:
            conditions.append("id > %s::uuid" if self.is_postgres else "id > ?")

        where = " AND ".join(conditions)
        ph = "%s" if self.is_postgres else "?"
        query = f"SELECT * FROM businesses WHERE {where} ORDER BY id ASC LIMIT {ph}"
        return query

    def fetch_for_enrichment(
        self,
        mode: str = "email",
        limit: int = 50,
        resume_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch businesses that need enrichment, with keyset pagination.

        Args:
            mode: Enrichment mode ("email", "ai", "tech", "all").
            limit: Maximum records to return per page.
            resume_id: Last record ID from previous page.

        Returns:
            List of business dicts ready for enrichment.
        """
        query = self._build_fetch_query(mode, resume_id)

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
            except Exception as e:  # Non-critical: return empty list so caller can continue
                logger.error("fetch_for_enrichment failed (mode=%s): %s", mode, e)
                return []

    # CSV import/export and upsert operations are inherited from _ForgeDBIOMixin (db_io.py)

    # ── Stats ────────────────────────────────────────────────────────────────

    def _build_stats_queries(self) -> Dict[str, str]:
        """Build dialect-specific stats queries."""
        if self.is_postgres:
            today_cond = "last_enriched_at >= NOW() - INTERVAL '24 hours'"
            tech_check = "tech_stack IS NOT NULL"
        else:
            today_cond = "last_enriched_at >= datetime('now', '-1 day')"
            tech_check = "tech_stack IS NOT NULL AND tech_stack != ''"

        return {
            "total_records": "SELECT COUNT(*) FROM businesses",
            "with_email": "SELECT COUNT(*) FROM businesses WHERE email IS NOT NULL AND email != ''",
            "with_tech_stack": f"SELECT COUNT(*) FROM businesses WHERE {tech_check}",
            "with_npi": "SELECT COUNT(*) FROM businesses WHERE npi_number IS NOT NULL AND npi_number != ''",
            "with_website": "SELECT COUNT(*) FROM businesses WHERE website_url IS NOT NULL AND website_url != ''",
            "with_ai_summary": "SELECT COUNT(*) FROM businesses WHERE ai_summary IS NOT NULL AND ai_summary != ''",
            "with_health_score": "SELECT COUNT(*) FROM businesses WHERE health_score IS NOT NULL",
            "with_industry": "SELECT COUNT(*) FROM businesses WHERE industry IS NOT NULL AND industry != ''",
            "enriched_today": f"SELECT COUNT(*) FROM businesses WHERE {today_cond}",
            "last_enriched": "SELECT MAX(last_enriched_at) FROM businesses",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics for the businesses table."""
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
        queries = self._build_stats_queries()
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
            except Exception as e:  # Non-critical: return partial stats on failure
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
            except Exception as e:  # Non-critical: return None so caller handles missing record
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
            except Exception as e:  # Non-critical: return -1 sentinel so caller knows it failed
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
            except Exception as e:  # Non-critical: return empty list so pipeline continues
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
        in_tx = getattr(self._in_transaction, "active", False)

        if self.is_postgres:
            # PG: pool gives a different connection, so auto-commit is safe
            # even when another connection holds an open transaction.
            conn = self._backend._pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                conn.commit()
                cur.close()
            except Exception:  # Catch-and-reraise: rollback before propagating
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
        in_tx = getattr(self._in_transaction, "active", False)

        if self.is_postgres:
            conn = self._backend._pool.getconn()
            broken = False
            try:
                cur = conn.cursor()
                cur.executemany(query, params_list)
                conn.commit()
                cur.close()
            except Exception:  # Catch-and-reraise: mark broken, attempt rollback, then propagate
                broken = True
                try:
                    conn.rollback()
                except Exception:  # Non-critical: connection may already be dead
                    pass
                raise
            finally:
                try:
                    self._backend._pool.putconn(conn, close=broken)
                except Exception:  # Non-critical: pool cleanup must not mask the real error
                    pass
        else:
            with self._backend.write_connection() as conn:
                conn.executemany(query, params_list)
                if not in_tx:
                    conn.commit()

    def commit(self) -> None:
        """No-op. execute() auto-commits. Use transaction() for multi-statement atomicity."""
        pass

    def rollback(self) -> None:
        """No-op. execute() auto-commits. Use transaction() for multi-statement atomicity."""
        pass

    @contextmanager
    def _pg_transaction(self):
        """PostgreSQL transaction context manager."""
        conn = self._backend._pool.getconn()
        broken = False
        self._in_transaction.active = True
        try:
            tx = _Transaction(conn, is_postgres=True, placeholder=self.placeholder)
            yield tx
            conn.commit()
        except Exception:  # Catch-and-reraise: rollback, mark connection broken, propagate
            broken = True
            try:
                conn.rollback()
            except Exception:  # Non-critical: connection may be dead after the failure
                pass
            raise
        finally:
            self._in_transaction.active = False
            try:
                self._backend._pool.putconn(conn, close=broken)
            except Exception:  # Non-critical: pool return must not mask the real error
                pass

    @contextmanager
    def _sqlite_transaction(self):
        """SQLite transaction context manager."""
        with self._backend._write_lock:
            conn = self._backend._conn
            self._in_transaction.active = True
            try:
                tx = _Transaction(conn, is_postgres=False, placeholder=self.placeholder)
                yield tx
                conn.commit()
            except Exception:  # Catch-and-reraise: rollback SQLite transaction, propagate
                conn.rollback()
                raise
            finally:
                self._in_transaction.active = False

    @contextmanager
    def transaction(self):
        """Context manager for a database transaction.

        Usage:
            with db.transaction() as tx:
                tx.execute("INSERT INTO ...", params)
                tx.execute("UPDATE ...", params)
        """
        ctx = self._pg_transaction() if self.is_postgres else self._sqlite_transaction()
        with ctx as tx:
            yield tx

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

    def _prepare_json_value(self, value: Any) -> str:
        """Serialize a value for a JSON column."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, str):
            try:
                json.loads(value)
                return value
            except (json.JSONDecodeError, ValueError):
                return json.dumps(value)
        return json.dumps(value)

    def _prepare_bool_value(self, value: Any) -> Any:
        """Normalize a value for a boolean column."""
        if isinstance(value, bool):
            return value if self.is_postgres else int(value)
        if isinstance(value, int):
            return bool(value) if self.is_postgres else value
        if isinstance(value, str):
            truthy = value.lower() in ("true", "1", "yes", "t")
            return truthy if self.is_postgres else int(truthy)
        return None

    def _prepare_value_for_write(self, column: str, value: Any) -> Any:
        """Prepare a value for writing to the database."""
        if value is None:
            return None
        if column in JSON_COLUMNS:
            return self._prepare_json_value(value)
        if column in BOOLEAN_COLUMNS:
            return self._prepare_bool_value(value)
        if isinstance(value, str):
            return value[:1000]
        if isinstance(value, (int, float)):
            return value
        return str(value)[:1000]


# _Transaction class is in db_schema.py but imported here for internal use
from forge.db_schema import _Transaction  # noqa: E402
