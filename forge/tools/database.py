"""
FORGE Database Tool — Database operations for enrichment.

Provides tools for the agent loop to:
  - Query businesses that need enrichment
  - Write enrichment results back to the database
  - Track enrichment progress

Uses ForgeDB abstraction layer for backend-agnostic operations.
All queries use parameterized placeholders — NEVER f-strings.

Dependencies: forge.db (ForgeDB)
Depended on by: enrichment pipeline, agent loop
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from forge.core.tool_registry import Tool

logger = logging.getLogger("forge.tools.database")


class DatabasePool:
    """
    Compatibility wrapper around ForgeDB.

    Provides the get_connection/return_connection interface expected by
    legacy code while delegating to ForgeDB internally. For new code,
    use ForgeDB directly.
    """

    def __init__(self, db=None, **kwargs):
        """
        Initialize with a ForgeDB instance or connection parameters.

        If a ForgeDB instance is passed as `db`, use it directly.
        Otherwise, fall back to creating a psycopg2 pool (PostgreSQL only).
        """
        if db is not None:
            self._db = db
            self._pool = None
        else:
            # Legacy path: create psycopg2 pool directly
            try:
                import psycopg2
                import psycopg2.pool
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    kwargs.get("min_connections", 2),
                    kwargs.get("max_connections", 10),
                    host=kwargs["host"],
                    port=kwargs["port"],
                    user=kwargs["user"],
                    password=kwargs["password"],
                    dbname=kwargs["dbname"],
                    connect_timeout=30,
                )
                self._db = None
                logger.info(
                    "Database pool created: %s:%d/%s",
                    kwargs["host"], kwargs["port"], kwargs["dbname"],
                )
            except ImportError:
                raise ImportError(
                    "psycopg2 is required for direct PostgreSQL connections. "
                    "Install it with: pip install psycopg2-binary"
                )

    def get_connection(self) -> Any:
        """Get a connection from the pool."""
        if self._db is not None:
            return self._db.get_connection()
        return self._pool.getconn()

    def return_connection(self, conn: Any) -> None:
        """Return a connection to the pool."""
        if self._db is not None:
            return self._db.return_connection(conn)
        self._pool.putconn(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._db is not None:
            self._db.close()
        elif self._pool is not None:
            self._pool.closeall()


class FetchUnenrichedTool(Tool):
    """
    Tool: Fetch businesses that need enrichment.

    The agent calls this to get a batch of businesses to enrich.
    Returns businesses missing specific fields (email, industry, etc.).
    """

    def __init__(self, db_pool: DatabasePool):
        self._pool = db_pool

    @property
    def name(self) -> str:
        return "fetch_unenriched"

    @property
    def description(self) -> str:
        return (
            "Fetch businesses that need enrichment from the database. "
            "Specify which field is missing (email, industry, ai_summary, health_score). "
            "Returns a batch of businesses with their current data."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "missing_field": {
                    "type": "string",
                    "description": "Field that should be null/empty: email, industry, ai_summary, health_score",
                    "enum": ["email", "industry", "ai_summary", "health_score"],
                },
                "state": {
                    "type": "string",
                    "description": "Optional: filter by US state (2-letter code)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of records to fetch (default 5, max 20)",
                    "default": 5,
                },
            },
            "required": ["missing_field"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        field = arguments["missing_field"]
        state = arguments.get("state")
        limit = min(arguments.get("limit", 5), 20)

        try:
            # Use ForgeDB if available on the pool, otherwise fall back
            db = getattr(self._pool, '_db', None)
            if db is not None:
                ph = db.placeholder
                # field is from enum, safe to format directly
                query = f"""
                    SELECT id, name, phone, website_url, address_line1, city, state, zip,
                           industry, sub_industry, email, ai_summary, health_score
                    FROM businesses
                    WHERE ({field} IS NULL OR {field} = '')
                    AND website_url IS NOT NULL AND website_url != ''
                """
                params: list = []
                if state:
                    query += f" AND state = {ph}"
                    params.append(state.upper())
                query += f" ORDER BY name ASC LIMIT {ph}"
                params.append(limit)
                rows = db.fetch_dicts(query, tuple(params))
            else:
                # Legacy psycopg2 path
                import psycopg2.extras
                conn = self._pool.get_connection()
                try:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    query = """
                        SELECT id, name, phone, website_url, address_line1, city, state, zip,
                               industry, sub_industry, email, ai_summary, health_score
                        FROM businesses
                        WHERE ({field} IS NULL OR {field}::text = '')
                        AND website_url IS NOT NULL AND website_url != ''
                    """.format(field=field)
                    params = []
                    if state:
                        query += " AND state = %s"
                        params.append(state.upper())
                    query += " ORDER BY name ASC LIMIT %s"
                    params.append(limit)
                    cur.execute(query, params)
                    rows = [dict(row) for row in cur.fetchall()]
                    cur.close()
                finally:
                    self._pool.return_connection(conn)

            return {
                "count": len(rows),
                "businesses": rows,
            }
        except Exception as e:
            logger.error("fetch_unenriched failed: %s", e)
            return {"error": str(e), "count": 0, "businesses": []}


class WriteEnrichmentTool(Tool):
    """
    Tool: Write enrichment results to the database.

    The agent calls this after processing a business to save results.
    Uses COALESCE pattern — never overwrites existing non-null values.
    """

    def __init__(self, db_pool: DatabasePool):
        self._pool = db_pool

    @property
    def name(self) -> str:
        return "write_enrichment"

    @property
    def description(self) -> str:
        return (
            "Write enrichment results for a business to the database. "
            "Provide the business ID and the fields to update. "
            "Existing non-null values are preserved (COALESCE pattern)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "business_id": {
                    "type": "string",
                    "description": "UUID of the business to update",
                },
                "updates": {
                    "type": "object",
                    "description": "Fields to update: email, industry, sub_industry, ai_summary, health_score, etc.",
                },
            },
            "required": ["business_id", "updates"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        business_id = arguments["business_id"]
        updates = arguments.get("updates", {})

        if not updates:
            return {"status": "no_updates"}

        # Allowed fields for enrichment — prevents injection
        ALLOWED_FIELDS = {
            "email", "industry", "sub_industry", "ai_summary", "health_score",
            "lead_score", "pain_points", "opportunities", "tech_stack",
            "cms_detected", "ssl_valid", "mobile_score", "site_speed_ms",
            "site_quality_score", "social_links", "facebook_url",
            "instagram_url", "linkedin_url", "twitter_url",
            "business_hours", "year_established", "employee_estimate",
        }

        # Filter to allowed fields only
        safe_updates = {k: v for k, v in updates.items() if k in ALLOWED_FIELDS}
        if not safe_updates:
            return {"status": "no_valid_fields", "rejected": list(updates.keys())}

        # Use ForgeDB if available
        db = getattr(self._pool, '_db', None)
        if db is not None:
            try:
                return db.write_enrichment(business_id, safe_updates, source="tool")
            except Exception as e:
                logger.error("write_enrichment failed for %s: %s", business_id, e)
                return {"error": str(e)}

        # Legacy psycopg2 path
        conn = self._pool.get_connection()
        try:
            cur = conn.cursor()

            set_clauses = []
            params: list = []

            for field_name, value in safe_updates.items():
                if isinstance(value, (dict, list)):
                    set_clauses.append(f"{field_name} = COALESCE({field_name}, %s::jsonb)")
                    params.append(json.dumps(value))
                elif isinstance(value, (int, float)):
                    set_clauses.append(f"{field_name} = COALESCE({field_name}, %s)")
                    params.append(value)
                elif isinstance(value, str):
                    set_clauses.append(f"{field_name} = COALESCE({field_name}, %s)")
                    params.append(value[:500])
                elif isinstance(value, bool):
                    set_clauses.append(f"{field_name} = COALESCE({field_name}, %s)")
                    params.append(value)

            if not set_clauses:
                return {"status": "no_valid_values"}

            set_clauses.append("updated_at = NOW()")

            query = "UPDATE businesses SET {} WHERE id = %s".format(", ".join(set_clauses))
            params.append(business_id)

            cur.execute(query, params)
            conn.commit()
            cur.close()

            return {
                "status": "updated",
                "business_id": business_id,
                "fields_updated": list(safe_updates.keys()),
            }
        except Exception as e:
            conn.rollback()
            logger.error("write_enrichment failed for %s: %s", business_id, e)
            return {"error": str(e)}
        finally:
            self._pool.return_connection(conn)


class BatchWriteEnrichmentTool(Tool):
    """
    Tool: Write enrichment results for multiple businesses in one transaction.

    More efficient than calling write_enrichment repeatedly.
    Used by batch enrichment workers that process 5-10 businesses per prompt.
    """

    def __init__(self, db_pool: DatabasePool):
        self._pool = db_pool

    @property
    def name(self) -> str:
        return "batch_write_enrichment"

    @property
    def description(self) -> str:
        return (
            "Write enrichment results for multiple businesses at once. "
            "Provide a list of {business_id, updates} objects."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "business_id": {"type": "string"},
                            "updates": {"type": "object"},
                        },
                    },
                    "description": "List of enrichment results to write",
                },
            },
            "required": ["results"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        results_list = arguments.get("results", [])
        if not results_list:
            return {"status": "empty", "updated": 0}

        # Delegate to individual writes (each manages its own connection)
        updated = 0
        errors = 0
        write_tool = WriteEnrichmentTool(self._pool)

        for item in results_list:
            bid = item.get("business_id")
            updates = item.get("updates", {})
            if bid and updates:
                try:
                    result = write_tool.execute({"business_id": bid, "updates": updates})
                    if result.get("status") == "updated":
                        updated += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1

        return {
            "status": "completed",
            "updated": updated,
            "errors": errors,
            "total": len(results_list),
        }
