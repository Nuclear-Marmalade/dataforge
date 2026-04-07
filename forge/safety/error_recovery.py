"""
FORGE Error Recovery & Corruption Prevention — Guardrails for database writes.

Provides:
  - Field validation before write (length, type, range)
  - Enrichment log table (before/after values for rollback)
  - Batch failure threshold (>10% fails → pause)
  - Rollback by timestamp range
  - COALESCE enforcement (never overwrite non-null with null)

Dependencies: psycopg2
Depended on by: enrichment/pipeline.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

# psycopg2 imported lazily in methods that need it

logger = logging.getLogger("forge.safety.error_recovery")


# ── Validation rules per field ───────────────────────────────────────────────

FIELD_VALIDATORS = {
    "email": {
        "type": str,
        "max_length": 254,
        "pattern_check": lambda v: "@" in v and "." in v.split("@")[-1],
    },
    "industry": {
        "type": str,
        "max_length": 100,
        "whitelist": [
            "restaurant", "salon", "real-estate", "dentist", "gym",
            "lawyer", "landscaping", "barber", "cleaning-service", "chiropractor",
            "veterinarian", "auto-repair", "tattoo-shop", "accountant", "plumber",
            "photographer", "dog-groomer", "electrician", "food-truck", "personal-trainer",
        ],
    },
    "ai_summary": {
        "type": str,
        "min_length": 10,
        "max_length": 500,
    },
    "health_score": {
        "type": int,
        "min_value": 0,
        "max_value": 100,
    },
    "tech_stack": {
        "type": str,
        "max_length": 2000,
    },
    "cms_detected": {
        "type": str,
        "max_length": 100,
    },
    "ssl_valid": {
        "type": bool,
    },
    "site_speed_ms": {
        "type": int,
        "min_value": 0,
        "max_value": 60000,
    },
    "pain_points": {
        "type": (list, dict),
        "max_length": 5000,
    },
}

# If >10% of a batch fails, pause
BATCH_FAILURE_THRESHOLD = 0.10


def validate_field(field_name: str, value: Any) -> tuple[bool, str]:
    """
    Validate a single field value against its rules.

    Returns (is_valid, error_message).
    """
    rules: Any = FIELD_VALIDATORS.get(field_name)
    if not rules:
        return True, ""  # No rules = allow

    # Type check
    expected_type = rules.get("type")
    if expected_type:
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                return False, f"{field_name}: expected {expected_type}, got {type(value).__name__}"
        elif not isinstance(value, expected_type):
            # Allow int for bool fields, string for int fields with conversion
            if expected_type is int and isinstance(value, (float, str)):
                try:
                    int(value)
                except (ValueError, TypeError):
                    return False, f"{field_name}: cannot convert to int"
            elif expected_type is bool and isinstance(value, int):
                pass  # int is ok for bool
            else:
                return False, f"{field_name}: expected {expected_type.__name__}, got {type(value).__name__}"

    # String length checks
    if isinstance(value, str):
        max_len = rules.get("max_length")
        if max_len and len(value) > max_len:
            return False, f"{field_name}: length {len(value)} exceeds max {max_len}"
        min_len = rules.get("min_length")
        if min_len and len(value) < min_len:
            return False, f"{field_name}: length {len(value)} below min {min_len}"

    # Numeric range checks
    if isinstance(value, (int, float)):
        min_val = rules.get("min_value")
        if min_val is not None and value < min_val:
            return False, f"{field_name}: value {value} below min {min_val}"
        max_val = rules.get("max_value")
        if max_val is not None and value > max_val:
            return False, f"{field_name}: value {value} above max {max_val}"

    # Whitelist check
    whitelist = rules.get("whitelist")
    if whitelist and isinstance(value, str) and value.lower() not in whitelist:
        return False, f"{field_name}: '{value}' not in whitelist"

    # Pattern check
    pattern_fn = rules.get("pattern_check")
    if pattern_fn and not pattern_fn(value):
        return False, f"{field_name}: failed pattern check"

    return True, ""


def validate_updates(updates: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """
    Validate all updates, returning (valid_updates, errors).
    """
    valid = {}
    errors = []

    for field, value in updates.items():
        if value is None:
            errors.append(f"{field}: null value rejected (COALESCE)")
            continue

        is_valid, err = validate_field(field, value)
        if is_valid:
            valid[field] = value
        else:
            errors.append(err)

    return valid, errors


class EnrichmentLogger:
    """
    Logs every enrichment write with before/after values for rollback.

    Creates and writes to the `enrichment_log` table.
    """

    def __init__(self, db_pool: Any):
        """
        Initialize the enrichment logger. Requires PostgreSQL backend.

        Raises ImportError if psycopg2 is not available.
        """
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError(
                "EnrichmentLogger requires PostgreSQL (pip install psycopg2-binary)"
            )
        self._db = db_pool
        self._ensure_table()
        self._batch_total = 0
        self._batch_failures = 0

    def _ensure_table(self) -> None:
        """Create enrichment_log table if not exists."""
        conn = self._db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS enrichment_log (
                    id SERIAL PRIMARY KEY,
                    business_id UUID NOT NULL,
                    field_name VARCHAR(100) NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    source VARCHAR(50) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_enrichment_log_biz
                ON enrichment_log (business_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_enrichment_log_time
                ON enrichment_log (created_at)
            """)
            conn.commit()
            cur.close()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to create enrichment_log table: %s", e)
        finally:
            self._db.return_connection(conn)

    def log_write(
        self,
        business_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        source: str,
    ) -> None:
        """Log a single field write for audit trail."""
        conn = self._db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO enrichment_log (business_id, field_name, old_value, new_value, source)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    business_id,
                    field_name,
                    str(old_value)[:500] if old_value is not None else None,
                    str(new_value)[:500] if new_value is not None else None,
                    source,
                ),
            )
            conn.commit()
            cur.close()
        except Exception as e:
            conn.rollback()
            logger.debug("log_write failed: %s", e)
        finally:
            self._db.return_connection(conn)

    def record_batch_result(self, success: bool) -> bool:
        """
        Track batch success/failure.

        Returns True if enrichment should continue, False if paused.
        """
        self._batch_total += 1
        if not success:
            self._batch_failures += 1

        if self._batch_total >= 10:
            failure_rate = self._batch_failures / self._batch_total
            if failure_rate > BATCH_FAILURE_THRESHOLD:
                logger.critical(
                    "BATCH FAILURE THRESHOLD: %.1f%% failures (%d/%d). PAUSING.",
                    failure_rate * 100,
                    self._batch_failures,
                    self._batch_total,
                )
                return False

        return True

    def reset_batch_counter(self) -> None:
        """Reset batch counters (call at start of each batch)."""
        self._batch_total = 0
        self._batch_failures = 0


def rollback_enrichment(
    db_pool: Any,
    start_time: str,
    end_time: str,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Rollback enrichment changes in a timestamp range.

    Reads enrichment_log entries and reverts fields to old_value.

    Args:
        db_pool: Database connection pool.
        start_time: ISO timestamp for range start.
        end_time: ISO timestamp for range end.
        dry_run: If True, only report what would be rolled back.

    Returns:
        Dict with rollback summary.
    """
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise ImportError("Safety module requires psycopg2: pip install psycopg2-binary")

    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """SELECT business_id, field_name, old_value, new_value, created_at
               FROM enrichment_log
               WHERE created_at >= %s AND created_at <= %s
               ORDER BY created_at DESC""",
            (start_time, end_time),
        )
        entries = cur.fetchall()

        if not entries:
            return {"status": "no_entries", "count": 0}

        if dry_run:
            return {
                "status": "dry_run",
                "count": len(entries),
                "affected_businesses": len(set(e["business_id"] for e in entries)),
                "fields": list(set(e["field_name"] for e in entries)),
            }

        # Execute rollback
        rolled_back = 0
        ALLOWED = {
            "email", "industry", "sub_industry", "ai_summary", "health_score",
            "tech_stack", "ssl_valid", "cms_detected", "site_speed_ms", "pain_points",
        }

        for entry in entries:
            field = entry["field_name"]
            if field not in ALLOWED:
                continue

            import re
            if not re.match(r'^[a-z_]+$', field):
                logger.error("Invalid field name in rollback: %s", field)
                continue

            old_val = entry["old_value"]
            biz_id = entry["business_id"]

            try:
                if old_val is None:
                    cur.execute(
                        f"UPDATE businesses SET {field} = NULL, updated_at = NOW() WHERE id = %s",
                        (biz_id,),
                    )
                else:
                    cur.execute(
                        f"UPDATE businesses SET {field} = %s, updated_at = NOW() WHERE id = %s",
                        (old_val, biz_id),
                    )
                rolled_back += 1
            except Exception as e:
                logger.error("Rollback failed for %s.%s: %s", biz_id, field, e)

        conn.commit()
        cur.close()

        return {
            "status": "rolled_back",
            "count": rolled_back,
            "total_entries": len(entries),
        }

    except Exception as e:
        conn.rollback()
        logger.error("Rollback failed: %s", e)
        return {"status": "error", "error": str(e)}
    finally:
        db_pool.return_connection(conn)
