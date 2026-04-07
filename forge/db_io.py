"""
FORGE Database I/O Mixin -- CSV import/export and upsert operations.

Extracted from db.py to keep it under 800 lines.
Provides _ForgeDBIOMixin which ForgeDB inherits from.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from forge.db_schema import ENRICHABLE_FIELDS, JSON_COLUMNS

if TYPE_CHECKING:
    pass

logger = logging.getLogger("forge.db")


class _ForgeDBIOMixin:
    """Mixin providing CSV import/export and upsert operations for ForgeDB.

    Type stubs below declare attributes that ForgeDB provides at runtime.
    This lets mypy verify the mixin without a circular import.
    """

    # -- provided by ForgeDB at runtime --
    _backend: Any

    @property
    def is_postgres(self) -> bool:  # type: ignore[empty-body]
        ...

    def _prepare_value_for_write(self, column: str, value: Any) -> Any:
        raise NotImplementedError

    def _resolve_where(self, where: Optional[str]) -> Optional[str]:
        raise NotImplementedError

    def fetch_dicts(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        raise NotImplementedError

    # ── Upsert operations ───────────────────────────────────────────────────

    def upsert_business(self, data: Dict[str, Any]) -> str:
        """Insert or update a business record. Returns business ID."""
        safe_data = {k: v for k, v in data.items() if k in ENRICHABLE_FIELDS or k == "id"}
        business_id = safe_data.pop("id", None) or str(uuid.uuid4())
        if not safe_data:
            logger.warning("upsert_business called with no valid columns")
            return business_id
        if self.is_postgres:
            return self._upsert_business_pg(business_id, safe_data)
        return self._upsert_business_sqlite(business_id, safe_data)

    def _upsert_business_sqlite(self, business_id: str, data: Dict[str, Any]) -> str:
        """SQLite upsert using INSERT OR REPLACE logic with COALESCE."""
        with self._backend.write_connection() as conn:
            row = conn.execute("SELECT id FROM businesses WHERE id = ?", (business_id,)).fetchone()
            if row:
                set_clauses = []
                params: list = []
                for col, val in data.items():
                    if col not in ENRICHABLE_FIELDS:
                        continue
                    set_clauses.append(f"{col} = COALESCE({col}, ?)")
                    params.append(self._prepare_value_for_write(col, val))
                if set_clauses:
                    set_clauses.append(f"updated_at = {self._backend.now_expr()}")
                    conn.execute(f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = ?", params + [business_id])
                    conn.commit()
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
                conn.execute(f"INSERT INTO businesses ({', '.join(columns)}) VALUES ({', '.join(placeholders)})", params)
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
            placeholders.append("%s::jsonb" if col in JSON_COLUMNS else "%s")
            params.append(self._prepare_value_for_write(col, val))
        conflict_sets = [f"{col} = COALESCE(businesses.{col}, EXCLUDED.{col})" for col in columns[1:]]
        conflict_sets.append("updated_at = NOW()")
        query = (f"INSERT INTO businesses ({', '.join(columns)}) VALUES ({', '.join(placeholders)}) "
                 f"ON CONFLICT (id) DO UPDATE SET {', '.join(conflict_sets)} RETURNING id")
        with self._backend.write_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            result = cur.fetchone()
            conn.commit()
            cur.close()
        return str(result[0]) if result else business_id

    def upsert_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert or update multiple business records in a single transaction."""
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
                    upsert_fn = self._upsert_single_pg_in_txn if self.is_postgres else self._upsert_single_sqlite_in_txn
                    if upsert_fn(conn, business_id, safe_data):
                        inserted += 1
                    else:
                        updated_count += 1
                conn.commit()
            except Exception as e:
                if self.is_postgres:
                    conn.rollback()
                logger.error("upsert_batch failed: %s", e)
                return {"status": "error", "inserted": inserted, "updated": updated_count, "ids": ids, "error": str(e)}
        return {"status": "completed", "inserted": inserted, "updated": updated_count, "ids": ids}

    def _upsert_single_sqlite_in_txn(self, conn: Any, business_id: str, data: Dict[str, Any]) -> bool:
        """SQLite upsert within existing transaction. Returns True if inserted."""
        row = conn.execute("SELECT id FROM businesses WHERE id = ?", (business_id,)).fetchone()
        if row:
            set_clauses = []
            params: list = []
            for col, val in data.items():
                if col not in ENRICHABLE_FIELDS:
                    continue
                set_clauses.append(f"{col} = COALESCE({col}, ?)")
                params.append(self._prepare_value_for_write(col, val))
            if set_clauses:
                set_clauses.append(f"updated_at = {self._backend.now_expr()}")
                conn.execute(f"UPDATE businesses SET {', '.join(set_clauses)} WHERE id = ?", params + [business_id])
            return False
        columns = ["id"]
        placeholders = ["?"]
        params = [business_id]
        for col, val in data.items():
            if col not in ENRICHABLE_FIELDS:
                continue
            columns.append(col)
            placeholders.append("?")
            params.append(self._prepare_value_for_write(col, val))
        conn.execute(f"INSERT INTO businesses ({', '.join(columns)}) VALUES ({', '.join(placeholders)})", params)
        return True

    def _upsert_single_pg_in_txn(self, conn: Any, business_id: str, data: Dict[str, Any]) -> bool:
        """PostgreSQL upsert within existing transaction. Returns True if inserted."""
        columns = ["id"]
        placeholders = ["%s"]
        params: list = [business_id]
        for col, val in data.items():
            if col not in ENRICHABLE_FIELDS:
                continue
            columns.append(col)
            placeholders.append("%s::jsonb" if col in JSON_COLUMNS else "%s")
            params.append(self._prepare_value_for_write(col, val))
        conflict_sets = [f"{col} = COALESCE(businesses.{col}, EXCLUDED.{col})" for col in columns[1:]]
        conflict_sets.append("updated_at = NOW()")
        query = (f"INSERT INTO businesses ({', '.join(columns)}) VALUES ({', '.join(placeholders)}) "
                 f"ON CONFLICT (id) DO UPDATE SET {', '.join(conflict_sets)} RETURNING (xmax = 0) AS was_insert")
        cur = conn.cursor()
        cur.execute(query, params)
        result = cur.fetchone()
        cur.close()
        return bool(result and result[0])

    # ── CSV Import/Export ────────────────────────────────────────────────────

    @staticmethod
    def _detect_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
        """Map CSV headers to canonical column names."""
        from forge.db_schema import COLUMN_ALIASES
        column_mapping: Dict[str, str] = {}
        for header in fieldnames:
            canonical = COLUMN_ALIASES.get(header.lower().strip())
            if canonical:
                column_mapping[header] = canonical
            elif header.lower().strip() in ENRICHABLE_FIELDS:
                column_mapping[header] = header.lower().strip()
        return column_mapping

    @staticmethod
    def _map_row(row: Dict[str, str], column_mapping: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Map a single CSV row to a canonical record dict."""
        record: Dict[str, Any] = {}
        for csv_col, db_col in column_mapping.items():
            val = row.get(csv_col, "").strip()
            if val:
                record[db_col] = val
        if not record:
            return None
        if "state" in record:
            state_val = record["state"].upper().strip()
            if len(state_val) == 2:
                record["state"] = state_val
            else:
                del record["state"]
        return record

    def _insert_batch(self, records_batch: List[Dict[str, Any]]) -> int:
        """Upsert a batch and return count of records written."""
        if not records_batch:
            return 0
        result = self.upsert_batch(records_batch)
        return result.get("inserted", 0) + result.get("updated", 0)

    def _read_and_batch_rows(self, filepath: str) -> tuple:
        """Read CSV, map columns, batch rows."""
        total_rows = imported = skipped = 0
        with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return 0, 0, 0, None
            column_mapping = self._detect_columns(reader.fieldnames)
            if not column_mapping:
                return 0, 0, 0, None
            logger.info("CSV import: %s -> mapped %d columns", filepath, len(column_mapping))
            records_batch: List[Dict[str, Any]] = []
            for row in reader:
                total_rows += 1
                record = self._map_row(row, column_mapping)
                if record is None:
                    skipped += 1
                    continue
                records_batch.append(record)
                if len(records_batch) >= 500:
                    imported += self._insert_batch(records_batch)
                    records_batch = []
            imported += self._insert_batch(records_batch)
        return total_rows, imported, skipped, column_mapping

    def import_csv(self, filepath: str, return_details: bool = False) -> Any:
        """Import business records from a CSV file."""
        if not os.path.exists(filepath):
            if return_details:
                return {"status": "error", "error": f"File not found: {filepath}"}
            raise FileNotFoundError(f"File not found: {filepath}")
        try:
            total_rows, imported, skipped, column_mapping = self._read_and_batch_rows(filepath)
            if column_mapping is None:
                return {"status": "error", "error": "No recognizable columns or headers"}
        except Exception as e:
            logger.error("CSV import failed: %s", e)
            return {"status": "error", "error": str(e)}
        logger.info("CSV import complete: %d total, %d imported, %d skipped", total_rows, imported, skipped)
        details = {"status": "completed", "total_rows": total_rows, "imported": imported,
                    "skipped": skipped, "new": imported, "updated": 0, "column_mapping": column_mapping}
        return details if return_details else imported

    def _write_rows_to_csv(self, filepath: str, rows: List[Any], fieldnames: List[str], as_dicts: bool) -> int:
        """Write rows to a CSV file. Returns count written."""
        exported = 0
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row) if as_dicts else dict(zip(fieldnames, row)))
                exported += 1
        return exported

    def export_csv(self, filepath: str, where: Optional[str] = None, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Export business records to a CSV file."""
        where = self._resolve_where(where)
        query = "SELECT * FROM businesses"
        query_params: list = params or []
        if where:
            query += f" WHERE {where}"
        query += " ORDER BY created_at DESC"
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
                    exported = self._write_rows_to_csv(filepath, rows, list(rows[0].keys()), True)
                else:
                    cursor = conn.execute(query, query_params)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    exported = self._write_rows_to_csv(filepath, rows, columns, False)
            except Exception as e:
                logger.error("CSV export failed: %s", e)
                return {"status": "error", "error": str(e), "row_count": 0}
        logger.info("CSV export complete: %d rows -> %s", exported, filepath)
        return {"status": "completed", "row_count": exported}

    def export_json(self, filepath: str, where: Optional[str] = None, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Export business records to a JSON file."""
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
            logger.info("JSON export complete: %d rows -> %s", len(rows), filepath)
            return {"status": "completed", "row_count": len(rows)}
        except Exception as e:
            logger.error("JSON export failed: %s", e)
            return {"status": "error", "error": str(e), "row_count": 0}
