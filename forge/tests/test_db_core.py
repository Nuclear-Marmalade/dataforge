"""Tests for ForgeDB core functionality."""
import uuid
import pytest

from forge.db import ForgeDB


class TestForgeDBSetup:
    def test_from_config_sqlite(self, tmp_path):
        db = ForgeDB.from_config({"db_path": str(tmp_path / "t.db")})
        assert not db.is_postgres
        db.close()

    def test_from_config_requires_db_path_or_host(self):
        with pytest.raises(ValueError, match="db_path.*db_host"):
            ForgeDB.from_config({})

    def test_ensure_schema_creates_table(self, sqlite_db):
        stats = sqlite_db.get_stats()
        assert int(stats["total_records"]) == 0

    def test_is_postgres_false_for_sqlite(self, sqlite_db):
        assert not sqlite_db.is_postgres

    def test_placeholder_is_question_mark_for_sqlite(self, sqlite_db):
        assert sqlite_db.placeholder == "?"

    def test_now_expr_for_sqlite(self, sqlite_db):
        assert "datetime" in sqlite_db.now_expr

    def test_interval_ago_sqlite(self, sqlite_db):
        expr = sqlite_db.interval_ago(7)
        assert "now" in expr and "-7 days" in expr


class TestUpsert:
    def test_upsert_returns_uuid(self, sqlite_db, sample_business):
        bid = sqlite_db.upsert_business(sample_business)
        assert bid
        assert len(bid) == 36  # UUID format

    def test_upsert_increments_count(self, sqlite_db, sample_business):
        sqlite_db.upsert_business(sample_business)
        stats = sqlite_db.get_stats()
        assert int(stats["total_records"]) == 1

    def test_upsert_second_time_does_not_duplicate(self, sqlite_db):
        """Upserting with the same ID should not create a new record."""
        bid = sqlite_db.upsert_business({"name": "Test", "state": "FL"})
        sqlite_db.upsert_business({"id": bid, "name": "Test Updated", "state": "FL"})
        assert int(sqlite_db.get_stats()["total_records"]) == 1

    def test_upsert_coalesce_does_not_overwrite(self, sqlite_db):
        """COALESCE: existing non-null values should not be overwritten."""
        bid = sqlite_db.upsert_business({"name": "Original", "state": "FL", "email": "first@test.com"})
        sqlite_db.upsert_business({"id": bid, "name": "New Name", "state": "TX", "email": "second@test.com"})
        record = sqlite_db.get_business(bid)
        # email was already set, COALESCE should preserve it
        assert record["email"] == "first@test.com"

    def test_upsert_batch(self, sqlite_db):
        records = [
            {"name": f"Biz {i}", "city": "Tampa", "state": "FL"}
            for i in range(10)
        ]
        result = sqlite_db.upsert_batch(records)
        assert result["inserted"] == 10
        stats = sqlite_db.get_stats()
        assert int(stats["total_records"]) == 10

    def test_upsert_batch_empty(self, sqlite_db):
        result = sqlite_db.upsert_batch([])
        assert result["status"] == "empty"
        assert result["inserted"] == 0

    def test_upsert_batch_returns_ids(self, sqlite_db):
        records = [{"name": f"Biz {i}", "state": "FL"} for i in range(3)]
        result = sqlite_db.upsert_batch(records)
        assert len(result["ids"]) == 3
        for bid in result["ids"]:
            assert len(bid) == 36  # UUID format

    def test_upsert_empty_record(self, sqlite_db):
        bid = sqlite_db.upsert_business({})
        # Should still return a UUID even with no valid columns
        assert bid and len(bid) == 36

    def test_upsert_ignores_unknown_columns(self, sqlite_db):
        bid = sqlite_db.upsert_business({"name": "Test", "state": "FL", "nonexistent_col": "value"})
        assert bid


class TestGetBusiness:
    def test_get_business_returns_record(self, sqlite_db, sample_business):
        bid = sqlite_db.upsert_business(sample_business)
        record = sqlite_db.get_business(bid)
        assert record is not None
        assert record["name"] == "Tampa Bay Dental"
        assert record["state"] == "FL"

    def test_get_business_returns_none_for_missing(self, sqlite_db):
        result = sqlite_db.get_business(str(uuid.uuid4()))
        assert result is None


class TestEnrichment:
    def test_write_enrichment_basic(self, sqlite_db):
        bid = sqlite_db.upsert_business({"name": "Test", "state": "FL"})
        result = sqlite_db.write_enrichment(bid, {"email": "a@test.com"}, "test")
        assert result["status"] == "updated"
        assert "email" in result["fields_updated"]

    def test_write_enrichment_coalesce(self, sqlite_db):
        bid = sqlite_db.upsert_business({"name": "Test", "state": "FL"})
        sqlite_db.write_enrichment(bid, {"email": "first@test.com"}, "test")
        sqlite_db.write_enrichment(bid, {"email": "second@test.com"}, "test2")
        # COALESCE: first email should survive
        record = sqlite_db.get_business(bid)
        assert record["email"] == "first@test.com"

    def test_write_enrichment_respects_allowed_fields(self, sqlite_db):
        bid = sqlite_db.upsert_business({"name": "Test", "state": "FL"})
        # Unknown field should be silently dropped
        result = sqlite_db.write_enrichment(bid, {"fake_field": "value"}, "test")
        assert result["status"] == "no_valid_fields"

    def test_write_enrichment_empty_updates(self, sqlite_db):
        bid = sqlite_db.upsert_business({"name": "Test", "state": "FL"})
        result = sqlite_db.write_enrichment(bid, {}, "test")
        assert result["status"] == "no_updates"

    def test_write_enrichment_batch(self, sqlite_db):
        bids = [sqlite_db.upsert_business({"name": f"B{i}", "state": "FL"}) for i in range(5)]
        batch = [(bid, {"email": f"e{i}@test.com"}) for i, bid in enumerate(bids)]
        result = sqlite_db.write_enrichment_batch(batch, source="test")
        assert result["status"] == "completed"
        assert result["updated"] == 5
        stats = sqlite_db.get_stats()
        assert int(stats["with_email"]) == 5

    def test_write_enrichment_batch_empty(self, sqlite_db):
        result = sqlite_db.write_enrichment_batch([], source="test")
        assert result["status"] == "empty"

    def test_write_enrichment_updates_last_enriched(self, sqlite_db):
        bid = sqlite_db.upsert_business({"name": "Test", "state": "FL"})
        sqlite_db.write_enrichment(bid, {"email": "a@test.com"}, "test")
        record = sqlite_db.get_business(bid)
        assert record["last_enriched_at"] is not None


class TestTransactions:
    def test_transaction_commits_on_clean_exit(self, sqlite_db):
        with sqlite_db.transaction() as tx:
            tx.execute(
                "INSERT INTO businesses (id, name, state) VALUES (?, ?, ?)",
                (str(uuid.uuid4()), "TxTest", "FL"),
            )
        assert int(sqlite_db.get_stats()["total_records"]) == 1

    def test_transaction_rollback_on_exception(self, sqlite_db):
        try:
            with sqlite_db.transaction() as tx:
                tx.execute(
                    "INSERT INTO businesses (id, name, state) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), "Rollback", "FL"),
                )
                raise RuntimeError("simulate failure")
        except RuntimeError:
            pass
        assert int(sqlite_db.get_stats()["total_records"]) == 0

    def test_execute_inside_transaction_no_premature_commit(self, sqlite_db):
        """Round 8 regression: db.execute() inside db.transaction() must not auto-commit."""
        try:
            with sqlite_db.transaction() as tx:
                tx.execute(
                    "INSERT INTO businesses (id, name, state) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), "A", "FL"),
                )
                sqlite_db.execute(
                    "INSERT INTO businesses (id, name, state) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), "B", "FL"),
                )
                raise RuntimeError("rollback all")
        except RuntimeError:
            pass
        assert int(sqlite_db.get_stats()["total_records"]) == 0

    def test_tx_fetch_dicts_preserves_row_factory(self, sqlite_db):
        """Round 8 regression: tx.fetch_dicts() must not permanently mutate row_factory."""

        sqlite_db.upsert_business({"name": "Test", "state": "FL"})
        factory_before = sqlite_db._backend._conn.row_factory
        with sqlite_db.transaction() as tx:
            rows = tx.fetch_dicts("SELECT name FROM businesses")
            assert len(rows) == 1
            assert rows[0]["name"] == "Test"
        factory_after = sqlite_db._backend._conn.row_factory
        assert factory_before == factory_after

    def test_transaction_multiple_inserts(self, sqlite_db):
        with sqlite_db.transaction() as tx:
            for i in range(5):
                tx.execute(
                    "INSERT INTO businesses (id, name, state) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), f"Multi{i}", "FL"),
                )
        assert int(sqlite_db.get_stats()["total_records"]) == 5


class TestExport:
    def test_export_csv(self, sqlite_db, tmp_path, sample_business):
        sqlite_db.upsert_business(sample_business)
        out = str(tmp_path / "out.csv")
        result = sqlite_db.export_csv(out)
        assert result["row_count"] >= 1
        assert result["status"] == "completed"

    def test_export_csv_file_exists(self, sqlite_db, tmp_path, sample_business):
        sqlite_db.upsert_business(sample_business)
        out = str(tmp_path / "out.csv")
        sqlite_db.export_csv(out)
        assert os.path.exists(out)
        import csv as csv_mod
        with open(out) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1

    def test_export_with_filter(self, sqlite_db, tmp_path):
        sqlite_db.upsert_business({"name": "A", "state": "FL", "email": "a@b.com"})
        sqlite_db.upsert_business({"name": "B", "state": "FL"})
        out = str(tmp_path / "out.csv")
        result = sqlite_db.export_csv(out, where="with_email")
        assert result["row_count"] == 1

    def test_export_rejects_raw_sql(self, sqlite_db, tmp_path):
        sqlite_db.upsert_business({"name": "A", "state": "FL"})
        out = str(tmp_path / "out.csv")
        result = sqlite_db.export_csv(out, where="1=1 UNION SELECT sql FROM sqlite_master")
        # Should export ALL records (filter rejected -> no WHERE), not the schema
        assert result["row_count"] == 1

    def test_export_json(self, sqlite_db, tmp_path, sample_business):
        sqlite_db.upsert_business(sample_business)
        out = str(tmp_path / "out.json")
        result = sqlite_db.export_json(out)
        assert result["status"] == "completed"
        assert result["row_count"] >= 1

    def test_export_empty_db(self, sqlite_db, tmp_path):
        out = str(tmp_path / "out.csv")
        result = sqlite_db.export_csv(out)
        assert result["row_count"] == 0

    def test_resolve_where_rejects_unknown(self, sqlite_db):
        result = sqlite_db._resolve_where("1=1; DROP TABLE businesses;")
        assert result is None

    def test_resolve_where_accepts_known_filters(self, sqlite_db):
        for name in sqlite_db.SAFE_WHERE_FILTERS:
            result = sqlite_db._resolve_where(name)
            # "all" maps to None, all others should return SQL
            if name == "all":
                assert result is None
            else:
                assert result is not None


class TestCSVImport:
    def test_import_csv(self, sqlite_db, sample_csv):
        result = sqlite_db.import_csv(sample_csv, return_details=True)
        assert result["imported"] == 3

    def test_import_csv_column_mapping(self, sqlite_db, sample_csv):
        result = sqlite_db.import_csv(sample_csv, return_details=True)
        assert "Business Name" in result.get("column_mapping", {})

    def test_import_csv_return_count(self, sqlite_db, sample_csv):
        """import_csv without return_details returns an int count."""
        result = sqlite_db.import_csv(sample_csv, return_details=False)
        assert result == 3

    def test_import_csv_missing_file(self, sqlite_db, tmp_path):
        result = sqlite_db.import_csv(str(tmp_path / "nonexistent.csv"), return_details=True)
        assert result["status"] == "error"

    def test_import_csv_no_headers(self, sqlite_db, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        result = sqlite_db.import_csv(str(path), return_details=True)
        assert result["status"] == "error"

    def test_import_csv_unrecognized_columns(self, sqlite_db, tmp_path):
        path = tmp_path / "unknown.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["FoozleBar", "BazQuux"])
            w.writerow(["val1", "val2"])
        result = sqlite_db.import_csv(str(path), return_details=True)
        assert result["status"] == "error"
        assert "No recognizable columns" in result.get("error", "")


class TestCount:
    def test_count_all(self, sqlite_db, sample_business):
        sqlite_db.upsert_business(sample_business)
        assert sqlite_db.count() >= 1

    def test_count_with_filter(self, sqlite_db):
        sqlite_db.upsert_business({"name": "A", "email": "a@b.com", "state": "FL"})
        sqlite_db.upsert_business({"name": "B", "state": "FL"})
        assert sqlite_db.count(where="with_email") >= 1

    def test_count_returns_zero_for_empty_db(self, sqlite_db):
        assert sqlite_db.count() == 0

    def test_count_rejects_sql_injection(self, sqlite_db):
        sqlite_db.upsert_business({"name": "A", "state": "FL"})
        # SQL injection filter should be rejected, returning all records count
        count = sqlite_db.count(where="1=1; DROP TABLE businesses;")
        assert count >= 0  # Must not crash


class TestFetchDicts:
    def test_fetch_dicts_returns_list_of_dicts(self, sqlite_db, sample_business):
        sqlite_db.upsert_business(sample_business)
        rows = sqlite_db.fetch_dicts("SELECT name, state FROM businesses")
        assert len(rows) == 1
        assert isinstance(rows[0], dict)
        assert rows[0]["name"] == "Tampa Bay Dental"

    def test_fetch_dicts_with_params(self, sqlite_db):
        sqlite_db.upsert_business({"name": "A", "state": "FL"})
        sqlite_db.upsert_business({"name": "B", "state": "TX"})
        rows = sqlite_db.fetch_dicts("SELECT name FROM businesses WHERE state = ?", ("FL",))
        assert len(rows) == 1
        assert rows[0]["name"] == "A"


class TestFetchForEnrichment:
    def test_fetch_email_mode(self, sqlite_db):
        sqlite_db.upsert_business({"name": "Has Website", "state": "FL", "website_url": "https://example.com"})
        sqlite_db.upsert_business({"name": "No Website", "state": "FL"})
        results = sqlite_db.fetch_for_enrichment(mode="email", limit=10)
        # Should include Has Website (has website, no email)
        assert any(r["name"] == "Has Website" for r in results)

    def test_fetch_with_limit(self, sqlite_db):
        for i in range(10):
            sqlite_db.upsert_business({"name": f"Biz{i}", "state": "FL", "website_url": "https://example.com"})
        results = sqlite_db.fetch_for_enrichment(mode="email", limit=3)
        assert len(results) == 3


class TestPrepareValue:
    def test_json_column_serializes_dict(self, sqlite_db):
        val = sqlite_db._prepare_value_for_write("tech_stack", ["React", "Node.js"])
        assert isinstance(val, str)
        import json
        assert json.loads(val) == ["React", "Node.js"]

    def test_boolean_column_returns_int_for_sqlite(self, sqlite_db):
        val = sqlite_db._prepare_value_for_write("ssl_valid", True)
        assert val == 1

    def test_string_truncated_to_1000(self, sqlite_db):
        val = sqlite_db._prepare_value_for_write("name", "x" * 2000)
        assert len(val) == 1000


import csv  # noqa: E402
import os  # noqa: E402
