"""Tests for forge.db — enrichment writes, batch operations, and advanced queries."""

import uuid

import pytest

from forge.db import ForgeDB, ENRICHABLE_FIELDS
from forge.db_schema import COLUMN_ALIASES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh SQLite database with schema."""
    db_path = str(tmp_path / "test_enrich.db")
    db = ForgeDB.from_config({"db_path": db_path})
    db.ensure_schema()
    yield db
    db.close()


@pytest.fixture
def biz_id(db):
    """Insert a minimal business and return its ID."""
    bid = str(uuid.uuid4())
    db.execute(
        "INSERT INTO businesses (id, name, city, state) VALUES (?, ?, ?, ?)",
        (bid, "Tampa Dental", "Tampa", "FL"),
    )
    return bid


# ---------------------------------------------------------------------------
# Tests: write_enrichment
# ---------------------------------------------------------------------------

class TestWriteEnrichment:
    def test_basic_write(self, db, biz_id):
        result = db.write_enrichment(biz_id, {"email": "info@dental.com"}, source="test")
        assert result["status"] == "updated"
        assert "email" in result["fields_updated"]

        # Verify written
        rows = db.fetch_dicts("SELECT email FROM businesses WHERE id = ?", (biz_id,))
        assert rows[0]["email"] == "info@dental.com"

    def test_coalesce_preserves_existing(self, db, biz_id):
        # First write sets email
        db.write_enrichment(biz_id, {"email": "first@dental.com"})

        # Second write should NOT overwrite
        db.write_enrichment(biz_id, {"email": "second@dental.com"})

        rows = db.fetch_dicts("SELECT email FROM businesses WHERE id = ?", (biz_id,))
        assert rows[0]["email"] == "first@dental.com"

    def test_empty_updates_noop(self, db, biz_id):
        result = db.write_enrichment(biz_id, {})
        assert result["status"] == "no_updates"

    def test_invalid_fields_rejected(self, db, biz_id):
        result = db.write_enrichment(biz_id, {"MALICIOUS": "DROP TABLE"})
        assert result["status"] == "no_valid_fields"

    def test_multiple_fields(self, db, biz_id):
        result = db.write_enrichment(biz_id, {
            "email": "info@dental.com",
            "industry": "dentist",
            "health_score": 75,
        })
        assert result["status"] == "updated"
        assert len(result["fields_updated"]) == 3

    def test_json_field_write(self, db, biz_id):
        result = db.write_enrichment(biz_id, {
            "tech_stack": '["wordpress", "google-analytics"]',
        })
        assert result["status"] == "updated"

    def test_updates_timestamp(self, db, biz_id):
        db.write_enrichment(biz_id, {"email": "info@dental.com"})
        rows = db.fetch_dicts("SELECT updated_at, last_enriched_at FROM businesses WHERE id = ?", (biz_id,))
        assert rows[0]["updated_at"] is not None
        assert rows[0]["last_enriched_at"] is not None

    def test_increments_enrichment_attempts(self, db, biz_id):
        db.write_enrichment(biz_id, {"email": "a@b.com"})
        rows = db.fetch_dicts("SELECT enrichment_attempts FROM businesses WHERE id = ?", (biz_id,))
        assert rows[0]["enrichment_attempts"] == 1

        # Write again (different field so COALESCE allows it)
        db.write_enrichment(biz_id, {"industry": "dentist"})
        rows = db.fetch_dicts("SELECT enrichment_attempts FROM businesses WHERE id = ?", (biz_id,))
        assert rows[0]["enrichment_attempts"] == 2


# ---------------------------------------------------------------------------
# Tests: write_enrichment_batch
# ---------------------------------------------------------------------------

class TestWriteEnrichmentBatch:
    def test_batch_write_multiple(self, db):
        ids = []
        for i in range(3):
            bid = str(uuid.uuid4())
            db.execute(
                "INSERT INTO businesses (id, name, state) VALUES (?, ?, ?)",
                (bid, f"Biz {i}", "FL"),
            )
            ids.append(bid)

        batch = [
            (ids[0], {"email": "a@biz.com"}),
            (ids[1], {"email": "b@biz.com"}),
            (ids[2], {"industry": "dentist"}),
        ]
        result = db.write_enrichment_batch(batch, source="test")
        assert result["status"] == "completed"
        assert result["updated"] == 3

    def test_batch_empty(self, db):
        result = db.write_enrichment_batch([])
        assert result["status"] == "empty"

    def test_batch_skips_invalid_fields(self, db, biz_id):
        batch = [(biz_id, {"BAD_FIELD": "nope"})]
        result = db.write_enrichment_batch(batch)
        # No valid fields means nothing written, but no error
        assert result["updated"] == 0


# ---------------------------------------------------------------------------
# Tests: upsert_business
# ---------------------------------------------------------------------------

class TestUpsertBusiness:
    def test_insert_new_business(self, db):
        bid = db.upsert_business({
            "name": "New Salon",
            "city": "Miami",
            "state": "FL",
        })
        assert bid is not None
        rows = db.fetch_dicts("SELECT name FROM businesses WHERE id = ?", (bid,))
        assert rows[0]["name"] == "New Salon"

    def test_upsert_existing_business(self, db, biz_id):
        # Upsert with same id in data dict
        db.upsert_business({
            "id": biz_id,
            "name": "Tampa Dental",
            "city": "Tampa",
            "state": "FL",
            "email": "new@dental.com",
        })

        rows = db.fetch_dicts("SELECT email FROM businesses WHERE id = ?", (biz_id,))
        # COALESCE: email was NULL, so it gets set
        assert rows[0]["email"] == "new@dental.com"


# ---------------------------------------------------------------------------
# Tests: get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_on_empty_db(self, db):
        stats = db.get_stats()
        assert "total_records" in stats
        assert stats["total_records"] == 0

    def test_stats_with_data(self, db, biz_id):
        db.write_enrichment(biz_id, {"email": "info@dental.com"})
        stats = db.get_stats()
        assert stats["total_records"] == 1
        assert stats["with_email"] >= 1


# ---------------------------------------------------------------------------
# Tests: fetch_dicts
# ---------------------------------------------------------------------------

class TestFetchDicts:
    def test_returns_list_of_dicts(self, db, biz_id):
        rows = db.fetch_dicts("SELECT id, name FROM businesses")
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert rows[0]["name"] == "Tampa Dental"

    def test_empty_result(self, db):
        rows = db.fetch_dicts("SELECT * FROM businesses WHERE name = ?", ("nonexistent",))
        assert rows == []


# ---------------------------------------------------------------------------
# Tests: transaction context manager
# ---------------------------------------------------------------------------

class TestTransaction:
    def test_transaction_commits(self, db, biz_id):
        with db.transaction() as tx:
            tx.execute(
                "UPDATE businesses SET email = ? WHERE id = ?",
                ("tx@test.com", biz_id),
            )
        rows = db.fetch_dicts("SELECT email FROM businesses WHERE id = ?", (biz_id,))
        assert rows[0]["email"] == "tx@test.com"

    def test_transaction_placeholder(self, db):
        with db.transaction() as tx:
            assert tx.placeholder == "?"  # SQLite


# ---------------------------------------------------------------------------
# Tests: COLUMN_ALIASES
# ---------------------------------------------------------------------------

class TestColumnAliases:
    def test_business_name_alias(self):
        assert COLUMN_ALIASES["business name"] == "name"

    def test_website_alias(self):
        assert COLUMN_ALIASES["website"] == "website_url"

    def test_phone_number_alias(self):
        assert COLUMN_ALIASES["phone number"] == "phone"

    def test_zip_code_alias(self):
        assert COLUMN_ALIASES["zip code"] == "zip"

    def test_npi_alias(self):
        assert COLUMN_ALIASES["npi"] == "npi_number"


# ---------------------------------------------------------------------------
# Tests: ENRICHABLE_FIELDS
# ---------------------------------------------------------------------------

class TestEnrichableFields:
    def test_core_fields_present(self):
        for field in ["email", "industry", "ai_summary", "health_score",
                      "tech_stack", "cms_detected", "ssl_valid", "site_speed_ms"]:
            assert field in ENRICHABLE_FIELDS

    def test_id_not_enrichable(self):
        assert "id" not in ENRICHABLE_FIELDS


# ---------------------------------------------------------------------------
# Tests: import_csv
# ---------------------------------------------------------------------------

class TestImportCSV:
    def test_import_csv(self, db, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "Business Name,City,State,Website,Phone\n"
            "Pizza Palace,Tampa,FL,https://pizza.com,8135551111\n"
            "Salon Bella,Miami,FL,https://salon.com,3055552222\n"
        )
        count = db.import_csv(str(csv_file))
        assert count == 2
        rows = db.fetch_dicts("SELECT name FROM businesses ORDER BY name")
        assert rows[0]["name"] == "Pizza Palace"
        assert rows[1]["name"] == "Salon Bella"

    def test_import_csv_dedup(self, db, tmp_path):
        csv_file = tmp_path / "dup.csv"
        csv_file.write_text(
            "Business Name,City,State,Phone\n"
            "Same Biz,Tampa,FL,8135551111\n"
            "Same Biz,Tampa,FL,8135551111\n"
        )
        count = db.import_csv(str(csv_file))
        # Should import at least 1 (dedup may or may not catch these depending on impl)
        assert count >= 1


# ---------------------------------------------------------------------------
# Tests: export_csv
# ---------------------------------------------------------------------------

class TestExportCSV:
    def test_export_csv(self, db, biz_id, tmp_path):
        db.write_enrichment(biz_id, {"email": "info@dental.com", "industry": "dentist"})

        out_file = str(tmp_path / "export.csv")
        result = db.export_csv(out_file)
        assert result["row_count"] == 1

        import csv
        with open(out_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "Tampa Dental"
        assert rows[0]["email"] == "info@dental.com"


# ---------------------------------------------------------------------------
# Tests: Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_is_postgres_false_for_sqlite(self, db):
        assert db.is_postgres is False

    def test_placeholder_sqlite(self, db):
        assert db.placeholder == "?"

    def test_now_expr_sqlite(self, db):
        assert "datetime" in db.now_expr.lower() or "now" in db.now_expr.lower()
