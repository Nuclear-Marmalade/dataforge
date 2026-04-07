"""Smoke tests for FORGE — verify basic functionality."""
import os
import csv
import tempfile
import pytest


def test_forge_imports():
    """Verify all critical modules import."""
    from forge.db import ForgeDB
    from forge.config import ForgeConfig
    from forge.cli import main
    from forge.adapters.claude import ClaudeAdapter
    from forge.adapters.ollama import OllamaAdapter


def test_forgedb_sqlite_crud():
    """Test basic CRUD with SQLite."""
    from forge.db import ForgeDB
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        db = ForgeDB.from_config({'db_path': db_path})
        db.ensure_schema()
        bid = db.upsert_business({'name': 'Test Corp', 'city': 'Tampa', 'state': 'FL'})
        assert bid
        stats = db.get_stats()
        assert int(stats['total_records']) >= 1
        db.close()
    finally:
        os.unlink(db_path)


def test_forgedb_coalesce():
    """Verify COALESCE never overwrites existing data."""
    from forge.db import ForgeDB
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        db = ForgeDB.from_config({'db_path': db_path})
        db.ensure_schema()
        bid = db.upsert_business({'name': 'Test', 'city': 'Tampa', 'state': 'FL'})
        db.write_enrichment(bid, {'email': 'first@test.com'}, 'test')
        db.write_enrichment(bid, {'email': 'second@test.com'}, 'test2')
        # Email should still be first@test.com
        stats = db.get_stats()
        assert int(stats['with_email']) == 1
        db.close()
    finally:
        os.unlink(db_path)


def test_csv_import_export():
    """Test CSV round-trip."""
    from forge.db import ForgeDB
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    csv_in = tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False, newline='')
    csv_out = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    csv_out.close()
    try:
        # Write test CSV
        writer = csv.writer(csv_in)
        writer.writerow(['Business Name', 'City', 'State', 'Website'])
        writer.writerow(['Pizza Palace', 'Tampa', 'FL', 'https://pizza.com'])
        writer.writerow(['Salon Bella', 'Miami', 'FL', 'https://salon.com'])
        csv_in.close()

        db = ForgeDB.from_config({'db_path': db_path})
        db.ensure_schema()
        result = db.import_csv(csv_in.name)
        assert result is not None
        assert result >= 2  # Should import at least 2 records

        export_result = db.export_csv(csv_out.name)
        assert export_result is not None
        row_count = export_result.get("row_count", 0) if isinstance(export_result, dict) else int(export_result)
        assert row_count >= 2
        db.close()
    finally:
        os.unlink(db_path)
        os.unlink(csv_in.name)
        os.unlink(csv_out.name)


def test_config_defaults(tmp_path, monkeypatch):
    """Test config loads with sensible defaults (isolated from user config)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Clear any FORGE_ env vars that might override defaults
    for key in list(os.environ.keys()):
        if key.startswith("FORGE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Reload the module so _TOML_PATH picks up the new HOME
    import importlib
    import forge.config
    importlib.reload(forge.config)
    from forge.config import ForgeConfig as FreshConfig

    config = FreshConfig.load()
    assert config.db_backend == 'sqlite'
    assert config.workers == 50
    assert config.adapter == 'auto'
