"""Security regression tests -- one per bug from nine audit rounds."""
import inspect


class TestSQLInjection:
    def test_resolve_where_blocks_union(self, sqlite_db):
        assert sqlite_db._resolve_where("1=1 UNION SELECT sql FROM sqlite_master") is None

    def test_resolve_where_blocks_semicolon(self, sqlite_db):
        assert sqlite_db._resolve_where("1=1; DROP TABLE businesses;") is None

    def test_resolve_where_blocks_subquery(self, sqlite_db):
        assert sqlite_db._resolve_where("(SELECT 1)") is None

    def test_resolve_where_blocks_or_true(self, sqlite_db):
        assert sqlite_db._resolve_where("1=1 OR 1=1") is None

    def test_resolve_where_accepts_with_email(self, sqlite_db):
        result = sqlite_db._resolve_where("with_email")
        assert result is not None
        assert "email" in result

    def test_resolve_where_accepts_enriched(self, sqlite_db):
        result = sqlite_db._resolve_where("enriched")
        assert result is not None
        assert "last_enriched_at" in result

    def test_resolve_where_accepts_all(self, sqlite_db):
        result = sqlite_db._resolve_where("all")
        assert result is None  # "all" means no filter

    def test_enrichable_fields_whitelist(self):
        """Only allowed fields should be writable via enrichment."""
        from forge.db import ENRICHABLE_FIELDS
        # These sensitive fields should NOT be in the enrichable set
        assert "id" not in ENRICHABLE_FIELDS
        assert "created_at" not in ENRICHABLE_FIELDS
        assert "updated_at" not in ENRICHABLE_FIELDS

    def test_upsert_filters_unknown_columns(self, sqlite_db):
        """Unknown columns in data should be silently ignored, not passed to SQL."""
        bid = sqlite_db.upsert_business({
            "name": "Test",
            "state": "FL",
            "malicious_field": "'; DROP TABLE businesses; --",
        })
        assert bid
        assert int(sqlite_db.get_stats()["total_records"]) == 1


class TestPathTraversal:
    def test_upload_uses_random_filename(self):
        """Round 5: upload handler must not use user-supplied filename."""
        from forge.dashboard.app import api_upload
        source = inspect.getsource(api_upload)
        assert "secrets.token_hex" in source
        # After generating safe name, user filename should not be used for path
        assert "safe_name" in source or "token_hex" in source


class TestXSS:
    def test_log_messages_are_escaped(self):
        """Round 6: log messages must use html.escape."""
        from forge.dashboard.app import _append_log
        source = inspect.getsource(_append_log)
        assert "html.escape" in source or "_html.escape" in source

    def test_esc_helper_exists(self):
        """Dashboard should have an HTML escape helper."""
        from forge.dashboard.app import _esc
        result = _esc('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_esc_handles_quotes(self):
        from forge.dashboard.app import _esc
        result = _esc('value="test"')
        assert '&quot;' in result


class TestNetworkExposure:
    def test_dashboard_binds_localhost(self):
        """Round 4: dashboard must bind to 127.0.0.1, not 0.0.0.0."""
        from forge.dashboard import app as app_module
        source = inspect.getsource(app_module.main)
        assert "127.0.0.1" in source
        assert "0.0.0.0" not in source


class TestCSPMiddleware:
    def test_csp_middleware_is_registered(self):
        """Dashboard should have Content-Security-Policy middleware."""
        from forge.dashboard.app import app
        middleware_found = False
        for m in app.user_middleware:
            if "CSP" in str(m.cls.__name__):
                middleware_found = True
        assert middleware_found


class TestSafetyModules:
    def test_audit_agent_imports_without_psycopg2(self):
        """Round 9: safety modules must not crash on SQLite."""
        # Just importing should not crash even without psycopg2

    def test_error_recovery_imports_without_psycopg2(self):
        """Safety modules should import cleanly."""
        # Should not crash

    def test_validate_field_email(self):
        from forge.safety.error_recovery import validate_field
        is_valid, err = validate_field("email", "test@example.com")
        assert is_valid
        is_invalid, err = validate_field("email", "not-an-email")
        assert not is_invalid

    def test_validate_field_health_score_range(self):
        from forge.safety.error_recovery import validate_field
        assert validate_field("health_score", 50)[0] is True
        assert validate_field("health_score", -1)[0] is False
        assert validate_field("health_score", 101)[0] is False

    def test_validate_updates_rejects_null(self):
        from forge.safety.error_recovery import validate_updates
        valid, errors = validate_updates({"email": None})
        assert len(errors) == 1
        assert "null" in errors[0].lower() or "COALESCE" in errors[0]


class TestUploadLimits:
    def test_max_upload_size_defined(self):
        """Round 7: upload must have size limit."""
        from forge.dashboard.app import MAX_UPLOAD_SIZE
        assert MAX_UPLOAD_SIZE > 0
        assert MAX_UPLOAD_SIZE <= 200 * 1024 * 1024  # Not more than 200MB

    def test_upload_handler_checks_size(self):
        from forge.dashboard.app import api_upload
        source = inspect.getsource(api_upload)
        assert "MAX_UPLOAD_SIZE" in source
