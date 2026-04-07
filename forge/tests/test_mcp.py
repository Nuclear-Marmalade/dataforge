"""Tests for MCP server tools and JSON-RPC protocol."""
import os
import uuid


class TestMCPToolDefinitions:
    def test_tool_definitions_exist(self):
        from forge.mcp_server import TOOL_DEFINITIONS
        assert len(TOOL_DEFINITIONS) >= 5

    def test_all_tools_have_name_and_schema(self):
        from forge.mcp_server import TOOL_DEFINITIONS
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "inputSchema" in tool
            assert "description" in tool

    def test_all_tools_have_handlers(self):
        from forge.mcp_server import TOOL_DEFINITIONS, TOOL_HANDLERS
        for tool in TOOL_DEFINITIONS:
            assert tool["name"] in TOOL_HANDLERS, f"No handler for tool {tool['name']}"


class TestMCPExport:
    def test_export_with_state_filter(self, sqlite_db, tmp_path, monkeypatch):
        from forge.mcp_server import _tool_forge_export
        import forge.mcp_server as mcp_mod

        sqlite_db.execute(
            "INSERT INTO businesses (id, name, state, email) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), "FL Biz", "FL", "a@b.com"),
        )
        sqlite_db.execute(
            "INSERT INTO businesses (id, name, state) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), "NY Biz", "NY"),
        )

        # Monkey-patch the module-level _db
        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            # Use /tmp which is in the allowed path list
            out = f"/tmp/forge_test_export_{uuid.uuid4().hex}.csv"
            result = _tool_forge_export({"output_path": out, "filter": "state=FL"})
            assert result["status"] == "success"
            assert result["row_count"] == 1
            if os.path.exists(out):
                os.unlink(out)
        finally:
            mcp_mod._db = old_db

    def test_export_no_filter(self, sqlite_db, tmp_path, monkeypatch):
        from forge.mcp_server import _tool_forge_export
        import forge.mcp_server as mcp_mod

        sqlite_db.upsert_business({"name": "Test", "state": "FL"})

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            out = f"/tmp/forge_test_export_{uuid.uuid4().hex}.csv"
            result = _tool_forge_export({"output_path": out})
            assert result["status"] == "success"
            assert result["row_count"] >= 1
            if os.path.exists(out):
                os.unlink(out)
        finally:
            mcp_mod._db = old_db

    def test_export_requires_output_path(self, sqlite_db):
        from forge.mcp_server import _tool_forge_export
        import forge.mcp_server as mcp_mod

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_export({"output_path": ""})
            assert "error" in result
        finally:
            mcp_mod._db = old_db

    def test_export_rejects_path_traversal(self, sqlite_db):
        from forge.mcp_server import _tool_forge_export
        import forge.mcp_server as mcp_mod

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_export({"output_path": "/tmp/../../../etc/passwd"})
            assert "error" in result
        finally:
            mcp_mod._db = old_db


class TestMCPStats:
    def test_stats_returns_counts(self, sqlite_db):
        from forge.mcp_server import _tool_forge_stats
        import forge.mcp_server as mcp_mod

        sqlite_db.upsert_business({"name": "Test", "state": "FL"})

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_stats({})
            assert "total_records" in result
            assert int(result["total_records"]) >= 1
        finally:
            mcp_mod._db = old_db


class TestMCPEnrichRecord:
    def test_enrich_record_creates_business(self, sqlite_db):
        from forge.mcp_server import _tool_forge_enrich_record
        import forge.mcp_server as mcp_mod

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_enrich_record({
                "name": "Test Biz",
                "city": "Tampa",
                "state": "FL",
            })
            assert result.get("status") == "created"
            assert "business_id" in result
        finally:
            mcp_mod._db = old_db

    def test_enrich_record_requires_name(self, sqlite_db):
        from forge.mcp_server import _tool_forge_enrich_record
        import forge.mcp_server as mcp_mod

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_enrich_record({"name": "", "city": "Tampa", "state": "FL"})
            assert "error" in result
        finally:
            mcp_mod._db = old_db

    def test_enrich_record_requires_valid_state(self, sqlite_db):
        from forge.mcp_server import _tool_forge_enrich_record
        import forge.mcp_server as mcp_mod

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_enrich_record({"name": "Test", "city": "Tampa", "state": "Florida"})
            assert "error" in result
        finally:
            mcp_mod._db = old_db


class TestMCPSearch:
    def test_search_requires_query(self, sqlite_db):
        from forge.mcp_server import _tool_forge_search
        import forge.mcp_server as mcp_mod

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_search({"query": ""})
            assert "error" in result
        finally:
            mcp_mod._db = old_db

    def test_search_finds_matching_business(self, sqlite_db):
        from forge.mcp_server import _tool_forge_search
        import forge.mcp_server as mcp_mod

        sqlite_db.upsert_business({"name": "Pizza Palace", "city": "Tampa", "state": "FL"})
        sqlite_db.upsert_business({"name": "Salon Bella", "city": "Miami", "state": "FL"})

        old_db = mcp_mod._db
        mcp_mod._db = sqlite_db
        try:
            result = _tool_forge_search({"query": "Pizza"})
            assert result["count"] == 1
            assert result["results"][0]["name"] == "Pizza Palace"
        finally:
            mcp_mod._db = old_db


class TestMCPDiscover:
    def test_discover_invalid_zip(self, sqlite_db):
        from forge.mcp_server import _tool_forge_discover
        result = _tool_forge_discover({"zip_code": "abc"})
        assert "error" in result

    def test_discover_empty_zip(self, sqlite_db):
        from forge.mcp_server import _tool_forge_discover
        result = _tool_forge_discover({"zip_code": ""})
        assert "error" in result


class TestMCPProtocol:
    def test_handle_initialize(self):
        from forge.mcp_server import handle_request
        response = handle_request({"method": "initialize", "id": 1, "params": {}})
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "forge"

    def test_handle_tools_list(self):
        from forge.mcp_server import handle_request
        response = handle_request({"method": "tools/list", "id": 2, "params": {}})
        assert response["id"] == 2
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) >= 5

    def test_handle_ping(self):
        from forge.mcp_server import handle_request
        response = handle_request({"method": "ping", "id": 3, "params": {}})
        assert response["id"] == 3
        assert "result" in response

    def test_handle_unknown_method(self):
        from forge.mcp_server import handle_request
        response = handle_request({"method": "unknown/method", "id": 4, "params": {}})
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_handle_notification_returns_none(self):
        from forge.mcp_server import handle_request
        response = handle_request({"method": "notifications/initialized", "params": {}})
        assert response is None

    def test_dispatch_unknown_tool(self):
        from forge.mcp_server import dispatch_tool
        result = dispatch_tool("nonexistent_tool", {})
        assert "error" in result


class TestParseFilter:
    def test_parse_state_filter(self, sqlite_db):
        from forge.mcp_server import _parse_filter
        conditions, params = _parse_filter("state=FL", sqlite_db)
        assert len(conditions) == 1
        assert "state" in conditions[0]
        assert params == ["FL"]

    def test_parse_has_email_filter(self, sqlite_db):
        from forge.mcp_server import _parse_filter
        conditions, params = _parse_filter("has_email=true", sqlite_db)
        assert len(conditions) == 1
        assert "email" in conditions[0]
        assert len(params) == 0  # has_email doesn't use params

    def test_parse_multiple_filters(self, sqlite_db):
        from forge.mcp_server import _parse_filter
        conditions, params = _parse_filter("state=FL,has_email=true", sqlite_db)
        assert len(conditions) == 2

    def test_parse_shorthand_has_email(self, sqlite_db):
        from forge.mcp_server import _parse_filter
        conditions, params = _parse_filter("has_email", sqlite_db)
        assert len(conditions) == 1
        assert "email" in conditions[0]
