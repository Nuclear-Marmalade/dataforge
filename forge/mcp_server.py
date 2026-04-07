"""
FORGE MCP Server — Exposes FORGE tools to Claude Code via Model Context Protocol.

Implements the MCP JSON-RPC protocol over stdin/stdout so Claude Code can
discover businesses, enrich records, search the database, get stats, and export
data directly through tool calls.

No external MCP SDK required — uses the standard JSON-RPC transport directly.

Usage:
    forge mcp-server          # Started by Claude Code automatically
    python mcp_server.py      # Direct execution for testing

Configuration in ~/.claude.json:
    {
      "mcpServers": {
        "forge": {
          "command": "forge",
          "args": ["mcp-server"]
        }
      }
    }

Dependencies: forge.db (ForgeDB), forge.config (ForgeConfig)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# All logging goes to stderr — stdout is the MCP transport
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("forge.mcp")


# ── Tool Definitions ────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "forge_discover",
        "description": (
            "Discover businesses by US ZIP code using the Overture Maps open dataset. "
            "Returns name, address, phone, website, and category for each business found. "
            "Optionally filter by industry (e.g., restaurant, healthcare, legal, beauty)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "zip_code": {
                    "type": "string",
                    "description": "5-digit US ZIP code to search around",
                },
                "industry": {
                    "type": "string",
                    "description": (
                        "Optional industry filter (e.g., restaurant, healthcare, legal, "
                        "beauty, automotive, retail, fitness, education)"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 100)",
                    "default": 100,
                },
            },
            "required": ["zip_code"],
        },
    },
    {
        "name": "forge_enrich_record",
        "description": (
            "Enrich a single business record by adding it to the FORGE database "
            "and returning the stored record. Provide at minimum a name and city/state. "
            "Optionally include a website URL for deeper enrichment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Business name",
                },
                "city": {
                    "type": "string",
                    "description": "City where the business is located",
                },
                "state": {
                    "type": "string",
                    "description": "Two-letter state abbreviation (e.g., FL, CA, NY)",
                },
                "website": {
                    "type": "string",
                    "description": "Optional website URL for deeper enrichment",
                },
            },
            "required": ["name", "city", "state"],
        },
    },
    {
        "name": "forge_stats",
        "description": (
            "Get current FORGE database statistics including total records, "
            "records with email, tech stack, NPI numbers, and enrichment activity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "forge_search",
        "description": (
            "Search the FORGE database for businesses matching criteria. "
            "Supports text search on name, filtering by state and industry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text to match against business names",
                },
                "state": {
                    "type": "string",
                    "description": "Optional two-letter state filter (e.g., FL, CA)",
                },
                "industry": {
                    "type": "string",
                    "description": "Optional industry filter",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20)",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "forge_export",
        "description": (
            "Export enriched business data from the FORGE database to a CSV file. "
            "Optionally filter by state, industry, or other criteria."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "File path for the exported CSV",
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Optional filter expression. Examples: "
                        "'state=FL', 'industry=restaurant', 'has_email=true'"
                    ),
                },
            },
            "required": ["output_path"],
        },
    },
]


# ── Database Initialization ─────────────────────────────────────────────────

_db = None


def _get_db():
    """
    Lazily initialize and return the ForgeDB instance.

    Uses SQLite by default, storing the database at ~/.forge/forge.db.
    Auto-creates the directory and schema if they don't exist.
    """
    global _db
    if _db is not None:
        return _db

    from forge.db import ForgeDB

    # Determine database path
    db_path = os.environ.get("FORGE_DB_PATH")
    if not db_path:
        forge_dir = Path.home() / ".forge"
        forge_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(forge_dir / "forge.db")

    logger.info("Initializing ForgeDB at: %s", db_path)
    _db = ForgeDB.from_config({"db_path": db_path})
    _db.ensure_schema()
    logger.info("ForgeDB ready (%d records)", _db.count())
    return _db


# ── Tool Implementations ────────────────────────────────────────────────────

def _tool_forge_discover(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Discover businesses by ZIP code using Overture Maps."""
    zip_code = arguments.get("zip_code", "")
    industry = arguments.get("industry")
    limit = arguments.get("limit", 100)

    if not zip_code or len(zip_code) != 5 or not zip_code.isdigit():
        return {"error": f"Invalid ZIP code: '{zip_code}'. Must be a 5-digit US ZIP code."}

    try:
        from forge.discovery.overture import OvertureDiscovery  # noqa: F401 - OvertureDiscoveryError removed (unused)
    except ImportError:
        return {
            "error": (
                "DuckDB is required for Overture Maps discovery. "
                "Install it with: pip install duckdb"
            )
        }

    try:
        disco = OvertureDiscovery()
        results = disco.search(
            zip_code=zip_code,
            industry=industry,
            limit=limit,
        )
        disco.close()

        # Also insert discovered businesses into the database
        db = _get_db()
        inserted = 0
        for biz in results:
            try:
                record = {
                    "name": biz.get("name"),
                    "address_line1": biz.get("address"),
                    "city": biz.get("city"),
                    "state": biz.get("state"),
                    "zip": biz.get("zip"),
                    "phone": biz.get("phone"),
                    "website_url": biz.get("website"),
                    "industry": biz.get("category"),
                    "latitude": biz.get("lat"),
                    "longitude": biz.get("lon"),
                }
                # Remove None values
                record = {k: v for k, v in record.items() if v is not None}
                if record.get("name"):
                    db.upsert_business(record)
                    inserted += 1
            except Exception as e:
                logger.warning("Failed to insert discovered business: %s", e)

        return {
            "businesses": results,
            "count": len(results),
            "inserted_to_db": inserted,
            "zip_code": zip_code,
            "industry": industry,
        }

    except Exception as e:
        return {"error": str(e)}


def _tool_forge_enrich_record(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a single business record."""
    name = arguments.get("name", "").strip()
    city = arguments.get("city", "").strip()
    state = arguments.get("state", "").strip().upper()
    website = arguments.get("website", "").strip()

    if not name:
        return {"error": "Business name is required."}
    if not city:
        return {"error": "City is required."}
    if not state or len(state) != 2:
        return {"error": f"Invalid state: '{state}'. Must be a 2-letter abbreviation."}

    db = _get_db()

    record = {
        "name": name,
        "city": city,
        "state": state,
    }
    if website:
        record["website_url"] = website

    try:
        business_id = db.upsert_business(record)
        # Fetch the full record back
        full_record = db.get_business(business_id)
        if full_record:
            # Convert non-serializable types
            clean: Dict[str, Any] = {}
            for k, v in full_record.items():
                if v is None:
                    clean[k] = None
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            return {
                "status": "created",
                "business_id": business_id,
                "record": clean,
            }
        else:
            return {
                "status": "created",
                "business_id": business_id,
                "note": "Record created but could not be retrieved.",
            }
    except Exception as e:
        return {"error": f"Failed to enrich record: {e}"}


def _tool_forge_stats(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get current enrichment statistics."""
    db = _get_db()
    try:
        stats = db.get_stats()
        # Ensure all values are JSON-serializable
        clean_stats: Dict[str, Any] = {}
        for k, v in stats.items():
            if v is None:
                clean_stats[k] = None
            elif isinstance(v, (str, int, float, bool)):
                clean_stats[k] = v
            else:
                clean_stats[k] = str(v)
        return clean_stats
    except Exception as e:
        return {"error": f"Failed to get stats: {e}"}


def _tool_forge_search(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Search the database for businesses matching criteria."""
    query = arguments.get("query", "").strip()
    state = arguments.get("state", "").strip().upper() if arguments.get("state") else None
    industry = arguments.get("industry", "").strip().lower() if arguments.get("industry") else None
    limit = min(arguments.get("limit", 20), 100)  # Cap at 100

    if not query:
        return {"error": "Search query is required."}

    db = _get_db()

    # Build WHERE clause with parameterized queries
    conditions = []
    params = []

    # Text search on name (LIKE for SQLite compatibility)
    if db.is_postgres:
        conditions.append("name ILIKE %s")
    else:
        conditions.append("name LIKE ?")
    params.append(f"%{query}%")

    if state:
        if db.is_postgres:
            conditions.append("state = %s")
        else:
            conditions.append("state = ?")
        params.append(state)

    if industry:
        if db.is_postgres:
            conditions.append("industry ILIKE %s")
        else:
            conditions.append("industry LIKE ?")
        params.append(f"%{industry}%")

    where_clause = " AND ".join(conditions)
    order_by = "ORDER BY name"
    limit_clause = f"LIMIT {limit}"

    full_query = f"SELECT * FROM businesses WHERE {where_clause} {order_by} {limit_clause}"

    try:
        with db._backend.connection() as conn:
            if db.is_postgres:
                import psycopg2.extras
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(full_query, params)
                rows = cur.fetchall()
                cur.close()
                results = [dict(row) for row in rows]
            else:
                cursor = conn.execute(full_query, params)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]

        # Clean results for JSON serialization
        clean_results = []
        for row in results:
            clean: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    clean[k] = None
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_results.append(clean)

        return {
            "results": clean_results,
            "count": len(clean_results),
            "query": query,
            "filters": {"state": state, "industry": industry},
        }

    except Exception as e:
        return {"error": f"Search failed: {e}"}


def _tool_forge_export(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Export enriched data to CSV."""
    import csv

    output_path = arguments.get("output_path", "").strip()
    filter_expr = arguments.get("filter", "").strip() if arguments.get("filter") else None

    if not output_path:
        return {"error": "output_path is required."}

    # Resolve relative paths
    output_path = os.path.abspath(output_path)
    # Reject paths outside current directory or home
    home = os.path.expanduser("~")
    cwd = os.getcwd()
    if not (output_path.startswith(cwd) or output_path.startswith(home) or output_path.startswith("/tmp")):
        return {"error": "Export path must be within home directory or current directory"}
    if ".." in output_path:
        return {"error": "Path traversal not allowed"}

    db = _get_db()

    # Build query with parameterized filters (same pattern as forge_search)
    query = "SELECT * FROM businesses"
    params: list = []

    if filter_expr:
        conditions, params = _parse_filter(filter_expr, db)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY name ASC"

    # Write CSV directly, bypassing db.export_csv's _resolve_where
    try:
        rows = db.fetch_dicts(query, tuple(params)) if params else db.fetch_dicts(query)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write("")  # empty file
        return {"status": "success", "row_count": len(rows), "output_path": output_path}
    except Exception as e:
        return {"error": f"Export failed: {e}", "row_count": 0}


def _parse_filter(filter_expr: str, db) -> tuple:
    """
    Parse a simple filter expression into a list of conditions and parameters.

    Supports:
        state=FL            -> state = ?/%s
        industry=restaurant -> industry LIKE ?/%s
        has_email=true      -> email IS NOT NULL AND email != ''
        has_website=true    -> website_url IS NOT NULL AND website_url != ''
        city=Tampa          -> city = ?/%s
        zip=33602           -> zip = ?/%s
        has_email           -> email IS NOT NULL AND email != ''
        has_website         -> website_url IS NOT NULL AND website_url != ''
        enriched            -> last_enriched_at IS NOT NULL

    Returns:
        (conditions_list, params_list) tuple — conditions are strings, not joined.
    """
    conditions = []
    params = []

    # Split on commas for multiple filters
    parts = [p.strip() for p in filter_expr.split(",")]

    ph = db.placeholder

    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "state":
                conditions.append(f"state = {ph}")
                params.append(value.upper())
            elif key == "industry":
                if db.is_postgres:
                    conditions.append(f"industry ILIKE {ph}")
                else:
                    conditions.append(f"industry LIKE {ph}")
                params.append(f"%{value}%")
            elif key == "has_email" and value.lower() == "true":
                conditions.append("email IS NOT NULL AND email != ''")
            elif key == "has_website" and value.lower() == "true":
                conditions.append("website_url IS NOT NULL AND website_url != ''")
            elif key == "city":
                conditions.append(f"city = {ph}")
                params.append(value)
            elif key == "zip":
                conditions.append(f"zip = {ph}")
                params.append(value)
        elif part == "has_email":
            conditions.append("email IS NOT NULL AND email != ''")
        elif part == "has_website":
            conditions.append("website_url IS NOT NULL AND website_url != ''")
        elif part == "enriched":
            conditions.append("last_enriched_at IS NOT NULL")

    return conditions, params


# ── Tool Dispatch ───────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "forge_discover": _tool_forge_discover,
    "forge_enrich_record": _tool_forge_enrich_record,
    "forge_stats": _tool_forge_stats,
    "forge_search": _tool_forge_search,
    "forge_export": _tool_forge_export,
}


def dispatch_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to the appropriate handler."""
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return handler(arguments)
    except Exception as e:
        logger.error("Tool '%s' raised an exception: %s", tool_name, e)
        logger.error(traceback.format_exc())
        return {"error": f"Tool execution failed: {e}"}


# ── JSON-RPC MCP Protocol Handler ───────────────────────────────────────────

def handle_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Handle a single MCP JSON-RPC request.

    Supports:
        - initialize: Server capability negotiation
        - initialized: Client acknowledgement (notification, no response)
        - tools/list: Return available tool definitions
        - tools/call: Execute a tool and return results
        - ping: Health check
    """
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    logger.info("Received request: method=%s id=%s", method, req_id)

    # Notifications (no id) don't get responses
    if req_id is None and method == "notifications/initialized":
        logger.info("Client initialized notification received")
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "forge",
                    "version": "1.0.0",
                },
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOL_DEFINITIONS,
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        result = dispatch_tool(tool_name, arguments)

        # Format as MCP tool result
        is_error = "error" in result
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2, default=str),
                    }
                ],
                "isError": is_error,
            },
        }

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        }

    else:
        # Unknown method
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        return None


# ── Main Loop ───────────────────────────────────────────────────────────────

def run_server():
    """
    Run the MCP server, reading JSON-RPC messages from stdin and writing
    responses to stdout.

    Uses Content-Length framed messages (the MCP standard transport).
    """
    logger.info("FORGE MCP Server starting...")
    logger.info("Reading JSON-RPC messages from stdin, writing to stdout.")

    while True:
        try:
            # Read headers until we get Content-Length
            content_length = None
            while True:
                line = sys.stdin.buffer.readline()
                if not line:
                    # EOF — client disconnected
                    logger.info("Client disconnected (EOF). Shutting down.")
                    return

                line_str = line.decode("utf-8").strip()

                if line_str == "":
                    # Empty line signals end of headers
                    break

                if line_str.lower().startswith("content-length:"):
                    content_length = int(line_str.split(":", 1)[1].strip())

            if content_length is None:
                logger.warning("No Content-Length header received, skipping message")
                continue

            # Read the body
            body = sys.stdin.buffer.read(content_length)
            if not body:
                logger.info("Client disconnected (empty body). Shutting down.")
                return

            # Parse JSON
            try:
                request = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON received: %s", e)
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}",
                    },
                }
                _send_response(error_response)
                continue

            # Handle the request
            response = handle_request(request)

            # Send response (if any — notifications don't get responses)
            if response is not None:
                _send_response(response)

        except KeyboardInterrupt:
            logger.info("Received interrupt. Shutting down.")
            return
        except Exception as e:
            logger.error("Unexpected error in main loop: %s", e)
            logger.error(traceback.format_exc())
            # Try to send an error response
            try:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {e}",
                    },
                }
                _send_response(error_response)
            except Exception:
                pass


def _send_response(response: Dict[str, Any]) -> None:
    """Send a JSON-RPC response with Content-Length framing to stdout."""
    body = json.dumps(response).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n"
    sys.stdout.buffer.write(header.encode("utf-8"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()
    logger.info("Sent response: id=%s", response.get("id"))


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_server()
