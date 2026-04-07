"""
FORGE Dashboard — FastAPI + Jinja2 + HTMX web interface.

No React, no npm, no build step. Server-rendered with HTMX for interactivity.

Usage:
    python -m forge.dashboard            # starts on port 8765
    uvicorn forge.dashboard.app:app      # direct uvicorn invocation
"""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("forge.dashboard")

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(title="FORGE Dashboard")


class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' https://cdn.tailwindcss.com https://unpkg.com; style-src 'self' 'unsafe-inline'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


app.add_middleware(CSPMiddleware)

_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

# ── Industry whitelist (matches enrichment/pipeline.py) ────────────────────

INDUSTRY_WHITELIST = [
    "restaurant", "salon", "real-estate", "dentist", "gym",
    "lawyer", "landscaping", "barber", "cleaning-service", "chiropractor",
    "veterinarian", "auto-repair", "tattoo-shop", "accountant", "plumber",
    "photographer", "dog-groomer", "electrician", "food-truck", "personal-trainer",
]

# ── Lazy-loaded singletons ─────────────────────────────────────────────────
#
# The dashboard should work even when the database or AI adapters are not
# configured -- pages render, but operations will return friendly errors.

_db_instance = None
_config_instance = None
_db_lock = threading.Lock()


def _get_config():
    """Load ForgeConfig (cached singleton)."""
    global _config_instance
    if _config_instance is None:
        try:
            from forge.config import ForgeConfig
            _config_instance = ForgeConfig.load()
        except Exception as e:
            logger.warning("Could not load ForgeConfig: %s", e)
            # Return a minimal mock so pages still render
            _config_instance = _MockConfig()
    return _config_instance


def _get_db():
    """Get or create the ForgeDB instance (thread-safe singleton)."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                try:
                    from forge.db import ForgeDB
                    config = _get_config()
                    db_config = config.to_db_config()
                    _db_instance = ForgeDB.from_config(db_config)
                    _db_instance.ensure_schema()
                except Exception as e:
                    logger.warning("Could not initialize ForgeDB: %s", e)
                    return None
    return _db_instance


class _MockConfig:
    """Minimal stand-in when forge.config is unavailable."""
    db_backend = "sqlite"
    db_path = "forge.db"
    db_host = ""
    db_port = 5432
    db_user = ""
    db_password = ""
    db_name = "forge"
    adapter = "auto"
    anthropic_api_key = ""
    ollama_url = "http://localhost:11434"
    ollama_model = "gemma4:26b"
    claude_model = "claude-sonnet-4-6"
    workers = 50
    batch_size = 5
    rate_limit = 100.0

    def to_db_config(self):
        return {"db_path": self.db_path}


# ── Enrichment state (global, mutable) ─────────────────────────────────────

_enrichment_thread: Optional[threading.Thread] = None
_enrichment_stop = threading.Event()
_enrichment_stats: Dict[str, Any] = {
    "running": False,
    "total_processed": 0,
    "emails_found": 0,
    "tech_stacks_found": 0,
    "rate_per_hour": 0,
    "progress_pct": 0,
    "log_messages": [],
    "started_at": None,
}
_enrichment_lock = threading.Lock()


def _append_log(msg: str):
    """Append a message to the enrichment log (keeps last 10)."""
    import html as _html
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = _html.escape(str(msg))
    with _enrichment_lock:
        _enrichment_stats["log_messages"].append(f"[{ts}] {safe_msg}")
        _enrichment_stats["log_messages"] = _enrichment_stats["log_messages"][-10:]


def _init_enrichment(mode: str, workers: int) -> tuple:
    """Initialize enrichment state and return (db, total) or (None, 0) on error."""
    global _enrichment_stats
    _append_log(f"Starting enrichment — mode={mode}, workers={workers}")

    with _enrichment_lock:
        _enrichment_stats["started_at"] = time.time()
        _enrichment_stats["total_processed"] = 0
        _enrichment_stats["emails_found"] = 0
        _enrichment_stats["tech_stacks_found"] = 0

    db = _get_db()
    if not db:
        _append_log("ERROR: Database not available")
        return None, 0

    total = db.count()
    _append_log(f"Total records in database: {total:,}")
    return db, total


def _run_enrichment_loop(db, mode: str, workers: int, total: int) -> None:
    """Build the pipeline and run enrichment."""
    from forge.enrichment.pipeline import EnrichmentPipeline
    from forge.tools.database import DatabasePool
    from forge.adapters.ollama import OllamaAdapter

    config = _get_config()
    pool = DatabasePool(db=db)

    ollama = None
    if mode in ("ai", "both"):
        try:
            ollama = OllamaAdapter(base_url=config.ollama_url, default_model=config.ollama_model)
            _append_log(f"Connected to Ollama at {config.ollama_url}")
        except Exception as e:
            _append_log(f"Ollama not available: {e}")
            if mode == "ai":
                _append_log("ERROR: AI mode requires Ollama")
                return
            mode = "email"
            _append_log("Falling back to email-only mode")

    pipeline = EnrichmentPipeline(db_pool=pool, ollama=ollama, web_scraper_workers=workers, batch_size=config.batch_size)
    _append_log("Pipeline initialized, starting enrichment...")

    def stop_watcher():
        _enrichment_stop.wait()
        pipeline.stop()
        _append_log("Stop signal received")

    watcher = threading.Thread(target=stop_watcher, daemon=True)
    watcher.start()

    stats = pipeline.run(mode=mode, resume=True)
    with _enrichment_lock:
        _enrichment_stats["total_processed"] = stats.total_processed
        _enrichment_stats["emails_found"] = stats.emails_found
        _enrichment_stats["tech_stacks_found"] = stats.tech_stacks_found
        _enrichment_stats["rate_per_hour"] = stats.rate_per_hour()
        if total > 0:
            _enrichment_stats["progress_pct"] = min(100.0, stats.total_processed / total * 100)
    _append_log(f"Enrichment complete: {stats.total_processed:,} processed")


def _run_enrichment_background(mode: str, workers: int):
    """Background thread that runs the enrichment pipeline."""
    db, total = _init_enrichment(mode, workers)
    if not db:
        with _enrichment_lock:
            _enrichment_stats["running"] = False
        return

    try:
        _run_enrichment_loop(db, mode, workers, total)
    except ImportError as e:
        _append_log(f"Pipeline dependencies not available: {e}")
        _append_log("Install dependencies: pip install aiohttp psycopg2-binary")
    except Exception as e:
        _append_log(f"ERROR: {e}")
        logger.exception("Enrichment failed")
    finally:
        with _enrichment_lock:
            _enrichment_stats["running"] = False
        _append_log("Enrichment stopped")


# ── Page Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    """Dashboard home page with stats overview."""
    db = _get_db()
    stats = db.get_stats() if db else {}
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_page": "dashboard",
        "stats": stats,
    })


@app.get("/discover", response_class=HTMLResponse)
async def page_discover(request: Request):
    """Discovery page — search for businesses by ZIP."""
    return templates.TemplateResponse("discover.html", {
        "request": request,
        "active_page": "discover",
        "industries": INDUSTRY_WHITELIST,
    })


@app.get("/enrich", response_class=HTMLResponse)
async def page_enrich(request: Request):
    """Enrichment control page."""
    return templates.TemplateResponse("enrich.html", {
        "request": request,
        "active_page": "enrich",
    })


@app.get("/import", response_class=HTMLResponse)
async def page_import(request: Request):
    """CSV upload page."""
    return templates.TemplateResponse("import.html", {
        "request": request,
        "active_page": "import",
    })


@app.get("/export", response_class=HTMLResponse)
async def page_export(request: Request):
    """Export page."""
    return templates.TemplateResponse("export.html", {
        "request": request,
        "active_page": "export",
    })


@app.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request):
    """Settings page."""
    config = _get_config()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "active_page": "settings",
        "config": config,
    })


# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    """Return database stats as JSON (polled by HTMX on dashboard)."""
    db = _get_db()
    if not db:
        return JSONResponse({"error": "Database not available", "total_records": 0})
    stats = db.get_stats()
    return JSONResponse(stats)


def _validate_discover_input(zip_code: str) -> Optional[HTMLResponse]:
    """Return an error HTMLResponse if zip_code is invalid, else None."""
    if not zip_code or len(zip_code) != 5:
        return HTMLResponse(
            '<div class="forge-card text-red-400 text-center py-4">'
            'Please enter a valid 5-digit ZIP code</div>'
        )
    return None


def _format_discover_results(results: list, zip_code: str, industry: str, radius: int, limit: int) -> str:
    """Build HTML table from discovery results."""
    rows_html = ""
    for r in results:
        rows_html += (
            f"<tr><td>{_esc(r.get('name', ''))}</td>"
            f"<td>{_esc(r.get('address_line1', ''))}</td>"
            f"<td>{_esc(r.get('city', ''))}</td>"
            f"<td>{_esc(r.get('state', ''))}</td>"
            f"<td>{_esc(r.get('industry', r.get('category', '')))}</td>"
            f"<td>{_esc(r.get('website_url', ''))}</td></tr>"
        )
    return (
        f'<div class="space-y-4">'
        f'<div class="flex items-center justify-between">'
        f'<p class="text-sm text-gray-400">Found <span class="text-amber-400 font-bold">{len(results)}</span> businesses</p>'
        f'<form hx-post="/api/import-results" hx-target="#discover-results" hx-swap="innerHTML">'
        f'<input type="hidden" name="zip_code" value="{_esc(zip_code)}">'
        f'<input type="hidden" name="industry" value="{_esc(industry)}">'
        f'<input type="hidden" name="radius" value="{_esc(str(radius))}">'
        f'<input type="hidden" name="limit" value="{_esc(str(limit))}">'
        f'<button type="submit" class="forge-btn text-sm">Import All {len(results)} Results</button>'
        f'</form></div>'
        f'<div class="overflow-x-auto"><table class="forge-table">'
        f'<thead><tr><th>Name</th><th>Address</th><th>City</th><th>State</th><th>Category</th><th>Website</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div></div>'
    )


@app.post("/api/discover", response_class=HTMLResponse)
async def api_discover(
    request: Request,
    zip_code: str = Form(""),
    industry: str = Form(""),
    radius: int = Form(10),
    limit: int = Form(100),
):
    """Execute Overture search and return HTML partial with results."""
    err = _validate_discover_input(zip_code)
    if err:
        return err

    results = []
    error = None
    try:
        from forge.discovery.overture import OvertureDiscovery
        discovery = OvertureDiscovery()
        results = discovery.search(zip_code=zip_code, industry=industry or None, radius_miles=radius, limit=limit)
    except ImportError:
        error = "Discovery module not available. Install DuckDB: pip install duckdb"
    except Exception as e:
        error = str(e)
        logger.exception("Discovery failed")

    if error:
        return HTMLResponse(f'<div class="forge-card text-red-400 text-center py-4">Discovery error: {_esc(str(error))}</div>')
    if not results:
        return HTMLResponse('<div class="forge-card text-gray-400 text-center py-4">No results found for this search</div>')
    return HTMLResponse(_format_discover_results(results, zip_code, industry, radius, limit))


@app.post("/api/import-results", response_class=HTMLResponse)
async def api_import_results(
    zip_code: str = Form(""),
    industry: str = Form(""),
    radius: int = Form(10),
    limit: int = Form(100),
):
    """Re-run discovery and import results to database."""
    db = _get_db()
    if not db:
        return HTMLResponse(
            '<div class="forge-card text-red-400 text-center py-4">'
            'Database not available</div>'
        )

    try:
        from forge.discovery.overture import OvertureDiscovery
        discovery = OvertureDiscovery()
        results = discovery.search(
            zip_code=zip_code,
            industry=industry or None,
            radius_miles=radius,
            limit=limit,
        )
    except Exception as e:
        return HTMLResponse(
            f'<div class="forge-card text-red-400 text-center py-4">'
            f'Discovery error: {_esc(str(e))}</div>'
        )

    imported = 0
    for record in results:
        try:
            db.upsert_business(record)
            imported += 1
        except Exception as e:
            logger.warning("Failed to import record: %s", e)

    return HTMLResponse(
        f'<div class="forge-card text-center py-6">'
        f'<p class="text-2xl font-bold text-amber-400">{imported:,}</p>'
        f'<p class="text-gray-400 mt-1">businesses imported to database</p>'
        f'<a href="/enrich" class="forge-btn inline-block mt-4">Start Enrichment</a>'
        f'</div>'
    )


@app.post("/api/enrich/start", response_class=HTMLResponse)
async def api_enrich_start(
    request: Request,
    mode: str = Form("both"),
    workers: int = Form(50),
):
    """Start enrichment in a background thread."""
    global _enrichment_thread

    with _enrichment_lock:
        if _enrichment_stats["running"]:
            return HTMLResponse(
                '<div class="forge-card text-amber-400 text-center py-3">'
                'Enrichment is already running</div>'
            )
        _enrichment_stats["running"] = True  # Set BEFORE releasing lock to prevent TOCTOU race

    _enrichment_stop.clear()
    _enrichment_thread = threading.Thread(
        target=_run_enrichment_background,
        args=(mode, workers),
        daemon=True,
        name="dashboard-enrichment",
    )
    _enrichment_thread.start()

    return HTMLResponse(
        '<div class="forge-card text-green-400 text-center py-3">'
        'Enrichment started! Stats will update below.</div>'
    )


@app.post("/api/enrich/stop", response_class=HTMLResponse)
async def api_enrich_stop():
    """Signal enrichment to stop."""
    _enrichment_stop.set()
    return HTMLResponse(
        '<div class="forge-card text-amber-400 text-center py-3">'
        'Stop signal sent. Enrichment will finish current batch and stop.</div>'
    )


@app.get("/api/enrich/status")
async def api_enrich_status():
    """Return enrichment status as JSON (polled by HTMX every 3s)."""
    with _enrichment_lock:
        stats_copy = dict(_enrichment_stats)
    return JSONResponse({"stats": stats_copy})


MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB


async def _save_upload_to_temp(file: UploadFile) -> tuple:
    """Stream upload to temp file. Returns (tmp_path, tmp_dir) or raises."""
    tmp_dir = tempfile.mkdtemp(prefix="forge_upload_")
    safe_name = secrets.token_hex(8) + ".csv"
    tmp_path = os.path.join(tmp_dir, safe_name)
    total = 0
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await file.read(8192)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                f.close()
                os.unlink(tmp_path)
                os.rmdir(tmp_dir)
                raise ValueError("File too large")
            f.write(chunk)
    return tmp_path, tmp_dir


def _format_upload_result(result: dict) -> HTMLResponse:
    """Format import result as HTML."""
    imported = result.get("imported", 0)
    total = result.get("total_rows", 0)
    mapping = result.get("column_mapping", {})
    mapped_cols = ", ".join(f"{k} -> {v}" for k, v in list(mapping.items())[:6])
    return HTMLResponse(
        f'<div class="forge-card text-center py-6 space-y-3">'
        f'<p class="text-3xl font-bold text-amber-400">{imported:,}</p>'
        f'<p class="text-gray-300">records imported from {total:,} rows</p>'
        f'<p class="text-sm text-gray-500">Columns mapped: {_esc(mapped_cols)}</p>'
        f'<a href="/enrich" class="forge-btn inline-block mt-4">Start Enrichment</a></div>'
    )


@app.post("/api/upload", response_class=HTMLResponse)
async def api_upload(file: UploadFile = File(...)):
    """Handle CSV file upload.

    Security: uses secrets.token_hex for filenames, enforces MAX_UPLOAD_SIZE.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        return HTMLResponse('<div class="forge-card text-red-400 text-center py-4">Please upload a .csv file</div>')

    db = _get_db()
    if not db:
        return HTMLResponse('<div class="forge-card text-red-400 text-center py-4">Database not available</div>')

    # Security: random filename via secrets.token_hex, size check via MAX_UPLOAD_SIZE
    tmp_path = tmp_dir = ""
    try:
        tmp_path, tmp_dir = await _save_upload_to_temp(file)
        result = db.import_csv(tmp_path, return_details=True)
    except ValueError as ve:
        return HTMLResponse(f'<div class="forge-card text-red-400 text-center py-4">{_esc(str(ve))}</div>')
    except Exception as e:
        logger.exception("Upload failed")
        return HTMLResponse(f'<div class="forge-card text-red-400 text-center py-4">Upload error: {_esc(str(e))}</div>')
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if tmp_dir and os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass

    if result.get("status") == "error":
        return HTMLResponse(f'<div class="forge-card text-red-400 text-center py-4">Import error: {_esc(result.get("error", "Unknown error"))}</div>')
    return _format_upload_result(result)


def _execute_export_query(db, where_clause: Optional[str], limit_clause: str = "") -> list:
    """Execute an export query and return a list of row dicts."""
    query = "SELECT * FROM businesses"
    if where_clause:
        query += f" WHERE {where_clause}"
    query += " ORDER BY created_at DESC"
    if limit_clause:
        query += limit_clause

    with db._backend.connection() as conn:
        if db.is_postgres:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(query)
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            return rows
        else:
            cursor = conn.execute(query)
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _generate_preview_html(db, where_clause: Optional[str], limit: int) -> Response:
    """Generate an HTML table preview for export."""
    safe_limit = max(1, min(limit, 100)) if limit > 0 else 10
    try:
        rows = _execute_export_query(db, where_clause, f" LIMIT {safe_limit}")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if not rows:
        return HTMLResponse(
            '<p class="text-gray-500 text-sm text-center py-4">No records match this filter</p>'
        )

    cols = list(rows[0].keys())
    show_cols = ["name", "city", "state", "email", "website_url", "industry", "health_score", "last_enriched_at"]
    display_cols = [c for c in show_cols if c in cols] or cols[:8]

    header = "".join(f"<th>{c}</th>" for c in display_cols)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{_esc(str(row.get(c, '') or ''))[:60]}</td>" for c in display_cols)
        body += f"<tr>{cells}</tr>"

    return HTMLResponse(
        f'<table class="forge-table"><thead><tr>{header}</tr></thead>'
        f'<tbody>{body}</tbody></table>'
        f'<p class="text-xs text-gray-500 mt-2">Showing {len(rows)} rows</p>'
    )


def _generate_json_download(db, where_clause: Optional[str]) -> StreamingResponse:
    """Generate a JSON file download for export."""
    rows = _execute_export_query(db, where_clause)

    def _default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    json_bytes = json.dumps(rows, indent=2, default=_default).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(json_bytes),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=forge_export.json"},
    )


def _generate_csv_download(db, filter_name: str) -> Response:
    """Generate a CSV file download for export."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, prefix="forge_export_")
    tmp_path = tmp.name
    tmp.close()

    def cleanup_file(path):
        try:
            os.unlink(path)
        except Exception:
            pass

    try:
        result = db.export_csv(filepath=tmp_path, where=filter_name)
        if result.get("status") == "error":
            cleanup_file(tmp_path)
            return JSONResponse({"error": result.get("error")}, status_code=500)

        return FileResponse(
            tmp_path,
            media_type="text/csv",
            filename="forge_export.csv",
            background=BackgroundTask(cleanup_file, tmp_path),
        )
    except Exception as e:
        cleanup_file(tmp_path)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/export/csv")
async def api_export_csv(
    filter: str = Query("all"),
    format: str = Query("csv"),
    preview: str = Query(""),
    limit: int = Query(0),
):
    """Export data with safe predefined filters only."""
    db = _get_db()
    if not db:
        return JSONResponse({"error": "Database not available"}, status_code=500)

    from forge.db import ForgeDB
    where_clause = ForgeDB.SAFE_WHERE_FILTERS.get(filter, None)

    if preview == "true":
        return _generate_preview_html(db, where_clause, limit)

    if format == "json":
        try:
            return _generate_json_download(db, where_clause)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return _generate_csv_download(db, filter)


@app.post("/api/settings", response_class=HTMLResponse)
async def api_settings(request: Request):
    """Save settings to ~/.forge/config.toml."""
    form = await request.form()
    saved = []

    try:
        from forge.config import cli_config_set
        for key in [
            "db_backend", "db_path", "db_port", "db_name",
            "adapter", "anthropic_api_key", "ollama_url", "ollama_model",
            "workers", "batch_size", "rate_limit",
        ]:
            value = form.get(key, "")
            if value is not None and str(value).strip():
                # Don't save masked API keys
                if key == "anthropic_api_key" and "****" in str(value):
                    continue
                cli_config_set(key, str(value))
                saved.append(key)

        # Reload config
        global _config_instance
        _config_instance = None

    except ImportError:
        return HTMLResponse(
            '<div class="forge-card text-red-400 text-center py-3">'
            'Could not import forge.config — settings not saved</div>'
        )
    except Exception as e:
        return HTMLResponse(
            f'<div class="forge-card text-red-400 text-center py-3">'
            f'Error saving settings: {_esc(str(e))}</div>'
        )

    return HTMLResponse(
        f'<div class="forge-card text-green-400 text-center py-3">'
        f'Settings saved: {", ".join(saved)}</div>'
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    """Run the dashboard server."""
    import uvicorn
    config = _get_config()
    port = getattr(config, "dashboard_port", 8765)
    logger.info("Starting FORGE Dashboard on port %d", port)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
