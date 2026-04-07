"""
FORGE CLI Enrich Helpers -- CSV validation, enrichment mode detection, pipeline orchestration.

Extracted from cli.py to keep it under 800 lines.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from forge.cli_helpers import die, warn, info, green, bold, dim

try:
    from forge import __version__
except ImportError:
    __version__ = "1.0.0"


def _validate_csv_input(file_path: str) -> tuple:
    """Validate the input CSV file exists and has data rows."""
    input_path = Path(file_path)
    if not input_path.exists():
        die(f"File not found: {file_path}", hint=f"Check the path. Current directory: {os.getcwd()}")
    if not input_path.suffix.lower() == ".csv":
        warn(f"File does not have .csv extension: {file_path}")
    if input_path.stat().st_size == 0:
        die("File is empty.", hint="Provide a CSV with at least a header row and one data row.")
    try:
        with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                die("No records found in file.", hint="The CSV appears to be empty.")
            if next(reader, None) is None:
                die("No records found in file.", hint="The CSV has a header but no data rows.")
            col_count = len(header)
    except UnicodeDecodeError:
        die("Cannot read file -- encoding error.", hint="Ensure the file is UTF-8 encoded.")
    except csv.Error as e:
        die(f"Invalid CSV format: {e}")
    return input_path, col_count


def _detect_enrichment_mode(args: argparse.Namespace, config: Any) -> tuple:
    """Detect AI adapter and enrichment mode. Returns (adapter, mode)."""
    adapter = config.get_adapter()
    if args.adapter:
        try:
            config.adapter = args.adapter
            adapter = config.get_adapter()
        except Exception as e:
            die(f"Could not initialize adapter '{args.adapter}': {e}")
    if args.mode:
        mode = args.mode
    elif adapter:
        mode = "both"
    else:
        mode = "email"
    if mode in ("ai", "both") and not adapter:
        warn("No AI backend available -- falling back to email enrichment only.")
        info("  Set ANTHROPIC_API_KEY or install Ollama for AI features.")
        mode = "email"
    return adapter, mode


def _run_enrichment_pipeline(db: Any, adapter: Any, args: argparse.Namespace, count: int, mode: str, default_workers: int) -> Any:
    """Run the enrichment pipeline with signal handling."""
    _stop_requested = threading.Event()

    def handle_signal(signum: int, frame: Any) -> None:
        warn("\nInterrupted -- stopping gracefully (Ctrl+C again to force quit)...")
        _stop_requested.set()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        from forge.enrichment.pipeline import EnrichmentPipeline
        workers = args.workers or default_workers
        pipeline = EnrichmentPipeline(db_pool=db.get_pool(), ollama=adapter,
                                      web_scraper_workers=workers, batch_size=args.batch_size or 5)
        info(f"Enriching {count:,} records with {workers} workers...")
        return pipeline.run(mode=mode, state_filter=getattr(args, "state", None), max_records=args.max, resume=args.resume)
    except ImportError as e:
        die(f"Missing dependency: {e}", hint="Run: pip install forge-enrichment")
    except KeyboardInterrupt:
        warn("Interrupted.")
        return None
    except ConnectionError:
        die("Could not connect to target.", hint="Web enrichment requires internet access.")
    except Exception as e:
        logging.getLogger("forge").error("Pipeline failed: %s", e, exc_info=True)
        die(f"Enrichment failed: {e}")


def _export_csv_results(db: Any, args: argparse.Namespace, input_path: Path) -> None:
    """Export enrichment results to a CSV file."""
    output_path = args.output or str(input_path).replace(".csv", "_enriched.csv")
    if output_path == str(input_path):
        output_path = str(input_path).replace(".csv", "_enriched.csv")
    try:
        result = db.export_csv(output_path)
        exported = result.get("row_count", 0) if isinstance(result, dict) else int(result)
    except Exception as e:
        die(f"Failed to export results: {e}")
    info(f"\n{green('Exported')} {exported:,} enriched records to {bold(output_path)}")


def _print_enrichment_summary(db: Any, count: int, stats: Any) -> None:
    """Print the enrichment summary after processing."""
    try:
        db_stats = db.get_stats()
    except Exception:
        db_stats = {}
    info(f"\n{bold('Enrichment Summary')}")
    info("  " + "-" * 40)
    info(f"  Records processed:    {db_stats.get('total_records', count):>8,}")
    info(f"  Emails found:         {db_stats.get('with_email', 0):>8,}")
    info(f"  Tech stacks detected: {db_stats.get('with_tech_stack', 0):>8,}")
    if db_stats.get("with_ai_summary"):
        info(f"  AI summaries:         {db_stats.get('with_ai_summary', 0):>8,}")
    if db_stats.get("with_health_score"):
        info(f"  Health scores:        {db_stats.get('with_health_score', 0):>8,}")
    info("  " + "-" * 40)
    if stats:
        elapsed = time.time() - stats.start_time if stats.start_time else 0
        if elapsed > 0:
            info(f"  Time elapsed:         {elapsed / 60:>7.1f}m")
            info(f"  Processing rate:      {stats.rate_per_hour():>7.0f}/hr")


def run_csv_enrich(args: argparse.Namespace, logger: logging.Logger) -> None:
    """CSV zero-config mode -- imports CSV, enriches, exports results."""
    from forge.config import ForgeConfig
    from forge.db import ForgeDB
    input_path, col_count = _validate_csv_input(args.file)
    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} -- CSV Enrichment Mode")
    info(f"  Input:   {input_path.name} ({col_count} columns)")
    config = ForgeConfig.load()
    tmp_dir = tempfile.mkdtemp(prefix="forge_")
    tmp_db_path = os.path.join(tmp_dir, "forge_temp.db")
    db = ForgeDB.from_config({"db_path": tmp_db_path})
    db.ensure_schema()
    try:
        count = db.import_csv(str(input_path))
    except Exception as e:
        die(f"Failed to import CSV: {e}")
    if count == 0:
        die("No records found in file.", hint="Check that the CSV has recognizable columns.")
    info(f"  Records: {count:,} imported")
    adapter, mode = _detect_enrichment_mode(args, config)
    info(f"  Mode:    {mode}")
    if adapter:
        info(f"  AI:      {adapter.name if hasattr(adapter, 'name') else type(adapter).__name__}")
    info("")
    stats = _run_enrichment_pipeline(db, adapter, args, count, mode, default_workers=30)
    _export_csv_results(db, args, input_path)
    _print_enrichment_summary(db, count, stats)
    if args.keep_db:
        info(f"\n{dim('Database kept at:')} {tmp_db_path}")
    else:
        try:
            os.remove(tmp_db_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def run_database_enrich(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Database mode -- run enrichment against a configured persistent database."""
    from forge.config import ForgeConfig
    from forge.db import ForgeDB
    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
        die("No database configured.",
            hint="Run 'forge enrich --file data.csv' for zero-config mode,\n"
                 "       or 'forge import --file data.csv' to load into a persistent database.")
    try:
        db = ForgeDB.from_config(db_config)
    except Exception as e:
        die(f"Could not connect to database: {e}")
    try:
        db_stats = db.get_stats()
    except Exception as e:
        die(f"Could not read database: {e}")
    total = db_stats.get("total_records", 0)
    if total == 0:
        die("Database is empty.", hint="Run 'forge import --file businesses.csv' to load data first.")
    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} -- Database Enrichment Mode")
    info(f"  Database: {db_config.get('db_path', db_config.get('db_host', 'configured'))}")
    info(f"  Records:  {total:,}")
    adapter, mode = _detect_enrichment_mode(args, config)
    info(f"  Mode:     {mode}")
    if adapter:
        info(f"  AI:       {adapter.name if hasattr(adapter, 'name') else type(adapter).__name__}")
    info("")
    stats = _run_enrichment_pipeline(db, adapter, args, total, mode, default_workers=50)
    if stats:
        info(f"\n{green('Enrichment complete.')}")
        info(stats.summary())
