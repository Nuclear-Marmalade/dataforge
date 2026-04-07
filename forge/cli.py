"""
FORGE CLI — The command-line interface for the FORGE enrichment engine.

This is the main entry point when a user types `forge` on the command line.
Provides subcommands for enrichment, import/export, discovery, and configuration.

Usage:
    forge enrich --file input.csv                         # Zero-config CSV mode
    forge enrich --mode email --workers 50 --resume       # Database mode
    forge enrich --mode ai --adapter claude                # AI with Claude
    forge import --file businesses.csv                     # Import CSV to database
    forge export --output results.csv                      # Export enriched data
    forge status                                           # Show enrichment stats
    forge config show                                      # Show configuration
    forge discover --zip 33602                             # Overture discovery
    forge dashboard                                        # Start web dashboard
    forge mcp-server                                       # Start MCP server

Dependencies: forge.db (ForgeDB), forge.config (ForgeConfig)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from forge import __version__
except ImportError:
    __version__ = "1.0.0"

# ---------------------------------------------------------------------------
# ANSI color helpers — disabled automatically when stdout is not a TTY
# ---------------------------------------------------------------------------

_COLOR_ENABLED: Optional[bool] = None


def _colors_enabled() -> bool:
    """Check if ANSI colors should be used."""
    global _COLOR_ENABLED
    if _COLOR_ENABLED is not None:
        return _COLOR_ENABLED

    # Respect NO_COLOR convention (https://no-color.org)
    if os.environ.get("NO_COLOR"):
        _COLOR_ENABLED = False
        return False

    # Force color if FORCE_COLOR is set
    if os.environ.get("FORCE_COLOR"):
        _COLOR_ENABLED = True
        return True

    # Auto-detect TTY
    _COLOR_ENABLED = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return _COLOR_ENABLED


def _c(text: str, code: str) -> str:
    """Wrap text in ANSI color code if colors are enabled."""
    if not _colors_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def red(text: str) -> str:
    return _c(text, "31")


def bold(text: str) -> str:
    return _c(text, "1")


def dim(text: str) -> str:
    return _c(text, "2")


def cyan(text: str) -> str:
    return _c(text, "36")


# ---------------------------------------------------------------------------
# Progress bar — simple character-based, no external dependencies
# ---------------------------------------------------------------------------

class ProgressBar:
    """
    Simple terminal progress bar using only built-in characters.

    Usage:
        bar = ProgressBar(total=100, label="Enriching")
        for i in range(100):
            bar.update(i + 1)
        bar.finish()
    """

    def __init__(self, total: int, label: str = "", width: int = 40):
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self._current = 0
        self._start_time = time.time()
        self._is_tty = _colors_enabled()

    def update(self, current: int) -> None:
        """Update progress bar to current value."""
        self._current = min(current, self.total)
        if self._is_tty:
            self._render()

    def _render(self) -> None:
        """Render the progress bar to stderr (so stdout stays clean for piping)."""
        fraction = self._current / self.total
        filled = int(self.width * fraction)
        bar = "=" * filled + "-" * (self.width - filled)

        elapsed = time.time() - self._start_time
        if self._current > 0 and elapsed > 0:
            rate = self._current / elapsed
            eta = (self.total - self._current) / rate if rate > 0 else 0
            time_str = f" {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
        else:
            time_str = ""

        label_prefix = f"{self.label}: " if self.label else ""
        line = f"\r{label_prefix}[{bar}] {self._current}/{self.total} ({fraction:.0%}){time_str}"
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self) -> None:
        """Complete the progress bar."""
        self._current = self.total
        if self._is_tty:
            self._render()
            sys.stderr.write("\n")
            sys.stderr.flush()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configure logging for FORGE CLI."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    return logging.getLogger("forge")


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def die(message: str, hint: str = "", exit_code: int = 1) -> None:
    """Print an error message and exit."""
    sys.stderr.write(f"{red('Error:')} {message}\n")
    if hint:
        sys.stderr.write(f"{dim('Hint:')} {hint}\n")
    sys.exit(exit_code)


def warn(message: str) -> None:
    """Print a warning to stderr."""
    sys.stderr.write(f"{yellow('Warning:')} {message}\n")


def info(message: str) -> None:
    """Print an info message to stdout."""
    print(message)


def success(message: str) -> None:
    """Print a success message to stdout."""
    print(f"{green('OK')} {message}")


# ---------------------------------------------------------------------------
# Subcommand: enrich
# ---------------------------------------------------------------------------

def cmd_enrich(args: argparse.Namespace) -> None:
    """
    Run the enrichment pipeline.

    Two modes:
      - CSV mode (--file): zero-config, imports CSV, enriches, exports results
      - Database mode (no --file): uses configured persistent database
    """
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)

    if args.file:
        _run_csv_enrich(args, logger)
    else:
        _run_database_enrich(args, logger)


def _run_csv_enrich(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    CSV zero-config mode — the killer feature.

    1. Validate the input CSV
    2. Create a temp SQLite database
    3. Import the CSV
    4. Detect available AI adapters
    5. Run enrichment pipeline
    6. Export results to output CSV
    7. Print summary
    8. Clean up (unless --keep-db)
    """
    from forge.config import ForgeConfig
    from forge.db import ForgeDB

    # 1. Validate input file
    input_path = Path(args.file)
    if not input_path.exists():
        die(
            f"File not found: {args.file}",
            hint=f"Check the path and try again. Current directory: {os.getcwd()}",
        )
    if not input_path.suffix.lower() == ".csv":
        warn(f"File does not have .csv extension: {args.file}")
    if input_path.stat().st_size == 0:
        die("File is empty.", hint="Provide a CSV with at least a header row and one data row.")

    # Quick-check: does the CSV have any data rows?
    try:
        with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                die("No records found in file.", hint="The CSV appears to be empty.")
            first_row = next(reader, None)
            if first_row is None:
                die("No records found in file.", hint="The CSV has a header but no data rows.")
            col_count = len(header)
    except UnicodeDecodeError:
        die(
            "Cannot read file — encoding error.",
            hint="Ensure the file is UTF-8 encoded. Try: file --mime-encoding " + str(input_path),
        )
    except csv.Error as e:
        die(f"Invalid CSV format: {e}")

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — CSV Enrichment Mode")
    info(f"  Input:   {input_path.name} ({col_count} columns)")

    # 2. Load config
    config = ForgeConfig.load()

    # 3. Create temp SQLite database
    tmp_dir = tempfile.mkdtemp(prefix="forge_")
    tmp_db_path = os.path.join(tmp_dir, "forge_temp.db")
    db = ForgeDB.from_config({"db_path": tmp_db_path})
    db.ensure_schema()

    logger.debug("Temp database created at: %s", tmp_db_path)

    # 4. Import CSV
    try:
        count = db.import_csv(str(input_path))
    except Exception as e:
        die(f"Failed to import CSV: {e}")

    if count == 0:
        die("No records found in file.", hint="Check that the CSV has recognizable columns (name, website, etc.).")

    info(f"  Records: {count:,} imported")

    # 5. Detect adapter
    adapter = config.get_adapter()
    if args.adapter:
        # User explicitly requested an adapter
        try:
            config.adapter = args.adapter
            adapter = config.get_adapter()
        except Exception as e:
            die(f"Could not initialize adapter '{args.adapter}': {e}")

    # Determine enrichment mode
    if args.mode:
        mode = args.mode
    elif adapter:
        mode = "both"
    else:
        mode = "email"

    if mode in ("ai", "both") and not adapter:
        warn("No AI backend available — falling back to email enrichment only.")
        info("  Set ANTHROPIC_API_KEY or install Ollama for AI features.")
        mode = "email"

    info(f"  Mode:    {mode}")
    if adapter:
        info(f"  AI:      {adapter.name if hasattr(adapter, 'name') else type(adapter).__name__}")
    info("")

    # 6. Run enrichment
    _stop_requested = threading.Event()

    def handle_signal(signum: int, frame: Any) -> None:
        warn("\nInterrupted — stopping gracefully (Ctrl+C again to force quit)...")
        _stop_requested.set()
        # Restore default handler for second Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        from forge.enrichment.pipeline import EnrichmentPipeline

        workers = args.workers or 30
        batch_size = args.batch_size or 5

        # Build pipeline from ForgeDB + adapter
        pipeline = EnrichmentPipeline(
            db_pool=db.get_pool(),
            ollama=adapter,
            web_scraper_workers=workers,
            batch_size=batch_size,
        )

        info(f"Enriching {count:,} records with {workers} workers...")

        stats = pipeline.run(
            mode=mode,
            max_records=args.max,
            resume=args.resume,
        )

    except ImportError as e:
        die(
            f"Missing dependency: {e}",
            hint="Run: pip install forge-enrichment",
        )
    except KeyboardInterrupt:
        warn("Interrupted.")
        stats = None
    except ConnectionError:
        die(
            "Could not connect to target. Check your internet connection.",
            hint="Web enrichment requires internet access to scrape websites.",
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        die(f"Enrichment failed: {e}")

    # 7. Export results
    output_path = args.output or str(input_path).replace(".csv", "_enriched.csv")
    if output_path == str(input_path):
        output_path = str(input_path).replace(".csv", "_enriched.csv")

    try:
        result = db.export_csv(output_path)
        exported = result.get("row_count", 0) if isinstance(result, dict) else int(result)
    except Exception as e:
        die(f"Failed to export results: {e}")

    info(f"\n{green('Exported')} {exported:,} enriched records to {bold(output_path)}")

    # 8. Print summary
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

    # 9. Cleanup
    if args.keep_db:
        info(f"\n{dim('Database kept at:')} {tmp_db_path}")
    else:
        try:
            os.remove(tmp_db_path)
            os.rmdir(tmp_dir)
            logger.debug("Temp database cleaned up")
        except OSError as e:
            logger.debug("Cleanup failed (non-critical): %s", e)


def _run_database_enrich(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Database mode — run enrichment against a configured persistent database.

    Requires prior import via `forge import` or data from `forge discover`.
    """
    from forge.config import ForgeConfig
    from forge.db import ForgeDB

    # Load config
    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
        die(
            "No database configured.",
            hint="Run 'forge enrich --file data.csv' for zero-config mode,\n"
                 "       or 'forge import --file data.csv' to load into a persistent database.",
        )

    # Connect to database
    try:
        db = ForgeDB.from_config(db_config)
    except Exception as e:
        die(f"Could not connect to database: {e}")

    # Check for records
    try:
        db_stats = db.get_stats()
    except Exception as e:
        die(f"Could not read database: {e}")

    total = db_stats.get("total_records", 0)
    if total == 0:
        die(
            "Database is empty — no records to enrich.",
            hint="Run 'forge import --file businesses.csv' to load data first.",
        )

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Database Enrichment Mode")
    info(f"  Database: {db_config.get('db_path', db_config.get('db_host', 'configured'))}")
    info(f"  Records:  {total:,}")

    # Detect adapter
    adapter = config.get_adapter()
    if args.adapter:
        try:
            config.adapter = args.adapter
            adapter = config.get_adapter()
        except Exception as e:
            die(f"Could not initialize adapter '{args.adapter}': {e}")

    # Determine mode
    if args.mode:
        mode = args.mode
    elif adapter:
        mode = "both"
    else:
        mode = "email"

    if mode in ("ai", "both") and not adapter:
        warn("No AI backend available — falling back to email enrichment only.")
        info("  Set ANTHROPIC_API_KEY or install Ollama for AI features.")
        mode = "email"

    info(f"  Mode:     {mode}")
    if adapter:
        info(f"  AI:       {adapter.name if hasattr(adapter, 'name') else type(adapter).__name__}")
    info("")

    # SIGINT handler
    _stop_requested = threading.Event()

    def handle_signal(signum: int, frame: Any) -> None:
        warn("\nInterrupted — stopping gracefully (Ctrl+C again to force quit)...")
        _stop_requested.set()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run pipeline
    try:
        from forge.enrichment.pipeline import EnrichmentPipeline

        workers = args.workers or 50
        batch_size = args.batch_size or 5

        pipeline = EnrichmentPipeline(
            db_pool=db.get_pool(),
            ollama=adapter,
            web_scraper_workers=workers,
            batch_size=batch_size,
        )

        info(f"Enriching {total:,} records with {workers} workers...")

        stats = pipeline.run(
            mode=mode,
            state_filter=args.state,
            max_records=args.max,
            resume=args.resume,
        )

        info(f"\n{green('Enrichment complete.')}")
        info(stats.summary())

    except KeyboardInterrupt:
        warn("Interrupted.")
    except ConnectionError:
        die(
            "Could not connect to target. Check your internet connection.",
            hint="Web enrichment requires internet access to scrape websites.",
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        die(f"Enrichment failed: {e}")


# ---------------------------------------------------------------------------
# Subcommand: import
# ---------------------------------------------------------------------------

def cmd_import(args: argparse.Namespace) -> None:
    """Import a CSV file into the persistent database."""
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)

    from forge.config import ForgeConfig
    from forge.db import ForgeDB

    # Validate input file
    input_path = Path(args.file)
    if not input_path.exists():
        die(
            f"File not found: {args.file}",
            hint=f"Check the path and try again. Current directory: {os.getcwd()}",
        )
    if input_path.stat().st_size == 0:
        die("File is empty.", hint="Provide a CSV with at least a header row and one data row.")

    # Quick sanity check
    try:
        with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                die("No records found in file.")
            first_row = next(reader, None)
            if first_row is None:
                die("No records found in file.", hint="The CSV has a header but no data rows.")
    except UnicodeDecodeError:
        die("Cannot read file — encoding error.", hint="Ensure the file is UTF-8 encoded.")
    except csv.Error as e:
        die(f"Invalid CSV format: {e}")

    # Load config and connect
    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
        # Default to local SQLite
        default_path = os.path.join(os.path.expanduser("~"), ".forge", "forge.db")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        db_config = {"db_path": default_path}
        info(f"No database configured — using default: {default_path}")

    try:
        db = ForgeDB.from_config(db_config)
        db.ensure_schema()
    except Exception as e:
        die(f"Could not connect to database: {e}")

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Import")
    info(f"  File:     {input_path.name}")

    # Import
    try:
        result = db.import_csv(str(input_path), return_details=True)
        if isinstance(result, dict):
            new_count = result.get("new", 0)
            updated_count = result.get("updated", 0)
            skipped_count = result.get("skipped", 0)
            total = new_count + updated_count + skipped_count
        else:
            total = int(result)
            new_count = total
            updated_count = 0
            skipped_count = 0
    except Exception as e:
        die(f"Import failed: {e}")

    info("")
    info(f"  {green('Imported:')}  {total:,} records")
    if new_count:
        info(f"    New:      {new_count:,}")
    if updated_count:
        info(f"    Updated:  {updated_count:,}")
    if skipped_count:
        info(f"    Skipped:  {skipped_count:,}")

    info(f"\nRun {bold('forge enrich')} to start enrichment, or {bold('forge status')} to check database stats.")


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> None:
    """Export enriched data from the database."""
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)

    from forge.config import ForgeConfig
    from forge.db import ForgeDB

    # Load config and connect
    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
        # Check default SQLite
        default_path = os.path.join(os.path.expanduser("~"), ".forge", "forge.db")
        if os.path.exists(default_path):
            db_config = {"db_path": default_path}
        else:
            die(
                "No database configured and no default database found.",
                hint="Run 'forge enrich --file data.csv' or 'forge import --file data.csv' first.",
            )

    try:
        db = ForgeDB.from_config(db_config)
    except Exception as e:
        die(f"Could not connect to database: {e}")

    output_path = args.output
    output_format = getattr(args, "format", "csv") or "csv"

    # Support both --filter (new, safe) and --where (legacy, deprecated)
    filter_name = getattr(args, "filter", None) or getattr(args, "where", None)

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Export")

    try:
        if output_format == "json":
            result = db.export_json(output_path, where=filter_name)
        else:
            result = db.export_csv(output_path, where=filter_name)
        exported = result.get("row_count", 0) if isinstance(result, dict) else int(result)
    except Exception as e:
        die(f"Export failed: {e}")

    if exported == 0:
        warn("No records matched the export criteria.")
        if filter_name:
            info(f"  Filter: {filter_name}")
            info("  Try removing the --filter flag to export all records.")
    else:
        info(f"  {green('Exported')} {exported:,} records to {bold(output_path)}")
        if filter_name:
            info(f"  Filter: {filter_name}")


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Show enrichment statistics for the current database."""
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)

    from forge.config import ForgeConfig
    from forge.db import ForgeDB

    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
        default_path = os.path.join(os.path.expanduser("~"), ".forge", "forge.db")
        if os.path.exists(default_path):
            db_config = {"db_path": default_path}
        else:
            die(
                "No database configured and no default database found.",
                hint="Run 'forge enrich --file data.csv' or 'forge import --file data.csv' first.",
            )

    try:
        db = ForgeDB.from_config(db_config)
        stats = db.get_stats()
    except Exception as e:
        die(f"Could not read database: {e}")

    total = stats.get("total_records", 0)
    if total == 0:
        info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Status")
        info("\n  Database is empty. Import data to get started:")
        info(f"    {bold('forge import --file businesses.csv')}")
        info(f"    {bold('forge enrich --file businesses.csv')}")
        return

    # Calculate percentages
    def pct(n: int) -> str:
        if total == 0:
            return "0%"
        return f"{n / total * 100:.1f}%"

    with_email = stats.get("with_email", 0)
    with_tech = stats.get("with_tech_stack", 0)
    with_summary = stats.get("with_ai_summary", 0)
    with_health = stats.get("with_health_score", 0)
    with_industry = stats.get("with_industry", 0)
    with_website = stats.get("with_website", 0)

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Status")
    info("")

    # Database info
    db_display = db_config.get("db_path", db_config.get("db_host", "configured"))
    info(f"  Database: {db_display}")
    info(f"  Total records: {bold(f'{total:,}')}")
    info("")

    # Enrichment table
    info(f"  {'Field':<24} {'Count':>8}  {'Rate':>6}")
    info(f"  {'-' * 24} {'-' * 8}  {'-' * 6}")
    info(f"  {'Website URL':<24} {with_website:>8,}  {pct(with_website):>6}")
    info(f"  {'Email':<24} {with_email:>8,}  {pct(with_email):>6}")
    info(f"  {'Tech stack':<24} {with_tech:>8,}  {pct(with_tech):>6}")
    info(f"  {'Industry':<24} {with_industry:>8,}  {pct(with_industry):>6}")
    info(f"  {'AI summary':<24} {with_summary:>8,}  {pct(with_summary):>6}")
    info(f"  {'Health score':<24} {with_health:>8,}  {pct(with_health):>6}")
    info("")

    # Overall enrichment rate
    enriched_fields = with_email + with_tech + with_summary + with_health + with_industry
    possible_fields = total * 5  # 5 key enrichment fields
    overall_rate = enriched_fields / possible_fields * 100 if possible_fields > 0 else 0
    info(f"  Overall enrichment rate: {bold(f'{overall_rate:.1f}%')}")
    info("")

    # Suggestions
    if with_email == 0 and with_website > 0:
        info(f"  {yellow('Tip:')} Run {bold('forge enrich --mode email')} to extract emails from websites.")
    if with_summary == 0 and total > 0:
        info(f"  {yellow('Tip:')} Run {bold('forge enrich --mode ai')} for AI-powered enrichment.")


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> None:
    """Show or set configuration values."""
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)

    from forge.config import ForgeConfig

    config = ForgeConfig.load()

    if args.config_action == "show":
        _config_show(config)
    elif args.config_action == "set":
        _config_set(config, args.key, args.value)
    else:
        die("Unknown config action. Use 'forge config show' or 'forge config set KEY VALUE'.")


def _config_show(config: Any) -> None:
    """Display current configuration."""
    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Configuration")
    info("")

    config_path = config.config_path if hasattr(config, "config_path") else "~/.forge/config.toml"
    info(f"  Config file: {config_path}")
    info("")

    # Database
    db_config = config.to_db_config()
    if db_config:
        db_type = "PostgreSQL" if db_config.get("db_host") else "SQLite"
        db_display = db_config.get("db_path", db_config.get("db_host", "unknown"))
        info(f"  Database:    {db_type} ({db_display})")
    else:
        default_path = os.path.join(os.path.expanduser("~"), ".forge", "forge.db")
        info(f"  Database:    SQLite (default: {default_path})")

    # AI adapter
    adapter = config.get_adapter()
    if adapter:
        adapter_name = adapter.name if hasattr(adapter, "name") else type(adapter).__name__
        info(f"  AI backend:  {green(adapter_name)}")
    else:
        info(f"  AI backend:  {dim('none (email enrichment only)')}")
        # Check which could be enabled
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        if has_anthropic:
            info(f"               {dim('ANTHROPIC_API_KEY is set — Claude adapter available')}")
        if has_openai:
            info(f"               {dim('OPENAI_API_KEY is set — OpenAI adapter available')}")
        if not has_anthropic and not has_openai:
            info(f"               {dim('Set ANTHROPIC_API_KEY or install Ollama to enable AI')}")

    # Workers
    workers = getattr(config, "workers", None)
    if workers:
        info(f"  Workers:     {workers}")
    else:
        info(f"  Workers:     {dim('50 (default)')}")

    info("")

    # Show all config values if available
    all_config = config.as_dict() if hasattr(config, "as_dict") else {}
    if all_config:
        info("  All settings:")
        for key, value in sorted(all_config.items()):
            # Mask sensitive values
            if any(secret in key.lower() for secret in ("key", "password", "secret", "token")):
                display = value[:4] + "..." + value[-4:] if isinstance(value, str) and len(value) > 8 else "***"
            else:
                display = str(value)
            info(f"    {key:<28} {display}")


def _config_set(config: Any, key: str, value: str) -> None:
    """Set a configuration value."""
    if not key:
        die("Key is required. Usage: forge config set KEY VALUE")

    try:
        from forge.config import cli_config_set
        cli_config_set(key, value)
    except Exception as e:
        die(f"Failed to set config value: {e}")


# ---------------------------------------------------------------------------
# Subcommand: discover (placeholder)
# ---------------------------------------------------------------------------

def cmd_discover(args: argparse.Namespace) -> None:
    """Discover businesses using Overture Maps data."""
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Discover")
    info("")

    try:
        from forge.discovery.overture import OvertureDiscovery
    except ImportError:
        die(
            "Discovery requires duckdb.",
            hint="pip install duckdb to enable discovery",
        )
        return

    if not args.zip and not args.city and not args.state:
        die(
            "No location specified.",
            hint="Use --zip 33602, --city Tampa, or --state FL",
        )

    try:
        disco = OvertureDiscovery()
        results = disco.search(
            zip_code=args.zip,
            city=getattr(args, "city", None),
            state=getattr(args, "state", None),
            category=getattr(args, "category", None),
        )
        disco.close()
    except Exception as e:
        die(f"Discovery failed: {e}")
        return

    info(f"  Found {len(results):,} businesses")

    # Export to CSV if --output specified
    if results and getattr(args, "output", None):
        output_path = args.output
        try:
            fieldnames = sorted({k for r in results for k in r.keys()})
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            info(f"  {green('Exported')} {len(results):,} businesses to {bold(output_path)}")
        except Exception as e:
            warn(f"Failed to export CSV: {e}")

    if results and hasattr(args, "enrich") and args.enrich:
        info("  --enrich flag set: importing discovered businesses for enrichment...")
        from forge.config import ForgeConfig
        from forge.db import ForgeDB

        config = ForgeConfig.load()
        db_config = config.to_db_config()
        if not db_config:
            default_path = os.path.join(os.path.expanduser("~"), ".forge", "forge.db")
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            db_config = {"db_path": default_path}

        db = ForgeDB.from_config(db_config)
        db.ensure_schema()
        imported = 0
        for record in results:
            try:
                db.upsert_business(record)
                imported += 1
            except Exception as e:
                logger.debug("Upsert failed for %s: %s", record.get("name", "?"), e)
        db.close()
        info(f"  {green('Imported')} {imported:,} businesses. Run {bold('forge enrich')} to enrich them.")
    elif results:
        # Print a preview
        for r in results[:10]:
            name = r.get("name", "Unknown")
            city = r.get("city", "")
            state = r.get("state", "")
            info(f"  - {name} ({city}, {state})")
        if len(results) > 10:
            info(f"  ... and {len(results) - 10:,} more")
        info(f"\n  Use --enrich to import and enrich these businesses.")


# ---------------------------------------------------------------------------
# Subcommand: dashboard (placeholder)
# ---------------------------------------------------------------------------

def cmd_dashboard(args: argparse.Namespace) -> None:
    """Start the FORGE web dashboard."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    port = getattr(args, "port", 8080) or 8080

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Dashboard")
    info(f"  Starting dashboard on http://127.0.0.1:{port}")
    info(f"  Press Ctrl+C to stop.\n")

    try:
        from forge.dashboard.app import app
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=port)
    except ImportError as e:
        die(
            f"Dashboard requires FastAPI and uvicorn: {e}",
            hint="pip install fastapi uvicorn jinja2",
        )
    except KeyboardInterrupt:
        info("\nDashboard stopped.")


# ---------------------------------------------------------------------------
# Subcommand: mcp-server (placeholder)
# ---------------------------------------------------------------------------

def cmd_mcp_server(args: argparse.Namespace) -> None:
    """Start the FORGE MCP server for AI assistant integration."""
    # MCP server uses stdin/stdout for JSON-RPC — logging goes to stderr
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    from forge.mcp_server import run_server
    run_server()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
  _____ ___  ____   ____ _____
 |  ___/ _ \|  _ \ / ___| ____|
 | |_ | | | | |_) | |  _|  _|
 |  _|| |_| |  _ <| |_| | |___
 |_|   \___/|_| \_\\____|_____|
"""


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the complete argument parser with all subcommands."""

    parser = argparse.ArgumentParser(
        prog="forge",
        description=(
            "FORGE — Free Open-source Runtime for Generalized Enrichment.\n"
            "The open-source alternative to Apollo, ZoomInfo, and Clearbit.\n\n"
            "Quick start:\n"
            "  forge enrich --file businesses.csv           # Enrich a CSV (zero config)\n"
            "  forge enrich --mode email --workers 50       # Database mode\n"
            "  forge status                                 # Check enrichment stats\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  forge enrich --file leads.csv --output enriched.csv\n"
            "  forge enrich --mode both --workers 50 --resume\n"
            "  forge import --file businesses.csv\n"
            "  forge export --output results.csv --filter with_email\n"
            "  forge status\n"
            "  forge config show\n"
            "\n"
            "Documentation: https://github.com/ghealysr/forge\n"
        ),
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"forge {__version__}",
    )

    # Global flags (available on all subcommands)
    # These are added to each subparser, not the root parser,
    # so they appear in subcommand help.

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Run 'forge <command> --help' for details on each command.",
        metavar="<command>",
    )

    # ── enrich ──────────────────────────────────────────────────────────────

    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Run the enrichment pipeline (CSV or database mode)",
        description=(
            "Enrich business data with emails, tech stacks, AI summaries, and more.\n\n"
            "CSV mode (--file):     Zero-config. Imports CSV, enriches, exports results.\n"
            "Database mode:         Uses configured database. Requires prior import or discover."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  forge enrich --file leads.csv                   # Enrich CSV, output leads_enriched.csv\n"
            "  forge enrich --file in.csv --output out.csv     # Custom output path\n"
            "  forge enrich --mode email --workers 50          # Email extraction only, 50 workers\n"
            "  forge enrich --mode ai --adapter claude         # AI enrichment with Claude\n"
            "  forge enrich --mode both --resume               # Both tracks, resume from last run\n"
        ),
    )
    enrich_parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Input CSV file (enables zero-config CSV mode)",
    )
    enrich_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: <input>_enriched.csv)",
    )
    enrich_parser.add_argument(
        "--mode", "-m",
        choices=["email", "ai", "both"],
        default=None,
        help="Enrichment mode: email (web scraping), ai (LLM), or both (default: auto-detect)",
    )
    enrich_parser.add_argument(
        "--adapter", "-a",
        type=str,
        default=None,
        help="AI adapter to use: ollama, claude, openai (default: auto-detect)",
    )
    enrich_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of concurrent web scraper workers (default: 30 CSV, 50 database)",
    )
    enrich_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of records per AI batch (default: 5)",
    )
    enrich_parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of records to process",
    )
    enrich_parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Filter by US state code (e.g. CA, FL) — database mode only",
    )
    enrich_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last run, skip already-enriched records (default: true)",
    )
    enrich_parser.add_argument(
        "--no-resume",
        action="store_true",
        dest="no_resume",
        default=False,
        help="Process all records including already-enriched",
    )
    enrich_parser.add_argument(
        "--keep-db",
        action="store_true",
        default=False,
        help="Keep temporary database after CSV mode (for debugging)",
    )
    _add_global_flags(enrich_parser)
    enrich_parser.set_defaults(func=cmd_enrich)

    # ── import ──────────────────────────────────────────────────────────────

    import_parser = subparsers.add_parser(
        "import",
        help="Import a CSV file into the persistent database",
        description=(
            "Import business records from a CSV file into FORGE's database.\n\n"
            "The CSV should contain columns like: name, website, phone, address, city, state, zip.\n"
            "Column mapping is automatic — FORGE recognizes common column name variations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  forge import --file businesses.csv\n"
            "  forge import --file leads.csv\n"
        ),
    )
    import_parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="CSV file to import",
    )
    _add_global_flags(import_parser)
    import_parser.set_defaults(func=cmd_import)

    # ── export ──────────────────────────────────────────────────────────────

    export_parser = subparsers.add_parser(
        "export",
        help="Export enriched data from the database",
        description=(
            "Export enriched business data to CSV or JSON.\n\n"
            "Use --filter to select a predefined filter.\n"
            "Available filters: all, with_email, with_tech, enriched, "
            "with_website, with_npi, with_ai"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  forge export --output results.csv\n"
            "  forge export --output results.csv --filter with_email\n"
            "  forge export --output results.json --format json\n"
            "  forge export --output enriched.csv --filter enriched\n"
        ),
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path",
    )
    export_parser.add_argument(
        "--filter",
        type=str,
        default=None,
        choices=["all", "with_email", "with_tech", "enriched",
                 "with_website", "with_npi", "with_ai"],
        help="Predefined filter name (e.g. with_email, enriched)",
    )
    export_parser.add_argument(
        "--where",
        type=str,
        default=None,
        help="(Deprecated — use --filter instead) Raw WHERE clause",
    )
    export_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    _add_global_flags(export_parser)
    export_parser.set_defaults(func=cmd_export)

    # ── discover ────────────────────────────────────────────────────────────

    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover businesses using Overture Maps data",
        description=(
            "Discover businesses in a geographic area using Overture Maps Foundation data.\n\n"
            "Requires duckdb: pip install duckdb"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    discover_parser.add_argument(
        "--zip",
        type=str,
        default=None,
        help="ZIP code to search",
    )
    discover_parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="City name to search",
    )
    discover_parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="State code to search (e.g. FL, CA)",
    )
    discover_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Business category filter (e.g. restaurant, dentist)",
    )
    discover_parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Automatically enrich discovered businesses",
    )
    discover_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Export discovered businesses to a CSV file",
    )
    _add_global_flags(discover_parser)
    discover_parser.set_defaults(func=cmd_discover)

    # ── status ──────────────────────────────────────────────────────────────

    status_parser = subparsers.add_parser(
        "status",
        help="Show enrichment statistics and database info",
        description="Display enrichment progress, data quality metrics, and database statistics.",
    )
    _add_global_flags(status_parser)
    status_parser.set_defaults(func=cmd_status)

    # ── config ──────────────────────────────────────────────────────────────

    config_parser = subparsers.add_parser(
        "config",
        help="Show or modify FORGE configuration",
        description=(
            "View and manage FORGE configuration.\n\n"
            "Actions:\n"
            "  show    Display current configuration\n"
            "  set     Set a configuration value"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  forge config show\n"
            "  forge config set workers 100\n"
            "  forge config set adapter ollama\n"
            "  forge config set db_path /path/to/forge.db\n"
        ),
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_action",
        title="actions",
        metavar="<action>",
    )

    config_show_parser = config_subparsers.add_parser(
        "show",
        help="Display current configuration",
    )
    _add_global_flags(config_show_parser)

    config_set_parser = config_subparsers.add_parser(
        "set",
        help="Set a configuration value",
    )
    config_set_parser.add_argument("key", help="Configuration key")
    config_set_parser.add_argument("value", help="Configuration value")
    _add_global_flags(config_set_parser)

    _add_global_flags(config_parser)
    config_parser.set_defaults(func=cmd_config)

    # ── dashboard ───────────────────────────────────────────────────────────

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start the FORGE web dashboard",
        description="Launch a local web dashboard for real-time enrichment monitoring.",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the dashboard server (default: 8080)",
    )
    _add_global_flags(dashboard_parser)
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # ── mcp-server ──────────────────────────────────────────────────────────

    mcp_parser = subparsers.add_parser(
        "mcp-server",
        help="Start the FORGE MCP server for AI assistant integration",
        description="Start a Model Context Protocol server that exposes FORGE tools to AI assistants like Claude Code.",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port for the MCP server (default: 3000)",
    )
    _add_global_flags(mcp_parser)
    mcp_parser.set_defaults(func=cmd_mcp_server)

    return parser


def _add_global_flags(parser: argparse.ArgumentParser) -> None:
    """Add global flags that are shared across all subcommands."""
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress all output except errors",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for the FORGE CLI.

    Called by __main__.py or the `forge` console script.
    Parses arguments, dispatches to the appropriate subcommand handler.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle --no-resume by setting resume=False
    if hasattr(args, "no_resume") and args.no_resume:
        args.resume = False

    # If no subcommand given, show help
    if not args.command:
        # Show the banner if TTY
        if _colors_enabled():
            sys.stderr.write(dim(BANNER))
            sys.stderr.write("\n")
        parser.print_help()
        sys.exit(0)

    # Config subcommand without action defaults to show
    if args.command == "config" and not getattr(args, "config_action", None):
        args.config_action = "show"

    # Dispatch to the handler function
    if hasattr(args, "func"):
        try:
            args.func(args)
        except KeyboardInterrupt:
            sys.stderr.write("\n")
            warn("Interrupted.")
            sys.exit(130)
        except BrokenPipeError:
            # Handle piping to head/less gracefully
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
