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
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

try:
    from forge import __version__
except ImportError:
    __version__ = "1.0.0"

# UI helpers (colors, progress bar, logging, messages) extracted to cli_helpers.py
from forge.cli_helpers import (  # noqa: E402
    setup_logging, die, warn, info,
)

# Re-export color functions that read _COLOR_ENABLED from this module
_COLOR_ENABLED: Optional[bool] = None


def _colors_enabled() -> bool:
    """Check if ANSI colors should be used."""
    global _COLOR_ENABLED
    if _COLOR_ENABLED is not None:
        return _COLOR_ENABLED
    if os.environ.get("NO_COLOR"):
        _COLOR_ENABLED = False
        return False
    if os.environ.get("FORCE_COLOR"):
        _COLOR_ENABLED = True
        return True
    _COLOR_ENABLED = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return _COLOR_ENABLED


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _colors_enabled() else text


def green(text: str) -> str: return _c(text, "32")
def yellow(text: str) -> str: return _c(text, "33")
def red(text: str) -> str: return _c(text, "31")
def bold(text: str) -> str: return _c(text, "1")
def dim(text: str) -> str: return _c(text, "2")
def cyan(text: str) -> str: return _c(text, "36")


# ---------------------------------------------------------------------------
# Subcommand: enrich
# ---------------------------------------------------------------------------

def cmd_enrich(args: argparse.Namespace) -> None:
    """Run the enrichment pipeline."""
    from forge.cli_enrich import run_csv_enrich, run_database_enrich
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)
    if args.file:
        run_csv_enrich(args, logger)
    else:
        run_database_enrich(args, logger)


# ---------------------------------------------------------------------------
# Subcommand: import
# ---------------------------------------------------------------------------

def _validate_import_file(file_path: str) -> Path:
    """Validate the import file exists and has data. Returns Path or dies."""
    input_path = Path(file_path)
    if not input_path.exists():
        die(f"File not found: {file_path}", hint=f"Check the path and try again. Current directory: {os.getcwd()}")
    if input_path.stat().st_size == 0:
        die("File is empty.", hint="Provide a CSV with at least a header row and one data row.")
    try:
        with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                die("No records found in file.")
            if next(reader, None) is None:
                die("No records found in file.", hint="The CSV has a header but no data rows.")
    except UnicodeDecodeError:
        die("Cannot read file — encoding error.", hint="Ensure the file is UTF-8 encoded.")
    except csv.Error as e:
        die(f"Invalid CSV format: {e}")
    return input_path


def _print_import_results(result: Any) -> None:
    """Print import result summary."""
    if isinstance(result, dict):
        new_count = result.get("new", 0)
        updated_count = result.get("updated", 0)
        skipped_count = result.get("skipped", 0)
        total = new_count + updated_count + skipped_count
    else:
        total = int(result)
        new_count, updated_count, skipped_count = total, 0, 0

    info("")
    info(f"  {green('Imported:')}  {total:,} records")
    if new_count:
        info(f"    New:      {new_count:,}")
    if updated_count:
        info(f"    Updated:  {updated_count:,}")
    if skipped_count:
        info(f"    Skipped:  {skipped_count:,}")
    info(f"\nRun {bold('forge enrich')} to start enrichment, or {bold('forge status')} to check database stats.")


def cmd_import(args: argparse.Namespace) -> None:
    """Import a CSV file into the persistent database."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    from forge.config import ForgeConfig
    from forge.db import ForgeDB

    input_path = _validate_import_file(args.file)

    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
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

    try:
        result = db.import_csv(str(input_path), return_details=True)
    except Exception as e:
        die(f"Import failed: {e}")

    _print_import_results(result)


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------

def _get_export_db():
    """Load config and return (db, db_config) or die."""
    from forge.config import ForgeConfig
    from forge.db import ForgeDB
    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
        default_path = os.path.join(os.path.expanduser("~"), ".forge", "forge.db")
        if os.path.exists(default_path):
            db_config = {"db_path": default_path}
        else:
            die("No database configured and no default database found.",
                hint="Run 'forge enrich --file data.csv' or 'forge import --file data.csv' first.")
    try:
        return ForgeDB.from_config(db_config)
    except Exception as e:
        die(f"Could not connect to database: {e}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export enriched data from the database."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    db = _get_export_db()
    output_path = args.output
    output_format = getattr(args, "format", "csv") or "csv"
    filter_name = getattr(args, "filter", None) or getattr(args, "where", None)

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Export")
    try:
        result = db.export_json(output_path, where=filter_name) if output_format == "json" else db.export_csv(output_path, where=filter_name)
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

def _print_status_table(stats: dict, db_config: dict) -> None:
    """Print the enrichment stats table."""
    total = stats.get("total_records", 0)
    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "0%"

    fields = [
        ("Website URL", stats.get("with_website", 0)),
        ("Email", stats.get("with_email", 0)),
        ("Tech stack", stats.get("with_tech_stack", 0)),
        ("Industry", stats.get("with_industry", 0)),
        ("AI summary", stats.get("with_ai_summary", 0)),
        ("Health score", stats.get("with_health_score", 0)),
    ]

    db_display = db_config.get("db_path", db_config.get("db_host", "configured"))
    info(f"  Database: {db_display}")
    info(f"  Total records: {bold(f'{total:,}')}")
    info("")
    info(f"  {'Field':<24} {'Count':>8}  {'Rate':>6}")
    info(f"  {'-' * 24} {'-' * 8}  {'-' * 6}")
    for label, count in fields:
        info(f"  {label:<24} {count:>8,}  {pct(count):>6}")
    info("")

    enriched_sum = sum(c for _, c in fields[1:])  # skip website
    possible = total * 5
    overall = enriched_sum / possible * 100 if possible > 0 else 0
    info(f"  Overall enrichment rate: {bold(f'{overall:.1f}%')}")
    info("")

    if stats.get("with_email", 0) == 0 and stats.get("with_website", 0) > 0:
        info(f"  {yellow('Tip:')} Run {bold('forge enrich --mode email')} to extract emails from websites.")
    if stats.get("with_ai_summary", 0) == 0 and total > 0:
        info(f"  {yellow('Tip:')} Run {bold('forge enrich --mode ai')} for AI-powered enrichment.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show enrichment statistics for the current database."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    from forge.config import ForgeConfig
    from forge.db import ForgeDB

    config = ForgeConfig.load()
    db_config = config.to_db_config()
    if not db_config:
        default_path = os.path.join(os.path.expanduser("~"), ".forge", "forge.db")
        if os.path.exists(default_path):
            db_config = {"db_path": default_path}
        else:
            die("No database configured and no default database found.",
                hint="Run 'forge enrich --file data.csv' or 'forge import --file data.csv' first.")

    try:
        db = ForgeDB.from_config(db_config)
        stats = db.get_stats()
    except Exception as e:
        die(f"Could not read database: {e}")

    total = stats.get("total_records", 0)
    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Status")
    if total == 0:
        info("\n  Database is empty. Import data to get started:")
        info(f"    {bold('forge import --file businesses.csv')}")
        info(f"    {bold('forge enrich --file businesses.csv')}")
        return

    info("")
    _print_status_table(stats, db_config)


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> None:
    """Show or set configuration values."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    from forge.config import ForgeConfig

    config = ForgeConfig.load()

    if args.config_action == "show":
        _config_show(config)
    elif args.config_action == "set":
        _config_set(config, args.key, args.value)
    else:
        die("Unknown config action. Use 'forge config show' or 'forge config set KEY VALUE'.")


def _print_config_header(config: Any) -> None:
    """Print config file path and database info."""
    config_path = config.config_path if hasattr(config, "config_path") else "~/.forge/config.toml"
    info(f"  Config file: {config_path}")
    info("")

    db_config = config.to_db_config()
    if db_config:
        db_type = "PostgreSQL" if db_config.get("db_host") else "SQLite"
        db_display = db_config.get("db_path", db_config.get("db_host", "unknown"))
        info(f"  Database:    {db_type} ({db_display})")
    else:
        info(f"  Database:    SQLite (default: {os.path.join(os.path.expanduser('~'), '.forge', 'forge.db')})")


def _print_adapter_info(config: Any) -> None:
    """Print AI adapter info."""
    adapter = config.get_adapter()
    if adapter:
        adapter_name = adapter.name if hasattr(adapter, "name") else type(adapter).__name__
        info(f"  AI backend:  {green(adapter_name)}")
    else:
        info(f"  AI backend:  {dim('none (email enrichment only)')}")
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        if has_anthropic:
            info(f"               {dim('ANTHROPIC_API_KEY is set — Claude adapter available')}")
        if has_openai:
            info(f"               {dim('OPENAI_API_KEY is set — OpenAI adapter available')}")
        if not has_anthropic and not has_openai:
            info(f"               {dim('Set ANTHROPIC_API_KEY or install Ollama to enable AI')}")


def _config_show(config: Any) -> None:
    """Display current configuration."""
    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Configuration")
    info("")
    _print_config_header(config)
    _print_adapter_info(config)

    workers = getattr(config, "workers", None)
    info(f"  Workers:     {workers}" if workers else f"  Workers:     {dim('50 (default)')}")
    info("")

    all_config = config.as_dict() if hasattr(config, "as_dict") else {}
    if all_config:
        info("  All settings:")
        for key, value in sorted(all_config.items()):
            if any(s in key.lower() for s in ("key", "password", "secret", "token")):
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

def _run_discovery(args: argparse.Namespace) -> list:
    """Execute the Overture search. Returns results list."""
    from forge.discovery.overture import OvertureDiscovery
    disco = OvertureDiscovery()
    results = disco.search(
        zip_code=args.zip,
        city=getattr(args, "city", None) or None,
        state=getattr(args, "state", None) or None,
        industry=getattr(args, "category", None) or None,
    )
    disco.close()
    return results


def _display_results(results: list, args: argparse.Namespace, logger: Any) -> None:
    """Export, import, or preview discovery results."""
    if results and getattr(args, "output", None):
        try:
            fieldnames = sorted({k for r in results for k in r.keys()})
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            info(f"  {green('Exported')} {len(results):,} businesses to {bold(args.output)}")
        except Exception as e:
            warn(f"Failed to export CSV: {e}")

    if results and hasattr(args, "enrich") and args.enrich:
        _import_discovered(results, logger)
    elif results:
        for r in results[:10]:
            info(f"  - {r.get('name', 'Unknown')} ({r.get('city', '')}, {r.get('state', '')})")
        if len(results) > 10:
            info(f"  ... and {len(results) - 10:,} more")
        info("\n  Use --enrich to import and enrich these businesses.")


def _import_discovered(results: list, logger: Any) -> None:
    """Import discovered businesses for enrichment."""
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


def cmd_discover(args: argparse.Namespace) -> None:
    """Discover businesses using Overture Maps data."""
    logger = setup_logging(verbose=args.verbose, quiet=args.quiet)
    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Discover")
    info("")

    try:
        from forge.discovery.overture import OvertureDiscovery  # noqa: F401
    except ImportError:
        die("Discovery requires duckdb.", hint="pip install duckdb to enable discovery")
        return
    if not args.zip and not args.city and not args.state:
        die("No location specified.", hint="Use --zip 33602, --city Tampa, or --state FL")

    try:
        results = _run_discovery(args)
    except Exception as e:
        die(f"Discovery failed: {e}")
        return

    info(f"  Found {len(results):,} businesses")
    _display_results(results, args, logger)


# ---------------------------------------------------------------------------
# Subcommand: dashboard (placeholder)
# ---------------------------------------------------------------------------

def cmd_dashboard(args: argparse.Namespace) -> None:
    """Start the FORGE web dashboard."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    port = getattr(args, "port", 8080) or 8080

    info(f"\n{bold('FORGE')} {dim(f'v{__version__}')} — Dashboard")
    info(f"  Starting dashboard on http://127.0.0.1:{port}")
    info("  Press Ctrl+C to stop.\n")

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


def build_parser() -> argparse.ArgumentParser:
    """Build the complete argument parser with all subcommands."""
    from forge.cli_parsers import build_parser as _build
    return _build({
        "enrich": cmd_enrich, "import": cmd_import, "export": cmd_export,
        "discover": cmd_discover, "status": cmd_status, "config": cmd_config,
        "dashboard": cmd_dashboard, "mcp_server": cmd_mcp_server,
    })


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
