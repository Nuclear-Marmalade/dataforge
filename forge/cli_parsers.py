"""
FORGE CLI Argument Parsers -- Extracted from cli.py to keep it under 800 lines.

Contains build_parser() and all _add_*_parser() helper functions.
"""

from __future__ import annotations

import argparse
from typing import Any

try:
    from forge import __version__
except ImportError:
    __version__ = "1.0.0"


def _add_global_flags(parser: argparse.ArgumentParser) -> None:
    """Add global flags that are shared across all subcommands."""
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Suppress all output except errors")


def _add_enrich_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'enrich' subcommand parser."""
    p = subparsers.add_parser(
        "enrich", help="Run the enrichment pipeline (CSV or database mode)",
        description="Enrich business data with emails, tech stacks, AI summaries, and more.\n\n"
                    "CSV mode (--file):     Zero-config. Imports CSV, enriches, exports results.\n"
                    "Database mode:         Uses configured database. Requires prior import or discover.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  forge enrich --file leads.csv\n  forge enrich --mode email --workers 50\n",
    )
    p.add_argument("--file", "-f", type=str, default=None, help="Input CSV file")
    p.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    p.add_argument("--mode", "-m", choices=["email", "ai", "both"], default=None, help="Enrichment mode")
    p.add_argument("--adapter", "-a", type=str, default=None, help="AI adapter: ollama, claude, openai")
    p.add_argument("--workers", "-w", type=int, default=None, help="Concurrent web scraper workers")
    p.add_argument("--batch-size", type=int, default=None, help="Records per AI batch (default: 5)")
    p.add_argument("--max", type=int, default=None, help="Maximum records to process")
    p.add_argument("--state", type=str, default=None, help="Filter by US state code")
    p.add_argument("--resume", action="store_true", default=True, help="Resume from last run (default: true)")
    p.add_argument("--no-resume", action="store_true", dest="no_resume", default=False, help="Process all records")
    p.add_argument("--keep-db", action="store_true", default=False, help="Keep temporary database after CSV mode")
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def _add_import_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'import' subcommand parser."""
    p = subparsers.add_parser(
        "import", help="Import a CSV file into the persistent database",
        description="Import business records from a CSV file into FORGE's database.\n\n"
                    "Column mapping is automatic -- FORGE recognizes common column name variations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  forge import --file businesses.csv\n",
    )
    p.add_argument("--file", "-f", type=str, required=True, help="CSV file to import")
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def _add_export_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'export' subcommand parser."""
    p = subparsers.add_parser(
        "export", help="Export enriched data from the database",
        description="Export enriched business data to CSV or JSON.\n\n"
                    "Available filters: all, with_email, with_tech, enriched, with_website, with_npi, with_ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  forge export --output results.csv --filter with_email\n",
    )
    p.add_argument("--output", "-o", type=str, required=True, help="Output file path")
    p.add_argument("--filter", type=str, default=None, choices=["all", "with_email", "with_tech", "enriched", "with_website", "with_npi", "with_ai"], help="Predefined filter name")
    p.add_argument("--where", type=str, default=None, help="(Deprecated) Raw WHERE clause")
    p.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format (default: csv)")
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def _add_discover_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'discover' subcommand parser."""
    p = subparsers.add_parser(
        "discover", help="Discover businesses using Overture Maps data",
        description="Discover businesses using Overture Maps Foundation data.\n\nRequires duckdb: pip install duckdb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--zip", type=str, default=None, help="ZIP code to search")
    p.add_argument("--city", type=str, default=None, help="City name to search")
    p.add_argument("--state", type=str, default=None, help="State code (e.g. FL, CA)")
    p.add_argument("--category", type=str, default=None, help="Business category filter")
    p.add_argument("--enrich", action="store_true", default=False, help="Auto-enrich discovered businesses")
    p.add_argument("--output", "-o", type=str, default=None, help="Export to CSV file")
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def _add_status_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'status' subcommand parser."""
    p = subparsers.add_parser("status", help="Show enrichment statistics and database info",
                              description="Display enrichment progress, data quality metrics, and database statistics.")
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def _add_config_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'config' subcommand parser."""
    p = subparsers.add_parser(
        "config", help="Show or modify FORGE configuration",
        description="View and manage FORGE configuration.\n\nActions:\n  show    Display current configuration\n  set     Set a configuration value",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  forge config show\n  forge config set workers 100\n",
    )
    sub = p.add_subparsers(dest="config_action", title="actions", metavar="<action>")
    show = sub.add_parser("show", help="Display current configuration")
    _add_global_flags(show)
    set_p = sub.add_parser("set", help="Set a configuration value")
    set_p.add_argument("key", help="Configuration key")
    set_p.add_argument("value", help="Configuration value")
    _add_global_flags(set_p)
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def _add_dashboard_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'dashboard' subcommand parser."""
    p = subparsers.add_parser("dashboard", help="Start the FORGE web dashboard",
                              description="Launch a local web dashboard for real-time enrichment monitoring.")
    p.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def _add_mcp_parser(subparsers: Any, cmd_func: Any) -> None:
    """Add the 'mcp-server' subcommand parser."""
    p = subparsers.add_parser("mcp-server", help="Start the FORGE MCP server for AI assistant integration",
                              description="Start a Model Context Protocol server that exposes FORGE tools to AI assistants.")
    p.add_argument("--port", type=int, default=3000, help="Port (default: 3000)")
    _add_global_flags(p)
    p.set_defaults(func=cmd_func)


def build_parser(cmd_handlers: dict) -> argparse.ArgumentParser:
    """Build the complete argument parser with all subcommands.

    Args:
        cmd_handlers: dict mapping command names to handler functions.
            Keys: enrich, import, export, discover, status, config, dashboard, mcp_server
    """
    parser = argparse.ArgumentParser(
        prog="forge",
        description="FORGE -- Free Open-source Runtime for Generalized Enrichment.\nThe open-source alternative to Apollo, ZoomInfo, and Clearbit.\n\nQuick start:\n  forge enrich --file businesses.csv\n  forge status\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Documentation: https://github.com/ghealysr/forge\n",
    )
    parser.add_argument("--version", "-V", action="version", version=f"forge {__version__}")
    sub = parser.add_subparsers(dest="command", title="commands", description="Run 'forge <command> --help' for details.", metavar="<command>")

    _add_enrich_parser(sub, cmd_handlers["enrich"])
    _add_import_parser(sub, cmd_handlers["import"])
    _add_export_parser(sub, cmd_handlers["export"])
    _add_discover_parser(sub, cmd_handlers["discover"])
    _add_status_parser(sub, cmd_handlers["status"])
    _add_config_parser(sub, cmd_handlers["config"])
    _add_dashboard_parser(sub, cmd_handlers["dashboard"])
    _add_mcp_parser(sub, cmd_handlers["mcp_server"])

    return parser
