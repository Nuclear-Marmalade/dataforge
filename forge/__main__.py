"""
FORGE — python -m forge entry point.

Delegates to forge.cli.main() which provides the full subcommand interface.

Usage:
    python -m forge enrich --file input.csv
    python -m forge status
    python -m forge --help
"""

from forge.cli import main

if __name__ == "__main__":
    main()
