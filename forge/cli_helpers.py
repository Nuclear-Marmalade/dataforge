"""
FORGE CLI Helpers -- Color output, progress bar, logging, and message functions.

Extracted from cli.py to keep it under 800 lines.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import NoReturn, Optional


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

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
# Progress bar
# ---------------------------------------------------------------------------

class ProgressBar:
    """Simple terminal progress bar using only built-in characters."""

    def __init__(self, total: int, label: str = "", width: int = 40):
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self._current = 0
        self._start_time = time.time()
        self._is_tty = _colors_enabled()

    def update(self, current: int) -> None:
        self._current = min(current, self.total)
        if self._is_tty:
            self._render()

    def _render(self) -> None:
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
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    return logging.getLogger("forge")


# ---------------------------------------------------------------------------
# Error / message helpers
# ---------------------------------------------------------------------------

def die(message: str, hint: str = "", exit_code: int = 1) -> "NoReturn":
    sys.stderr.write(f"{red('Error:')} {message}\n")
    if hint:
        sys.stderr.write(f"{dim('Hint:')} {hint}\n")
    sys.exit(exit_code)


def warn(message: str) -> None:
    sys.stderr.write(f"{yellow('Warning:')} {message}\n")


def info(message: str) -> None:
    print(message)


def success(message: str) -> None:
    print(f"{green('OK')} {message}")
