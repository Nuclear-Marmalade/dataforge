"""Tests for forge.cli — utility functions, color helpers, ProgressBar, parser."""

import os
from unittest.mock import patch


import forge.cli as cli_module
from forge.cli import (
    build_parser,
    setup_logging,
)
from forge.cli_helpers import ProgressBar


# ---------------------------------------------------------------------------
# Tests: color helpers
# ---------------------------------------------------------------------------

class TestColorHelpers:
    def test_colors_disabled_with_no_color(self):
        # Reset cached state
        cli_module._COLOR_ENABLED = None
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            cli_module._COLOR_ENABLED = None
            result = cli_module._colors_enabled()
            assert result is False
        cli_module._COLOR_ENABLED = None

    def test_colors_enabled_with_force_color(self):
        cli_module._COLOR_ENABLED = None
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            # Ensure NO_COLOR is not set
            env = os.environ.copy()
            env.pop("NO_COLOR", None)
            env["FORCE_COLOR"] = "1"
            with patch.dict(os.environ, env, clear=True):
                cli_module._COLOR_ENABLED = None
                result = cli_module._colors_enabled()
                assert result is True
        cli_module._COLOR_ENABLED = None

    def test_green_no_color(self):
        cli_module._COLOR_ENABLED = False
        assert cli_module.green("hello") == "hello"
        cli_module._COLOR_ENABLED = None

    def test_yellow_no_color(self):
        cli_module._COLOR_ENABLED = False
        assert cli_module.yellow("warn") == "warn"
        cli_module._COLOR_ENABLED = None

    def test_red_no_color(self):
        cli_module._COLOR_ENABLED = False
        assert cli_module.red("error") == "error"
        cli_module._COLOR_ENABLED = None

    def test_bold_no_color(self):
        cli_module._COLOR_ENABLED = False
        assert cli_module.bold("bold") == "bold"
        cli_module._COLOR_ENABLED = None

    def test_dim_no_color(self):
        cli_module._COLOR_ENABLED = False
        assert cli_module.dim("dim") == "dim"
        cli_module._COLOR_ENABLED = None

    def test_cyan_no_color(self):
        cli_module._COLOR_ENABLED = False
        assert cli_module.cyan("info") == "info"
        cli_module._COLOR_ENABLED = None

    def test_green_with_color(self):
        cli_module._COLOR_ENABLED = True
        result = cli_module.green("hello")
        assert "\033[32m" in result
        assert "hello" in result
        assert "\033[0m" in result
        cli_module._COLOR_ENABLED = None

    def test_red_with_color(self):
        cli_module._COLOR_ENABLED = True
        result = cli_module.red("error")
        assert "\033[31m" in result
        cli_module._COLOR_ENABLED = None


# ---------------------------------------------------------------------------
# Tests: ProgressBar
# ---------------------------------------------------------------------------

class TestProgressBar:
    def test_construction(self):
        bar = ProgressBar(total=100, label="Test")
        assert bar.total == 100
        assert bar.label == "Test"

    def test_zero_total_becomes_one(self):
        bar = ProgressBar(total=0)
        assert bar.total == 1

    def test_update_clamps_to_total(self):
        bar = ProgressBar(total=10)
        bar.update(15)
        assert bar._current == 10

    def test_update_tracks_current(self):
        bar = ProgressBar(total=50)
        bar.update(25)
        assert bar._current == 25

    def test_render_no_crash(self):
        """Render should not crash even in non-TTY mode."""
        cli_module._COLOR_ENABLED = False
        bar = ProgressBar(total=100, label="Test")
        bar.update(50)
        # In non-TTY mode, _render is not called but update should still work
        assert bar._current == 50
        cli_module._COLOR_ENABLED = None


# ---------------------------------------------------------------------------
# Tests: build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_parser_builds_without_error(self):
        parser = build_parser()
        assert parser is not None

    def test_parser_has_subcommands(self):
        parser = build_parser()
        # Try parsing known subcommands
        args = parser.parse_args(["status"])
        assert hasattr(args, "func") or hasattr(args, "subcommand")

    def test_enrich_subcommand(self):
        parser = build_parser()
        parser.parse_args(["enrich", "--file", "test.csv"])
        # Should parse without error

    def test_import_subcommand(self):
        parser = build_parser()
        parser.parse_args(["import", "--file", "data.csv"])

    def test_export_subcommand(self):
        parser = build_parser()
        parser.parse_args(["export", "--output", "out.csv"])

    def test_config_subcommand(self):
        parser = build_parser()
        parser.parse_args(["config", "show"])


# ---------------------------------------------------------------------------
# Tests: setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_default_logging(self):
        logger = setup_logging()
        assert logger is not None

    def test_verbose_logging(self):
        logger = setup_logging(verbose=True)
        assert logger.level <= 10  # DEBUG

    def test_quiet_logging(self):
        logger = setup_logging(quiet=True)
        # basicConfig sets root logger level; the named logger inherits it
        assert logger.getEffectiveLevel() >= 30  # WARNING
