"""Tests for CLI commands."""
import subprocess
import sys


class TestCLIHelp:
    def test_forge_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "forge", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "enrich" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_forge_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "forge", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "1.1.0" in result.stdout


class TestCLIImports:
    def test_cli_module_imports(self):
        from forge.cli import main
        assert callable(main)

    def test_main_module_imports(self):
        from forge.__main__ import main
        assert callable(main)

    def test_version_constant(self):
        from forge import __version__
        assert __version__ == "1.1.0"
