"""
FORGE Configuration -- Centralized config with multi-source loading.

Reads configuration from (highest priority first):
  1. CLI arguments (--db-host, --workers, etc.)
  2. Environment variables (FORGE_DB_HOST, FORGE_WORKERS, etc.)
  3. .env file in the current directory
  4. ~/.forge/config.toml (persistent user config)
  5. Compiled defaults

No external dependencies -- uses stdlib only for .env and TOML parsing.

Usage:
    from forge.config import ForgeConfig
    config = ForgeConfig.load()
    adapter = config.get_adapter()
    db_config = config.to_db_config()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("forge.config")

# ── Env var prefix ──────────────────────────────────────────────────────
_ENV_PREFIX = "FORGE_"

# ── Config file locations ───────────────────────────────────────────────
_DOTENV_PATH = Path(".env")
_TOML_PATH = Path.home() / ".forge" / "config.toml"


# ── Lightweight parsers (stdlib only) ───────────────────────────────────

def _parse_dotenv(path: Path) -> Dict[str, str]:
    """
    Parse a .env file into a dict.

    Handles:
      - KEY=VALUE
      - KEY="VALUE" and KEY='VALUE' (strips outer quotes)
      - # comments and blank lines
      - No interpolation or multiline
    """
    result: Dict[str, str] = {}
    if not path.is_file():
        return result

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                result[key] = value
    except OSError as e:
        logger.debug("Could not read %s: %s", path, e)

    return result


def _parse_toml(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse a minimal TOML file with [sections] and key = value pairs.

    Only supports flat sections (no nested tables, no arrays, no inline
    tables).  This covers the config.toml use case without pulling in a
    TOML library.

    Returns:
        {"section": {"key": "value", ...}, ...}
        Keys without a section go under "default".
    """
    result: Dict[str, Dict[str, str]] = {"default": {}}
    if not path.is_file():
        return result

    current_section = "default"

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Section header
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip()
                    if current_section not in result:
                        result[current_section] = {}
                    continue
                # Key = value
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                    # Unescape TOML escape sequences (quotes first, then backslashes)
                    value = value.replace('\\"', '"').replace('\\\\', '\\')
                # Strip inline comments
                for comment_char in (" #", "\t#"):
                    idx = value.find(comment_char)
                    if idx >= 0:
                        value = value[:idx].strip()
                result[current_section][key] = value
    except OSError as e:
        logger.debug("Could not read %s: %s", path, e)

    return result


def _flatten_toml(parsed: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Flatten sectioned TOML into a flat key map.

    [database]
    host = "..."     ->  db_host = "..."

    [ai]
    adapter = "..."  ->  adapter = "..."

    Section mapping:
      database -> db_*
      ai       -> (as-is, keys already match dataclass fields)
      enrichment -> (as-is)
      smtp     -> smtp_*
      sam      -> sam_gov_*
      dashboard -> dashboard_*
    """
    flat: Dict[str, str] = {}

    section_prefix = {
        "database": "db_",
        "smtp": "smtp_",
        "sam": "sam_gov_",
        "dashboard": "dashboard_",
    }

    for section, kv in parsed.items():
        prefix = section_prefix.get(section, "")
        for key, value in kv.items():
            flat_key = f"{prefix}{key}" if prefix and not key.startswith(prefix) else key
            flat[flat_key] = value

    return flat


def _mask_secret(value: str) -> str:
    """Mask a secret, showing at most the first 4 characters."""
    if not value:
        return "(not set)"
    if len(value) <= 4:
        return "****"
    return value[:4] + "****"


def _safe_toml_value(v: str) -> str:
    """Escape a value for safe TOML output."""
    v = str(v).replace('\n', '').replace('\r', '')
    v = v.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{v}"'


# ── ForgeConfig dataclass ──────────────────────────────────────────────

@dataclass
class ForgeConfig:
    """
    Centralized FORGE configuration.

    All settings have sensible defaults.  Override via environment
    variables (FORGE_DB_HOST, etc.), .env file, or ~/.forge/config.toml.
    """

    # Database
    db_backend: str = "sqlite"
    db_path: str = "forge.db"
    db_host: str = ""
    db_port: int = 5432
    db_user: str = ""
    db_password: str = ""
    db_name: str = "forge"

    # AI
    adapter: str = "auto"
    anthropic_api_key: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:26b"
    claude_model: str = "claude-sonnet-4-6"

    # Enrichment
    workers: int = 50
    batch_size: int = 5
    rate_limit: float = 100.0

    # SMTP
    smtp_from: str = "verify@example.com"
    smtp_ehlo: str = "localhost"

    # SAM.gov
    sam_gov_api_key: str = ""

    # Dashboard
    dashboard_port: int = 8765

    # ── Secrets that should be masked in output ─────────────────────
    _SECRET_FIELDS = frozenset({
        "db_password",
        "anthropic_api_key",
        "sam_gov_api_key",
    })

    # ── Loading ─────────────────────────────────────────────────────

    @classmethod
    def load(cls, cli_args: Optional[Dict[str, Any]] = None) -> "ForgeConfig":
        """
        Load config from multiple sources (highest priority first):

          1. cli_args dict (if provided)
          2. Environment variables (FORGE_DB_HOST, FORGE_WORKERS, ...)
          3. .env file in CWD
          4. ~/.forge/config.toml
          5. Dataclass defaults

        Returns a fully-resolved ForgeConfig instance.
        """
        # Start with defaults
        config = cls()

        # Layer 5 → 2: build a merged flat dict (lowest priority first)
        layers: List[Dict[str, str]] = []

        # Layer 4: TOML config file
        toml_data = _parse_toml(_TOML_PATH)
        layers.append(_flatten_toml(toml_data))

        # Layer 3: .env file
        layers.append(_parse_dotenv(_DOTENV_PATH))

        # Layer 2: environment variables
        env_layer: Dict[str, str] = {}
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            env_key = _ENV_PREFIX + f.name.upper()
            val = os.environ.get(env_key)
            if val is not None:
                env_layer[f.name] = val
        # Also check ANTHROPIC_API_KEY directly (common convention)
        if "ANTHROPIC_API_KEY" in os.environ and "anthropic_api_key" not in env_layer:
            env_layer["anthropic_api_key"] = os.environ["ANTHROPIC_API_KEY"]
        layers.append(env_layer)

        # Layer 1: CLI args
        if cli_args:
            cli_layer = {k: str(v) for k, v in cli_args.items() if v is not None}
            layers.append(cli_layer)

        # Merge layers onto config (later layers = higher priority)
        field_types = {f.name: f.type for f in fields(cls) if not f.name.startswith("_")}
        for layer in layers:
            for key, value in layer.items():
                if key not in field_types:
                    continue
                ftype = field_types[key]
                try:
                    if ftype == "int":
                        setattr(config, key, int(value))
                    elif ftype == "float":
                        setattr(config, key, float(value))
                    elif ftype == "bool":
                        setattr(config, key, value.lower() in ("true", "1", "yes"))
                    else:
                        setattr(config, key, value)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid config value %s=%r: %s", key, value, e)

        logger.info(
            "Config loaded — backend=%s, adapter=%s, workers=%d",
            config.db_backend,
            config.adapter,
            config.workers,
        )

        return config

    # ── Persistence helpers ─────────────────────────────────────────

    @property
    def config_path(self) -> str:
        """Return the path to the user config file."""
        return str(_TOML_PATH)

    def set(self, key: str, value: str) -> None:
        """
        Set a config value by attribute name.

        Converts the value to the correct type based on the field definition.
        Call save() afterwards to persist.
        """
        valid_keys = {f.name for f in fields(self.__class__) if not f.name.startswith("_")}
        if key not in valid_keys:
            raise ValueError(f"Unknown config key: {key}")
        field_type = type(getattr(self, key))
        if field_type is int:
            setattr(self, key, int(value))
        elif field_type is float:
            setattr(self, key, float(value))
        elif field_type is bool:
            setattr(self, key, value.lower() in ('true', '1', 'yes'))
        else:
            setattr(self, key, value)

    def save(self) -> None:
        """Save current config to ~/.forge/config.toml in a single write."""
        config_dir = os.path.expanduser("~/.forge")
        os.makedirs(config_dir, exist_ok=True)
        path = os.path.join(config_dir, "config.toml")

        d = self.as_dict()

        section_map = {
            'db_backend': 'database', 'db_path': 'database', 'db_host': 'database',
            'db_port': 'database', 'db_user': 'database', 'db_password': 'database', 'db_name': 'database',
            'adapter': 'ai', 'anthropic_api_key': 'ai', 'ollama_url': 'ai',
            'ollama_model': 'ai', 'claude_model': 'ai',
            'workers': 'enrichment', 'batch_size': 'enrichment', 'rate_limit': 'enrichment',
            'smtp_from': 'smtp', 'smtp_ehlo': 'smtp',
            'sam_gov_api_key': 'samgov',
            'dashboard_port': 'dashboard',
        }

        sections: dict[str, list[tuple[str, str]]] = {}
        for key, value in d.items():
            if key.startswith('_'):
                continue
            v = str(value)
            if not v or v == '0' or v == '0.0':
                continue
            section = section_map.get(key, 'general')
            if section not in sections:
                sections[section] = []
            sections[section].append((key, v))

        with open(path, 'w') as f:
            for section, pairs in sections.items():
                f.write(f"[{section}]\n")
                for k, v in pairs:
                    f.write(f"{k} = {_safe_toml_value(v)}\n")
                f.write("\n")

    def as_dict(self) -> dict:
        """Return config as a flat dictionary of all settings."""
        result = {}
        for f in fields(self.__class__):
            if f.name.startswith("_"):
                continue
            result[f.name] = getattr(self, f.name)
        return result

    # ── Output helpers ──────────────────────────────────────────────

    def show(self) -> str:
        """
        Return a human-readable config dump with secrets masked.

        Suitable for `forge config show` CLI command.
        """
        lines = ["FORGE Configuration", "=" * 40]
        for f in fields(self.__class__):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if f.name in self._SECRET_FIELDS:
                display = _mask_secret(str(value))
            else:
                display = str(value)
            lines.append(f"  {f.name:25s} = {display}")
        return "\n".join(lines)

    def to_db_config(self) -> Dict[str, Any]:
        """
        Return config suitable for ForgeDB.from_config().

        For sqlite backend, returns {"db_path": ...}.
        For postgresql, returns {"db_host": ..., "db_port": ..., ...}.
        """
        if self.db_backend == "sqlite":
            return {
                "db_path": self.db_path,
            }
        else:
            return {
                "db_host": self.db_host,
                "db_port": self.db_port,
                "db_user": self.db_user,
                "db_password": self.db_password,
                "db_name": self.db_name,
            }

    def get_adapter(self):
        """
        Return the appropriate AI adapter based on configuration.

        Auto-detection order:
          1. If anthropic_api_key is set -> ClaudeAdapter
          2. If Ollama is healthy at ollama_url -> OllamaAdapter
          3. None (email-only mode)

        If adapter is explicitly set to "claude" or "ollama", skip
        auto-detection and use that adapter directly.
        """
        # Explicit adapter selection
        if self.adapter == "claude":
            from forge.adapters.claude import ClaudeAdapter
            return ClaudeAdapter(
                api_key=self.anthropic_api_key,
                default_model=self.claude_model,
            )

        if self.adapter == "ollama":
            from forge.adapters.ollama import OllamaAdapter
            return OllamaAdapter(
                base_url=self.ollama_url,
                default_model=self.ollama_model,
            )

        if self.adapter == "none":
            logger.info("Adapter explicitly set to 'none' — email-only mode")
            return None

        # Auto-detection
        if self.anthropic_api_key:
            try:
                from forge.adapters.claude import ClaudeAdapter
                adapter = ClaudeAdapter(
                    api_key=self.anthropic_api_key,
                    default_model=self.claude_model,
                )
                logger.info("Auto-detected Claude adapter (API key present)")
                return adapter
            except Exception as e:
                logger.warning("Claude adapter init failed: %s", e)

        # Try Ollama
        try:
            from forge.adapters.ollama import OllamaAdapter
            adapter = OllamaAdapter(
                base_url=self.ollama_url,
                default_model=self.ollama_model,
            )
            if adapter.is_healthy():
                logger.info("Auto-detected Ollama adapter at %s", self.ollama_url)
                return adapter
            else:
                adapter.close()
        except Exception as e:
            logger.debug("Ollama not available: %s", e)

        logger.info("No AI adapter available — running in email-only mode")
        return None


# ── CLI subcommands ─────────────────────────────────────────────────────

def cli_config_show() -> None:
    """Handle `forge config show` command."""
    config = ForgeConfig.load()
    print(config.show())


def cli_config_set(key: str, value: str) -> None:
    """
    Handle `forge config set KEY VALUE` command.

    Writes the key-value pair to ~/.forge/config.toml under the
    appropriate section.
    """
    # Map config keys to TOML sections
    section_map = {
        "db_backend": "database",
        "db_path": "database",
        "db_host": "database",
        "db_port": "database",
        "db_user": "database",
        "db_password": "database",
        "db_name": "database",
        "adapter": "ai",
        "anthropic_api_key": "ai",
        "ollama_url": "ai",
        "ollama_model": "ai",
        "claude_model": "ai",
        "workers": "enrichment",
        "batch_size": "enrichment",
        "rate_limit": "enrichment",
        "smtp_from": "smtp",
        "smtp_ehlo": "smtp",
        "sam_gov_api_key": "sam",
        "dashboard_port": "dashboard",
    }

    # Validate key
    valid_keys = {f.name for f in fields(ForgeConfig) if not f.name.startswith("_")}
    if key not in valid_keys:
        print(f"ERROR: Unknown config key '{key}'")
        print(f"Valid keys: {', '.join(sorted(valid_keys))}")
        sys.exit(1)

    section = section_map.get(key, "default")
    # Strip db_ prefix for the TOML key in the [database] section (etc.)
    section_prefix = {
        "database": "db_",
        "smtp": "smtp_",
        "sam": "sam_gov_",
        "dashboard": "dashboard_",
    }
    prefix = section_prefix.get(section, "")
    toml_key = key[len(prefix):] if prefix and key.startswith(prefix) else key

    # Read existing file
    toml_path = _TOML_PATH
    toml_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _parse_toml(toml_path)

    # Update
    if section not in existing:
        existing[section] = {}
    existing[section][toml_key] = value

    # Write back
    with open(toml_path, "w") as f:
        for sec_name, sec_data in existing.items():
            if sec_name == "default" and not sec_data:
                continue
            if sec_name != "default":
                f.write(f"\n[{sec_name}]\n")
            for k, v in sec_data.items():
                f.write(f"{k} = {_safe_toml_value(v)}\n")

    print(f"Set {key} = {_mask_secret(value) if key in ForgeConfig._SECRET_FIELDS else value}")
    print(f"Written to {toml_path}")
