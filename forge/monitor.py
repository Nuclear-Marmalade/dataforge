"""
FORGE Monitor — Hourly health check that auto-restarts crashed processes.

Checks:
  1. All services are running (platform-aware)
  2. DB enrichment progress since last check
  3. Log files for errors
  4. Restarts any crashed process

Usage:
    python -m forge.monitor
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from forge.db import ForgeDB
from forge.config import ForgeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FORGE-MONITOR] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forge.monitor")

# Configurable service label prefix (e.g. "com.yourcompany" or "com.forge")
SERVICE_PREFIX = os.environ.get("FORGE_SERVICE_PREFIX", "com.forge")

# Services to monitor
SERVICES = {
    f"{SERVICE_PREFIX}.forge-scraper-1": {
        "name": "Scraper 1",
        "log": "/tmp/forge_scraper_1.log",
        "plist": os.path.expanduser(f"~/Library/LaunchAgents/{SERVICE_PREFIX}.forge-scraper-1.plist"),
    },
    f"{SERVICE_PREFIX}.forge-scraper-2": {
        "name": "Scraper 2",
        "log": "/tmp/forge_scraper_2.log",
        "plist": os.path.expanduser(f"~/Library/LaunchAgents/{SERVICE_PREFIX}.forge-scraper-2.plist"),
    },
    f"{SERVICE_PREFIX}.forge-scraper-3": {
        "name": "Scraper 3",
        "log": "/tmp/forge_scraper_3.log",
        "plist": os.path.expanduser(f"~/Library/LaunchAgents/{SERVICE_PREFIX}.forge-scraper-3.plist"),
    },
    f"{SERVICE_PREFIX}.forge-scraper-4": {
        "name": "Scraper 4",
        "log": "/tmp/forge_scraper_4.log",
        "plist": os.path.expanduser(f"~/Library/LaunchAgents/{SERVICE_PREFIX}.forge-scraper-4.plist"),
    },
    f"{SERVICE_PREFIX}.fcc-import": {
        "name": "FCC Import",
        "log": "/tmp/fcc_import.log",
        "plist": os.path.expanduser(f"~/Library/LaunchAgents/{SERVICE_PREFIX}.fcc-import.plist"),
    },
    f"{SERVICE_PREFIX}.npi-import": {
        "name": "NPI Import",
        "log": "/tmp/npi_import.log",
        "plist": os.path.expanduser(f"~/Library/LaunchAgents/{SERVICE_PREFIX}.npi-import.plist"),
    },
    f"{SERVICE_PREFIX}.smtp-verifier": {
        "name": "SMTP Verifier",
        "log": "/tmp/smtp_verifier.log",
        "plist": os.path.expanduser(f"~/Library/LaunchAgents/{SERVICE_PREFIX}.smtp-verifier.plist"),
    },
}

STATUS_FILE = "/tmp/forge_monitor_status.json"


def check_service_running(label: str) -> dict:
    """Check if a service is running. Platform-aware."""
    if platform.system() == "Darwin" and shutil.which("launchctl"):
        # macOS launchd
        try:
            result = subprocess.run(
                ["launchctl", "list"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 3 and parts[2] == label:
                    pid = parts[0]
                    exit_code = parts[1]
                    return {
                        "running": pid != "-",
                        "pid": pid if pid != "-" else None,
                        "exit_code": exit_code,
                    }
        except Exception as e:
            logger.error("Failed to check service %s: %s", label, e)
        return {"running": False, "pid": None, "exit_code": "unknown"}
    else:
        # Cross-platform: check for running Python processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", label],
                capture_output=True, text=True, timeout=5,
            )
            return {
                "running": result.returncode == 0,
                "pid": result.stdout.strip().split('\n')[0] if result.returncode == 0 else None,
            }
        except Exception:
            return {
                "running": False,
                "pid": None,
                "note": "Process monitoring not available on this platform",
            }


def restart_service(label: str, plist: str) -> bool:
    """Restart a service. Platform-aware with fallback message."""
    if platform.system() == "Darwin" and shutil.which("launchctl"):
        # macOS launchd
        try:
            subprocess.run(["launchctl", "unload", plist],
                           capture_output=True, timeout=10)
            time.sleep(1)
            subprocess.run(["launchctl", "load", plist],
                           capture_output=True, timeout=10)
            time.sleep(3)
            status = check_service_running(label)
            if status["running"]:
                logger.info("Successfully restarted %s (PID %s)", label, status["pid"])
                return True
            else:
                logger.error("Failed to restart %s", label)
                return False
        except Exception as e:
            logger.error("Error restarting %s: %s", label, e)
            return False
    else:
        logger.warning(
            "Cannot auto-restart %s on this platform (%s). "
            "Please restart manually or configure a systemd unit.",
            label, platform.system(),
        )
        return False


def get_db_stats() -> dict:
    """Query current enrichment stats from the database via ForgeDB."""
    try:
        config = ForgeConfig.load()
        db = ForgeDB.from_config(config.to_db_config())
        db.ensure_schema()
        stats = db.get_stats()
        db.close()
        return stats
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return {"error": str(e)}


def load_previous_status() -> dict:
    """Load status from previous run."""
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_status(status: dict) -> None:
    """Save current status for next run comparison."""
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2, default=str)


def tail_log(logfile: str, lines: int = 5) -> str:
    """Get last N lines of a log file."""
    try:
        result = subprocess.run(
            ["tail", f"-{lines}", logfile],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "(no log)"


def run_monitor():
    """Main monitoring loop — runs once per invocation."""
    logger.info("=" * 60)
    logger.info("FORGE MONITOR CHECK — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    previous = load_previous_status()
    current = {
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "db_stats": {},
        "actions_taken": [],
    }

    # Check each service
    for label, info in SERVICES.items():
        status = check_service_running(label)
        current["services"][label] = status

        if status["running"]:
            logger.info("[OK] %s — PID %s", info["name"], status["pid"])
        else:
            logger.warning("[DOWN] %s — exit code %s", info["name"], status.get("exit_code", "unknown"))

            # Check log for error details
            log_tail = tail_log(info["log"])
            if log_tail:
                logger.info("  Last log: %s", log_tail[-200:])

            # Auto-restart
            logger.info("  Restarting %s...", info["name"])
            success = restart_service(label, info["plist"])
            action = f"Restarted {info['name']}: {'SUCCESS' if success else 'FAILED'}"
            current["actions_taken"].append(action)
            logger.info("  %s", action)

    # Check DB progress
    db_stats = get_db_stats()
    current["db_stats"] = db_stats

    if "error" not in db_stats:
        logger.info("")
        logger.info("DB ENRICHMENT STATE:")
        logger.info("  Total records:  %s", db_stats.get("total_records", "?"))
        logger.info("  With email:     %s", db_stats.get("with_email", "?"))
        logger.info("  With tech_stack: %s", db_stats.get("with_tech_stack", "?"))
        logger.info("  With NPI:       %s", db_stats.get("with_npi", "?"))
        logger.info("  Enriched today: %s", db_stats.get("enriched_today", "?"))
        logger.info("  Last enriched:  %s", db_stats.get("last_enriched", "?"))

        # Compare with previous run
        prev_email = int(previous.get("db_stats", {}).get("with_email", "0"))
        curr_email = int(db_stats.get("with_email", "0"))
        delta = curr_email - prev_email
        if prev_email > 0:
            logger.info("  Email delta since last check: +%d", delta)
            if delta == 0:
                logger.warning("  WARNING: No new emails since last check!")
    else:
        logger.error("DB CONNECTION FAILED: %s", db_stats["error"])

    # Save status
    save_status(current)

    logger.info("")
    if current["actions_taken"]:
        logger.info("ACTIONS TAKEN: %s", "; ".join(current["actions_taken"]))
    else:
        logger.info("No corrective actions needed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_monitor()
