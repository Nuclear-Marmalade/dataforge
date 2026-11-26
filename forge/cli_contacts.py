"""forge contacts: who to reach at each company, and how, written to a spreadsheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from forge.cli_helpers import die, info, setup_logging, success, warn

_WEBSITE_COLUMNS = (
    "website",
    "url",
    "domain",
    "site",
    "homepage",
    "company website",
    "web",
    "website_url",
)
_MAIL_COLUMNS = ("email_domain", "email domain", "mail_domain", "mail domain")


def read_targets(path: str) -> List[Tuple[str, str]]:
    """(website, email domain) from a CSV with a website column, or a text file with one per line."""
    file = Path(path)
    if not file.is_file():
        die(f"File not found: {path}")
    text = file.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = [h.strip().lower() for h in next(csv.reader([lines[0]]))]
    site_col = next((i for i, h in enumerate(header) if h in _WEBSITE_COLUMNS), None)
    if site_col is None:
        return [
            (line.strip().split(",")[0].strip(), "") for line in lines if not line.startswith("#")
        ]
    mail_col = next((i for i, h in enumerate(header) if h in _MAIL_COLUMNS), None)
    targets = []
    for row in csv.reader(lines[1:]):
        if len(row) > site_col and row[site_col].strip():
            mail = row[mail_col].strip() if mail_col is not None and len(row) > mail_col else ""
            targets.append((row[site_col].strip(), mail))
    return targets


def cmd_contacts(args: argparse.Namespace) -> None:
    """Find people and ways in for each company and write the spreadsheet."""
    from forge.contacts.export import write_companies_csv, write_json, write_people_csv
    from forge.contacts.finder import run_contacts

    verbose = bool(getattr(args, "verbose", False))
    setup_logging(verbose=verbose, quiet=not verbose)
    targets: List[Tuple[str, str]] = [(t, "") for t in (args.targets or [])]
    if args.file:
        targets += read_targets(args.file)
    if not targets:
        die(
            "Give at least one company website.",
            hint="forge contacts zyto.com eng3corp.com --output contacts.csv",
        )

    info(
        f"Looking for people and ways in at {len(targets)} compan{'y' if len(targets) == 1 else 'ies'}..."
    )
    results = run_contacts(
        targets,
        archive=not args.no_archive,
        max_pages=args.max_pages,
        concurrency=args.workers,
    )
    for r in results:
        mark = "-" if r.status != "ok" else "+"
        info(f"  {mark} {r.company or r.website} [{r.status}]: {r.best_way_in()}")

    output = args.output or ("contacts.json" if args.format == "json" else "contacts.csv")
    if args.format == "json":
        write_json(results, output)
    else:
        write_companies_csv(results, output)
    success(f"Wrote {len(results)} compan{'y' if len(results) == 1 else 'ies'} to {output}")
    if args.people_output:
        write_people_csv(results, args.people_output)
        success(f"Wrote {sum(len(r.people) for r in results)} people to {args.people_output}")
    empty = [r for r in results if r.best_way_in() == "nothing public found"]
    if empty:
        warn(
            f"{len(empty)} compan{'y' if len(empty) == 1 else 'ies'} showed nothing public: see the Notes column"
        )


__all__ = ["cmd_contacts", "read_targets"]
