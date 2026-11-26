"""Archived copies of a company's contact, team and press pages (the Internet Archive's Wayback Machine).

Used when a site turns automated visits away or shows nothing on its live pages. The CDX index lists
what was captured; only the newest capture of each useful page is read, a few per company, and every
result keeps the snapshot date so a reader knows how old it is.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from forge.tools.web_scraper import read_capped

logger = logging.getLogger("forge.contacts.archive")

WAYBACK = "https://web.archive.org"
CAPTURES_SINCE = "2023"
RETRY_PAUSE = 3.0

# Lower comes first. A page counts by whole words in the last part of its address, so "about-us" and
# "find-a-practitioner" count and a blog post that merely mentions people ("...-people-who-win-...") does not.
PAGE_KINDS: List[Tuple[int, List[str]]] = [
    (0, ["impressum", "imprint", "legal-notice", "mentions-legales"]),
    (1, ["about", "team", "leadership", "our-story", "founder", "founders", "who-we-are", "people", "staff",
         "management", "ueber-uns", "uber-uns"]),
    (2, ["contact", "kontakt", "reach-us", "get-in-touch"]),
    (3, ["press", "media", "newsroom", "presse", "press-room"]),
    (4, ["partner", "partners", "partnership", "partnerships", "partnerprogramm", "affiliate", "affiliates",
         "ambassador", "ambassadors", "wholesale", "influencer", "influencers", "creator", "creators",
         "collab", "collabs", "sponsor", "sponsors", "sponsorship", "sponsorships", "practitioner",
         "practitioners", "reseller", "resellers", "dealer", "dealers", "distributor", "distributors"]),
]  # fmt: skip
_MAX_SLUG_WORDS = 5


def page_kind(path: str) -> Optional[int]:
    """The priority of a page for finding people and inboxes, from the last part of its address (or a
    link's text written with dashes), or None when it is not useful."""
    segments = [seg for seg in (path or "").lower().split("/") if seg]
    if not segments:
        return None
    last = re.sub(r"\.(html?|php|aspx?)$", "", segments[-1])
    words = [w for w in re.split(r"[-_.\s]+", last) if w]
    if not words or len(words) > _MAX_SLUG_WORDS:
        return None
    for rank, phrases in PAGE_KINDS:
        for phrase in phrases:
            target = phrase.split("-")
            n = len(target)
            if any(words[i : i + n] == target for i in range(len(words) - n + 1)):
                return rank
    return None


def _path_key(original: str) -> str:
    parsed = urlparse(original)
    return (parsed.path or "/").rstrip("/").lower() or "/"


def pick_snapshots(rows: List[List[str]], limit: int) -> List[Tuple[str, str]]:
    """(timestamp, original) of the newest capture of each useful page, most useful first."""
    newest: Dict[str, Tuple[str, str]] = {}
    for row in rows[1:] if rows and rows[0] and rows[0][0] == "timestamp" else rows:
        if len(row) < 2:
            continue
        timestamp, original = row[0], row[1]
        key = _path_key(original)
        if page_kind(key) is None:
            continue
        if key not in newest or timestamp > newest[key][0]:
            newest[key] = (timestamp, original)
    ranked = sorted(newest.items(), key=lambda kv: (page_kind(kv[0]), -int(kv[1][0][:14] or 0)))
    return [snap for _key, snap in ranked[:limit]]


def snapshot_date(timestamp: str) -> str:
    return (
        f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}" if len(timestamp) >= 8 else timestamp
    )


async def archived_pages(
    session: aiohttp.ClientSession,
    host: str,
    *,
    base: str = WAYBACK,
    limit: int = 4,
) -> Tuple[List[Tuple[str, str, str]], Optional[str]]:
    """([(html, original url, snapshot date)], problem). problem is None when the archive answered."""
    query = {
        "url": f"{host}/*",
        "output": "json",
        "fl": "timestamp,original",
        "filter": ["statuscode:200", "mimetype:text/html"],
        "from": CAPTURES_SINCE,
        "collapse": "urlkey",
        "limit": "5000",
    }
    params: List[Tuple[str, str]] = []
    for key, value in query.items():
        for item in value if isinstance(value, list) else [value]:
            params.append((key, item))
    rows: Any = None
    for attempt in range(2):  # the index is often briefly busy (429/503): one retry after a pause
        try:
            async with session.get(f"{base}/cdx/search/cdx", params=params) as resp:
                if resp.status in (429, 502, 503, 504) and attempt == 0:
                    await asyncio.sleep(RETRY_PAUSE)
                    continue
                if resp.status == 429:
                    return [], "the archive asked us to slow down"
                if resp.status != 200:
                    return [], f"the archive answered {resp.status}"
                rows = await resp.json(content_type=None)
                break
        except Exception as exc:
            logger.info("Archive index for %s not reached: %s", host, exc)
            if attempt == 0:
                await asyncio.sleep(RETRY_PAUSE)
                continue
            return [], "the archive did not answer"
    if not isinstance(rows, list):
        return [], None
    pages: List[Tuple[str, str, str]] = []
    for timestamp, original in pick_snapshots(rows, limit):
        try:
            async with session.get(f"{base}/web/{timestamp}id_/{original}") as resp:
                if resp.status == 429:
                    return pages, "the archive asked us to slow down"
                if resp.status != 200:
                    continue
                body = await read_capped(resp.content, 2 * 1024 * 1024)
        except Exception as exc:
            logger.info("Archived copy %s of %s not read: %s", timestamp, original, exc)
            continue
        pages.append((body.decode("utf-8", errors="replace"), original, snapshot_date(timestamp)))
    return pages, None


__all__ = ["WAYBACK", "archived_pages", "page_kind", "pick_snapshots", "snapshot_date"]
