"""Who to reach at a company, and how, from its own public pages.

For each company: read the home page, pick the pages most likely to name people or give a way in
(Impressum, about and team, contact, press, partner and affiliate programs), read them politely
(robots.txt is obeyed, one request at a time per site), and fall back to archived copies when the site
turns automated visits away or shows nothing. Out come the named people with titles, the inboxes on the
company's own domain, phone, contact form, partner program, social pages, and the email format when a
named address shows it. Every item keeps the page it came from. Nothing is guessed silently: a built
address says it was built from the format, and an empty answer says nothing public was found.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp

from forge.contacts.archive import WAYBACK, archived_pages, page_kind
from forge.contacts.page import Page, parse_page
from forge.contacts.people import (
    LEADERSHIP,
    MARKETING,
    Person,
    extract_people,
    name_key,
    name_parts,
)
from forge.tools.web_scraper import extract_emails, keep_business_emails, rank_emails, read_capped

logger = logging.getLogger("forge.contacts.finder")

USER_AGENT = (
    "Mozilla/5.0 (compatible; ForgeBot/1.0; +https://github.com/Nuclear-Marmalade/dataforge)"
)
ROBOTS_AGENT = "ForgeBot"

MAX_PAGES = 8
ARCHIVE_PAGES = 4
HOST_DELAY = 0.5
REQUEST_TIMEOUT = 15.0
COMPANY_TIMEOUT = 180.0
_MAX_BODY = 3 * 1024 * 1024

_COUNTRY_SECOND_LEVELS = {"ac", "co", "com", "edu", "gov", "net", "org"}
_FALLBACK_PATHS = [
    "/about",
    "/about-us",
    "/team",
    "/contact",
    "/contact-us",
    "/press",
    "/impressum",
]
_SKIP_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".zip",
    ".mp4",
    ".mp3",
    ".css",
    ".js",
)
_CONTACT_KIND = 2
_PARTNER_KIND = 4
_BLOCKED_STATUSES = {401, 403, 406, 429, 451, 503}

# Inboxes a partnership or sponsorship pitch should go to first.
_PITCH_INBOXES = re.compile(
    r"^(partner|partners|partnerships?|marketing|press|media|pr|sponsor|sponsorships?|affiliates?|"
    r"collab|collabs|collaborations?|influencers?|creators?|brand|brands|wholesale|business|bd|presse)$"
)
_GENERAL_INBOXES = re.compile(
    r"^(hello|hi|info|contact|team|office|mail|inquiries|enquiries|kontakt|sales)$"
)

_SOCIAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("linkedin", re.compile(r"^https?://([a-z]{2,3}\.)?(www\.)?linkedin\.com/(company|school|showcase)/[^/?#]+", re.I)),
    ("instagram", re.compile(r"^https?://(www\.)?instagram\.com/(?!p/|reel/|share|explore)[A-Za-z0-9_.]+/?$", re.I)),
    ("facebook", re.compile(r"^https?://(www\.|m\.)?facebook\.com/(?!sharer|share|dialog|plugins|tr\b)[^/?#]+/?$", re.I)),
    ("x", re.compile(r"^https?://(www\.)?(twitter|x)\.com/(?!intent|share|home)[A-Za-z0-9_]+/?$", re.I)),
    ("youtube", re.compile(r"^https?://(www\.)?youtube\.com/(channel/|c/|user/|@)[^/?#]+", re.I)),
    ("tiktok", re.compile(r"^https?://(www\.)?tiktok\.com/@[^/?#]+", re.I)),
]  # fmt: skip

_PARTNER_PLATFORMS: List[Tuple[str, re.Pattern]] = [
    ("GoAffPro", re.compile(r"goaffpro", re.I)),
    ("Refersion", re.compile(r"refersion", re.I)),
    ("Impact", re.compile(r"impactradius|impact\.com/campaign|impact-affiliate", re.I)),
    ("ShareASale", re.compile(r"shareasale", re.I)),
    ("Awin", re.compile(r"awin1\.com|dwin1\.com", re.I)),
    ("UpPromote", re.compile(r"uppromote", re.I)),
    ("Social Snowball", re.compile(r"socialsnowball", re.I)),
    ("GRIN", re.compile(r"grin\.co\b", re.I)),
    ("Tapfiliate", re.compile(r"tapfiliate", re.I)),
    ("LeadDyno", re.compile(r"leaddyno", re.I)),
    ("PartnerStack", re.compile(r"partnerstack", re.I)),
    ("Rewardful", re.compile(r"rewardful", re.I)),
    ("FirstPromoter", re.compile(r"firstpromoter", re.I)),
]  # fmt: skip

_ORG_TYPES = (
    "Organization",
    "Corporation",
    "LocalBusiness",
    "Store",
    "Brand",
    "MedicalBusiness",
    "OnlineStore",
)

# How a named address is written, most specific first.
_FORMATS = ["first.last", "first_last", "firstlast", "f.last", "flast", "firstl", "last", "first"]

Target = Union[str, Tuple[str, str]]


# ---------------------------------------------------------------------------------------------
# Company input
# ---------------------------------------------------------------------------------------------


def _root_domain(host: str) -> str:
    host = host.lower().strip(".")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _COUNTRY_SECOND_LEVELS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def normalize_company(value: str) -> Tuple[str, str, str]:
    """(site base URL, host, root domain) from a domain, a home page or any page on the site."""
    value = (value or "").strip()
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    netloc = parsed.netloc.split("@")[-1].lower()
    return f"{parsed.scheme}://{netloc}", host, _root_domain(host)


def brand_label(root: str) -> str:
    """The company's name as the domain spells it: zyto.com -> zyto, shop.airnergy.de -> airnergy."""
    return re.sub(r"[^a-z0-9]", "", (root or "").split(".")[0].lower())


def _same_site(url: str, host: str, root: str) -> bool:
    link_host = (urlparse(url).hostname or "").lower()
    return bool(link_host) and (link_host in (host, root) or link_host.endswith("." + root))


def _clean_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or ""
    return f"{parsed.scheme}://{parsed.netloc}{path}" + (f"?{parsed.query}" if parsed.query else "")


def url_kind(url: str, root: str) -> Optional[int]:
    """What kind of page an address is, from its path, or from its subdomain when the path says
    nothing: Shiftwave's partner program is partners.shiftwave.co/, whose path is just "/"."""
    parsed = urlparse(url)
    kind = page_kind(parsed.path)
    if kind is not None:
        return kind
    host = (parsed.hostname or "").lower()
    if not root or not host.endswith("." + root):
        return None
    for label in host[: -len(root) - 1].split("."):
        kind = page_kind(label)
        if kind is not None:
            return kind
    return None


def pick_pages(home: Page, host: str, root: str, limit: int = MAX_PAGES) -> List[str]:
    """The pages on the same site most likely to name people or give a way in, best first.

    A link counts by the last part of its address ("/about/contact-us" is a contact page,
    "/about/blog/some-post" is not) or, for a short address, by its text. A site that does not link
    an about or contact page still often has one, so those addresses are tried as well.
    """
    ranked: List[Tuple[int, int, int, str]] = []
    seen = set()
    home_clean = _clean_url(home.url).rstrip("/")
    for order, link in enumerate(home.links):
        url = link.url
        if not url.startswith(("http://", "https://")) or not _same_site(url, host, root):
            continue
        path = urlparse(url).path.lower()
        if path.endswith(_SKIP_EXTENSIONS):
            continue
        clean = _clean_url(url)
        if clean in seen or clean.rstrip("/") == home_clean:
            continue
        segments = [seg for seg in path.split("/") if seg]
        kind = url_kind(url, root)
        if kind is None and len(segments) <= 2:
            kind = page_kind(link.text.lower().replace(" ", "-"))
        if kind is None:
            continue
        seen.add(clean)
        ranked.append((kind, len(segments), order, clean))
    ranked.sort()
    picked = [url for _kind, _depth, _order, url in ranked]
    parsed = urlparse(home.url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    kinds = {kind for kind, _depth, _order, _url in ranked}
    if not ranked:
        guesses = list(_FALLBACK_PATHS)
    else:
        guesses = []
        if 1 not in kinds:
            guesses += ["/about", "/about-us", "/team"]
        if 2 not in kinds:
            guesses += ["/contact", "/contact-us"]
        if 0 not in kinds and root.endswith((".de", ".at", ".ch")):
            guesses += ["/impressum"]
    for path in guesses:
        if base + path not in picked:
            picked.append(base + path)
    return picked[:limit]


# ---------------------------------------------------------------------------------------------
# Email format
# ---------------------------------------------------------------------------------------------


def _local_for(first: str, last: str, fmt: str) -> str:
    if not first:
        return ""
    if fmt == "first":
        return first
    if not last:
        return ""
    return {
        "first.last": f"{first}.{last}",
        "first_last": f"{first}_{last}",
        "firstlast": f"{first}{last}",
        "f.last": f"{first[0]}.{last}",
        "flast": f"{first[0]}{last}",
        "firstl": f"{first}{last[0]}",
        "last": last,
    }.get(fmt, "")


def _on_domain(email: str, domain: str) -> bool:
    email_domain = email.rpartition("@")[2].lower()
    return email_domain == domain or email_domain.endswith("." + domain)


def match_person(person_name: str, email: str) -> Optional[str]:
    """The format the address uses if it is this person's, else None."""
    first, last = name_parts(person_name)
    local = email.partition("@")[0].lower()
    for fmt in _FORMATS:
        candidate = _local_for(first, last, fmt)
        if candidate and candidate == local:
            return fmt
    return None


def infer_email_format(
    people: Sequence[Person], emails: Iterable[str], domain: str
) -> Optional[Tuple[str, str]]:
    """(format, the address that shows it) when an address on the domain belongs to a named person."""
    for email in emails:
        if not _on_domain(email, domain):
            continue
        for person in people:
            fmt = match_person(person.name, email)
            if fmt:
                return fmt, email
    return None


def email_for(name: str, fmt: str, domain: str) -> Optional[str]:
    first, last = name_parts(name)
    local = _local_for(first, last, fmt)
    return f"{local}@{domain}" if local else None


# ---------------------------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------------------------


@dataclass
class CompanyContacts:
    input: str
    website: str
    host: str
    root: str
    email_domain: str = ""
    company: str = ""
    status: str = ""
    people: List[Person] = field(default_factory=list)
    emails: Dict[str, str] = field(default_factory=dict)  # address -> the page it was on
    phones: List[str] = field(default_factory=list)
    contact_form: str = ""
    partner_pages: List[str] = field(default_factory=list)
    partner_platforms: List[str] = field(default_factory=list)
    socials: Dict[str, str] = field(default_factory=dict)
    email_format: str = ""
    email_format_from: str = ""
    notes: List[str] = field(default_factory=list)
    pages_read: List[str] = field(default_factory=list)

    @property
    def mail_domain(self) -> str:
        return self.email_domain or self.root

    @property
    def marketing_contact(self) -> Optional[Person]:
        marketing = [p for p in self.people if p.role == MARKETING]
        senior = re.compile(r"(?i)chief|head|director|\bvp\b|vice president|lead|leiter|leitung")
        marketing.sort(key=lambda p: (not senior.search(p.title), not p.email))
        return marketing[0] if marketing else None

    @property
    def leaders(self) -> List[Person]:
        return [p for p in self.people if p.role == LEADERSHIP]

    @property
    def inboxes(self) -> List[str]:
        """Addresses not tied to a named person, the best one for a partnership pitch first."""
        personal = {p.email for p in self.people if p.email}
        ranked = rank_emails([e for e in self.emails if e not in personal], self.mail_domain)

        def kind(email: str) -> int:
            local = email.partition("@")[0]
            if _PITCH_INBOXES.match(local):
                return 0
            if _GENERAL_INBOXES.match(local):
                return 1
            return 2

        return sorted(ranked, key=kind)

    def best_way_in(self) -> str:
        """One line: who to write to and how, or that nothing public was found."""
        who = self.marketing_contact or (self.leaders[0] if self.leaders else None)
        inbox = self.inboxes[0] if self.inboxes else ""
        how = ""
        if who and who.email:
            how = who.email + (
                " (built from the email format)" if who.email_basis.startswith("guessed") else ""
            )
        elif inbox:
            how = f"via {inbox}"
        elif self.contact_form:
            how = f"via the contact form {self.contact_form}"
        elif self.phones:
            how = f"via {self.phones[0]}"
        if who:
            return f"{who.name} ({who.title}) {how}".strip()
        if how:
            return how[4:] if how.startswith("via ") else how
        return "nothing public found"


# ---------------------------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------------------------


class _Site:
    """One company's site: robots.txt, a pause between requests, and the pages read."""

    def __init__(self, session: aiohttp.ClientSession, base: str, host_delay: float) -> None:
        self.session = session
        self.base = base
        self.host_delay = host_delay
        self._last = 0.0
        self._robots: Optional[RobotFileParser] = None
        self.robots_problem = ""

    async def _pause(self) -> None:
        wait = self.host_delay - (time.monotonic() - self._last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = time.monotonic()

    async def load_robots(self) -> None:
        parser = RobotFileParser()
        try:
            await self._pause()
            async with self.session.get(self.base + "/robots.txt", allow_redirects=True) as resp:
                if resp.status == 200:
                    text = (await read_capped(resp.content, 512 * 1024)).decode(
                        "utf-8", errors="replace"
                    )
                    parser.parse(text.splitlines())
                elif 400 <= resp.status < 500:
                    parser.parse([])  # no robots.txt: everything is allowed
                else:
                    parser.parse(["User-agent: *", "Disallow: /"])
                    self.robots_problem = f"robots.txt answered {resp.status}, so nothing was read"
        except Exception as exc:
            logger.debug("robots.txt for %s not read: %s", self.base, exc)
            parser.parse([])
        self._robots = parser

    def allowed(self, url: str) -> bool:
        return self._robots is None or self._robots.can_fetch(ROBOTS_AGENT, url)

    async def get(self, url: str) -> Tuple[Optional[int], str, str]:
        """(status, html, final url). status None means the site could not be reached."""
        await self._pause()
        try:
            async with self.session.get(url, allow_redirects=True, max_redirects=5) as resp:
                kind = resp.headers.get("Content-Type", "").lower()
                if resp.status != 200 or ("html" not in kind and "text" not in kind and kind):
                    return resp.status, "", str(resp.url)
                body = await read_capped(resp.content, _MAX_BODY)
                return resp.status, body.decode("utf-8", errors="replace"), str(resp.url)
        except Exception as exc:
            logger.debug("Fetch %s failed: %s", url, exc)
            return None, "", url


def _phones(page: Page) -> List[str]:
    found = []
    for link in page.links:
        if link.url.lower().startswith("tel:"):
            found.append(link.url[4:])
    for data in page.jsonld:
        for value in _walk_key(data, "telephone"):
            found.append(value)
    out = []
    for raw in found:
        digits = re.sub(r"[^\d+]", "", raw)
        digits = ("+" if digits.startswith("+") else "") + digits.replace("+", "")
        if len(digits.lstrip("+")) >= 7 and digits not in out:
            out.append(digits)
    return out


def _walk_key(data: object, key: str) -> Iterable[str]:
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key and isinstance(v, str):
                yield v
            else:
                yield from _walk_key(v, key)
    elif isinstance(data, list):
        for item in data:
            yield from _walk_key(item, key)


def _jsonld_items(data: object) -> Iterable[dict]:
    if isinstance(data, list):
        for item in data:
            yield from _jsonld_items(item)
    elif isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if isinstance(graph, list):
            yield from _jsonld_items(graph)


def _company_name(page: Page, root: str) -> str:
    """The organisation's name: from JSON-LD, the site name or a part of the page title, whichever first
    matches the domain's name ("ZYTO" on zyto.com, "Intelligent Bioenergetics" on intellbio.com), else the
    domain itself. A slogan is never used."""
    label = brand_label(root)
    stem = label[:5]
    candidates: List[str] = []
    for data in page.jsonld:
        for item in _jsonld_items(data):
            kind = item.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            name = item.get("name")
            if isinstance(name, str) and any(isinstance(k, str) and k in _ORG_TYPES for k in kinds):
                candidates.append(name.strip())
    if page.site_name:
        candidates.append(page.site_name)
    if page.title:
        candidates.extend(
            part.strip() for part in re.split(" [|\u2013\u2014\\-\u00b7:] ", page.title)
        )
    for name in candidates:
        letters = re.sub(r"[^a-z0-9]", "", name.lower())
        if len(stem) >= 3 and stem in letters and len(name) <= 60:
            return name
    return root


def _absorb(result: CompanyContacts, page: Page, source: str, live: bool) -> None:
    """Add one page's people, own-domain addresses, phones, socials and partner signals."""
    known = {name_key(p.name) for p in result.people}
    for person in extract_people(page, brand=brand_label(result.mail_domain)):
        person.source = source
        if name_key(person.name) not in known:
            known.add(name_key(person.name))
            result.people.append(person)
    for email in keep_business_emails(extract_emails(page.html), result.mail_domain):
        result.emails.setdefault(email, source)
    for phone in _phones(page):
        if phone not in result.phones:
            result.phones.append(phone)
    page_kind_here = url_kind(page.url, result.root)
    # a partner program's sign-up form is listed with the partner program, never as the contact form
    if live and page.has_contact_form and page_kind_here != _PARTNER_KIND:
        # the form on the contact page wins over one met on the way (a newsletter, a quote form)
        if not result.contact_form or (
            page_kind_here == _CONTACT_KIND
            and url_kind(result.contact_form, result.root) != _CONTACT_KIND
        ):
            result.contact_form = page.url
    for link in page.links:
        for network, pattern in _SOCIAL_PATTERNS:
            if network not in result.socials and pattern.match(link.url):
                result.socials[network] = link.url.rstrip("/")
        if live and link.url.startswith("http") and _same_site(link.url, result.host, result.root):
            if (
                url_kind(link.url.lower(), result.root) == _PARTNER_KIND
                and _clean_url(link.url) not in result.partner_pages
            ):
                result.partner_pages.append(_clean_url(link.url))
    if live and page_kind_here == _PARTNER_KIND:
        if _clean_url(page.url) not in result.partner_pages:
            result.partner_pages.append(_clean_url(page.url))
    for name, pattern in _PARTNER_PLATFORMS:
        if name not in result.partner_platforms and pattern.search(page.html):
            result.partner_platforms.append(name)


def _attach_emails(result: CompanyContacts) -> None:
    """A named address goes to its person; the format it shows builds the others, labelled as built."""
    for person in result.people:
        for email in result.emails:
            if _on_domain(email, result.mail_domain) and match_person(person.name, email):
                person.email, person.email_basis = email, "on their site"
                break
    shown = infer_email_format(result.people, result.emails, result.mail_domain)
    if not shown:
        return
    result.email_format, result.email_format_from = shown
    for person in result.people:
        if not person.email:
            built = email_for(person.name, result.email_format, result.mail_domain)
            if built:
                person.email = built
                person.email_basis = f"guessed from the format of {result.email_format_from}"


async def _find_one(
    session: aiohttp.ClientSession,
    archive_session: aiohttp.ClientSession,
    result: CompanyContacts,
    *,
    archive: bool,
    archive_base: str,
    max_pages: int,
    host_delay: float,
) -> None:
    site = _Site(session, result.website, host_delay)
    await site.load_robots()
    home_url = result.website + "/"
    if site.robots_problem:
        result.status = "robots_unreadable"
        result.notes.append(site.robots_problem)
        return
    if not site.allowed(home_url):
        result.status = "robots_disallowed"
        result.notes.append(
            "robots.txt asks crawlers to stay out, so nothing was read (not even archived copies)"
        )
        return
    status, html, final_url = await site.get(home_url)
    if status == 200:
        result.status = "ok"
        home = parse_page(html, final_url)
        result.company = _company_name(home, result.mail_domain)
        result.pages_read.append(final_url)
        _absorb(result, home, final_url, live=True)
        skipped = 0
        read = {_clean_url(final_url)}
        for url in pick_pages(home, result.host, result.root, limit=max_pages):
            if not site.allowed(url):
                skipped += 1
                continue
            page_status, page_html, page_url = await site.get(url)
            if page_status != 200 or not page_html or _clean_url(page_url) in read:
                continue
            page = parse_page(page_html, page_url)
            if page.title == home.title and abs(len(page_html) - len(html)) <= 0.01 * len(html):
                continue  # a missing page that answers with the home page
            read.add(_clean_url(page_url))
            result.pages_read.append(page_url)
            _absorb(result, page, page_url, live=True)
        if skipped:
            result.notes.append(
                f"{skipped} page(s) skipped because robots.txt asks crawlers not to read them"
            )
    elif status is None:
        result.status = "unreachable"
        result.notes.append("the site could not be reached")
    elif status in _BLOCKED_STATUSES:
        result.status = "blocked"
    else:
        result.status = f"http_{status}"
        result.notes.append(f"the home page answered {status}")

    need_archive = result.status != "ok" or (not result.people and not result.emails)
    if archive and need_archive:
        pages, problem = await archived_pages(
            archive_session, result.host, base=archive_base, limit=ARCHIVE_PAGES
        )
        for page_html, original, date in pages:
            page = parse_page(page_html, original)
            _absorb(result, page, f"{original} (archived copy from {date})", live=False)
        why = (
            "the site turns automated visits away"
            if result.status == "blocked"
            else "the live site showed nothing"
        )
        if pages:
            copies = "1 archived copy" if len(pages) == 1 else f"{len(pages)} archived copies"
            result.notes.append(f"{why}; read {copies}")
        elif problem:
            result.notes.append(f"{why}; {problem}")
        else:
            result.notes.append(f"{why}; the archive has no useful pages since 2023")
    elif result.status == "blocked":
        result.notes.append("the site turns automated visits away")

    _attach_emails(result)
    if not result.company:
        result.company = result.root


def _target(value: Target) -> Tuple[str, str]:
    if isinstance(value, tuple):
        return value[0], (value[1] or "").lower().strip()
    return value, ""


async def find_contacts(
    targets: Sequence[Target],
    *,
    archive: bool = True,
    archive_base: Optional[str] = None,
    max_pages: int = MAX_PAGES,
    concurrency: int = 4,
    host_delay: float = HOST_DELAY,
    request_timeout: float = REQUEST_TIMEOUT,
    company_timeout: float = COMPANY_TIMEOUT,
) -> List[CompanyContacts]:
    """One CompanyContacts per target, in the same order.

    A target is a website (domain, home page or any page on the site), or (website, email domain)
    when the company's mail is on another domain (a shop.example.de that writes from example.com).
    """
    results = []
    for value in targets:
        website, email_domain = _target(value)
        base, host, root = normalize_company(website)
        results.append(
            CompanyContacts(
                input=website, website=base, host=host, root=root, email_domain=email_domain
            )
        )
    timeout = aiohttp.ClientTimeout(total=request_timeout, connect=min(10.0, request_timeout))
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    limit = asyncio.Semaphore(max(1, concurrency))
    async with (
        aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
        aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=max(request_timeout, 30.0)),
            headers={"User-Agent": USER_AGENT},
        ) as archive_session,
    ):

        async def run(result: CompanyContacts) -> None:
            async with limit:
                try:
                    await asyncio.wait_for(
                        _find_one(
                            session,
                            archive_session,
                            result,
                            archive=archive,
                            archive_base=archive_base or WAYBACK,
                            max_pages=max_pages,
                            host_delay=host_delay,
                        ),
                        timeout=company_timeout,
                    )
                except asyncio.TimeoutError:
                    result.notes.append(
                        f"stopped after {company_timeout:.0f} s; results are partial"
                    )
                    _attach_emails(result)
                    result.status = result.status or "timeout"
                except Exception as exc:
                    logger.warning("Contacts for %s failed: %s", result.website, exc)
                    result.status = "error"
                    result.notes.append(f"failed: {str(exc)[:200]}")

        await asyncio.gather(*(run(r) for r in results))
    return results


def run_contacts(targets: Sequence[Target], **kwargs: object) -> List[CompanyContacts]:
    """Synchronous wrapper for scripts and the CLI."""
    return asyncio.run(find_contacts(targets, **kwargs))  # type: ignore[arg-type]


__all__ = [
    "USER_AGENT",
    "CompanyContacts",
    "email_for",
    "find_contacts",
    "infer_email_format",
    "match_person",
    "normalize_company",
    "pick_pages",
    "run_contacts",
]
