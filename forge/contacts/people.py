"""Named people and their job titles on a company's own pages.

Only a name next to a job title counts as a person: "Hans Eng, Founder and CEO", a team card with the
name above the title, "Meet our founder: Dr. James Hardt", a German Impressum's "Geschäftsführer: Max
Mustermann", or a schema.org Person in the page's JSON-LD. A capitalised phrase alone is never a person.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

from forge.contacts.page import Page

MARKETING = "marketing"
LEADERSHIP = "leadership"
OTHER = "other"
# Named on the site, but the title belongs to another organisation: an endorsement or a quote.
OUTSIDE = "another company"

# Checked first, so "Chief Marketing Officer" is marketing, not leadership.
_MARKETING_TERMS = (
    r"\bmarketing|\bcmo\b|\bbrand\b|\bgrowth\b|\bcommunications?\b|\bpublic relations|\bpr\b|"
    r"\bmedia relations|\bpress\b|\bpartnerships?\b|\bpartner (?:manager|relations|marketing|program)|"
    r"\bbusiness development|\bsponsorships?|\baffiliates?\b|\binfluencers?\b|\bcreators?\b|\bcommunity\b|"
    r"\bsocial media|\bcontent\b|\bpressesprecher|\bkommunikation|\böffentlichkeitsarbeit|\bmarketingleiter"
)
_LEADERSHIP_TERMS = (
    r"\bceo\b|\bcoo\b|\bcto\b|\bcfo\b|\bfounder|\bco-?founder|\bowner\b|\bpresident\b|"
    r"\bchief [a-z ]{0,30}officer|\bmanaging director|\bgeneral manager|\bmanaging partner|\bprincipal\b|"
    r"\bproprietor|\bchairman|\bchair\b|\bgeschäftsführer(?:in)?|\binhaber(?:in)?|\bvorstand|"
    r"\bgründer(?:in)?|\bvorsitzende[rn]?"
)
# Any job title, including ones that are neither marketing nor leadership.
_OTHER_TITLE_TERMS = (
    r"\bdirector\b|\bhead of|\bmanager\b|\blead\b|\bvp\b|\bvice president|\bofficer\b|\bchief\b|"
    r"\bexecutive\b|\bcoordinator|\bspecialist|\bhead\b|\bleiter(?:in)?\b|\bleitung|\bpartner\b|"
    r"\bengineer\b|\bscientist|\bphysician\b|\bdoctor\b|\baccountant|\badvisor\b|\bassociate\b"
)
# Words that make a phrase a job, not just a subject ("community", "content", "brand" are subjects).
_JOB_TERMS = (
    r"\bdirector\b|\bhead\b|\bmanager\b|\blead\b|\bvp\b|\bvice president|\bofficer\b|\bchief\b|"
    r"\bexecutive\b|\bcoordinator|\bspecialist|\bfounder|\bco-?founder|\bceo\b|\bcoo\b|\bcto\b|"
    r"\bcfo\b|\bcmo\b|\bowner\b|\bpresident\b|\bprincipal\b|\bpartner\b|\bchair|\bstrategist|"
    r"\bassociate\b|\bengineer\b|\bscientist|\bphysician\b|\bdoctor\b|\baccountant|\badvisor\b|"
    r"\bconsultant|\bproprietor|\bgeschäftsführer|\binhaber|\bvorstand|\bgründer|\bleiter|\bleitung|"
    r"\bpressesprecher|\bmarketingleiter|\bmanaging director|\bgeneral manager"
)
_JOB_RE = re.compile(_JOB_TERMS, re.IGNORECASE)
_MARKETING_RE = re.compile(_MARKETING_TERMS, re.IGNORECASE)
_LEADERSHIP_RE = re.compile(_LEADERSHIP_TERMS, re.IGNORECASE)
_TITLE_RE = re.compile(
    "|".join((_MARKETING_TERMS, _LEADERSHIP_TERMS, _OTHER_TITLE_TERMS)), re.IGNORECASE
)
# A section heading that happens to hold a job word ("Our Founders", "Meet the Team").
_HEADING_RE = re.compile(
    r"(?i)\b(our|the|team|meet|leadership|who we are|people|staff|board)\b|"
    r"\b(founders|directors|officers|executives|advisors|managers|partners)\b"
)

_PREFIXES = {"dr", "dr.", "prof", "prof.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "mx", "mx."}
_SUFFIXES = {
    "md", "m.d.", "phd", "ph.d.", "dc", "nd", "jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "cpa",
    "mba", "rn", "dds", "do", "esq", "esq.", "lac", "l.ac.",
}  # fmt: skip
_PARTICLES = {
    "van",
    "von",
    "de",
    "da",
    "del",
    "della",
    "di",
    "der",
    "den",
    "la",
    "le",
    "du",
    "bin",
    "al",
}
# Words that turn a capitalised phrase into a menu item, a heading or a product, never a name.
_NOT_NAME_WORDS = {
    "about", "account", "add", "affiliate", "all", "and", "best", "blog", "book", "buy", "call", "cart",
    "careers", "center", "centre", "click", "clinic", "co", "company", "contact", "copyright", "corp",
    "device", "devices", "email", "faq", "follow", "for", "foundation", "free", "get", "gmbh", "group",
    "health", "helmet", "help", "here", "home", "inc", "institute", "jobs", "lab", "labs", "learn",
    "light", "links", "llc", "login", "ltd", "media", "meet", "menu", "mission", "more", "my", "new",
    "news", "newsletter", "now", "of", "order", "our", "partner", "partners", "policy", "press",
    "privacy", "product", "products", "program", "quick", "read", "red", "research", "reserved",
    "review", "reviews", "rights", "sale", "science", "search", "see", "service", "services", "shipping",
    "shop", "sign", "started", "store", "story", "studio", "subscribe", "support", "team", "technology",
    "terms", "the", "therapy", "to", "university", "us", "view", "welcome", "wellness", "with", "your",
    "programs", "certification", "certified", "screening", "solutions", "systems", "therapies", "sessions",
    "scans", "training", "courses", "events", "returns", "warranty", "financing", "gift", "cards",
    "testimonials", "stories", "results", "benefits", "features", "how", "works", "why", "what",
    "specialist", "specialists", "practitioner", "practitioners", "provider", "providers", "dealers",
    "custom", "virtual", "items", "item", "olympic", "player", "players", "official", "pro", "plus",
    "werden", "sie", "ihr", "ihre", "unser", "unsere", "jetzt", "mehr", "uns", "kontakt", "als", "und",
    "für", "mit", "eine", "einen", "bewerben", "anmelden",
}  # fmt: skip
# Endings of nouns that are not names ("Certification", "Screening Results", "Biology").
_NOUN_ENDINGS = ("tion", "sion", "ment", "ology", "ism", "ics", "ship", "ances", "ences", "ings")
_UPPER = "A-ZÀ-ÖØ-Þ"
_LOWER = "a-zß-öø-ÿ"
_WORD_RE = re.compile(
    rf"^[{_UPPER}][{_LOWER}]*(?:[{_UPPER}][{_LOWER}]+)?(?:['\u2019\-][{_UPPER}]?[{_LOWER}]+)*$"
)
_INITIAL_RE = re.compile(rf"^[{_UPPER}]\.?$")
_INLINE_SEPARATORS = (", ", " \u2013 ", " \u2014 ", " - ", " | ", " \u00b7 ")
_MAX_TITLE_WORDS = 10


@dataclass
class Person:
    name: str
    title: str
    role: str
    source: str
    email: str = ""
    email_basis: str = ""


def classify_role(title: str) -> str:
    """marketing (the people a partnership or sponsorship pitch is for), leadership, or other."""
    if _MARKETING_RE.search(title or ""):
        return MARKETING
    if _LEADERSHIP_RE.search(title or ""):
        return LEADERSHIP
    return OTHER


def looks_like_name(text: str, brand: str = "") -> bool:
    """Two to four capitalised words (initials and particles allowed), with no menu or product words,
    no job title ("Chief Executive Officer") and not the company's own name ("Shiftwave Pro")."""
    text = (text or "").strip().strip(",")
    if not text or len(text) > 40 or any(ch.isdigit() for ch in text):
        return False
    if _JOB_RE.search(text):
        return False
    if brand and any(
        re.sub(r"[^a-z0-9]", "", part.lower()) == brand for part in re.split(r"[\s\-]+", text)
    ):
        return False
    tokens = text.replace(",", " ").split()
    while tokens and tokens[0].lower() in _PREFIXES:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in _SUFFIXES:
        tokens = tokens[:-1]
    if not 2 <= len(tokens) <= 4:
        return False
    if any(t.lower().strip(".") in _NOT_NAME_WORDS for t in tokens):
        return False
    if any(len(t) >= 7 and t.lower().endswith(_NOUN_ENDINGS) for t in tokens):
        return False
    for i, tok in enumerate(tokens):
        if _WORD_RE.match(tok) and len(tok) >= 2:
            continue
        if _INITIAL_RE.match(tok) and i < len(tokens) - 1:
            continue
        if tok.lower() in _PARTICLES and 0 < i < len(tokens) - 1:
            continue
        return False
    return True


def is_title(text: str) -> bool:
    """A short phrase that names a job: a job word ("Director", "Founder"), or a subject alone in at most
    three words ("Marketing", "PR & Communications"). Never a sentence or a quote."""
    text = (text or "").strip()
    if not text or len(text) > 80 or len(text.split()) > _MAX_TITLE_WORDS:
        return False
    if (
        text.endswith(("?", "!"))
        or (text.endswith(".") and len(text.split()) > 3)
        or text.endswith('."')
    ):
        return False
    if any(q in text for q in ('"', "“", "”", "„")):
        return False
    if _JOB_RE.search(text):
        return True
    return len(text.split()) <= 3 and bool(_TITLE_RE.search(text))


# Words a real title may start with; another capitalised word right before "President" or "CEO"
# ("Exos President and Founder", "Google CEO") names a different company.
_TITLE_LEAD_WORDS = {
    "co", "co-founder", "cofounder", "founder", "founding", "vice", "executive", "senior", "sr", "junior",
    "global", "regional", "national", "international", "managing", "general", "medical", "deputy",
    "interim", "acting", "board", "chief", "head", "lead", "associate", "assistant", "creative", "digital",
    "brand", "marketing", "communications", "partnerships", "business", "sales", "product", "operations",
    "customer", "clinical", "scientific", "company", "owner", "president", "ceo", "chairman", "principal",
    "technical", "commercial", "financial", "public", "community", "content", "growth", "media", "press",
}  # fmt: skip
_KNOWN_ACRONYMS = {
    "ceo",
    "coo",
    "cto",
    "cfo",
    "cmo",
    "vp",
    "svp",
    "evp",
    "pr",
    "md",
    "rn",
    "hr",
    "it",
    "qa",
    "cpo",
    "cro",
}
_DEPARTMENT_WORDS = {
    "marketing", "sales", "business", "operations", "partnerships", "communications", "product",
    "engineering", "research", "growth", "brand", "content", "customer", "science", "medicine", "finance",
    "development", "people", "strategy", "technology", "education", "wellness", "the",
}  # fmt: skip
_COMPANY_SUFFIX_RE = re.compile(r"\b(inc|llc|ltd|gmbh|corp|corporation|co\.)\b\.?", re.IGNORECASE)
_OF_ORG_RE = re.compile(r"\b(?:of|at|@)\s+([A-Z][\w&.\-]*)")
_LEADER_WORD_RE = re.compile(
    r"\b(president|ceo|founder|co-?founder|owner|chairman)\b", re.IGNORECASE
)


def names_another_company(title: str, brand: str = "") -> bool:
    """True when the title belongs to some other organisation ("CEO of Kwik Learning", "NFL Players Inc.
    VP", "Exos President and Founder"): a quote or an endorsement, not this company's staff."""
    words = re.findall(r"[\w&.\-]+", title or "")
    if not words:
        return False
    brand = brand.lower()
    mentions_brand = bool(brand) and brand in re.sub(r"[^a-z0-9]", "", (title or "").lower())
    if _COMPANY_SUFFIX_RE.search(title) and not mentions_brand:
        return True
    for match in _OF_ORG_RE.finditer(title):
        org = match.group(1).lower().strip(".")
        if org not in _DEPARTMENT_WORDS and not (brand and org.startswith(brand[:5])):
            return True
    first = words[0]
    if first.isupper() and 2 <= len(first) <= 5 and first.lower() not in _KNOWN_ACRONYMS:
        return not mentions_brand
    if len(words) > 1 and first[0].isupper() and first.lower() not in _TITLE_LEAD_WORDS:
        if _LEADER_WORD_RE.fullmatch(words[1]) and not (
            brand and first.lower().startswith(brand[:5])
        ):
            return True
    return False


def _title_from_heading(text: str) -> Optional[str]:
    """'Meet Our Founder' -> 'Founder'; 'Vertreten durch den Geschäftsführer' -> 'Geschäftsführer'."""
    if not is_title(text):
        return None
    match = _TITLE_RE.search(text)
    return text[match.start() :].strip() if match else None


def _inline(block: str, brand: str = "") -> Optional[Tuple[str, str]]:
    """One block holding both: 'Name, Title' or 'Title: Name'."""
    if ":" in block:
        left, _, right = block.partition(":")
        title = _title_from_heading(left.strip())
        if title and looks_like_name(right.strip(), brand):
            return right.strip(), title
    for sep in _INLINE_SEPARATORS:
        if sep in block:
            left, _, right = block.partition(sep)
            if looks_like_name(left, brand) and is_title(right):
                return left.strip(), right.strip()
    return None


def _jsonld_people(data: Any) -> Iterator[Tuple[str, str]]:
    """(name, title) for each schema.org Person with a job title, and each founder."""
    if isinstance(data, list):
        for item in data:
            yield from _jsonld_people(item)
        return
    if not isinstance(data, dict):
        return
    kind = data.get("@type")
    kinds = kind if isinstance(kind, list) else [kind]
    name = data.get("name")
    if "Person" in kinds and isinstance(name, str) and isinstance(data.get("jobTitle"), str):
        yield name.strip(), data["jobTitle"].strip()
    for key, value in data.items():
        if key.lower() in ("founder", "founders"):
            for founder in value if isinstance(value, list) else [value]:
                if isinstance(founder, str):
                    yield founder.strip(), "Founder"
                elif isinstance(founder, dict) and isinstance(founder.get("name"), str):
                    title = founder.get("jobTitle")
                    yield (
                        founder["name"].strip(),
                        title.strip() if isinstance(title, str) else "Founder",
                    )
        elif isinstance(value, (dict, list)):
            yield from _jsonld_people(value)


def name_key(name: str) -> str:
    """The same person however the name is written: no titles, accents, case or middle initials."""
    first, last = name_parts(name)
    return f"{first} {last}".strip()


def name_parts(name: str) -> Tuple[str, str]:
    """(first, last) in plain lowercase letters: 'Dr. José V. Núñez, PhD' -> ('jose', 'nunez')."""
    folded = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    tokens = [t for t in re.split(r"[\s,]+", folded) if t]
    while tokens and tokens[0].lower() in _PREFIXES:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in _SUFFIXES:
        tokens = tokens[:-1]
    words = [re.sub(r"[^a-z]", "", t.lower()) for t in tokens if len(t.strip(".")) > 1]
    words = [w for w in words if w and w not in _PARTICLES]
    if not words:
        return "", ""
    return words[0], (words[-1] if len(words) > 1 else "")


def extract_people(page: Page, brand: str = "") -> List[Person]:
    """Every named person with a job title on the page, once each, in the order they appear.

    `brand` is the company's name as letters only ("shiftwave"): a phrase holding it is a product or
    a program, not a person, and a title holding it is this company's title.
    """
    found: List[Tuple[str, str]] = []
    for data in page.jsonld:
        found.extend(_jsonld_people(data))
    blocks = page.blocks
    i = 0
    while i < len(blocks):
        block = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else ""
        pair = _inline(block, brand)
        if pair:
            found.append(pair)
            i += 1
        elif looks_like_name(block, brand) and is_title(nxt):
            found.append((block, nxt))
            i += 2
        elif is_title(block) and not _HEADING_RE.search(block) and looks_like_name(nxt, brand):
            found.append((nxt, block))
            i += 2
        else:
            i += 1
    people: List[Person] = []
    seen = set()
    for name, title in found:
        key = name_key(name)
        if not key or key in seen or not looks_like_name(name, brand):
            continue
        seen.add(key)
        role = OUTSIDE if names_another_company(title, brand) else classify_role(title)
        people.append(Person(name=name, title=title, role=role, source=page.url))
    return people


__all__ = [
    "LEADERSHIP",
    "MARKETING",
    "OTHER",
    "OUTSIDE",
    "Person",
    "classify_role",
    "extract_people",
    "is_title",
    "looks_like_name",
    "name_key",
    "name_parts",
    "names_another_company",
]
