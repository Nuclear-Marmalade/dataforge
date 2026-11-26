"""Read one HTML page into the parts the contacts finder needs.

Standard library only: the text in reading order, split where the page starts a new block (a heading,
a paragraph, a list item, a line break), the links with absolute addresses, the JSON-LD data, and
whether the page has a form a person can write a message in.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

# Elements whose start or end begins a new run of text.
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
    "nav", "ol", "p", "section", "table", "td", "th", "tr", "ul", "button", "label", "option",
}  # fmt: skip
# Elements whose text is never read (JSON-LD scripts are collected separately).
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe"}

logger = logging.getLogger("forge.contacts.page")

_SPACE_RE = re.compile(r"\s+")
_MAX_BLOCK = 400


@dataclass
class Link:
    url: str
    text: str


@dataclass
class Page:
    url: str
    html: str = ""
    title: str = ""
    site_name: str = ""
    blocks: List[str] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    jsonld: List[Any] = field(default_factory=list)
    has_contact_form: bool = False


class _Reader(HTMLParser):
    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.url = url
        self.page = Page(url=url)
        self._skip = 0
        self._buf: List[str] = []
        self._in_title = False
        self._title: List[str] = []
        self._jsonld: Optional[List[str]] = None
        self._link: Optional[Dict[str, Any]] = None
        self._form: Optional[Dict[str, bool]] = None

    # text runs ---------------------------------------------------------------------------------
    def _flush(self) -> None:
        text = _SPACE_RE.sub(" ", "".join(self._buf)).strip()
        self._buf = []
        if text:
            self.page.blocks.append(text[:_MAX_BLOCK])

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "script" and "ld+json" in a.get("type", "").lower():
            self._jsonld = []
            return
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            prop = (a.get("property") or a.get("name") or "").lower()
            if prop == "og:site_name" and a.get("content"):
                self.page.site_name = a["content"].strip()
        elif tag == "a":
            href = a.get("href", "").strip()
            if href and not href.startswith(("#", "javascript:")):
                self._link = {"url": urljoin(self.url, href), "text": []}
        elif tag == "form":
            # a blog's comment form has an email and a message box too; it is not a way to reach anyone
            self._form = {
                "email": False,
                "message": False,
                "comment": "comment" in a.get("action", "").lower(),
            }
        elif tag in ("input", "textarea") and self._form is not None:
            kind = a.get("type", "").lower()
            name = (
                a.get("name", "") + " " + a.get("id", "") + " " + a.get("placeholder", "")
            ).lower()
            if kind == "email" or "email" in name or "e-mail" in name:
                self._form["email"] = True
            if tag == "textarea" or "message" in name or "nachricht" in name:
                self._form["message"] = True
            if "comment" in name:
                self._form["comment"] = True
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._jsonld is not None:
            raw = "".join(self._jsonld).strip()
            self._jsonld = None
            try:
                self.page.jsonld.append(json.loads(raw))
            except (ValueError, TypeError):
                pass  # broken JSON-LD on the page: nothing to read
            return
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._link is not None:
            text = _SPACE_RE.sub(" ", "".join(self._link["text"])).strip()
            self.page.links.append(Link(url=self._link["url"], text=text))
            self._link = None
        elif tag == "form" and self._form is not None:
            if self._form["email"] and self._form["message"] and not self._form["comment"]:
                self.page.has_contact_form = True
            self._form = None
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._jsonld is not None:
            self._jsonld.append(data)
            return
        if self._skip:
            return
        if self._in_title:
            self._title.append(data)
            return
        self._buf.append(data)
        if self._link is not None:
            self._link["text"].append(data)

    def close(self) -> None:
        super().close()
        self._flush()
        self.page.title = _SPACE_RE.sub(" ", "".join(self._title)).strip()


def parse_page(html: str, url: str) -> Page:
    """The page's text blocks, links, JSON-LD and contact form, for one fetched document."""
    reader = _Reader(url)
    try:
        reader.feed(html or "")
        reader.close()
    except Exception as exc:
        logger.debug("Page %s could not be read to the end: %s", url, exc)
        reader._flush()
    page = reader.page
    page.html = html or ""
    return page


__all__ = ["Link", "Page", "parse_page"]
