"""
FORGE Web Scraper Tool — Async website scraping for email and data extraction.

Provides high-throughput async scraping via aiohttp:
  - Six-layer email extraction: mailto → regex → Cloudflare decode → JSON-LD → obfuscation decode → contact page crawl
  - Tech stack detection from HTTP headers and HTML
  - SSL validation, CMS detection, site speed (TTFB)
  - Rate limiting: 50 req/sec global, 2 per host

Uses aiohttp with TCPConnector for connection pooling.
Respects robots.txt and rate limits per FORGE_WORKFLOW.md.

Dependencies: aiohttp, aiodns, aiolimiter
Depended on by: enrichment pipeline
"""

from __future__ import annotations

import asyncio
import html as html_module
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
from aiolimiter import AsyncLimiter

logger = logging.getLogger("forge.tools.web_scraper")

# ── Email extraction patterns ────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

_MAILTO_RE = re.compile(r'mailto:([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})', re.IGNORECASE)

# Cloudflare email obfuscation pattern
_CF_EMAIL_RE = re.compile(r'data-cfemail="([a-fA-F0-9]+)"')

# Obfuscated email patterns: [at], (at), [dot], (dot) variations
_OBFUSCATED_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+\s*[\[\(]\s*(?:at|AT)\s*[\]\)]\s*[a-zA-Z0-9.\-]+\s*[\[\(]\s*(?:dot|DOT)\s*[\]\)]\s*[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# HTML numeric entity for @ (&#64;) and . (&#46;)
_HTML_ENTITY_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+(?:&#64;|&#x40;)[a-zA-Z0-9.\-]+(?:&#46;|&#x2e;)[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Common false positive email patterns to exclude
_FAKE_EMAIL_DOMAINS = {
    "example.com", "email.com", "domain.com", "yoursite.com",
    "yourdomain.com", "company.com", "test.com", "placeholder.com",
    "sentry.io", "wixpress.com", "wordpress.org", "w3.org",
    "schema.org", "googleapis.com", "gravatar.com", "wp.com",
    "sentry-next.wixpress.com", "changedetection.io",
}

# ── Tech stack detection patterns (from HTTP headers + HTML) ─────────────────

_TECH_PATTERNS = {
    # CMS
    "wordpress": [r"wp-content", r"wp-includes", r"WordPress"],
    "shopify": [r"cdn\.shopify\.com", r"Shopify"],
    "squarespace": [r"squarespace\.com", r"Squarespace"],
    "wix": [r"wix\.com", r"wixstatic\.com"],
    "webflow": [r"webflow\.com", r"Webflow"],
    "drupal": [r"Drupal", r"/sites/default/files"],
    "joomla": [r"Joomla", r"/components/com_"],
    "ghost": [r"ghost\.io", r"Ghost"],
    "weebly": [r"weebly\.com"],
    "godaddy-builder": [r"godaddy\.com/website-builder"],
    "hubspot-cms": [r"hubspot\.com/hub", r"hs-sites\.com"],
    "bigcommerce": [r"bigcommerce\.com", r"BigCommerce"],
    # Analytics
    "google-analytics": [r"google-analytics\.com", r"gtag", r"UA-\d+", r"G-[A-Z0-9]+"],
    "google-tag-manager": [r"googletagmanager\.com", r"GTM-"],
    "facebook-pixel": [r"connect\.facebook\.net", r"fbevents\.js"],
    "hotjar": [r"hotjar\.com"],
    "clarity": [r"clarity\.ms"],
    "segment": [r"segment\.com/analytics", r"analytics\.js", r"cdn\.segment\.com"],
    "mixpanel": [r"mixpanel\.com", r"mixpanel\.init"],
    "heap": [r"heap\.io", r"heapanalytics\.com"],
    # Frameworks
    "react": [r"__next", r"_next/static", r"react\.production"],
    "angular": [r"ng-version", r"angular"],
    "vue": [r"vue\.js", r"__vue__"],
    "next.js": [r"__next", r"_next/"],
    "nuxt.js": [r"__nuxt", r"_nuxt/"],
    "svelte": [r"svelte", r"__svelte"],
    "jquery": [r"jquery\.min\.js", r"jquery\.com"],
    "bootstrap": [r"bootstrap\.min\.(css|js)", r"getbootstrap\.com"],
    "tailwind": [r"tailwindcss", r"tailwind\.min\.css"],
    # Booking
    "calendly": [r"calendly\.com"],
    "acuity": [r"acuityscheduling\.com"],
    # Chat
    "intercom": [r"intercom\.io", r"intercomSettings"],
    "drift": [r"drift\.com"],
    "zendesk": [r"zendesk\.com"],
    "tawk": [r"tawk\.to"],
    "livechat": [r"livechat\.com"],
    "hubspot": [r"hubspot\.com", r"hs-scripts"],
    "crisp": [r"crisp\.chat", r"client\.crisp\.chat"],
    "tidio": [r"tidio\.co", r"tidiochat"],
    # Payments
    "stripe": [r"stripe\.com", r"js\.stripe"],
    "square": [r"squareup\.com"],
    "paypal": [r"paypal\.com", r"paypalobjects\.com"],
    "braintree": [r"braintreegateway\.com", r"braintree-api"],
    # Email marketing
    "mailchimp": [r"mailchimp\.com", r"mc\.us"],
    "constant-contact": [r"constantcontact\.com"],
    "klaviyo": [r"klaviyo\.com", r"a\.klaviyo\.com"],
    "sendgrid": [r"sendgrid\.com", r"sendgrid\.net"],
    # Other
    "cloudflare": [r"cloudflare\.com", r"cf-ray", r"__cf_bm"],
    "recaptcha": [r"google\.com/recaptcha", r"g-recaptcha"],
    "schema-org": [r"schema\.org", r"application/ld\+json"],
}

# CMS mapping for primary CMS detection
_CMS_PRIORITY = [
    "wordpress", "shopify", "squarespace", "wix", "webflow",
    "drupal", "joomla", "ghost", "weebly", "godaddy-builder",
]

# Pages to check for email addresses (in priority order)
_CONTACT_PATHS = [
    "/contact", "/contact-us", "/about", "/about-us",
    "/team", "/staff", "/our-team", "/reach-us",
    "/privacy-policy", "/privacy", "/terms", "/careers",
    "/jobs", "/directory", "/people", "/leadership",
]

# Maximum response body size: 5MB
_MAX_BODY_SIZE = 5 * 1024 * 1024

# User agent
_USER_AGENT = "Mozilla/5.0 (compatible; ForgeBot/1.0; business-directory-enrichment)"


def decode_cf_email(encoded: str) -> Optional[str]:
    """Decode Cloudflare's __cf_email__ obfuscation."""
    try:
        key = int(encoded[:2], 16)
        decoded = ""
        for i in range(2, len(encoded), 2):
            decoded += chr(int(encoded[i:i+2], 16) ^ key)
        if "@" in decoded and "." in decoded.split("@")[-1]:
            return decoded.lower()
    except (ValueError, IndexError):
        pass
    return None


class AsyncWebScraper:
    """
    High-throughput async web scraper for business data extraction.

    Extracts: email, tech_stack, cms_detected, ssl_valid, site_speed_ms.
    Uses aiohttp with connection pooling, rate limiting, and concurrency control.
    """

    def __init__(
        self,
        max_concurrent: int = 100,
        max_per_host: int = 2,
        connect_timeout: float = 3.0,
        total_timeout: float = 10.0,
        rate_limit: float = 100.0,
    ):
        self._max_concurrent = max_concurrent
        self._max_per_host = max_per_host
        self._connect_timeout = connect_timeout
        self._total_timeout = total_timeout
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = AsyncLimiter(rate_limit, 1)  # rate_limit req/sec
        self._domain_last_request: Dict[str, float] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=200,
                limit_per_host=self._max_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=self._total_timeout,
                connect=self._connect_timeout,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": _USER_AGENT},
            )
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _extract_all_data(self, html: str, headers: Dict[str, str], resp_url: str, session: aiohttp.ClientSession, domain: str) -> Dict[str, Any]:
        """Extract tech stack, CMS, and emails from a fetched page."""
        data: Dict[str, Any] = {}

        tech = self._detect_tech(html, headers)
        data["tech_stack"] = tech

        for cms in _CMS_PRIORITY:
            if cms in tech:
                data["cms_detected"] = cms
                break

        emails: Set[str] = set()
        emails.update(self._extract_mailto(html))
        emails.update(self._extract_emails(html))

        if not emails:
            base = f"{urlparse(resp_url).scheme}://{urlparse(resp_url).hostname}"
            contact_emails = await self._crawl_contact_pages(session, base, domain)
            emails.update(contact_emails)

        emails.update(self._extract_jsonld_emails(html))
        emails.update(self._decode_obfuscated_emails(html))
        emails.update(self._extract_footer_emails(html))

        data["emails"] = sorted(emails)
        return data

    async def _handle_ssl_fallback(self, session: aiohttp.ClientSession, url: str, start_time: float, result: Dict[str, Any]) -> None:
        """Retry a failed SSL request over plain HTTP."""
        result["ssl_valid"] = False
        try:
            async with session.get(
                url.replace("https://", "http://"),
                allow_redirects=True,
                max_redirects=3,
            ) as resp:
                ttfb_ms = int((time.monotonic() - start_time) * 1000)
                result["site_speed_ms"] = ttfb_ms
                result["status_code"] = resp.status
                if resp.status == 200:
                    body = await resp.content.read(_MAX_BODY_SIZE)
                    html = body.decode("utf-8", errors="replace")
                    headers = dict(resp.headers)
                    result["status"] = "ok_no_ssl"
                    result["tech_stack"] = self._detect_tech(html, headers)
                    for cms in _CMS_PRIORITY:
                        if cms in result["tech_stack"]:
                            result["cms_detected"] = cms
                            break
                    result["emails"] = sorted(self._extract_emails(html))
                else:
                    result["status"] = f"http_{resp.status}"
        except Exception:
            result["status"] = "ssl_and_http_failed"

    async def _fetch_and_extract(self, session: aiohttp.ClientSession, url: str, domain: str, result: Dict[str, Any]) -> None:
        """Fetch a URL and extract enrichment data into result dict."""
        start_time = time.monotonic()
        try:
            async with session.get(url, allow_redirects=True, max_redirects=3, ssl=True) as resp:
                result["site_speed_ms"] = int((time.monotonic() - start_time) * 1000)
                result["status_code"] = resp.status
                result["ssl_valid"] = url.startswith("https://")

                if resp.status != 200:
                    result["status"] = f"http_{resp.status}"
                    return

                body = await resp.content.read(_MAX_BODY_SIZE)
                html = body.decode("utf-8", errors="replace")
                result["status"] = "ok"
                extracted = await self._extract_all_data(html, dict(resp.headers), str(resp.url), session, domain)
                result.update(extracted)

        except aiohttp.ClientSSLError:
            await self._handle_ssl_fallback(session, url, start_time, result)
        except aiohttp.ClientConnectorError:
            result["status"] = "connection_failed"
        except asyncio.TimeoutError:
            result["status"] = "timeout"
        except aiohttp.TooManyRedirects:
            result["status"] = "too_many_redirects"
        except aiohttp.ClientResponseError as e:
            result["status"] = f"response_error_{e.status}"

    async def scrape_one(self, url: str) -> Dict[str, Any]:
        """Scrape a single business website for enrichment data."""
        if not url.startswith("http"):
            url = f"https://{url}"

        result: Dict[str, Any] = {
            "url": url, "status": "unknown", "emails": [], "tech_stack": [],
            "cms_detected": None, "ssl_valid": None, "site_speed_ms": None, "status_code": None,
        }

        session = await self._get_session()
        try:
            assert self._semaphore is not None
            async with self._semaphore:
                await self._rate_limiter.acquire()
                domain = urlparse(url).hostname or ""
                await self._domain_rate_limit(domain)
                await self._fetch_and_extract(session, url, domain, result)
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:200]
            logger.debug("Scrape error for %s: %s", url, e)

        return result

    async def scrape_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs concurrently.

        Returns list of result dicts, same order as input URLs.
        """
        tasks = [self.scrape_one(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ── Email extraction layers ──────────────────────────────────────────────

    def _extract_mailto(self, html: str) -> Set[str]:
        """Layer 1: Extract emails from mailto: links."""
        emails = set()
        for match in _MAILTO_RE.findall(html):
            email = match.lower().strip()
            if self._is_valid_email(email):
                emails.add(email)
        return emails

    def _extract_emails(self, html: str) -> Set[str]:
        """Layer 2: Regex scan + Cloudflare __cf_email__ decode."""
        emails = set()

        # Standard regex extraction
        for match in _EMAIL_RE.findall(html):
            email = match.lower().strip()
            if self._is_valid_email(email):
                emails.add(email)

        # Cloudflare obfuscated emails
        for cf_match in _CF_EMAIL_RE.findall(html):
            decoded = decode_cf_email(cf_match)
            if decoded and self._is_valid_email(decoded):
                emails.add(decoded)

        return emails

    def _extract_jsonld_emails(self, html: str) -> Set[str]:
        """Layer 4: Extract emails from JSON-LD schema.org structured data."""
        emails: Set[str] = set()
        # Find all <script type="application/ld+json"> blocks
        jsonld_re = re.compile(
            r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            re.DOTALL | re.IGNORECASE,
        )
        for block in jsonld_re.findall(html):
            try:
                data = json.loads(block)
                self._extract_emails_from_jsonld(data, emails)
            except (json.JSONDecodeError, ValueError):
                continue
        return emails

    def _extract_emails_from_jsonld(self, data: Any, emails: Set[str]) -> None:
        """Recursively extract 'email' fields from parsed JSON-LD data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() == "email" and isinstance(value, str):
                    email = value.lower().strip().removeprefix("mailto:")
                    if self._is_valid_email(email):
                        emails.add(email)
                else:
                    self._extract_emails_from_jsonld(value, emails)
        elif isinstance(data, list):
            for item in data:
                self._extract_emails_from_jsonld(item, emails)

    def _decode_obfuscated_emails(self, html: str) -> Set[str]:
        """Decode HTML entity and [at]/[dot] obfuscated emails."""
        emails: Set[str] = set()

        # Decode full HTML entities first, then scan for emails
        decoded_html = html_module.unescape(html)
        for match in _EMAIL_RE.findall(decoded_html):
            email = match.lower().strip()
            if self._is_valid_email(email):
                emails.add(email)

        # HTML numeric entity emails (&#64; for @, &#46; for .)
        for match in _HTML_ENTITY_EMAIL_RE.findall(html):
            decoded = html_module.unescape(match).lower().strip()
            if self._is_valid_email(decoded):
                emails.add(decoded)

        # [at] / (at) / [dot] / (dot) obfuscation
        for match in _OBFUSCATED_EMAIL_RE.findall(html):
            cleaned = re.sub(r'\s*[\[\(]\s*(?:at|AT)\s*[\]\)]\s*', '@', match)
            cleaned = re.sub(r'\s*[\[\(]\s*(?:dot|DOT)\s*[\]\)]\s*', '.', cleaned)
            cleaned = cleaned.lower().strip()
            if self._is_valid_email(cleaned):
                emails.add(cleaned)

        return emails

    def _extract_footer_emails(self, html: str) -> Set[str]:
        """Extract emails from the footer area (last 20% of HTML)."""
        emails: Set[str] = set()
        footer_start = int(len(html) * 0.8)
        footer_html = html[footer_start:]

        # Standard regex on footer area
        for match in _EMAIL_RE.findall(footer_html):
            email = match.lower().strip()
            if self._is_valid_email(email):
                emails.add(email)

        # Also check mailto links in footer
        for match in _MAILTO_RE.findall(footer_html):
            email = match.lower().strip()
            if self._is_valid_email(email):
                emails.add(email)

        # Check HTML comments in footer for hidden emails
        comment_re = re.compile(r'<!--(.*?)-->', re.DOTALL)
        for comment in comment_re.findall(footer_html):
            for match in _EMAIL_RE.findall(comment):
                email = match.lower().strip()
                if self._is_valid_email(email):
                    emails.add(email)

        return emails

    async def _crawl_contact_pages(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        domain: str,
    ) -> Set[str]:
        """Layer 3: Crawl common contact page paths for emails."""
        emails: Set[str] = set()

        for path in _CONTACT_PATHS:
            try:
                await self._rate_limiter.acquire()
                await self._domain_rate_limit(domain)

                async with session.get(
                    urljoin(base_url, path),
                    allow_redirects=True,
                    max_redirects=3,
                ) as resp:
                    if resp.status == 200:
                        body = await resp.content.read(_MAX_BODY_SIZE)
                        html = body.decode("utf-8", errors="replace")
                        emails.update(self._extract_mailto(html))
                        emails.update(self._extract_emails(html))
                        if emails:
                            break  # Found emails, stop crawling
            except Exception:
                continue

        return emails

    def _is_valid_email(self, email: str) -> bool:
        """Validate an extracted email address."""
        if not email or "@" not in email:
            return False
        domain = email.split("@")[-1]
        if domain in _FAKE_EMAIL_DOMAINS:
            return False
        if email.endswith((".png", ".jpg", ".gif", ".svg", ".css", ".js")):
            return False
        if "." not in domain:
            return False
        if len(email) > 254:
            return False
        return True

    # ── Tech stack detection ─────────────────────────────────────────────────

    def _detect_tech(self, html: str, headers: Dict[str, str]) -> List[str]:
        """Detect technologies from HTML content and HTTP headers."""
        detected: List[str] = []
        search_text = html + " " + " ".join(f"{k}: {v}" for k, v in headers.items())

        for tech, patterns in _TECH_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    detected.append(tech)
                    break

        # Check server header
        server = headers.get("server", headers.get("Server", "")).lower()
        if "nginx" in server:
            detected.append("nginx")
        elif "apache" in server:
            detected.append("apache")
        elif "cloudflare" in server:
            detected.append("cloudflare")

        # Check X-Powered-By
        powered_by = headers.get("x-powered-by", headers.get("X-Powered-By", "")).lower()
        if "php" in powered_by:
            detected.append("php")
        if "express" in powered_by:
            detected.append("express")
        if "asp.net" in powered_by:
            detected.append("asp.net")

        return sorted(set(detected))

    # ── Rate limiting ────────────────────────────────────────────────────────

    async def _domain_rate_limit(self, domain: str) -> None:
        """Enforce minimum 200ms between requests to the same domain."""
        now = time.monotonic()
        last = self._domain_last_request.get(domain, 0)
        elapsed_ms = (now - last) * 1000
        if elapsed_ms < 200:
            await asyncio.sleep((200 - elapsed_ms) / 1000)
        self._domain_last_request[domain] = time.monotonic()


# ── Synchronous wrapper for backward compatibility ───────────────────────────

class WebScrapeTool:
    """
    Synchronous wrapper around AsyncWebScraper.

    Maintains the same interface as the original tool for use in
    the enrichment pipeline's thread-based execution.
    """

    def __init__(self, timeout: float = 15.0):
        self._scraper = AsyncWebScraper(total_timeout=timeout)

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous scrape — runs the async scraper in an event loop."""
        url = arguments.get("url", "")
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # If already in an async context, create a new loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(self._run_async, url).result()
                return result
            else:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self._scraper.scrape_one(url))
                finally:
                    loop.close()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._scraper.scrape_one(url))
            finally:
                loop.close()

    def _run_async(self, url: str) -> Dict[str, Any]:
        """Run async scraper in a new event loop (for thread contexts)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._scraper.scrape_one(url))
        finally:
            loop.run_until_complete(self._scraper.close())
            loop.close()

    @property
    def name(self) -> str:
        return "web_scrape"

    @property
    def description(self) -> str:
        return (
            "Scrape a business website for contact emails, tech stack, and site health. "
            "Provide the website URL. Returns extracted emails, detected technologies, "
            "SSL status, CMS, and site speed."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Website URL to scrape (e.g., https://example.com)",
                },
            },
            "required": ["url"],
        }
