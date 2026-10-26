"""Tests for forge.tools.web_scraper (no network calls)."""

import pytest

from forge.tools.web_scraper import (
    AsyncWebScraper,
    WebScrapeTool,
    decode_cf_email,
    rank_emails,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def scraper():
    """Create an AsyncWebScraper instance for calling internal methods."""
    return AsyncWebScraper()


# ---------------------------------------------------------------------------
# Tests: decode_cf_email
# ---------------------------------------------------------------------------


class TestDecodeCfEmail:
    def test_known_encoded_value(self):
        # Cloudflare encodes "test@example.com" by XOR-ing each byte with a key.
        # Key byte = 0x2A (42 decimal). t=0x74, e=0x65, s=0x73, t=0x74, @=0x40, ...
        # Encoded: "2a" + hex(0x74^0x2a) + hex(0x65^0x2a) + ...
        # Let's test a simpler manually-computed example.
        # For email "a@b.c":  a=0x61, @=0x40, b=0x62, .=0x2e, c=0x63
        # Key = 0x10 -> encoded = "10" + hex(0x61^0x10) + hex(0x40^0x10) + hex(0x62^0x10) + hex(0x2e^0x10) + hex(0x63^0x10)
        #   = "10" + "71" + "50" + "72" + "3e" + "73"
        #   = "1071507273e73"... wait, let me be precise:
        # 0x61^0x10 = 0x71, 0x40^0x10 = 0x50, 0x62^0x10 = 0x72,
        # 0x2e^0x10 = 0x3e, 0x63^0x10 = 0x73
        # Note: hex digits need to be correct. Let me just compute directly.
        # Actually the format expects hex pairs after the key.
        # "10" "71" "50" "72" "3e" "73" -> "107150723e73"
        result = decode_cf_email("107150723e73")
        assert result == "a@b.c"

    def test_another_encoded_value(self):
        # Email "hi@ab.cd", key=0x20
        # h=0x68^0x20=0x48, i=0x69^0x20=0x49, @=0x40^0x20=0x60
        # a=0x61^0x20=0x41, b=0x62^0x20=0x42, .=0x2e^0x20=0x0e, c=0x63^0x20=0x43, d=0x64^0x20=0x44
        encoded = "20484960414200e4344"  # wrong length, let me fix
        # "20" + "48" + "49" + "60" + "41" + "42" + "0e" + "43" + "44"
        encoded = "2048496041420e4344"
        result = decode_cf_email(encoded)
        assert result == "hi@ab.cd"

    def test_invalid_encoding_returns_none(self):
        assert decode_cf_email("xyz") is None

    def test_empty_string_returns_none(self):
        assert decode_cf_email("") is None

    def test_no_at_sign_returns_none(self):
        # Key=0xFF, all same char -> no @ sign in result
        assert decode_cf_email("ff") is None


# ---------------------------------------------------------------------------
# Tests: _is_valid_email
# ---------------------------------------------------------------------------


class TestIsValidEmail:
    def test_good_emails(self, scraper):
        assert scraper._is_valid_email("john@tampadental.com") is True
        assert scraper._is_valid_email("info@mybiz.net") is True
        assert scraper._is_valid_email("contact+sales@company.org") is True

    def test_rejects_fake_domains(self, scraper):
        for domain in ["example.com", "test.com", "sentry.io", "wordpress.org"]:
            assert scraper._is_valid_email(f"user@{domain}") is False

    def test_rejects_no_at_sign(self, scraper):
        assert scraper._is_valid_email("notanemail") is False

    def test_rejects_empty(self, scraper):
        assert scraper._is_valid_email("") is False

    def test_rejects_file_extensions(self, scraper):
        assert scraper._is_valid_email("icon@site.png") is False
        assert scraper._is_valid_email("bg@site.jpg") is False
        assert scraper._is_valid_email("style@site.css") is False

    def test_rejects_no_dot_in_domain(self, scraper):
        assert scraper._is_valid_email("user@localhost") is False

    def test_rejects_too_long(self, scraper):
        assert scraper._is_valid_email("a" * 250 + "@x.com") is False


# ---------------------------------------------------------------------------
# Tests: _detect_tech
# ---------------------------------------------------------------------------


class TestDetectTech:
    def test_finds_wordpress(self, scraper):
        html = '<link rel="stylesheet" href="/wp-content/themes/theme/style.css">'
        tech = scraper._detect_tech(html, {})
        assert "wordpress" in tech

    def test_finds_shopify(self, scraper):
        html = '<script src="https://cdn.shopify.com/s/files/1/store.js"></script>'
        tech = scraper._detect_tech(html, {})
        assert "shopify" in tech

    def test_finds_squarespace(self, scraper):
        html = '<meta content="Squarespace" name="generator">'
        tech = scraper._detect_tech(html, {})
        assert "squarespace" in tech

    def test_finds_react_nextjs(self, scraper):
        html = '<div id="__next"><script src="/_next/static/chunks/main.js"></script></div>'
        tech = scraper._detect_tech(html, {})
        assert "react" in tech or "next.js" in tech

    def test_finds_google_analytics(self, scraper):
        html = '<script src="https://www.google-analytics.com/analytics.js"></script>'
        tech = scraper._detect_tech(html, {})
        assert "google-analytics" in tech

    def test_finds_cloudflare_from_header(self, scraper):
        tech = scraper._detect_tech("", {"server": "cloudflare"})
        assert "cloudflare" in tech

    def test_finds_nginx_from_header(self, scraper):
        tech = scraper._detect_tech("", {"server": "nginx/1.24.0"})
        assert "nginx" in tech

    def test_finds_apache_from_header(self, scraper):
        tech = scraper._detect_tech("", {"server": "Apache/2.4.57"})
        assert "apache" in tech

    def test_finds_php_from_powered_by(self, scraper):
        tech = scraper._detect_tech("", {"x-powered-by": "PHP/8.2"})
        assert "php" in tech

    def test_finds_express_from_powered_by(self, scraper):
        tech = scraper._detect_tech("", {"X-Powered-By": "Express"})
        assert "express" in tech

    def test_finds_stripe(self, scraper):
        html = '<script src="https://js.stripe.com/v3/"></script>'
        tech = scraper._detect_tech(html, {})
        assert "stripe" in tech

    def test_finds_multiple_technologies(self, scraper):
        html = (
            '<link href="/wp-content/themes/style.css">'
            '<script src="https://www.google-analytics.com/analytics.js"></script>'
            '<script src="https://js.stripe.com/v3/"></script>'
        )
        tech = scraper._detect_tech(html, {"server": "nginx/1.24"})
        assert "wordpress" in tech
        assert "google-analytics" in tech
        assert "stripe" in tech
        assert "nginx" in tech

    def test_empty_html_and_headers(self, scraper):
        tech = scraper._detect_tech("", {})
        assert tech == []

    def test_returns_sorted_deduplicated(self, scraper):
        html = '<script src="/wp-content/test.js"></script><link href="/wp-includes/style.css">'
        tech = scraper._detect_tech(html, {})
        # WordPress should appear only once despite two matching patterns
        assert tech.count("wordpress") == 1
        assert tech == sorted(tech)


# ---------------------------------------------------------------------------
# Tests: _extract_mailto
# ---------------------------------------------------------------------------


class TestExtractMailto:
    def test_finds_mailto_links(self, scraper):
        html = '<a href="mailto:info@dental.com">Email us</a>'
        emails = scraper._extract_mailto(html)
        assert "info@dental.com" in emails

    def test_multiple_mailto(self, scraper):
        html = (
            '<a href="mailto:sales@biz.com">Sales</a><a href="mailto:support@biz.com">Support</a>'
        )
        emails = scraper._extract_mailto(html)
        assert "sales@biz.com" in emails
        assert "support@biz.com" in emails

    def test_no_mailto(self, scraper):
        html = '<a href="https://example.com">Link</a>'
        emails = scraper._extract_mailto(html)
        assert len(emails) == 0

    def test_filters_fake_domains(self, scraper):
        html = '<a href="mailto:user@example.com">Fake</a>'
        emails = scraper._extract_mailto(html)
        assert len(emails) == 0


# ---------------------------------------------------------------------------
# Tests: _extract_emails
# ---------------------------------------------------------------------------


class TestExtractEmails:
    def test_finds_regex_emails(self, scraper):
        html = "Contact us at info@tampadental.com for appointments."
        emails = scraper._extract_emails(html)
        assert "info@tampadental.com" in emails

    def test_finds_multiple_emails(self, scraper):
        html = "john@biz.com or jane@biz.com"
        emails = scraper._extract_emails(html)
        assert "john@biz.com" in emails
        assert "jane@biz.com" in emails

    def test_decodes_cloudflare_emails(self, scraper):
        # Cloudflare obfuscated email for "a@b.c" with key 0x10
        html = 'Contact: <span data-cfemail="107150723e73">[email protected]</span>'
        emails = scraper._extract_emails(html)
        assert "a@b.c" in emails

    def test_filters_fake_emails(self, scraper):
        html = "user@example.com or test@sentry.io"
        emails = scraper._extract_emails(html)
        assert len(emails) == 0


# ---------------------------------------------------------------------------
# Tests: _extract_jsonld_emails
# ---------------------------------------------------------------------------


class TestExtractJsonldEmails:
    def test_finds_schema_org_email(self, scraper):
        html = """
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Dentist",
            "name": "Tampa Dental",
            "email": "info@tampadental.com"
        }
        </script>
        """
        emails = scraper._extract_jsonld_emails(html)
        assert "info@tampadental.com" in emails

    def test_handles_mailto_prefix(self, scraper):
        html = """
        <script type="application/ld+json">
        {"email": "mailto:contact@biz.com"}
        </script>
        """
        emails = scraper._extract_jsonld_emails(html)
        assert "contact@biz.com" in emails

    def test_nested_jsonld(self, scraper):
        html = """
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Organization",
            "contactPoint": {
                "@type": "ContactPoint",
                "email": "support@company.net"
            }
        }
        </script>
        """
        emails = scraper._extract_jsonld_emails(html)
        assert "support@company.net" in emails

    def test_no_jsonld(self, scraper):
        html = "<html><body>No structured data here</body></html>"
        emails = scraper._extract_jsonld_emails(html)
        assert len(emails) == 0

    def test_invalid_json_in_script(self, scraper):
        html = '<script type="application/ld+json">{broken json</script>'
        emails = scraper._extract_jsonld_emails(html)
        assert len(emails) == 0


# ---------------------------------------------------------------------------
# Tests: _decode_obfuscated_emails
# ---------------------------------------------------------------------------


class TestDecodeObfuscatedEmails:
    def test_at_dot_obfuscation(self, scraper):
        html = "Email: info [at] mybiz [dot] com"
        emails = scraper._decode_obfuscated_emails(html)
        assert "info@mybiz.com" in emails

    def test_parenthesis_obfuscation(self, scraper):
        html = "Email: john (AT) company (DOT) org"
        emails = scraper._decode_obfuscated_emails(html)
        assert "john@company.org" in emails

    def test_html_entity_email(self, scraper):
        # &#64; = @, &#46; = .
        html = "Contact: info&#64;business&#46;com"
        emails = scraper._decode_obfuscated_emails(html)
        assert "info@business.com" in emails

    def test_no_obfuscation(self, scraper):
        html = "<p>No obfuscated emails here.</p>"
        # May find regular emails via html.unescape scan, but no obfuscated ones
        emails = scraper._decode_obfuscated_emails(html)
        # No valid emails in this text
        assert len(emails) == 0


# ---------------------------------------------------------------------------
# Tests: WebScrapeTool
# ---------------------------------------------------------------------------


class TestWebScrapeTool:
    def test_name_property(self):
        tool = WebScrapeTool()
        assert tool.name == "web_scrape"

    def test_description_property(self):
        tool = WebScrapeTool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 10

    def test_parameters_property(self):
        tool = WebScrapeTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "url" in params["properties"]
        assert "url" in params["required"]


# ---------------------------------------------------------------------------
# Tests: _extract_footer_emails
# ---------------------------------------------------------------------------


class TestExtractFooterEmails:
    def test_finds_email_in_footer(self, scraper):
        # Build a page where the email is in the last 20%
        padding = "x" * 1000
        footer = "<footer>Contact: hello@realbiz.com</footer>"
        html = padding + footer
        emails = scraper._extract_footer_emails(html)
        assert "hello@realbiz.com" in emails

    def test_no_email_in_footer(self, scraper):
        html = "x" * 100 + "<footer>No emails here</footer>"
        emails = scraper._extract_footer_emails(html)
        assert len(emails) == 0


# ---------------------------------------------------------------------------
# Tests: rank_emails (best contact first, nothing dropped)
# ---------------------------------------------------------------------------


class TestRankEmails:
    def test_low_value_inbox_ranked_last(self):
        ranked = rank_emails(["privacy@thorne.com", "info@thorne.com"], "www.thorne.com")
        assert ranked == ["info@thorne.com", "privacy@thorne.com"]

    def test_own_domain_before_third_party(self):
        ranked = rank_emails(["owner@gmail.com", "hello@biz.com"], "biz.com")
        assert ranked == ["hello@biz.com", "owner@gmail.com"]

    def test_third_party_contact_beats_own_domain_low_value(self):
        ranked = rank_emails(["privacy@biz.com", "owner@gmail.com"], "biz.com")
        assert ranked == ["owner@gmail.com", "privacy@biz.com"]

    def test_subdomain_site_counts_as_own_domain(self):
        ranked = rank_emails(["a@other.com", "sales@biz.com"], "shop.biz.com")
        assert ranked[0] == "sales@biz.com"

    def test_keeps_every_email_once(self):
        ranked = rank_emails(["Info@Biz.com", "info@biz.com", "careers@biz.com"], "biz.com")
        assert ranked == ["info@biz.com", "careers@biz.com"]

    def test_empty(self):
        assert rank_emails([], "biz.com") == []
