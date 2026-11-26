"""Tests for the contacts finder (forge contacts): who to reach at a company, from its own public pages."""

import asyncio
import csv
import json

from aiohttp import web

from forge.contacts.archive import pick_snapshots
from forge.contacts.export import COMPANY_COLUMNS, write_companies_csv, write_people_csv
from forge.contacts.finder import (
    CompanyContacts,
    _absorb,
    email_for,
    find_contacts,
    infer_email_format,
    normalize_company,
    pick_pages,
)
from forge.contacts.page import parse_page
from forge.contacts.people import Person, classify_role, extract_people, looks_like_name

# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------


class TestParsePage:
    def test_blocks_split_at_block_elements_and_skip_scripts(self):
        page = parse_page(
            "<html><head><title>Acme | Home</title><style>.x{}</style></head><body>"
            "<div><h3>Jane Roe</h3><p>Head of Marketing</p></div>"
            "<script>var a = 'Not A Person';</script>Footer<br>Line two</body></html>",
            "https://acme.com/",
        )
        assert page.blocks == ["Jane Roe", "Head of Marketing", "Footer", "Line two"]
        assert page.title == "Acme | Home"

    def test_links_are_made_absolute(self):
        page = parse_page(
            '<a href="/about-us">About us</a><a href="https://x.com/acme">X</a><a href="mailto:a@acme.com">mail</a>',
            "https://acme.com/products/thing/",
        )
        assert [(link.url, link.text) for link in page.links] == [
            ("https://acme.com/about-us", "About us"),
            ("https://x.com/acme", "X"),
            ("mailto:a@acme.com", "mail"),
        ]

    def test_jsonld_is_parsed_and_bad_jsonld_is_skipped(self):
        page = parse_page(
            '<script type="application/ld+json">{"@type": "Organization", "name": "Acme"}</script>'
            '<script type="application/ld+json">{not json</script>',
            "https://acme.com/",
        )
        assert page.jsonld == [{"@type": "Organization", "name": "Acme"}]

    def test_contact_form_is_detected(self):
        with_form = parse_page(
            '<form action="/send"><input type="email" name="email"><textarea name="message"></textarea></form>',
            "https://acme.com/contact",
        )
        search_only = parse_page(
            '<form action="/search"><input type="text" name="q"></form>', "https://acme.com/"
        )
        assert with_form.has_contact_form is True
        assert search_only.has_contact_form is False

    def test_site_name_from_open_graph(self):
        page = parse_page(
            '<meta property="og:site_name" content="Acme Wellness">', "https://acme.com/"
        )
        assert page.site_name == "Acme Wellness"


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------


def _people(html, url="https://acme.com/about"):
    return [(p.name, p.title, p.role) for p in extract_people(parse_page(html, url))]


class TestClassifyRole:
    def test_marketing_titles(self):
        for title in (
            "Director of Marketing",
            "CMO",
            "Chief Marketing Officer",
            "Head of Partnerships",
            "VP Business Development",
            "PR & Communications Manager",
            "Leitung Marketing",
            "Marketingleiterin",
        ):
            assert classify_role(title) == "marketing", title

    def test_leadership_titles(self):
        for title in ("Founder & CEO", "Co-Founder", "President", "Geschäftsführer", "Owner"):
            assert classify_role(title) == "leadership", title

    def test_other_titles(self):
        assert classify_role("Staff Accountant") == "other"


class TestLooksLikeName:
    def test_real_names(self):
        for name in (
            "Hans Eng",
            "Dr. James V. Hardt",
            "Rowena Gates",
            "Anne-Marie O'Neil",
            "José Núñez",
        ):
            assert looks_like_name(name), name

    def test_not_names(self):
        for text in (
            "Contact Us",
            "Read More",
            "Our Team",
            "Red Light Therapy",
            "hans eng",
            "Shop",
            "A B C D E F",
        ):
            assert not looks_like_name(text), text


class TestExtractPeople:
    def test_name_comma_title_on_one_line(self):
        assert _people("<p>Hans Eng, Founder and CEO</p>") == [
            ("Hans Eng", "Founder and CEO", "leadership")
        ]

    def test_team_card_name_then_title(self):
        assert _people("<div><h4>Rowena Gates</h4><span>VP Business Development</span></div>") == [
            ("Rowena Gates", "VP Business Development", "marketing")
        ]

    def test_title_colon_name(self):
        assert _people("<h2>Meet Our Founder: Dr. James V. Hardt</h2>") == [
            ("Dr. James V. Hardt", "Founder", "leadership")
        ]

    def test_german_impressum(self):
        assert _people(
            "<p>Geschäftsführer: Max Mustermann</p><p>Registergericht: Amtsgericht Köln</p>"
        ) == [("Max Mustermann", "Geschäftsführer", "leadership")]

    def test_jsonld_founder_and_employee(self):
        html = (
            '<script type="application/ld+json">{"@type": "Organization", "name": "Acme",'
            ' "founder": {"@type": "Person", "name": "Jane Roe"},'
            ' "employee": [{"@type": "Person", "name": "Sam Poe", "jobTitle": "Head of Marketing"}]}</script>'
        )
        assert sorted(_people(html)) == [
            ("Jane Roe", "Founder", "leadership"),
            ("Sam Poe", "Head of Marketing", "marketing"),
        ]

    def test_sentences_and_lone_titles_are_not_people(self):
        html = (
            "<p>Google Tag Manager simplifies the management of marketing tags on your website.</p>"
            "<p>VP Business Development</p><p>Contact Us</p><p>Founder</p><p>Read More</p>"
        )
        assert _people(html) == []

    def test_same_person_twice_is_kept_once(self):
        html = "<p>Hans Eng, Founder and CEO</p><div><h4>Hans Eng</h4><p>Founder and CEO</p></div>"
        assert _people(html) == [("Hans Eng", "Founder and CEO", "leadership")]

    def test_source_page_is_kept(self):
        people = extract_people(
            parse_page("<p>Hans Eng, Founder and CEO</p>", "https://acme.com/about")
        )
        assert people[0].source == "https://acme.com/about"


# ---------------------------------------------------------------------------
# Company input, page choice, email format
# ---------------------------------------------------------------------------


class TestNormalizeCompany:
    def test_product_url_becomes_the_site(self):
        assert normalize_company("https://sunpowerled.com/product/helmet/") == (
            "https://sunpowerled.com",
            "sunpowerled.com",
            "sunpowerled.com",
        )

    def test_bare_domain_and_subdomain(self):
        assert normalize_company("shop.airnergy.de") == (
            "https://shop.airnergy.de",
            "shop.airnergy.de",
            "airnergy.de",
        )
        assert normalize_company("www.zyto.com")[2] == "zyto.com"

    def test_country_second_level(self):
        assert normalize_company("https://www.acme.co.uk/x")[2] == "acme.co.uk"


class TestPickPages:
    def test_picks_people_and_contact_pages_on_the_same_site(self):
        home = parse_page(
            '<a href="/products/helmet">Helmet</a><a href="/about-us">About</a><a href="/pages/team">Our team</a>'
            '<a href="/contact">Contact</a><a href="/impressum">Impressum</a><a href="/affiliates">Affiliates</a>'
            '<a href="/press">Press</a><a href="https://www.instagram.com/acme">IG</a>'
            '<a href="https://shop.acme.com/contact-us">Shop contact</a><a href="mailto:a@acme.com">mail</a>',
            "https://acme.com/",
        )
        pages = pick_pages(home, "acme.com", "acme.com", limit=10)
        assert "https://acme.com/products/helmet" not in pages
        assert "https://www.instagram.com/acme" not in pages
        assert set(pages) == {
            "https://acme.com/about-us",
            "https://acme.com/pages/team",
            "https://acme.com/contact",
            "https://acme.com/impressum",
            "https://acme.com/affiliates",
            "https://acme.com/press",
            "https://shop.acme.com/contact-us",
        }
        assert pages.index("https://acme.com/impressum") < pages.index(
            "https://acme.com/affiliates"
        )

    def test_falls_back_to_common_paths_and_respects_the_limit(self):
        home = parse_page("<p>No links here</p>", "https://acme.com/")
        pages = pick_pages(home, "acme.com", "acme.com", limit=3)
        assert pages == [
            "https://acme.com/about",
            "https://acme.com/about-us",
            "https://acme.com/team",
        ]


class TestPartnerSubdomain:
    """A partner program on its own subdomain is a partner page, and its sign-up form is not the
    company's contact form. Shiftwave's lives at partners.shiftwave.co/, whose address is just "/"."""

    _SIGNUP = (
        "<html><head><title>Join Acme Light's Partner Program</title></head><body>"
        "<form><input type='email' name='email'><textarea name='message'></textarea></form>"
        "</body></html>"
    )
    _CONTACT = (
        "<html><head><title>Contact Acme Light</title></head><body>"
        "<form><input type='email' name='email'><textarea name='message'></textarea></form>"
        "</body></html>"
    )

    @staticmethod
    def _result():
        return CompanyContacts(
            input="acmelight.co",
            website="https://acmelight.co",
            host="acmelight.co",
            root="acmelight.co",
        )

    def test_the_address_kind_reads_the_subdomain_when_the_path_says_nothing(self):
        from forge.contacts.finder import url_kind

        assert url_kind("https://partners.acmelight.co/", "acmelight.co") == 4
        assert url_kind("https://affiliates.acmelight.co/login", "acmelight.co") == 4
        assert url_kind("https://shop.acmelight.co/pages/contact", "acmelight.co") == 2
        assert url_kind("https://www.acmelight.co/", "acmelight.co") is None
        assert url_kind("https://acmelight.co/", "acmelight.co") is None
        assert url_kind("https://partners.othersite.com/", "acmelight.co") is None

    def test_a_partner_subdomain_page_is_listed_as_the_partner_program_not_the_contact_form(self):
        r = self._result()
        page = parse_page(self._SIGNUP, "https://partners.acmelight.co/")
        _absorb(r, page, page.url, live=True)
        assert r.partner_pages == ["https://partners.acmelight.co"]
        assert r.contact_form == ""

    def test_a_link_to_a_partner_subdomain_is_a_partner_page(self):
        r = self._result()
        home = parse_page(
            '<a href="https://partners.acmelight.co/">Partner Program</a>', "https://acmelight.co/"
        )
        _absorb(r, home, home.url, live=True)
        assert r.partner_pages == ["https://partners.acmelight.co"]

    def test_a_partner_subdomain_is_picked_even_when_its_link_text_says_nothing(self):
        home = parse_page(
            '<a href="https://partners.acmelight.co/">Join us</a>', "https://acmelight.co/"
        )
        assert "https://partners.acmelight.co" in pick_pages(home, "acmelight.co", "acmelight.co")

    def test_the_contact_page_form_is_still_the_contact_form(self):
        r = self._result()
        for html, url in (
            (self._SIGNUP, "https://partners.acmelight.co/"),
            (self._CONTACT, "https://acmelight.co/contact"),
        ):
            page = parse_page(html, url)
            _absorb(r, page, page.url, live=True)
        assert r.contact_form == "https://acmelight.co/contact"
        assert r.partner_pages == ["https://partners.acmelight.co"]


class TestEmailFormat:
    def _p(self, name):
        return Person(name=name, title="CEO", role="leadership", source="https://acme.com/about")

    def test_formats_learned_from_a_matching_address(self):
        cases = {
            "john@acme.com": "first",
            "john.thorp@acme.com": "first.last",
            "jthorp@acme.com": "flast",
            "johnthorp@acme.com": "firstlast",
        }
        for email, fmt in cases.items():
            assert infer_email_format([self._p("Dr. John V. Thorp")], [email], "acme.com") == (
                fmt,
                email,
            )

    def test_no_format_without_a_named_address(self):
        assert infer_email_format([self._p("John Thorp")], ["info@acme.com"], "acme.com") is None

    def test_email_for_builds_the_address(self):
        assert email_for("Rowena Gates", "first.last", "acme.com") == "rowena.gates@acme.com"
        assert email_for("José Núñez", "flast", "acme.com") == "jnunez@acme.com"


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


class TestPickSnapshots:
    def test_newest_capture_per_useful_path(self):
        rows = [
            ["timestamp", "original"],
            ["20230101000000", "https://acme.com/contact"],
            ["20240501000000", "https://acme.com/contact/"],
            ["20240601000000", "https://acme.com/products/helmet"],
            ["20240301000000", "https://acme.com/press"],
        ]
        assert pick_snapshots(rows, limit=5) == [
            ("20240501000000", "https://acme.com/contact/"),
            ("20240301000000", "https://acme.com/press"),
        ]


# ---------------------------------------------------------------------------
# A whole run against a local site
# ---------------------------------------------------------------------------

_HOME = """<html><head><title>Acme Light | Red light for recovery</title>
<script type="application/ld+json">{"@type": "Organization", "name": "Acme Light", "telephone": "+1 801 555 0100"}</script>
</head><body>
<a href="/about">About</a><a href="/contact">Contact</a><a href="/affiliates">Affiliate program</a>
<a href="/private/team">Staff</a>
<a href="https://www.linkedin.com/company/acme-light">LinkedIn</a>
<a href="https://www.facebook.com/sharer/sharer.php?u=x">Share</a>
<a href="https://www.instagram.com/acmelight">Instagram</a>
<script src="https://static.goaffpro.com/loader.js"></script>
</body></html>"""

_ABOUT = """<html><body>
<div><h3>Jane Roe</h3><p>Founder &amp; CEO</p></div>
<div><h3>Sam Poe</h3><p>Director of Marketing</p></div>
<p>Write to Jane at jane.roe@acmelight.test</p>
</body></html>"""

_CONTACT = """<html><body>
<a href="mailto:hello@acmelight.test">hello@acmelight.test</a>
<a href="mailto:press@acmelight.test">Press</a>
<a href="tel:+18015550100">Call us</a>
<p>Built by studio@webagency.example.org</p>
<form><input type="email" name="email"><textarea name="message"></textarea></form>
</body></html>"""


async def _serve(routes, hits):
    app = web.Application()

    def make(body, status=200, content_type="text/html"):
        async def handler(request):
            hits.append(request.path)
            return web.Response(text=body, status=status, content_type=content_type)

        return handler

    for path, (body, status) in routes.items():
        content_type = "text/plain" if path == "/robots.txt" else "text/html"
        if path.startswith("/cdx"):
            content_type = "application/json"
        app.router.add_get(path, make(body, status, content_type))

    async def missing(request):
        hits.append(request.path)
        return web.Response(text="not here", status=404)

    app.router.add_get("/{tail:.*}", missing)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


def _run_local(routes, archive=False, archive_routes=None):
    hits: list = []
    archive_hits: list = []

    async def go():
        runner, port = await _serve(routes, hits)
        archive_runner = archive_port = None
        if archive_routes is not None:
            archive_runner, archive_port = await _serve(archive_routes(port), archive_hits)
        try:
            results = await find_contacts(
                # the site is on 127.0.0.1, its mail on acmelight.test (the website/mail-domain pair)
                [(f"http://127.0.0.1:{port}/products/thing", "acmelight.test")],
                archive=archive,
                archive_base=f"http://127.0.0.1:{archive_port}" if archive_port else None,
                host_delay=0.0,
            )
            return results[0], port
        finally:
            await runner.cleanup()
            if archive_runner:
                await archive_runner.cleanup()

    result, port = asyncio.run(go())
    return result, hits, archive_hits, port


class TestFindContactsLocalSite:
    ROUTES = {
        "/": (_HOME, 200),
        "/about": (_ABOUT, 200),
        "/contact": (_CONTACT, 200),
        "/affiliates": ("<p>Join our affiliate program</p>", 200),
        "/private/team": ("<p>Hidden Person, Chief Marketing Officer</p>", 200),
        "/robots.txt": ("User-agent: *\nDisallow: /private/\n", 200),
    }

    def test_people_inboxes_phone_form_and_partner_program(self):
        result, hits, _, port = _run_local(self.ROUTES)
        assert result.status == "ok"
        assert result.company == "Acme Light"
        names = {(p.name, p.title, p.role) for p in result.people}
        assert names == {
            ("Jane Roe", "Founder & CEO", "leadership"),
            ("Sam Poe", "Director of Marketing", "marketing"),
        }
        assert result.marketing_contact.name == "Sam Poe"
        # press@ ranks above hello@ for a partnership pitch; the web agency's address is dropped
        assert result.inboxes == ["press@acmelight.test", "hello@acmelight.test"]
        assert "+18015550100" in result.phones
        assert result.contact_form == f"http://127.0.0.1:{port}/contact"
        assert result.partner_pages == [f"http://127.0.0.1:{port}/affiliates"]
        assert result.partner_platforms == ["GoAffPro"]
        assert result.socials == {
            "linkedin": "https://www.linkedin.com/company/acme-light",
            "instagram": "https://www.instagram.com/acmelight",
        }

    def test_named_address_teaches_the_format(self):
        result, _, _, _ = _run_local(self.ROUTES)
        jane = next(p for p in result.people if p.name == "Jane Roe")
        sam = next(p for p in result.people if p.name == "Sam Poe")
        assert (jane.email, jane.email_basis) == ("jane.roe@acmelight.test", "on their site")
        assert result.email_format == "first.last"
        assert sam.email == "sam.poe@acmelight.test"
        assert sam.email_basis.startswith("guessed from the format of jane.roe@")

    def test_robots_txt_is_respected(self):
        result, hits, _, _ = _run_local(self.ROUTES)
        assert "/private/team" not in hits
        assert all(p.name != "Hidden Person" for p in result.people)

    def test_best_way_in_prefers_the_marketing_person(self):
        result, _, _, _ = _run_local(self.ROUTES)
        assert result.best_way_in().startswith("Sam Poe (Director of Marketing)")


class TestFindContactsBlockedSite:
    def test_blocked_site_reads_archived_copies_and_keeps_own_domain_only(self):
        routes = {"/": ("Forbidden", 403), "/robots.txt": ("", 404)}

        def archive_routes(site_port):
            original = f"http://127.0.0.1:{site_port}/contact"
            cdx = json.dumps([["timestamp", "original"], ["20240501000000", original]])
            snapshot = (
                "<p>Partnerships: partnerships@acmelight.test</p>"
                "<p>Site by hello@someagency.example.org</p>"
                "<p>Lena Park, Head of Partnerships</p>"
            )
            return {
                "/cdx/search/cdx": (cdx, 200),
                "/web/{tail:.*}": (snapshot, 200),
            }

        result, _, archive_hits, _ = _run_local(routes, archive=True, archive_routes=archive_routes)
        assert result.status == "blocked"
        assert result.inboxes == ["partnerships@acmelight.test"]
        assert [(p.name, p.role) for p in result.people] == [("Lena Park", "marketing")]
        assert "archived copy from 2024-05-01" in result.people[0].source
        assert any("read 1 archived copy" in note for note in result.notes)

    def test_blocked_site_without_archive_says_so(self):
        routes = {"/": ("Forbidden", 403), "/robots.txt": ("", 404)}
        result, _, _, _ = _run_local(routes, archive=False)
        assert result.status == "blocked"
        assert result.people == [] and result.inboxes == []
        assert result.best_way_in() == "nothing public found"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def _result(self):
        r = CompanyContacts(
            input="acme.com", website="https://acme.com", host="acme.com", root="acme.com"
        )
        r.company = "Acme"
        r.status = "ok"
        r.people = [
            Person(
                "Jane Roe",
                "Founder & CEO",
                "leadership",
                "https://acme.com/about",
                "jane@acme.com",
                "on their site",
            )
        ]
        r.emails = {"hello@acme.com": "https://acme.com/contact"}
        return r

    def test_company_csv_has_one_row_per_company(self, tmp_path):
        path = tmp_path / "contacts.csv"
        write_companies_csv([self._result()], str(path))
        rows = list(csv.DictReader(open(path, newline="")))
        assert list(rows[0].keys()) == COMPANY_COLUMNS
        assert rows[0]["Company"] == "Acme"
        assert rows[0]["Marketing person"] == "none listed publicly"
        assert "Jane Roe (Founder & CEO) jane@acme.com" in rows[0]["Leaders"]
        assert rows[0]["Inboxes"] == "hello@acme.com"

    def test_people_csv_has_one_row_per_person(self, tmp_path):
        path = tmp_path / "people.csv"
        write_people_csv([self._result()], str(path))
        rows = list(csv.DictReader(open(path, newline="")))
        assert rows == [
            {
                "Company": "Acme",
                "Website": "https://acme.com",
                "Name": "Jane Roe",
                "Title": "Founder & CEO",
                "Role": "leadership",
                "Email": "jane@acme.com",
                "Email basis": "on their site",
                "Found on": "https://acme.com/about",
            }
        ]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


class TestContactsCommand:
    def test_help(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "forge", "contacts", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "robots.txt" in result.stdout

    def test_read_targets_from_csv_and_from_lines(self, tmp_path):
        from forge.cli_contacts import read_targets

        table = tmp_path / "companies.csv"
        table.write_text(
            "Company,Website,Email domain\nAirnergy,shop.airnergy.de,airnergy.com\nZyto,zyto.com,\n"
        )
        lines = tmp_path / "sites.txt"
        lines.write_text("zyto.com\n\nhttps://eng3corp.com/about\n")
        assert read_targets(str(table)) == [("shop.airnergy.de", "airnergy.com"), ("zyto.com", "")]
        assert read_targets(str(lines)) == [("zyto.com", ""), ("https://eng3corp.com/about", "")]

    def test_writes_both_spreadsheets(self, tmp_path, monkeypatch):
        import forge.contacts.finder as finder
        from forge.cli import main

        seen = {}

        def fake_run(targets, **kwargs):
            seen["targets"], seen["kwargs"] = targets, kwargs
            return [TestExport()._result()]

        monkeypatch.setattr(finder, "run_contacts", fake_run)
        out, people = tmp_path / "contacts.csv", tmp_path / "people.csv"
        main(
            [
                "contacts",
                "acme.com",
                "--no-archive",
                "--output",
                str(out),
                "--people-output",
                str(people),
            ]
        )
        assert seen["targets"] == [("acme.com", "")]
        assert seen["kwargs"]["archive"] is False
        assert next(csv.DictReader(open(out, newline="")))["Company"] == "Acme"
        assert next(csv.DictReader(open(people, newline="")))["Name"] == "Jane Roe"


# ---------------------------------------------------------------------------
# Found on the first real run (Gabe's eight companies, 2026-11-26)
# ---------------------------------------------------------------------------


class TestFirstRealRun:
    def test_blog_posts_under_about_are_not_people_pages(self):
        home = parse_page(
            '<a href="/about-eng3-corp">About</a><a href="/about/contact-us">Contact</a>'
            '<a href="/about/blog/wellness-mama">Wellness Mama</a><a href="/about/blog/author/rolen">Rolen</a>',
            "https://eng3corp.com/",
        )
        assert pick_pages(home, "eng3corp.com", "eng3corp.com", limit=8) == [
            "https://eng3corp.com/about-eng3-corp",
            "https://eng3corp.com/about/contact-us",
        ]

    def test_unlinked_about_and_contact_pages_are_tried(self):
        home = parse_page('<a href="/practitioners">Practitioners</a>', "https://zyto.com/")
        assert pick_pages(home, "zyto.com", "zyto.com", limit=8) == [
            "https://zyto.com/practitioners",
            "https://zyto.com/about",
            "https://zyto.com/about-us",
            "https://zyto.com/team",
            "https://zyto.com/contact",
            "https://zyto.com/contact-us",
        ]

    def test_product_phrases_are_not_names_and_biocommunication_is_not_a_title(self):
        assert not looks_like_name("Certification Programs")
        assert not looks_like_name("Electrodermal Screening")
        assert classify_role("Biocommunication") == "other"
        assert _people("<p>Electrodermal Screening</p><p>Biocommunication</p>") == []

    def test_company_name_is_never_a_slogan(self):
        from forge.contacts.finder import _company_name

        zyto = parse_page(
            "<title>Improving Wellness through Biocommunication | ZYTO</title>", "https://zyto.com/"
        )
        eng3 = parse_page(
            "<title>Oxidative Stress Relief | Repair Cell Damage with NanoVi</title>",
            "https://eng3corp.com/",
        )
        assert _company_name(zyto, "zyto.com") == "ZYTO"
        assert _company_name(eng3, "eng3corp.com") == "eng3corp.com"

    def test_a_missing_page_that_serves_the_home_page_is_not_read_twice(self):
        home = "<html><head><title>Hoolest</title></head><body><a href='/about'>About</a><p>Hello</p></body></html>"
        routes = {"/": (home, 200), "/about": (home, 200), "/robots.txt": ("", 404)}
        result, _, _, port = _run_local(routes)
        assert result.pages_read == [f"http://127.0.0.1:{port}/"]

    def test_the_same_person_on_two_pages_is_one_person(self):
        routes = {
            "/": (
                "<a href='/about'>About</a><p>Dr. James V. Hardt, Founder and President</p>",
                200,
            ),
            "/about": ("<p>Dr James Hardt, Founder</p>", 200),
            "/robots.txt": ("", 404),
        }
        result, _, _, _ = _run_local(routes)
        assert [(p.name, p.title) for p in result.people] == [
            ("Dr. James V. Hardt", "Founder and President")
        ]


class TestSecondRealRun:
    def test_menu_labels_quotes_and_job_titles_are_not_people(self):
        html = (
            "<p>Custom Virtual Items</p><p>Practitioner training &amp; community</p>"
            "<p>Chief Executive Officer</p><p>Previously VP and Controller, Browz, LLC</p>"
            "<p>Olympic Volleyball Player</p><p>“The technology is amazing and makes me a better physician.”</p>"
        )
        assert _people(html) == []

    def test_the_brand_is_not_a_person(self):
        page = parse_page(
            "<p>Shiftwave Pro</p><p>Official brand tagline.</p><p>Werden Sie Airnergy-Partner</p><p>Partner &amp; Login</p>",
            "https://shiftwave.co/",
        )
        assert extract_people(page, brand="shiftwave") == []
        assert extract_people(page, brand="airnergy") == []

    def test_endorsers_from_other_companies_are_kept_apart(self):
        from forge.contacts.people import names_another_company

        for title in (
            "Exos President and Founder",
            "NFL Players Inc. VP of Business",
            "THP Performance Director & Founder",
            "CEO of Kwik Learning, speaker, and podcaster",
        ):
            assert names_another_company(title, "shiftwave"), title
        for title in (
            "Founder & Medical Director",
            "Co-founder",
            "Global Communications Lead",
            "VP of Business Development",
            "Head of Marketing",
            "CEO",
        ):
            assert not names_another_company(title, "shiftwave"), title
        people = extract_people(
            parse_page(
                "<p>Mark Verstegen</p><p>Exos President and Founder</p><p>Will McNulty</p><p>Co-founder</p>",
                "https://shiftwave.co/",
            ),
            brand="shiftwave",
        )
        assert [(p.name, p.role) for p in people] == [
            ("Mark Verstegen", "another company"),
            ("Will McNulty", "leadership"),
        ]

    def test_a_subject_alone_is_a_title_only_when_short(self):
        from forge.contacts.people import is_title

        assert is_title("Marketing")
        assert is_title("PR & Communications")
        assert not is_title("Practitioner training & community")
        assert is_title("Global Communications Lead")


class TestThirdRealRun:
    def test_blog_slugs_that_mention_people_are_not_people_pages(self):
        from forge.contacts.archive import page_kind

        assert (
            page_kind("/about/blog/there-are-people-who-win-in-life-and-people-who-win-mma-fights")
            is None
        )
        assert page_kind("/find-a-practitioner") == 4
        assert page_kind("/policies/legal-notice") == 0
        assert page_kind("/pages/als-partner-bewerben") == 4
        assert page_kind("/pages/partnerprogramm-login") == 4
        assert page_kind("/about-us.html") == 1

    def test_a_comment_form_is_not_a_contact_form(self):
        page = parse_page(
            '<form action="https://x.com/wp-comments-post.php"><input type="email" name="email">'
            '<textarea name="comment"></textarea></form>',
            "https://x.com/blog/post",
        )
        assert page.has_contact_form is False

    def test_the_contact_page_form_wins(self):
        form = '<form><input type="email" name="email"><textarea name="message"></textarea></form>'
        routes = {
            "/": ("<a href='/partners'>Partners</a><a href='/contact'>Contact</a>", 200),
            "/partners": ("<p>Apply</p>" + form, 200),
            "/contact": ("<p>Write to us</p>" + form, 200),
            "/robots.txt": ("", 404),
        }
        result, _, _, port = _run_local(routes)
        assert result.contact_form == f"http://127.0.0.1:{port}/contact"
