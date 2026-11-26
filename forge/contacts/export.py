"""Write contacts results as spreadsheets (one row per company, or one row per person) or JSON."""

from __future__ import annotations

import csv
import json
from typing import Any, Dict, List, Sequence

from forge.contacts.finder import CompanyContacts
from forge.contacts.people import OUTSIDE, Person

COMPANY_COLUMNS = [
    "Company",
    "Website",
    "Status",
    "Best way in",
    "Marketing person",
    "Leaders",
    "Other people",
    "Inboxes",
    "Phone",
    "Contact form",
    "Partner or affiliate program",
    "LinkedIn",
    "Other social",
    "Email format",
    "Notes",
    "Pages read",
]

PEOPLE_COLUMNS = ["Company", "Website", "Name", "Title", "Role", "Email", "Email basis", "Found on"]


def _person(p: Person) -> str:
    text = f"{p.name} ({p.title})"
    if p.role == OUTSIDE:
        return text + " [title names another company]"
    if p.email:
        text += f" {p.email}"
        if p.email_basis.startswith("guessed"):
            text += " [built from the email format]"
    return text


def company_row(r: CompanyContacts) -> Dict[str, str]:
    marketing = r.marketing_contact
    others = [p for p in r.people if p is not marketing and p not in r.leaders]
    partner = "; ".join(r.partner_pages + [f"runs on {name}" for name in r.partner_platforms])
    socials = {k: v for k, v in r.socials.items() if k != "linkedin"}
    fmt = (
        f"{r.email_format}@{r.mail_domain} (seen in {r.email_format_from})"
        if r.email_format
        else ""
    )
    return {
        "Company": r.company,
        "Website": r.website,
        "Status": r.status,
        "Best way in": r.best_way_in(),
        "Marketing person": _person(marketing) if marketing else "none listed publicly",
        "Leaders": "; ".join(_person(p) for p in r.leaders),
        "Other people": "; ".join(_person(p) for p in others),
        "Inboxes": "; ".join(r.inboxes),
        "Phone": "; ".join(r.phones[:3])
        + (f" (+{len(r.phones) - 3} more)" if len(r.phones) > 3 else ""),
        "Contact form": r.contact_form,
        "Partner or affiliate program": partner,
        "LinkedIn": r.socials.get("linkedin", ""),
        "Other social": "; ".join(f"{k}: {v}" for k, v in socials.items()),
        "Email format": fmt,
        "Notes": "; ".join(r.notes),
        "Pages read": "; ".join(r.pages_read),
    }


def write_companies_csv(results: Sequence[CompanyContacts], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPANY_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(company_row(r))


def write_people_csv(results: Sequence[CompanyContacts], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PEOPLE_COLUMNS)
        writer.writeheader()
        for r in results:
            for p in r.people:
                writer.writerow(
                    {
                        "Company": r.company,
                        "Website": r.website,
                        "Name": p.name,
                        "Title": p.title,
                        "Role": p.role,
                        "Email": p.email,
                        "Email basis": p.email_basis,
                        "Found on": p.source,
                    }
                )


def to_json(results: Sequence[CompanyContacts]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        out.append(
            {
                "company": r.company,
                "website": r.website,
                "status": r.status,
                "best_way_in": r.best_way_in(),
                "people": [vars(p) for p in r.people],
                "inboxes": [{"email": e, "found_on": r.emails.get(e, "")} for e in r.inboxes],
                "phones": r.phones,
                "contact_form": r.contact_form,
                "partner_pages": r.partner_pages,
                "partner_platforms": r.partner_platforms,
                "socials": r.socials,
                "email_format": r.email_format,
                "email_format_from": r.email_format_from,
                "notes": r.notes,
                "pages_read": r.pages_read,
            }
        )
    return out


def write_json(results: Sequence[CompanyContacts], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_json(results), fh, indent=2, ensure_ascii=False)


__all__ = [
    "COMPANY_COLUMNS",
    "PEOPLE_COLUMNS",
    "company_row",
    "to_json",
    "write_companies_csv",
    "write_json",
    "write_people_csv",
]
