"""forge contacts: who to reach at a company, and how, from its own public pages.

from forge.contacts import run_contacts
results = run_contacts(["zyto.com", "https://eng3corp.com"])
results[0].best_way_in()
"""

from forge.contacts.finder import CompanyContacts, find_contacts, run_contacts
from forge.contacts.people import Person

__all__ = ["CompanyContacts", "Person", "find_contacts", "run_contacts"]
