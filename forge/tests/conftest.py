"""Shared test fixtures for the FORGE test suite."""
import csv
import pytest

from forge.db import ForgeDB


@pytest.fixture
def sqlite_db(tmp_path):
    """Fresh SQLite database for each test."""
    db_path = str(tmp_path / "test.db")
    db = ForgeDB.from_config({"db_path": db_path})
    db.ensure_schema()
    yield db
    db.close()


@pytest.fixture
def sample_business():
    """A typical business record."""
    return {
        "name": "Tampa Bay Dental",
        "city": "Tampa",
        "state": "FL",
        "website_url": "https://tampabaydental.com",
        "phone": "8135551234",
        "email": "info@tampabaydental.com",
    }


@pytest.fixture
def sample_csv(tmp_path):
    """A small test CSV file."""
    path = tmp_path / "test_leads.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Business Name", "City", "State", "Website", "Phone"])
        w.writerow(["Pizza Palace", "Tampa", "FL", "https://pizza.com", "8135551111"])
        w.writerow(["Salon Bella", "Miami", "FL", "https://salon.com", "3055552222"])
        w.writerow(["Austin BBQ", "Austin", "TX", "https://bbq.com", "5125553333"])
    return str(path)
