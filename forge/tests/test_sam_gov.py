"""Tests for forge.importers.sam_gov."""

from forge.importers.sam_gov import (
    _build_params,
    _extract_entity,
    _process_entities,
    normalize_name,
)

# ---------------------------------------------------------------------------
# Tests: normalize_name
# ---------------------------------------------------------------------------


class TestSamNormalizeName:
    def test_removes_llc(self):
        assert "LLC" not in normalize_name("Acme Solutions LLC")

    def test_removes_inc(self):
        assert "INC" not in normalize_name("Tech Corp Inc")

    def test_removes_corp(self):
        assert "CORP" not in normalize_name("Global Corp")

    def test_uppercase_and_strip(self):
        result = normalize_name("  hello world  ")
        assert result == "HELLO WORLD"

    def test_preserves_core_name(self):
        result = normalize_name("Tampa Bay Dental")
        assert result == "TAMPA BAY DENTAL"


# ---------------------------------------------------------------------------
# Tests: _build_params
# ---------------------------------------------------------------------------


class TestBuildParams:
    def test_basic_params(self):
        params = _build_params(page=0, page_size=100)
        assert params["page"] == "0"
        assert params["size"] == "100"
        assert params["registrationStatus"] == "A"
        assert "stateCode" not in params

    def test_with_state_filter(self):
        params = _build_params(page=1, state_filter="CA")
        assert params["stateCode"] == "CA"

    def test_state_filter_uppercased(self):
        params = _build_params(page=0, state_filter="fl")
        assert params["stateCode"] == "FL"

    def test_includes_sections(self):
        params = _build_params(page=0)
        assert "entityRegistration" in params["includeSections"]
        assert "coreData" in params["includeSections"]
        assert "pointsOfContact" in params["includeSections"]


# ---------------------------------------------------------------------------
# Tests: _extract_entity
# ---------------------------------------------------------------------------


class TestExtractEntity:
    def _make_entity(
        self,
        org_name="ACME CORP",
        state="FL",
        city="TAMPA",
        zip_code="33602",
        email="info@acme.com",
        first="John",
        last="Smith",
    ):
        return {
            "entityRegistration": {
                "legalBusinessName": org_name,
            },
            "coreData": {
                "physicalAddress": {
                    "stateOrProvinceCode": state,
                    "city": city,
                    "zipCode": zip_code,
                },
                "naicsCode": [
                    {"naicsCode": "541511"},
                    {"naicsCode": "541512"},
                ],
            },
            "pointsOfContact": {
                "governmentBusinessPOC": {
                    "email": email,
                    "firstName": first,
                    "lastName": last,
                },
            },
        }

    def test_extracts_valid_entity(self):
        entity = self._make_entity()
        result = _extract_entity(entity)
        assert result is not None
        assert result["org_name"] == "ACME CORP"
        assert result["state"] == "FL"
        assert result["city"] == "TAMPA"
        assert result["poc_email"] == "info@acme.com"
        assert result["poc_name"] == "John Smith"
        assert "541511" in result["naics_codes"]

    def test_normalizes_name(self):
        entity = self._make_entity(org_name="Acme Solutions LLC")
        result = _extract_entity(entity)
        assert result["name_normalized"] == "ACME SOLUTIONS"

    def test_returns_none_without_org_name(self):
        entity = self._make_entity(org_name="")
        result = _extract_entity(entity)
        assert result is None

    def test_returns_none_without_state(self):
        entity = self._make_entity(state="")
        result = _extract_entity(entity)
        assert result is None

    def test_returns_none_without_email(self):
        entity = self._make_entity(email="")
        result = _extract_entity(entity)
        assert result is None

    def test_email_lowercased(self):
        entity = self._make_entity(email="Info@ACME.COM")
        result = _extract_entity(entity)
        assert result["poc_email"] == "info@acme.com"

    def test_city_uppercased(self):
        entity = self._make_entity(city="tampa")
        result = _extract_entity(entity)
        assert result["city"] == "TAMPA"

    def test_zip_truncated_to_5(self):
        entity = self._make_entity(zip_code="33602-1234")
        result = _extract_entity(entity)
        assert result["zip_code"] == "33602"

    def test_fallback_to_electronic_poc(self):
        entity = {
            "entityRegistration": {"legalBusinessName": "Test Corp"},
            "coreData": {
                "physicalAddress": {
                    "stateOrProvinceCode": "CA",
                    "city": "LA",
                    "zipCode": "90001",
                },
                "naicsCode": [],
            },
            "pointsOfContact": {
                "governmentBusinessPOC": {},
                "electronicBusinessPOC": {
                    "email": "electronic@test.com",
                    "firstName": "Jane",
                    "lastName": "Doe",
                },
            },
        }
        result = _extract_entity(entity)
        assert result is not None
        assert result["poc_email"] == "electronic@test.com"
        assert result["poc_name"] == "Jane Doe"

    def test_naics_codes_as_strings(self):
        entity = self._make_entity()
        entity["coreData"]["naicsCode"] = ["541511", "541512"]
        result = _extract_entity(entity)
        assert "541511" in result["naics_codes"]

    def test_naics_codes_as_ints(self):
        entity = self._make_entity()
        entity["coreData"]["naicsCode"] = [541511]
        result = _extract_entity(entity)
        assert "541511" in result["naics_codes"]

    def test_empty_naics(self):
        entity = self._make_entity()
        entity["coreData"]["naicsCode"] = []
        result = _extract_entity(entity)
        assert result["naics_codes"] == []

    def test_poc_with_middle_initial(self):
        entity = self._make_entity()
        entity["pointsOfContact"]["governmentBusinessPOC"]["middleInitial"] = "Q"
        result = _extract_entity(entity)
        assert result["poc_name"] == "John Q Smith"

    def test_poc_email_must_contain_at(self):
        entity = self._make_entity(email="nope")
        result = _extract_entity(entity)
        assert result is None

    def test_empty_entity(self):
        result = _extract_entity({})
        assert result is None

    def test_missing_core_data(self):
        entity = {
            "entityRegistration": {"legalBusinessName": "Test"},
            "coreData": {},
            "pointsOfContact": {
                "governmentBusinessPOC": {"email": "a@b.com"},
            },
        }
        result = _extract_entity(entity)
        # No state -> returns None
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _process_entities (a name match must be at the same location)
# ---------------------------------------------------------------------------


def _sam_entity(city, zip_code, org_name="ACME CORP", email="poc@acme.com"):
    return {
        "entityRegistration": {"legalBusinessName": org_name},
        "coreData": {
            "physicalAddress": {"stateOrProvinceCode": "FL", "city": city, "zipCode": zip_code}
        },
        "pointsOfContact": {
            "governmentBusinessPOC": {"email": email, "firstName": "Jo", "lastName": "Lee"}
        },
    }


def _sam_stats():
    return {"entities_with_email": 0, "matches_found": 0}


class TestProcessEntities:
    def _index(self):
        return {
            "ACME|FL": [
                {"id": "miami", "name": "Acme", "state": "FL", "city": "MIAMI", "zip": "33101"},
                {"id": "tampa", "name": "Acme", "state": "FL", "city": "TAMPA", "zip": "33602"},
            ]
        }

    def test_contact_goes_to_the_business_at_the_same_zip(self):
        batch: list = []
        stats = _sam_stats()
        _process_entities([_sam_entity("TAMPA", "33602")], self._index(), batch, stats)
        assert batch == [("poc@acme.com", "Jo Lee", "poc@acme.com", "tampa")]
        assert stats["matches_found"] == 1

    def test_same_name_in_another_city_gets_nothing(self):
        batch: list = []
        stats = _sam_stats()
        _process_entities([_sam_entity("ORLANDO", "32801")], self._index(), batch, stats)
        assert batch == []
        assert stats["matches_found"] == 0

    def test_each_business_written_once(self):
        batch: list = []
        index = self._index()
        entity = _sam_entity("TAMPA", "33602")
        _process_entities([entity, entity], index, batch, _sam_stats())
        assert len(batch) == 1
        assert [b["id"] for b in index["ACME|FL"]] == ["miami"]
