"""Tests for forge.discovery."""

import pytest

from forge.discovery import overture
from forge.discovery.overture import (
    _CATEGORY_TO_INDUSTRY,
    _INDUSTRY_TO_CATEGORIES,
    MILES_TO_DEGREES,
    OvertureDiscovery,
    OvertureDiscoveryError,
)
from forge.discovery.zip_centroids import get_zip_centroid

# ---------------------------------------------------------------------------
# Tests: get_zip_centroid
# ---------------------------------------------------------------------------


class TestGetZipCentroid:
    def test_known_zip_tampa(self):
        result = get_zip_centroid("33602")
        assert result is not None
        assert result["city"] == "Tampa"
        assert result["state"] == "FL"
        assert abs(result["lat"] - 27.9517) < 0.01
        assert abs(result["lon"] - (-82.4588)) < 0.01

    def test_known_zip_nyc(self):
        result = get_zip_centroid("10001")
        assert result is not None
        assert result["city"] == "New York"
        assert result["state"] == "NY"

    def test_known_zip_chicago(self):
        result = get_zip_centroid("60601")
        assert result is not None
        assert result["city"] == "Chicago"
        assert result["state"] == "IL"

    def test_known_zip_la(self):
        result = get_zip_centroid("90001")
        assert result is not None
        assert result["city"] == "Los Angeles"
        assert result["state"] == "CA"

    def test_known_zip_sf(self):
        result = get_zip_centroid("94102")
        assert result is not None
        assert result["city"] == "San Francisco"
        assert result["state"] == "CA"

    def test_known_zip_houston(self):
        result = get_zip_centroid("77001")
        assert result is not None
        assert result["city"] == "Houston"
        assert result["state"] == "TX"

    def test_known_zip_phoenix(self):
        result = get_zip_centroid("85001")
        assert result is not None
        assert result["city"] == "Phoenix"
        assert result["state"] == "AZ"

    def test_known_zip_miami(self):
        result = get_zip_centroid("33101")
        assert result is not None
        assert result["city"] == "Miami"
        assert result["state"] == "FL"

    def test_known_zip_dc(self):
        result = get_zip_centroid("20001")
        assert result is not None
        assert result["city"] == "Washington"
        assert result["state"] == "DC"

    def test_known_zip_atlanta(self):
        result = get_zip_centroid("30301")
        assert result is not None
        assert result["city"] == "Atlanta"
        assert result["state"] == "GA"

    def test_unknown_zip_returns_none(self):
        result = get_zip_centroid("00000")
        assert result is None

    def test_unknown_rural_zip(self):
        result = get_zip_centroid("99999")
        assert result is None

    def test_zero_padded(self):
        # Should pad to 5 digits
        result = get_zip_centroid("10001")
        assert result is not None

    def test_numeric_input(self):
        # Accepts numeric string
        result = get_zip_centroid("33602")
        assert result is not None

    def test_result_keys(self):
        result = get_zip_centroid("33602")
        assert set(result.keys()) == {"lat", "lon", "city", "state"}


# ---------------------------------------------------------------------------
# Tests: Overture Maps category mapping
# ---------------------------------------------------------------------------


class TestCategoryMapping:
    def test_restaurant_maps_to_restaurant(self):
        assert _CATEGORY_TO_INDUSTRY["restaurant"] == "restaurant"

    def test_dentist_maps_to_healthcare(self):
        assert _CATEGORY_TO_INDUSTRY["dental_clinic"] == "healthcare"

    def test_hair_salon_maps_to_beauty(self):
        assert _CATEGORY_TO_INDUSTRY["hair_salon"] == "beauty"

    def test_gym_maps_to_fitness(self):
        assert _CATEGORY_TO_INDUSTRY["gym"] == "fitness"

    def test_lawyer_maps_to_legal(self):
        assert _CATEGORY_TO_INDUSTRY["attorney_or_law_firm"] == "legal"

    def test_plumber_maps_to_home_services(self):
        assert _CATEGORY_TO_INDUSTRY["plumbing"] == "home_services"

    def test_hotel_maps_to_hospitality(self):
        assert _CATEGORY_TO_INDUSTRY["hotel"] == "hospitality"

    def test_reverse_mapping_completeness(self):
        """Every FORGE industry should have at least one category."""
        all_industries = set(_CATEGORY_TO_INDUSTRY.values())
        for ind in all_industries:
            assert ind in _INDUSTRY_TO_CATEGORIES
            assert len(_INDUSTRY_TO_CATEGORIES[ind]) >= 1

    def test_category_count(self):
        """Should have a substantial number of category mappings."""
        assert len(_CATEGORY_TO_INDUSTRY) > 40


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_miles_to_degrees_reasonable(self):
        # At mid-latitudes, 1 mile ~ 0.0145 degrees
        assert 0.01 < MILES_TO_DEGREES < 0.02

    def test_overture_error_is_exception(self):
        with pytest.raises(OvertureDiscoveryError):
            raise OvertureDiscoveryError("test error")


# ---------------------------------------------------------------------------
# Tests: which Overture release to read
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestLatestRelease:
    def test_reads_latest_from_the_catalog(self, monkeypatch):
        monkeypatch.delenv("FORGE_OVERTURE_RELEASE", raising=False)
        monkeypatch.setattr(
            overture.httpx, "get", lambda *a, **k: _FakeResponse({"latest": "2026-10-21.0"})
        )
        assert overture.latest_release() == "2026-10-21.0"

    def test_env_pin_wins(self, monkeypatch):
        monkeypatch.setenv("FORGE_OVERTURE_RELEASE", " 2026-09-23.1 ")
        assert overture.latest_release() == "2026-09-23.1"

    def test_catalog_failure_names_the_override(self, monkeypatch):
        monkeypatch.delenv("FORGE_OVERTURE_RELEASE", raising=False)

        def boom(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(overture.httpx, "get", boom)
        with pytest.raises(OvertureDiscoveryError, match="FORGE_OVERTURE_RELEASE"):
            overture.latest_release()

    def test_rejects_a_malformed_release(self, monkeypatch):
        monkeypatch.setenv("FORGE_OVERTURE_RELEASE", "x'; DROP TABLE t; --")
        with pytest.raises(OvertureDiscoveryError):
            overture.latest_release()


# ---------------------------------------------------------------------------
# Tests: the query against Overture's current places schema
# ---------------------------------------------------------------------------


class TestCategoryFilter:
    def test_mapped_industry_matches_the_hierarchy(self):
        clause = OvertureDiscovery._build_category_filter("healthcare")
        assert "list_has_any(taxonomy.hierarchy" in clause
        assert "'dental_clinic'" in clause

    def test_raw_category_matches_the_hierarchy(self):
        clause = OvertureDiscovery._build_category_filter("Dental Clinic")
        assert clause == "AND list_contains(taxonomy.hierarchy, 'dental_clinic')"


def _write_places(path):
    """A two-place Parquet file laid out like Overture's places theme (2026 schema)."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
          SELECT * FROM (VALUES
            ('p1', {{'primary': 'Bay Dental'}},
             [{{'freeform': '1 Main St', 'locality': 'Tampa', 'postcode': '33602',
                'region': 'FL', 'country': 'US'}}],
             ['+18135550100'], ['https://baydental.example'],
             {{'primary': 'general_dentistry',
               'hierarchy': ['health_care', 'outpatient_care_facility', 'dental_clinic',
                             'general_dentistry'], 'alternates': []}},
             {{'xmin': -82.46, 'xmax': -82.46, 'ymin': 27.95, 'ymax': 27.95}}),
            ('p2', {{'primary': 'Far Away Pizza'}},
             [{{'freeform': '9 Elm St', 'locality': 'Orlando', 'postcode': '32801',
                'region': 'FL', 'country': 'US'}}],
             [], [],
             {{'primary': 'pizza_restaurant',
               'hierarchy': ['food_and_drink', 'restaurant', 'pizza_restaurant'],
               'alternates': []}},
             {{'xmin': -81.38, 'xmax': -81.38, 'ymin': 28.54, 'ymax': 28.54}})
          ) AS t(id, names, addresses, phones, websites, taxonomy, bbox)
        ) TO '{path}' (FORMAT parquet)
        """
    )
    con.close()


class TestSearchAgainstLocalParquet:
    @pytest.fixture
    def disco(self, tmp_path, monkeypatch):
        path = str(tmp_path / "places.parquet")
        _write_places(path)
        monkeypatch.setattr(OvertureDiscovery, "_setup_extensions", lambda self: None)
        d = OvertureDiscovery()
        d._places_path = path
        yield d
        d.close()

    def test_finds_the_place_in_the_box_with_db_column_names(self, disco):
        results = disco.search(latitude=27.95, longitude=-82.46, radius_miles=5)
        assert len(results) == 1
        r = results[0]
        assert r["name"] == "Bay Dental"
        assert r["website_url"] == "https://baydental.example"
        assert r["address_line1"] == "1 Main St"
        assert r["zip"] == "33602"
        assert r["industry"] == "healthcare"
        assert r["sub_industry"] == "general_dentistry"
        assert abs(r["latitude"] - 27.95) < 1e-9
        assert abs(r["longitude"] + 82.46) < 1e-9
        assert "hierarchy" not in r

    def test_industry_filter_uses_the_hierarchy(self, disco):
        assert (
            disco.search(latitude=28.54, longitude=-81.38, industry="restaurant")[0]["name"]
            == "Far Away Pizza"
        )
        assert disco.search(latitude=28.54, longitude=-81.38, industry="healthcare") == []

    def test_results_import_with_website(self, disco, sqlite_db):
        record = disco.search(latitude=27.95, longitude=-82.46, radius_miles=5)[0]
        bid = sqlite_db.upsert_business(record)
        saved = sqlite_db.get_business(bid)
        assert saved["website_url"] == "https://baydental.example"
        assert saved["address_line1"] == "1 Main St"
        assert saved["industry"] == "healthcare"
