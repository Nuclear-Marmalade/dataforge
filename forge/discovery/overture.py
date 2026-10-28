"""
Overture Maps discovery client for FORGE.

Queries the Overture Maps places dataset (free, open GeoParquet on S3)
via DuckDB to discover businesses by location and industry. Results use the
businesses table's column names, so they can be passed to upsert_business as-is.

Usage:
    from discovery.overture import OvertureDiscovery

    disco = OvertureDiscovery()
    results = disco.search(zip_code="33602", industry="restaurant", limit=500)
    disco.close()
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import List, Optional, Tuple

import httpx

from forge.discovery.zip_centroids import get_zip_centroid

logger = logging.getLogger(__name__)

# Overture keeps only the last few releases on S3, so the release is looked up at run
# time from its STAC catalog (or pinned with FORGE_OVERTURE_RELEASE), never hardcoded.
OVERTURE_RELEASE_BUCKET = "s3://overturemaps-us-west-2/release"
OVERTURE_STAC_CATALOG = "https://stac.overturemaps.org/catalog.json"
_RELEASE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[.-]\d+$")

# Approximate conversion: 1 mile ~ 0.0145 degrees at mid-latitudes
MILES_TO_DEGREES = 0.0145

# Maximum query time in seconds
QUERY_TIMEOUT_SECONDS = 60

# Map Overture taxonomy categories -> FORGE industry slugs. A place matches a category
# anywhere in its taxonomy.hierarchy, so "dental_clinic" also covers general_dentistry,
# pediatric_dentistry and the rest of that branch. Names checked against release
# 2026-10-21.0.
_CATEGORY_TO_INDUSTRY: dict[str, str] = {
    # Food & Drink
    "restaurant": "restaurant",
    "fast_food_restaurant": "restaurant",
    "bar": "bar",
    "pub": "bar",
    "cafe": "cafe",
    "coffee_shop": "cafe",
    "bakery": "bakery",
    "pizza_restaurant": "restaurant",
    "ice_cream_shop": "restaurant",
    "food_court": "restaurant",
    # Retail
    "clothing_store": "retail",
    "shoe_store": "retail",
    "jewelry_store": "retail",
    "department_store": "retail",
    "shopping_mall": "retail",
    "convenience_store": "retail",
    "grocery_store": "grocery",
    "discount_store": "retail",
    "electronics_store": "retail",
    "furniture_store": "retail",
    "hardware_store": "retail",
    "pet_store": "retail",
    "bookstore": "retail",
    "toy_store": "retail",
    # Health & Medical
    "doctors_office": "healthcare",
    "primary_care_or_general_clinic": "healthcare",
    "dental_clinic": "healthcare",
    "hospital": "healthcare",
    "pharmacy": "healthcare",
    "vision_or_eye_care_clinic": "healthcare",
    "veterinarian": "veterinary",
    "physical_therapy": "healthcare",
    "chiropractic": "healthcare",
    # Professional Services
    "attorney_or_law_firm": "legal",
    "accountant": "accounting",
    "insurance_agency": "insurance",
    "real_estate_service": "real_estate",
    "bank": "finance",
    "financial_advising": "finance",
    # Automotive
    "auto_dealer": "automotive",
    "automotive_repair": "automotive",
    "car_wash": "automotive",
    "gas_station": "automotive",
    # Beauty & Personal Care
    "hair_salon": "beauty",
    "beauty_salon": "beauty",
    "spa": "beauty",
    "nail_salon": "beauty",
    "barber": "beauty",
    # Fitness & Recreation
    "gym": "fitness",
    "sport_or_fitness_facility": "fitness",
    "yoga_studio": "fitness",
    "sport_or_recreation_club": "fitness",
    # Accommodation
    "hotel": "hospitality",
    "motel": "hospitality",
    "bed_and_breakfast": "hospitality",
    "resort": "hospitality",
    # Home Services
    "plumbing": "home_services",
    "electrician": "home_services",
    "key_and_locksmith": "home_services",
    "mover": "home_services",
    "storage_facility": "home_services",
    # Education
    "school": "education",
    "college_university": "education",
    "tutoring_service": "education",
    "driving_school": "education",
    # Entertainment
    "movie_theater": "entertainment",
    "nightlife_venue": "entertainment",
    "bowling_alley": "entertainment",
    "amusement_park": "entertainment",
}

# Reverse lookup: FORGE industry -> list of Overture categories
_INDUSTRY_TO_CATEGORIES: dict[str, list[str]] = {}
for _cat, _ind in _CATEGORY_TO_INDUSTRY.items():
    _INDUSTRY_TO_CATEGORIES.setdefault(_ind, []).append(_cat)


class OvertureDiscoveryError(Exception):
    """Base error for Overture discovery issues."""


def latest_release() -> str:
    """The Overture release to read: FORGE_OVERTURE_RELEASE if set, else the catalog's latest."""
    release = os.environ.get("FORGE_OVERTURE_RELEASE", "").strip()
    if not release:
        try:
            resp = httpx.get(OVERTURE_STAC_CATALOG, timeout=15.0)
            resp.raise_for_status()
            release = str(resp.json().get("latest") or "")
        except Exception as exc:  # Catch-and-reraise: say how to work around it
            raise OvertureDiscoveryError(
                f"Could not look up the latest Overture release ({exc}). "
                "Set FORGE_OVERTURE_RELEASE, e.g. 2026-10-21.0."
            ) from exc
    if not _RELEASE_RE.match(release):
        raise OvertureDiscoveryError(
            f"Not an Overture release name: {release!r}. Set FORGE_OVERTURE_RELEASE, "
            "e.g. 2026-10-21.0."
        )
    return release


def places_path(release: str) -> str:
    """S3 glob for one release's places files."""
    return f"{OVERTURE_RELEASE_BUCKET}/{release}/theme=places/type=place/*"


class OvertureDiscovery:
    """
    Discovers businesses from the Overture Maps open dataset via DuckDB.

    DuckDB queries remote GeoParquet files on S3 directly -- no bulk download
    required. First queries may be slow while DuckDB fetches Parquet metadata.
    """

    def __init__(self):
        """Initialize DuckDB with spatial and httpfs extensions."""
        try:
            import duckdb
        except ImportError:
            raise OvertureDiscoveryError(
                "DuckDB is required for Overture Maps discovery. "
                "Install it with: pip install duckdb"
            )

        self._conn = duckdb.connect(database=":memory:")
        self._places_path: Optional[str] = None  # resolved on first search
        self._setup_extensions()

    def _setup_extensions(self):
        """Install and load the DuckDB extension for reading S3 (no spatial needed: the
        query filters and positions on the bbox column)."""
        logger.info("Setting up DuckDB extension (httpfs)...")
        try:
            self._conn.execute("INSTALL httpfs; LOAD httpfs;")
            self._conn.execute("SET s3_region='us-west-2';")
            # Anonymous access -- Overture data is public
            self._conn.execute("SET s3_access_key_id='';")
            self._conn.execute("SET s3_secret_access_key='';")
        except Exception as exc:  # Catch-and-reraise: wrap in domain-specific error
            raise OvertureDiscoveryError(
                f"Failed to initialize DuckDB extensions: {exc}. Check your internet connection."
            )
        logger.info("DuckDB extension ready.")

    def geocode_zip(self, zip_code: str) -> Tuple[float, float]:
        """
        Convert a US ZIP code to (latitude, longitude).

        Uses the bundled centroid database for fast lookup.

        Args:
            zip_code: 5-digit US ZIP code.

        Returns:
            Tuple of (latitude, longitude).

        Raises:
            OvertureDiscoveryError: If the ZIP code is unknown.
        """
        result = get_zip_centroid(zip_code)
        if result is None:
            raise OvertureDiscoveryError(
                f"ZIP code not found in database: {zip_code}. "
                "Try providing latitude/longitude directly."
            )
        return result["lat"], result["lon"]

    def _resolve_location(
        self,
        zip_code: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> tuple:
        """Resolve search inputs to (lat, lon) coordinates."""
        if latitude is not None and longitude is not None:
            return latitude, longitude
        elif zip_code:
            lat, lon = self.geocode_zip(zip_code)
            logger.info("Geocoded ZIP %s -> (%.4f, %.4f)", zip_code, lat, lon)
            return lat, lon
        else:
            raise OvertureDiscoveryError("Provide either zip_code or (latitude, longitude).")

    @staticmethod
    def _build_category_filter(industry: Optional[str]) -> str:
        """Build a SQL category filter clause from a FORGE industry slug."""
        if not industry:
            return ""

        if industry.lower() in _INDUSTRY_TO_CATEGORIES:
            cats = _INDUSTRY_TO_CATEGORIES[industry.lower()]
            cat_list = ", ".join(f"'{c}'" for c in cats)
            logger.info("Filtering to industry '%s' -> categories: %s", industry, cats)
            return f"AND list_has_any(taxonomy.hierarchy, [{cat_list}])"

        # Overture category names are snake_case: "Dental Clinic" -> dental_clinic.
        safe_industry = re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s\-]+", "_", industry.lower().strip()))
        if not safe_industry:
            logger.warning("Industry '%s' rejected, invalid characters.", industry)
            return ""
        logger.info(
            "Industry '%s' has no mapping; filtering raw category (sanitized: '%s').",
            industry,
            safe_industry,
        )
        return f"AND list_contains(taxonomy.hierarchy, '{safe_industry}')"

    @staticmethod
    def _build_overture_sql(
        path: str,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        category_filter: str,
        limit: int,
    ) -> str:
        """Build the DuckDB SQL query for Overture Maps (column names match businesses).

        Filters on bbox, which DuckDB can use to skip whole row groups; places are
        points, so the bbox centre is the location.
        """
        return f"""
        SELECT
            id                                          AS overture_id,
            names.primary                               AS name,
            addresses[1].freeform                       AS address_line1,
            addresses[1].locality                       AS city,
            addresses[1].region                         AS state,
            addresses[1].postcode                       AS zip,
            (bbox.ymin + bbox.ymax) / 2                 AS latitude,
            (bbox.xmin + bbox.xmax) / 2                 AS longitude,
            CASE WHEN phones IS NOT NULL AND len(phones) > 0
                 THEN phones[1] ELSE NULL END           AS phone,
            CASE WHEN websites IS NOT NULL AND len(websites) > 0
                 THEN websites[1] ELSE NULL END         AS website_url,
            taxonomy.primary                            AS category,
            taxonomy.hierarchy                          AS hierarchy
        FROM read_parquet('{path}', hive_partitioning=1)
        WHERE
            bbox.xmin BETWEEN {min_lon} AND {max_lon}
            AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
            {category_filter}
        LIMIT {limit}
        """

    def _execute_query(self, sql: str, label: str) -> tuple:
        """Execute a DuckDB query and return (rows, columns).

        Raises OvertureDiscoveryError on timeout or connection failure.
        """
        start = time.time()
        try:
            result = self._conn.execute(sql)
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]
        except Exception as exc:  # Catch-and-reraise: wrap in domain-specific error with context
            elapsed = time.time() - start
            err_str = str(exc).lower()
            if elapsed >= QUERY_TIMEOUT_SECONDS or "timeout" in err_str:
                raise OvertureDiscoveryError(
                    f"Query timed out after {elapsed:.0f}s. Try a smaller radius or more specific industry."
                )
            if "unable to connect" in err_str or "http" in err_str:
                raise OvertureDiscoveryError(
                    f"Could not read Overture Maps data ({exc}). Check internet connection."
                )
            raise OvertureDiscoveryError(f"DuckDB query failed: {exc}")

        elapsed = time.time() - start
        logger.info("Overture query returned %d rows in %.1fs", len(rows), elapsed)
        return rows, columns

    @staticmethod
    def _format_results(rows: list, columns: list) -> List[dict]:
        """Convert raw DuckDB rows to business records.

        industry is the FORGE slug of the most specific mapped category in the place's
        taxonomy hierarchy; sub_industry is Overture's own category name.
        """
        results: List[dict] = []
        for row in rows:
            record = dict(zip(columns, row))
            hierarchy = record.pop("hierarchy", None) or []
            record["industry"] = next(
                (
                    _CATEGORY_TO_INDUSTRY[c]
                    for c in reversed(hierarchy)
                    if c in _CATEGORY_TO_INDUSTRY
                ),
                None,
            )
            record["sub_industry"] = record.get("category")
            results.append(record)
        return results

    def search(
        self,
        zip_code: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_miles: float = 10,
        industry: Optional[str] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Search Overture Maps for businesses near a location.

        Provide EITHER zip_code OR (latitude, longitude).
        """
        lat, lon = self._resolve_location(zip_code, latitude, longitude)

        delta = radius_miles * MILES_TO_DEGREES
        category_filter = self._build_category_filter(industry)
        if industry and not category_filter:
            return []

        if self._places_path is None:
            self._places_path = places_path(latest_release())
            logger.info("Reading Overture places from %s", self._places_path)
        sql = self._build_overture_sql(
            self._places_path,
            lat - delta,
            lat + delta,
            lon - delta,
            lon + delta,
            category_filter,
            limit,
        )

        label = f"({lat:.4f}, {lon:.4f}) r={radius_miles}mi"
        if industry:
            label += f" industry={industry}"
        logger.info("Querying Overture Maps: %s ...", label)

        rows, columns = self._execute_query(sql, label)
        if not rows:
            logger.warning("No businesses found matching criteria: %s", label)
            return []

        return self._format_results(rows, columns)

    def close(self):
        """Close the DuckDB connection and release resources."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:  # Non-critical: best-effort cleanup on close
                pass
            self._conn = None
            logger.info("DuckDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
