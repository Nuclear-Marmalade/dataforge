"""
Overture Maps discovery client for FORGE.

Queries the Overture Maps places dataset (free, open GeoParquet on S3)
via DuckDB to discover businesses by location and industry.

Usage:
    from discovery.overture import OvertureDiscovery

    disco = OvertureDiscovery()
    results = disco.search(zip_code="33602", industry="restaurant", limit=500)
    disco.close()
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional, Tuple

from forge.discovery.zip_centroids import get_zip_centroid

logger = logging.getLogger(__name__)

# Overture Maps S3 release path (update when new releases land)
OVERTURE_PLACES_PATH = (
    "s3://overturemaps-us-west-2/release/2025-03-02-0/theme=places/type=place/*"
)

# Approximate conversion: 1 mile ~ 0.0145 degrees at mid-latitudes
MILES_TO_DEGREES = 0.0145

# Maximum query time in seconds
QUERY_TIMEOUT_SECONDS = 60

# Map Overture primary categories -> FORGE industry slugs
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
    "supermarket": "grocery",
    "discount_store": "retail",
    "electronics_store": "retail",
    "furniture_store": "retail",
    "hardware_store": "retail",
    "pet_store": "retail",
    "bookstore": "retail",
    "toy_store": "retail",
    # Health & Medical
    "doctor": "healthcare",
    "dentist": "healthcare",
    "hospital": "healthcare",
    "pharmacy": "healthcare",
    "optician": "healthcare",
    "veterinarian": "veterinary",
    "physiotherapist": "healthcare",
    "chiropractor": "healthcare",
    # Professional Services
    "lawyer": "legal",
    "accountant": "accounting",
    "insurance_agency": "insurance",
    "real_estate_agency": "real_estate",
    "bank": "finance",
    "financial_advisor": "finance",
    # Automotive
    "car_dealer": "automotive",
    "car_repair": "automotive",
    "car_wash": "automotive",
    "gas_station": "automotive",
    # Beauty & Personal Care
    "hair_salon": "beauty",
    "beauty_salon": "beauty",
    "spa": "beauty",
    "nail_salon": "beauty",
    "barber_shop": "beauty",
    # Fitness & Recreation
    "gym": "fitness",
    "fitness_center": "fitness",
    "yoga_studio": "fitness",
    "sports_club": "fitness",
    # Accommodation
    "hotel": "hospitality",
    "motel": "hospitality",
    "bed_and_breakfast": "hospitality",
    "resort": "hospitality",
    # Home Services
    "plumber": "home_services",
    "electrician": "home_services",
    "locksmith": "home_services",
    "moving_company": "home_services",
    "storage_facility": "home_services",
    # Education
    "school": "education",
    "university": "education",
    "tutoring_service": "education",
    "driving_school": "education",
    # Entertainment
    "movie_theater": "entertainment",
    "night_club": "entertainment",
    "bowling_alley": "entertainment",
    "amusement_park": "entertainment",
}

# Reverse lookup: FORGE industry -> list of Overture categories
_INDUSTRY_TO_CATEGORIES: dict[str, list[str]] = {}
for _cat, _ind in _CATEGORY_TO_INDUSTRY.items():
    _INDUSTRY_TO_CATEGORIES.setdefault(_ind, []).append(_cat)


class OvertureDiscoveryError(Exception):
    """Base error for Overture discovery issues."""


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
        self._setup_extensions()

    def _setup_extensions(self):
        """Install and load required DuckDB extensions."""
        logger.info("Setting up DuckDB extensions (httpfs, spatial)...")
        try:
            self._conn.execute("INSTALL httpfs; LOAD httpfs;")
            self._conn.execute("INSTALL spatial; LOAD spatial;")
            self._conn.execute("SET s3_region='us-west-2';")
            # Anonymous access -- Overture data is public
            self._conn.execute("SET s3_access_key_id='';")
            self._conn.execute("SET s3_secret_access_key='';")
        except Exception as exc:
            raise OvertureDiscoveryError(
                f"Failed to initialize DuckDB extensions: {exc}. "
                "Check your internet connection."
            )
        logger.info("DuckDB extensions ready.")

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
        """
        Search Overture Maps for businesses near a location.

        Provide EITHER zip_code OR (latitude, longitude). City/state are
        informational and used for logging only when lat/lon are given.

        Args:
            zip_code:      5-digit US ZIP code (geocoded automatically).
            city:          City name (informational; not used for filtering).
            state:         State abbreviation (informational; not used for filtering).
            latitude:      Center latitude for the search.
            longitude:     Center longitude for the search.
            radius_miles:  Search radius in miles (default 10).
            industry:      Optional FORGE industry slug to filter by.
            limit:         Max number of results (default 1000).

        Returns:
            List of dicts, each with keys:
                name, address, city, state, zip, lat, lon,
                phone, website, category, overture_id

        Raises:
            OvertureDiscoveryError: On invalid input, network, or timeout issues.
        """
        # --- Resolve location ---
        if latitude is not None and longitude is not None:
            lat, lon = latitude, longitude
        elif zip_code:
            lat, lon = self.geocode_zip(zip_code)
            logger.info("Geocoded ZIP %s -> (%.4f, %.4f)", zip_code, lat, lon)
        else:
            raise OvertureDiscoveryError(
                "Provide either zip_code or (latitude, longitude)."
            )

        # --- Compute bounding box ---
        delta = radius_miles * MILES_TO_DEGREES
        min_lat = lat - delta
        max_lat = lat + delta
        min_lon = lon - delta
        max_lon = lon + delta

        # --- Build category filter ---
        category_filter = ""
        if industry:
            # Check known industries first
            if industry.lower() in _INDUSTRY_TO_CATEGORIES:
                cats = _INDUSTRY_TO_CATEGORIES[industry.lower()]
                cat_list = ", ".join(f"'{c}'" for c in cats)
                category_filter = (
                    f"AND categories.primary IN ({cat_list})"
                )
                logger.info(
                    "Filtering to industry '%s' -> categories: %s",
                    industry,
                    cats,
                )
            else:
                # Sanitize unknown values
                safe_industry = re.sub(r'[^a-zA-Z0-9_\-\s]', '', industry.lower().strip())
                if not safe_industry:
                    logger.warning(
                        "Industry '%s' rejected — invalid characters.", industry
                    )
                    return []
                category_filter = (
                    f"AND LOWER(categories.primary) = '{safe_industry}'"
                )
                logger.info(
                    "Industry '%s' has no mapping; filtering raw category (sanitized: '%s').",
                    industry,
                    safe_industry,
                )

        # --- Build SQL ---
        sql = f"""
        SELECT
            id                                          AS overture_id,
            names.primary                               AS name,
            addresses[1].freeform                       AS address,
            addresses[1].locality                       AS city,
            addresses[1].region                         AS state,
            addresses[1].postcode                       AS zip,
            ST_Y(ST_GeomFromWKB(geometry))              AS lat,
            ST_X(ST_GeomFromWKB(geometry))              AS lon,
            CASE WHEN phones IS NOT NULL AND len(phones) > 0
                 THEN phones[1] ELSE NULL END           AS phone,
            CASE WHEN websites IS NOT NULL AND len(websites) > 0
                 THEN websites[1] ELSE NULL END         AS website,
            categories.primary                          AS category
        FROM read_parquet('{OVERTURE_PLACES_PATH}', filename=true, hive_partitioning=1)
        WHERE
            ST_Y(ST_GeomFromWKB(geometry)) BETWEEN {min_lat} AND {max_lat}
            AND ST_X(ST_GeomFromWKB(geometry)) BETWEEN {min_lon} AND {max_lon}
            {category_filter}
        LIMIT {limit}
        """

        # --- Execute with timeout ---
        label = f"({lat:.4f}, {lon:.4f}) r={radius_miles}mi"
        if industry:
            label += f" industry={industry}"
        logger.info("Querying Overture Maps: %s ...", label)

        start = time.time()
        try:
            result = self._conn.execute(sql)
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]
        except Exception as exc:
            elapsed = time.time() - start
            err_str = str(exc).lower()
            if elapsed >= QUERY_TIMEOUT_SECONDS or "timeout" in err_str:
                raise OvertureDiscoveryError(
                    f"Query timed out after {elapsed:.0f}s. "
                    "Try a smaller radius or more specific industry."
                )
            if "unable to connect" in err_str or "http" in err_str:
                raise OvertureDiscoveryError(
                    "Could not reach Overture Maps. Check internet connection."
                )
            raise OvertureDiscoveryError(f"DuckDB query failed: {exc}")

        elapsed = time.time() - start
        logger.info(
            "Overture query returned %d rows in %.1fs", len(rows), elapsed
        )

        if not rows:
            logger.warning("No businesses found matching criteria: %s", label)
            return []

        # --- Convert to dicts ---
        results: List[dict] = []
        for row in rows:
            record = dict(zip(columns, row))
            # Map Overture category to FORGE industry
            raw_cat = record.get("category") or ""
            record["forge_industry"] = _CATEGORY_TO_INDUSTRY.get(
                raw_cat.lower(), None
            )
            results.append(record)

        return results

    def close(self):
        """Close the DuckDB connection and release resources."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            logger.info("DuckDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
