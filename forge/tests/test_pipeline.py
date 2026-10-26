"""Tests for forge.enrichment.pipeline."""

import json
import time
from unittest.mock import MagicMock, patch

from forge.enrichment.pipeline import (
    INDUSTRY_WHITELIST,
    EnrichmentPipeline,
    EnrichmentStats,
)

# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockDB:
    """Mock database pool that records calls."""

    def __init__(self, businesses=None):
        self._businesses = businesses or []
        self._fetch_call_count = 0
        self.written = []
        self.tracked = []
        self.placeholder = "%s"
        self.now_expr = "NOW()"
        self.is_postgres = True
        self._transaction_calls = []

    def fetch_dicts(self, query, params=None):
        self._fetch_call_count += 1
        # Return businesses on first call, empty on second (to stop the loop)
        if self._fetch_call_count <= 1:
            return self._businesses
        return []

    def execute(self, query, params=None):
        pass

    def commit(self):
        pass

    def write_enrichment(self, business_id, updates, source="unknown"):
        self.written.append((business_id, updates, source))

    def write_enrichment_batch(self, batch, source="unknown"):
        for bid, updates in batch:
            self.written.append((bid, updates, source))

    def interval_ago(self, days):
        return f"NOW() - INTERVAL '{days} days'"

    def transaction(self):
        return MockTransaction()

    def close(self):
        pass


class MockTransaction:
    def __init__(self):
        self.placeholder = "%s"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, params=None):
        pass


class MockScraper:
    """Mock AsyncWebScraper."""

    def __init__(self, results=None):
        self._results = results or []

    async def scrape_batch(self, urls):
        return self._results

    async def close(self):
        pass


class MockOllama:
    """Mock Ollama adapter for AI enrichment."""

    def __init__(
        self,
        response='{"summary": "Great business", "industry": "dentist", "health_score": 75, "pain_points": ["no website"]}',
    ):
        self._response = response

    def generate_simple(self, prompt, timeout=None):
        return self._response


# ---------------------------------------------------------------------------
# Tests: EnrichmentStats
# ---------------------------------------------------------------------------


class TestEnrichmentStats:
    def test_summary_returns_string(self):
        stats = EnrichmentStats(
            total_processed=100,
            emails_found=50,
            tech_stacks_found=30,
            start_time=time.time() - 60,
        )
        s = stats.summary()
        assert "Processed: 100" in s
        assert "Emails: 50" in s
        assert "TechStacks: 30" in s

    def test_rate_per_hour_calculation(self):
        stats = EnrichmentStats(
            total_processed=3600,
            start_time=time.time() - 3600,  # 1 hour ago
        )
        rate = stats.rate_per_hour()
        assert abs(rate - 3600.0) < 10  # Allow some timing imprecision

    def test_rate_per_hour_zero_elapsed(self):
        stats = EnrichmentStats(
            total_processed=100,
            start_time=time.time(),  # Just now
        )
        assert stats.rate_per_hour() == 0.0

    def test_summary_includes_all_fields(self):
        stats = EnrichmentStats(
            total_processed=10,
            emails_found=5,
            tech_stacks_found=3,
            categories_classified=2,
            summaries_generated=4,
            health_scores_set=1,
            scrape_failures=2,
            llm_failures=1,
            skipped_resume=3,
            start_time=time.time() - 120,
        )
        s = stats.summary()
        assert "Categories: 2" in s
        assert "Summaries: 4" in s
        assert "HealthScores: 1" in s
        assert "ScrapeFailures: 2" in s
        assert "LLMFailures: 1" in s
        assert "Skipped(resume): 3" in s
        assert "Rate:" in s
        assert "Elapsed:" in s


# ---------------------------------------------------------------------------
# Tests: EnrichmentPipeline construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_basic_construction(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MagicMock()
        db = MockDB()
        pipeline = EnrichmentPipeline(db_pool=db, ollama=None, web_scraper_workers=10)
        assert pipeline._running is False
        assert pipeline._web_workers == 10
        assert pipeline._db is db


# ---------------------------------------------------------------------------
# Tests: run() with mode="email"
# ---------------------------------------------------------------------------


class TestRunEmailMode:
    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_email_mode_starts_thread(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        db = MockDB()
        pipeline = EnrichmentPipeline(db_pool=db)

        # Pipeline will find no businesses and exit quickly
        stats = pipeline.run(mode="email", max_records=0)
        assert isinstance(stats, EnrichmentStats)

    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_email_mode_no_businesses(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        db = MockDB(businesses=[])
        pipeline = EnrichmentPipeline(db_pool=db)

        stats = pipeline.run(mode="email")
        assert stats.total_processed == 0


# ---------------------------------------------------------------------------
# Tests: run() with mode="ai"
# ---------------------------------------------------------------------------


class TestRunAIMode:
    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_ai_mode_no_businesses(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        db = MockDB(businesses=[])
        pipeline = EnrichmentPipeline(db_pool=db, ollama=MockOllama())

        stats = pipeline.run(mode="ai")
        assert stats.total_processed == 0

    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    @patch("forge.enrichment.pipeline.extract_json_from_response")
    @patch("forge.enrichment.pipeline.EnrichmentPipeline._update_enrichment_tracking")
    def test_ai_mode_enriches_business(self, mock_tracking, mock_extract, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        mock_extract.return_value = {
            "summary": "A solid dental practice in Tampa Bay area",
            "industry": "dentist",
            "health_score": 72,
            "pain_points": ["outdated website"],
        }

        biz = {
            "id": "test-uuid-1",
            "name": "Tampa Dental",
            "phone": "8135551234",
            "website_url": "https://tampadental.com",
            "address_line1": "123 Main St",
            "city": "Tampa",
            "state": "FL",
            "zip": "33602",
            "industry": None,
            "sub_industry": None,
            "email": None,
            "ai_summary": None,
            "health_score": None,
        }
        db = MockDB(businesses=[biz])
        pipeline = EnrichmentPipeline(db_pool=db, ollama=MockOllama())

        stats = pipeline.run(mode="ai")

        assert stats.total_processed >= 1
        # Verify the write was called
        assert len(db.written) >= 1
        written_id, written_updates, _ = db.written[0]
        assert written_id == "test-uuid-1"
        assert written_updates.get("industry") == "dentist"


# ---------------------------------------------------------------------------
# Tests: stop()
# ---------------------------------------------------------------------------


class TestPipelineStop:
    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_stop_sets_running_false(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        pipeline = EnrichmentPipeline(db_pool=MockDB())
        pipeline._running = True
        pipeline.stop()
        assert pipeline._running is False


# ---------------------------------------------------------------------------
# Tests: _write_enrichment
# ---------------------------------------------------------------------------


class TestWriteEnrichment:
    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_respects_allowed_fields(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        db = MockDB()
        pipeline = EnrichmentPipeline(db_pool=db)

        # Try to write both allowed and disallowed fields
        pipeline._write_enrichment(
            "uuid-1",
            {
                "email": "test@example.com",
                "industry": "dentist",
                "DANGEROUS_FIELD": "should be filtered",
                "sql_injection": "DROP TABLE",
            },
            source="test",
        )

        assert len(db.written) == 1
        _, updates, _ = db.written[0]
        assert "email" in updates
        assert "industry" in updates
        assert "DANGEROUS_FIELD" not in updates
        assert "sql_injection" not in updates

    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_empty_updates_noop(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        db = MockDB()
        pipeline = EnrichmentPipeline(db_pool=db)

        pipeline._write_enrichment("uuid-1", {}, source="test")
        assert len(db.written) == 0

    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_all_disallowed_fields_noop(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        db = MockDB()
        pipeline = EnrichmentPipeline(db_pool=db)

        pipeline._write_enrichment("uuid-1", {"bad_field": "value"}, source="test")
        assert len(db.written) == 0


# ---------------------------------------------------------------------------
# Tests: empty business list
# ---------------------------------------------------------------------------


class TestEmptyBusinessList:
    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_handles_empty_list_gracefully(self, mock_scraper_cls):
        mock_scraper_cls.return_value = MockScraper()
        db = MockDB(businesses=[])
        pipeline = EnrichmentPipeline(db_pool=db, ollama=MockOllama())

        stats = pipeline.run(mode="both")
        assert stats.total_processed == 0
        assert stats.emails_found == 0
        assert stats.scrape_failures == 0


# ---------------------------------------------------------------------------
# Tests: INDUSTRY_WHITELIST
# ---------------------------------------------------------------------------


class TestIndustryWhitelist:
    def test_has_20_categories(self):
        assert len(INDUSTRY_WHITELIST) == 20

    def test_key_categories_present(self):
        for cat in ["dentist", "lawyer", "plumber", "restaurant", "salon"]:
            assert cat in INDUSTRY_WHITELIST


# ---------------------------------------------------------------------------
# Tests: scrape result -> updates (keeps every email, not just the first)
# ---------------------------------------------------------------------------


class TestProcessScrapeResult:
    @patch("forge.enrichment.pipeline.AsyncWebScraper")
    def test_saves_best_email_and_all_emails(self, mock_scraper_cls):
        pipeline = EnrichmentPipeline(db_pool=MockDB(), ollama=None, web_scraper_workers=1)
        url = "https://biz.com"
        result = {
            "url": url,
            "status": "ok",
            "emails": ["partnerships@biz.com", "info@biz.com", "privacy@biz.com"],
        }
        biz_id, updates, is_failure = pipeline._process_scrape_result(result, {url: {"id": "b1"}})
        assert biz_id == "b1"
        assert is_failure is False
        assert updates["email"] == "partnerships@biz.com"
        assert json.loads(updates["all_emails"]) == [
            "partnerships@biz.com",
            "info@biz.com",
            "privacy@biz.com",
        ]
