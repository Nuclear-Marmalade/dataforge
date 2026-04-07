"""Enrichment pipeline: web scraping + AI enrichment in parallel tracks.

Writes results immediately per-record using COALESCE (never overwrites good data).
"""

from __future__ import annotations

import asyncio
import json
import logging
from forge.errors import EnrichmentError
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from forge.core.output_parser import extract_json_from_response
from forge.tools.web_scraper import AsyncWebScraper

logger = logging.getLogger("forge.enrichment")


# ── Industry whitelist (20 categories) ───────────────────────────────────────

INDUSTRY_WHITELIST = [
    "restaurant",
    "salon",
    "real-estate",
    "dentist",
    "gym",
    "lawyer",
    "landscaping",
    "barber",
    "cleaning-service",
    "chiropractor",
    "veterinarian",
    "auto-repair",
    "tattoo-shop",
    "accountant",
    "plumber",
    "photographer",
    "dog-groomer",
    "electrician",
    "food-truck",
    "personal-trainer",
]


@dataclass
class EnrichmentStats:
    """Tracks enrichment progress across both tracks."""

    total_processed: int = 0
    emails_found: int = 0
    tech_stacks_found: int = 0
    categories_classified: int = 0
    summaries_generated: int = 0
    health_scores_set: int = 0
    scrape_failures: int = 0
    llm_failures: int = 0
    skipped_resume: int = 0
    start_time: float = 0.0

    def rate_per_hour(self) -> float:
        """Calculate processing rate."""
        elapsed = time.time() - self.start_time
        if elapsed < 1:
            return 0.0
        return self.total_processed / (elapsed / 3600)

    def summary(self) -> str:
        """Human-readable progress summary."""
        elapsed = time.time() - self.start_time
        return (
            f"Processed: {self.total_processed:,} | "
            f"Emails: {self.emails_found:,} | "
            f"TechStacks: {self.tech_stacks_found:,} | "
            f"Categories: {self.categories_classified:,} | "
            f"Summaries: {self.summaries_generated:,} | "
            f"HealthScores: {self.health_scores_set:,} | "
            f"ScrapeFailures: {self.scrape_failures:,} | "
            f"LLMFailures: {self.llm_failures:,} | "
            f"Skipped(resume): {self.skipped_resume:,} | "
            f"Rate: {self.rate_per_hour():.0f}/hr | "
            f"Elapsed: {elapsed / 60:.1f}m"
        )


class EnrichmentPipeline:
    """
    Main enrichment orchestrator.

    Runs two parallel tracks:
    1. Web scraping (async aiohttp, concurrent)
    2. AI enrichment (sequential, Gemma 26B-A4B)

    Call run() to start. Call stop() to gracefully stop.
    """

    def __init__(
        self,
        db_pool,
        ollama=None,
        web_scraper_workers: int = 50,
        batch_size: int = 5,
    ):
        self._db = db_pool  # ForgeDB instance (or anything with fetch_dicts/execute/commit)
        self._ollama = ollama
        self._scraper = AsyncWebScraper(max_concurrent=web_scraper_workers)
        self._web_workers = web_scraper_workers
        self._batch_size = batch_size
        self._stats = EnrichmentStats()
        self._running = False
        self._lock = threading.Lock()

    def _start_track_threads(
        self, mode: str, state_filter: Optional[str], max_records: Optional[int], resume: bool
    ) -> List[threading.Thread]:
        """Start enrichment track threads based on mode. Returns thread list."""
        threads = []
        if mode in ("email", "both"):
            t = threading.Thread(
                target=self._run_email_extraction_thread,
                args=(state_filter, max_records, resume),
                name="email-extractor",
                daemon=True,
            )
            threads.append(t)
            t.start()
        if mode in ("ai", "both"):
            t = threading.Thread(
                target=self._run_ai_enrichment,
                args=(state_filter, max_records, resume),
                name="ai-enricher",
                daemon=True,
            )
            threads.append(t)
            t.start()
        return threads

    def run(
        self,
        mode: str = "both",
        state_filter: Optional[str] = None,
        max_records: Optional[int] = None,
        resume: bool = True,
    ) -> EnrichmentStats:
        """Run the enrichment pipeline.

        Returns:
            EnrichmentStats with results.
        """
        self._running = True
        self._stats = EnrichmentStats(start_time=time.time())
        logger.info(
            "Enrichment pipeline starting: mode=%s, state=%s, max=%s, workers=%d, resume=%s",
            mode,
            state_filter or "all",
            max_records or "unlimited",
            self._web_workers,
            resume,
        )

        threads = self._start_track_threads(mode, state_filter, max_records, resume)
        for t in threads:
            t.join()
        logger.info("Enrichment complete: %s", self._stats.summary())
        return self._stats

    def stop(self) -> None:
        """Gracefully stop the pipeline."""
        self._running = False

    # ── Track 1: Email/Web Extraction (Async) ────────────────────────────────

    def _run_email_extraction_thread(
        self,
        state_filter: Optional[str],
        max_records: Optional[int],
        resume: bool,
    ) -> None:
        """Thread wrapper for async email extraction."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run_email_extraction(state_filter, max_records, resume))
        finally:
            loop.run_until_complete(self._scraper.close())
            loop.close()

    def _fetch_and_build_url_map(
        self,
        state_filter: Optional[str],
        resume: bool,
    ) -> tuple:
        """Fetch businesses for scraping and build URL-to-business mapping.

        Returns (urls, url_biz_map) or ([], {}) if nothing to process.
        """
        businesses = self._fetch_businesses_for_scrape(
            state=state_filter,
            limit=self._web_workers * 2,
            resume=resume,
        )
        if not businesses:
            return [], {}

        url_biz_map: Dict[str, dict] = {}
        urls: List[str] = []
        for biz in businesses:
            url = biz.get("website_url", "")
            if url:
                urls.append(url)
                url_biz_map[url] = biz
        return urls, url_biz_map

    def _process_scrape_result(self, result: Dict[str, Any], url_biz_map: Dict[str, dict]) -> tuple:
        """Process a single scrape result into enrichment updates.

        Returns (biz_id, updates_dict_or_None, is_failure).
        """
        url = result.get("url", "")
        matched_biz = url_biz_map.get(url)
        if not matched_biz:
            return None, None, False

        status = result.get("status", "unknown")
        if status not in ("ok", "ok_no_ssl"):
            logger.info("Scrape failed for %s: %s", url[:80], status)
            return matched_biz["id"], None, True

        updates: Dict[str, Any] = {}
        if result.get("emails"):
            updates["email"] = result["emails"][0]
            with self._lock:
                self._stats.emails_found += 1
        if result.get("tech_stack"):
            updates["tech_stack"] = json.dumps(result["tech_stack"])
            with self._lock:
                self._stats.tech_stacks_found += 1
        if result.get("cms_detected"):
            updates["cms_detected"] = result["cms_detected"]
        if result.get("ssl_valid") is not None:
            updates["ssl_valid"] = result["ssl_valid"]
        if result.get("site_speed_ms") is not None:
            updates["site_speed_ms"] = result["site_speed_ms"]

        return matched_biz["id"], updates if updates else None, False

    def _flush_batch(self, batch_enrichments: List[tuple], batch_tracking_ids: List[str]) -> None:
        """Write enrichment data and tracking updates to DB."""
        if batch_enrichments:
            try:
                self._write_enrichment_batch(batch_enrichments, source="scraper")
            except Exception as e:  # Non-critical: count failures, continue processing
                with self._lock:
                    self._stats.scrape_failures += len(batch_enrichments)
                logger.error("Batch enrichment write failed: %s", e)

        if batch_tracking_ids:
            try:
                self._update_enrichment_tracking_batch(batch_tracking_ids)
            except Exception as e:  # Non-critical: tracking is best-effort
                logger.error("Batch tracking update failed: %s", e)

    def _collect_scrape_results(
        self,
        results: List[Dict[str, Any]],
        url_biz_map: Dict[str, dict],
        processed: int,
        max_records: Optional[int],
    ) -> tuple:
        """Collect enrichment data from scrape results.

        Returns (batch_enrichments, batch_tracking_ids, new_processed).
        """
        batch_enrichments: List[tuple] = []
        batch_tracking_ids: List[str] = []

        for result in results:
            if not self._running:
                break
            try:
                biz_id, updates, is_failure = self._process_scrape_result(result, url_biz_map)
            except Exception as e:  # Non-critical: skip bad result, continue batch
                with self._lock:
                    self._stats.scrape_failures += 1
                logger.debug("Scrape result processing failed: %s", e)
                continue

            if biz_id is None:
                continue

            if is_failure:
                with self._lock:
                    self._stats.scrape_failures += 1
            elif updates:
                batch_enrichments.append((biz_id, updates))
            batch_tracking_ids.append(biz_id)

            with self._lock:
                self._stats.total_processed += 1
                processed += 1
            if max_records and processed >= max_records:
                self._running = False
                break

        return batch_enrichments, batch_tracking_ids, processed

    async def _run_email_extraction(
        self,
        state_filter: Optional[str],
        max_records: Optional[int],
        resume: bool,
    ) -> None:
        """Async scrape websites for email, tech stack, CMS, SSL, speed."""
        logger.info("Email extraction track starting (%d async workers)", self._web_workers)
        processed = 0

        while self._running:
            urls, url_biz_map = self._fetch_and_build_url_map(state_filter, resume)
            if not urls:
                logger.info("Email extraction: no more businesses to process")
                break

            results = await self._scraper.scrape_batch(urls)
            enrichments, tracking_ids, processed = self._collect_scrape_results(
                results, url_biz_map, processed, max_records
            )
            self._flush_batch(enrichments, tracking_ids)

            with self._lock:
                logger.info("Email track: %s", self._stats.summary())

    # ── Track 2: AI Enrichment (Gemma 26B-A4B) ──────────────────────────────

    def _run_ai_enrichment(
        self,
        state_filter: Optional[str],
        max_records: Optional[int],
        resume: bool,
    ) -> None:
        """
        Use Gemma 26B-A4B to generate AI summaries, classify industries,
        score health, and identify pain points.
        """
        logger.info(
            "AI enrichment track starting (batch_size=%d, model=gemma4:26b)", self._batch_size
        )
        processed = 0

        while self._running:
            # Fetch businesses needing AI enrichment
            businesses = self._fetch_businesses_for_ai(
                state=state_filter,
                limit=self._batch_size,
                resume=resume,
            )

            if not businesses:
                logger.info("AI enrichment: no more businesses to process")
                break

            # Process each business individually (more reliable than batch)
            for biz in businesses:
                if not self._running:
                    break

                try:
                    self._enrich_single_ai(biz)
                except Exception as e:  # Non-critical: skip failed business, continue batch
                    logger.warning("AI enrichment failed for %s: %s", biz.get("name", "?"), e)
                    with self._lock:
                        self._stats.llm_failures += 1

                with self._lock:
                    processed += 1
                    self._stats.total_processed += 1

                if max_records and processed >= max_records:
                    self._running = False
                    break

            # Progress log every batch
            if processed % 10 == 0:
                with self._lock:
                    logger.info("AI track: %s", self._stats.summary())

    def _validate_ai_response(self, parsed: dict) -> Dict[str, Any]:
        """Extract and validate fields from AI response."""
        updates: Dict[str, Any] = {}
        summary = parsed.get("summary", "")
        if isinstance(summary, str) and 10 <= len(summary) <= 500:
            updates["ai_summary"] = summary
        industry = parsed.get("industry", "")
        if isinstance(industry, str) and industry.lower() in INDUSTRY_WHITELIST:
            updates["industry"] = industry.lower()
        health_score = parsed.get("health_score")
        if health_score is not None:
            try:
                score = int(health_score)
                if 0 <= score <= 100:
                    updates["health_score"] = score
            except (ValueError, TypeError):
                pass
        pain_points = parsed.get("pain_points", [])
        if isinstance(pain_points, list) and pain_points:
            updates["pain_points"] = pain_points
        return updates

    def _enrich_single_ai(self, business: dict) -> None:
        """Enrich a single business with Gemma 26B-A4B."""
        from forge.enrichment.prompts import build_single_enrichment_prompt

        prompt = build_single_enrichment_prompt(business)
        response = self._ollama.generate_simple(prompt, timeout=90.0)
        logger.debug("Gemma raw output for %s: %s", business.get("name", "?"), response[:200])

        parsed = extract_json_from_response(response)
        if not parsed:
            with self._lock:
                self._stats.llm_failures += 1
            logger.warning("Failed to parse JSON for %s", business.get("name", "?"))
            self._update_enrichment_tracking(business["id"])
            return

        updates = self._validate_ai_response(parsed)
        if updates:
            self._write_enrichment(business["id"], updates, source="gemma")
            with self._lock:
                if "ai_summary" in updates:
                    self._stats.summaries_generated += 1
                if "industry" in updates:
                    self._stats.categories_classified += 1
                if "health_score" in updates:
                    self._stats.health_scores_set += 1
        else:
            with self._lock:
                self._stats.llm_failures += 1
        self._update_enrichment_tracking(business["id"])

    # ── Database helpers ─────────────────────────────────────────────────────

    # Whitelist of fields that can be used in WHERE clauses
    _QUERYABLE_FIELDS = {
        "email",
        "industry",
        "ai_summary",
        "health_score",
        "sub_industry",
        "website_url",
        "phone",
        "tech_stack",
        "ssl_valid",
        "cms_detected",
        "site_speed_ms",
        "pain_points",
    }

    def _fetch_businesses_for_scrape(
        self,
        state: Optional[str] = None,
        limit: int = 100,
        resume: bool = True,
    ) -> List[dict]:
        """Fetch businesses that have a website but need web enrichment."""
        ph = self._db.placeholder

        query = """
            SELECT id, name, phone, website_url, address_line1, city, state, zip,
                   industry, email, tech_stack, cms_detected, ssl_valid, site_speed_ms
            FROM businesses
            WHERE website_url IS NOT NULL AND website_url != ''
            AND (email IS NULL OR email = '')
        """
        params: list = []

        if resume:
            interval_expr = self._db.interval_ago(7)
            query += f" AND (last_enriched_at IS NULL OR last_enriched_at < {interval_expr})"
            query += " AND (enrichment_attempts < 3 OR enrichment_attempts IS NULL)"

        if state:
            query += f" AND state = {ph}"
            params.append(state.upper())

        query += f" ORDER BY name ASC LIMIT {ph}"
        params.append(limit)

        try:
            return self._db.fetch_dicts(query, tuple(params))
        except Exception as e:  # Non-critical: return empty list so extraction loop stops cleanly
            logger.error("fetch_businesses_for_scrape failed: %s", e)
            return []

    def _fetch_businesses_for_ai(
        self,
        state: Optional[str] = None,
        limit: int = 5,
        resume: bool = True,
    ) -> List[dict]:
        """Fetch businesses that need AI enrichment (no summary yet)."""
        ph = self._db.placeholder

        query = """
            SELECT id, name, phone, website_url, address_line1, city, state, zip,
                   industry, sub_industry, email, ai_summary, health_score
            FROM businesses
            WHERE (ai_summary IS NULL OR ai_summary = '')
        """
        params: list = []

        if resume:
            query += " AND (enrichment_attempts < 3 OR enrichment_attempts IS NULL)"

        if state:
            query += f" AND state = {ph}"
            params.append(state.upper())

        query += f" ORDER BY name ASC LIMIT {ph}"
        params.append(limit)

        try:
            return self._db.fetch_dicts(query, tuple(params))
        except Exception as e:  # Non-critical: return empty list so AI loop stops cleanly
            logger.error("fetch_businesses_for_ai failed: %s", e)
            return []

    def _write_enrichment(
        self,
        business_id: str,
        updates: Dict[str, Any],
        source: str = "unknown",
    ) -> None:
        """
        Write enrichment data to DB using COALESCE pattern.

        Never overwrites existing non-null values.
        Logs every write for audit trail.
        """
        if not updates:
            return

        ALLOWED = {
            "email",
            "industry",
            "sub_industry",
            "ai_summary",
            "health_score",
            "tech_stack",
            "ssl_valid",
            "cms_detected",
            "lead_score",
            "site_speed_ms",
            "pain_points",
            "opportunities",
        }

        safe_updates = {k: v for k, v in updates.items() if k in ALLOWED}
        if not safe_updates:
            return

        # Delegate to ForgeDB's write_enrichment which handles dialect differences
        try:
            self._db.write_enrichment(business_id, safe_updates, source=source)
        except Exception as e:  # Non-critical: log and continue; data loss is acceptable
            logger.error("write_enrichment failed for %s: %s", business_id, e)

    def _update_enrichment_tracking(self, business_id: str) -> None:
        """Update enrichment attempt count and timestamp."""
        ph = self._db.placeholder
        now = self._db.now_expr
        try:
            self._db.execute(
                f"""UPDATE businesses
                   SET enrichment_attempts = COALESCE(enrichment_attempts, 0) + 1,
                       last_enriched_at = {now}
                   WHERE id = {ph}""",
                (business_id,),
            )
        except Exception as e:  # Non-critical: tracking is best-effort, continue
            logger.error("update_enrichment_tracking failed for %s: %s", business_id, e)

    def _write_enrichment_batch(
        self,
        batch: List[tuple],
        source: str = "unknown",
    ) -> None:
        """
        Batch-write enrichment data for multiple businesses in a single transaction.

        Uses COALESCE pattern (never overwrites existing non-null values).
        Each entry in batch is (business_id, updates_dict).
        """
        if not batch:
            return

        # Delegate to ForgeDB's batch write which handles dialect differences
        formatted_batch = [(bid, updates) for bid, updates in batch]
        try:
            self._db.write_enrichment_batch(formatted_batch, source=source)
        except Exception as e:  # Catch-and-reraise: log context, then propagate to flush handler
            logger.error("write_enrichment_batch failed (%d records): %s", len(batch), e)
            raise

    def _update_enrichment_tracking_batch(self, business_ids: List[str]) -> None:
        """Batch-update enrichment attempt counts and timestamps in a single transaction."""
        if not business_ids:
            return

        ph = self._db.placeholder
        now = self._db.now_expr
        try:
            with self._db.transaction() as tx:
                for bid in business_ids:
                    tx.execute(
                        f"""UPDATE businesses
                           SET enrichment_attempts = COALESCE(enrichment_attempts, 0) + 1,
                               last_enriched_at = {now}
                           WHERE id = {ph}""",
                        (bid,),
                    )
            # auto-commits on clean context exit

            logger.debug("Batch tracking updated: %d records", len(business_ids))

        except Exception as e:  # Catch-and-reraise: log context, then propagate to flush handler
            logger.error(
                "update_enrichment_tracking_batch failed (%d records): %s", len(business_ids), e
            )
            raise
