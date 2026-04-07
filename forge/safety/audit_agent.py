"""
FORGE Haiku Audit Agent — Verification layer for Gemma's enrichment output.

After every N enrichments (configurable, default 50), samples 3 records
and sends the original business data + Gemma's output to Claude Haiku
for validation.

Checks:
  - Is the summary accurate and factual?
  - Is the industry classification correct?
  - Is the health score reasonable given the data?
  - Are the pain points plausible?

If >20% of audits fail, PAUSES enrichment and logs an alert.

Dependencies: anthropic (Claude API)
Depended on by: enrichment/pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# psycopg2 imported lazily in methods that need it

logger = logging.getLogger("forge.safety.audit")

# Haiku model
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Audit sampling: check 3 out of every N enrichments
DEFAULT_AUDIT_INTERVAL = 50
DEFAULT_SAMPLE_SIZE = 3

# If >20% of audits fail, pause enrichment
FAILURE_THRESHOLD = 0.20


@dataclass
class AuditResult:
    """Result of a single Haiku audit check."""
    business_id: str
    business_name: str
    passed: bool
    issues: List[str]
    confidence: float  # 0.0-1.0
    haiku_feedback: str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AuditStats:
    """Aggregated audit statistics."""
    total_audited: int = 0
    total_passed: int = 0
    total_failed: int = 0
    issues_by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_audited == 0:
            return 1.0
        return self.total_passed / self.total_audited

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.pass_rate

    def summary(self) -> str:
        return (
            f"Audited: {self.total_audited} | "
            f"Passed: {self.total_passed} | "
            f"Failed: {self.total_failed} | "
            f"PassRate: {self.pass_rate:.1%}"
        )


class HaikuAuditAgent:
    """
    Claude Haiku-powered verification agent for Gemma enrichment output.

    Samples enriched records and validates quality via Haiku API calls.
    Logs all results to forge_audit_log for tracking.
    """

    def __init__(
        self,
        api_key: str,
        db_pool: Any,
        audit_interval: int = DEFAULT_AUDIT_INTERVAL,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self._db = db_pool
        self._audit_interval = audit_interval
        self._sample_size = sample_size
        self._stats = AuditStats()
        self._enrichment_count = 0
        self._audit_log: List[AuditResult] = []
        self._paused = False

        logger.info(
            "Haiku Audit Agent initialized — interval=%d, sample=%d, model=%s",
            audit_interval, sample_size, HAIKU_MODEL,
        )

    @property
    def is_paused(self) -> bool:
        """True if enrichment should be paused due to high failure rate."""
        return self._paused

    @property
    def stats(self) -> AuditStats:
        return self._stats

    def record_enrichment(self) -> bool:
        """
        Record that an enrichment was performed.

        Returns True if an audit should be triggered (every N enrichments).
        """
        self._enrichment_count += 1
        return self._enrichment_count % self._audit_interval == 0

    def run_audit(self, state_filter: Optional[str] = None) -> List[AuditResult]:
        """
        Sample recently enriched records and validate with Haiku.

        Requires PostgreSQL backend.

        Returns list of AuditResult objects.
        """
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            logger.warning("Audit agent requires PostgreSQL (pip install psycopg2-binary)")
            return []

        if self._paused:
            logger.warning("Audit agent is paused due to high failure rate")
            return []

        # Fetch recently enriched records
        samples = self._fetch_recent_enrichments(state_filter)
        if not samples:
            logger.info("No recently enriched records to audit")
            return []

        results = []
        for biz in samples:
            try:
                result = self._audit_one(biz)
                results.append(result)
                self._audit_log.append(result)

                # Update stats
                self._stats.total_audited += 1
                if result.passed:
                    self._stats.total_passed += 1
                else:
                    self._stats.total_failed += 1
                    for issue in result.issues:
                        self._stats.issues_by_type[issue] = (
                            self._stats.issues_by_type.get(issue, 0) + 1
                        )

                # Log to DB
                self._log_audit_result(result)

            except Exception as e:
                logger.error("Audit failed for %s: %s", biz.get("name", "?"), e)

        # Check failure threshold
        if self._stats.failure_rate > FAILURE_THRESHOLD and self._stats.total_audited >= 10:
            self._paused = True
            logger.critical(
                "AUDIT ALERT: Failure rate %.1f%% exceeds threshold %.1f%%. "
                "Enrichment PAUSED. Review audit log.",
                self._stats.failure_rate * 100,
                FAILURE_THRESHOLD * 100,
            )

        logger.info("Audit complete: %s", self._stats.summary())
        return results

    def resume(self) -> None:
        """Resume enrichment after pause."""
        self._paused = False
        logger.info("Audit agent resumed")

    def _audit_one(self, business: dict) -> AuditResult:
        """Send a single business to Haiku for validation."""
        prompt = self._build_audit_prompt(business)

        response = self._client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse Haiku's response
        haiku_text = response.content[0].text
        logger.debug("Haiku audit response for %s: %s", business.get("name", "?"), haiku_text[:200])

        return self._parse_audit_response(
            business_id=business["id"],
            business_name=business.get("name", "Unknown"),
            haiku_text=haiku_text,
        )

    def _build_audit_prompt(self, business: dict) -> str:
        """Build the prompt for Haiku to validate Gemma's output."""
        return f"""You are a data quality auditor. Validate this AI-generated business enrichment.

ORIGINAL DATA:
- Name: {business.get('name', 'Unknown')}
- Address: {business.get('address_line1', '')}, {business.get('city', '')}, {business.get('state', '')} {business.get('zip', '')}
- Phone: {business.get('phone', 'none')}
- Website: {business.get('website_url', 'none')}

AI-GENERATED ENRICHMENT:
- Summary: {business.get('ai_summary', 'none')}
- Industry: {business.get('industry', 'none')}
- Health Score: {business.get('health_score', 'none')}
- Pain Points: {json.dumps(business.get('pain_points', []))}

VALIDATE and respond with ONLY this JSON:
{{"passed": true/false, "issues": ["list of specific issues found"], "confidence": 0.0-1.0, "feedback": "brief explanation"}}

CHECK:
1. Is the summary factual for this business type? (not hallucinated)
2. Does the industry category match the business name/type?
3. Is the health score reasonable (higher if more data, lower if sparse)?
4. Are the pain points plausible for this business type?

If everything looks reasonable, set passed=true with empty issues."""

    def _parse_audit_response(
        self,
        business_id: str,
        business_name: str,
        haiku_text: str,
    ) -> AuditResult:
        """Parse Haiku's audit response into an AuditResult."""
        from forge.core.output_parser import extract_json_from_response

        parsed = extract_json_from_response(haiku_text)

        if parsed:
            return AuditResult(
                business_id=business_id,
                business_name=business_name,
                passed=bool(parsed.get("passed", False)),
                issues=parsed.get("issues", []),
                confidence=float(parsed.get("confidence", 0.5)),
                haiku_feedback=parsed.get("feedback", haiku_text[:500]),
            )

        # If parsing fails, treat as a failure
        return AuditResult(
            business_id=business_id,
            business_name=business_name,
            passed=False,
            issues=["haiku_response_parse_error"],
            confidence=0.0,
            haiku_feedback=haiku_text[:500],
        )

    def _fetch_recent_enrichments(
        self,
        state_filter: Optional[str] = None,
    ) -> List[dict]:
        """Fetch recently enriched records for audit sampling."""
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError("Safety module requires psycopg2: pip install psycopg2-binary")

        conn = self._db.get_connection()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            query = """
                SELECT id, name, phone, website_url, address_line1, city, state, zip,
                       industry, email, ai_summary, health_score, pain_points
                FROM businesses
                WHERE ai_summary IS NOT NULL AND ai_summary != ''
                AND last_enriched_at > NOW() - INTERVAL '1 day'
            """
            params: list = []

            if state_filter:
                query += " AND state = %s"
                params.append(state_filter.upper())

            query += " ORDER BY last_enriched_at DESC LIMIT %s"
            params.append(self._sample_size)

            cur.execute(query, params)
            rows = cur.fetchall()
            cur.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error("fetch_recent_enrichments failed: %s", e)
            return []
        finally:
            self._db.return_connection(conn)

    def _log_audit_result(self, result: AuditResult) -> None:
        """Log audit result to the forge_audit_log table."""
        conn = self._db.get_connection()
        try:
            cur = conn.cursor()

            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forge_audit_log (
                    id SERIAL PRIMARY KEY,
                    business_id UUID NOT NULL,
                    business_name VARCHAR(255),
                    passed BOOLEAN NOT NULL,
                    issues JSONB,
                    confidence NUMERIC(3,2),
                    haiku_feedback TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cur.execute(
                """INSERT INTO forge_audit_log
                   (business_id, business_name, passed, issues, confidence, haiku_feedback)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    result.business_id,
                    result.business_name[:255],
                    result.passed,
                    json.dumps(result.issues),
                    result.confidence,
                    result.haiku_feedback[:2000],
                ),
            )
            conn.commit()
            cur.close()
        except Exception as e:
            conn.rollback()
            logger.error("log_audit_result failed: %s", e)
        finally:
            self._db.return_connection(conn)
