"""
FORGE Enrichment Prompts — Optimized system prompts for Gemma 26B-A4B.

Each prompt includes:
  - Exact JSON schema example
  - Strict output format specification
  - Concise instructions (Gemma 26B has limited context vs Claude)
  - Field validation constraints

Depended on by: enrichment/pipeline.py
"""

from __future__ import annotations

from typing import Dict


# ── Industry whitelist ───────────────────────────────────────────────────────

INDUSTRY_LIST = (
    "restaurant, salon, real-estate, dentist, gym, lawyer, landscaping, "
    "barber, cleaning-service, chiropractor, veterinarian, auto-repair, "
    "tattoo-shop, accountant, plumber, photographer, dog-groomer, "
    "electrician, food-truck, personal-trainer"
)


def build_single_enrichment_prompt(business: Dict) -> str:
    """
    Build a prompt for single-business AI enrichment.

    Generates: summary, industry, health_score, pain_points.
    Output must be a single JSON object.
    """
    name = business.get("name", "Unknown")
    address = business.get("address_line1", "")
    city = business.get("city", "")
    state = business.get("state", "")
    zipcode = business.get("zip", "")
    phone = business.get("phone", "none")
    website = business.get("website_url", "none")
    current_industry = business.get("industry", "unknown")

    return f"""You are a business data analyst. Analyze this business and respond with ONLY a JSON object.

Business:
- Name: {name}
- Address: {address}, {city}, {state} {zipcode}
- Phone: {phone}
- Website: {website}
- Current category: {current_industry}

Generate this exact JSON structure:
{{"summary": "2-sentence description of what this business does and who it serves", "industry": "one of: {INDUSTRY_LIST}", "health_score": 0-100, "pain_points": ["issue1", "issue2"]}}

Rules:
- summary: 10-500 characters, factual, no speculation
- industry: MUST be from the list above, or null if none fit
- health_score: 0-100 integer based on data completeness (has phone +20, has website +20, has address +20, has category +20, has email +20)
- pain_points: 1-5 strings identifying likely business challenges

Respond with ONLY the JSON object. No explanation, no markdown."""


def build_batch_enrichment_prompt(businesses: list) -> str:
    """
    Build a prompt for batch enrichment (5 businesses per call).

    Only use if the 26B model is fast enough for batch processing.
    """
    biz_lines = []
    for biz in businesses:
        biz_lines.append(
            f"ID: {biz['id']}\n"
            f"Name: {biz.get('name', '')}\n"
            f"Category: {biz.get('industry', 'unknown')}\n"
            f"City: {biz.get('city', '')}, {biz.get('state', '')} {biz.get('zip', '')}\n"
            f"Phone: {biz.get('phone', 'none')}\n"
            f"Website: {biz.get('website_url', 'none')}"
        )

    return f"""You are a business data analyst. For each business below, generate enrichment data.

Respond with ONLY a JSON array. No other text.

Example output:
[{{"id": "abc-123", "summary": "2-sentence description", "industry": "dentist", "health_score": 85, "pain_points": ["no website", "few reviews"]}}]

Industry must be one of: {INDUSTRY_LIST}
health_score: 0-100 integer
summary: 10-500 characters
pain_points: 1-5 strings

Businesses:
{"---".join(biz_lines)}

Respond with ONLY the JSON array."""


def build_industry_classification_prompt(business: Dict) -> str:
    """Focused prompt for just industry classification."""
    name = business.get("name", "Unknown")
    current = business.get("industry", "unknown")

    return f"""Classify this business into one category.

Business: {name}
Current category: {current}

Categories: {INDUSTRY_LIST}

Respond with ONLY a JSON object:
{{"industry": "category_name"}}

If no category fits, respond: {{"industry": null}}"""


def build_health_score_prompt(business: Dict) -> str:
    """Focused prompt for health score calculation."""
    fields = {
        "phone": bool(business.get("phone")),
        "website": bool(business.get("website_url")),
        "email": bool(business.get("email")),
        "address": bool(business.get("address_line1")),
        "industry": bool(business.get("industry")),
        "ssl": business.get("ssl_valid"),
        "tech_stack": bool(business.get("tech_stack")),
    }

    return f"""Score this business's data completeness from 0-100.

Data present: {fields}

Scoring guide:
- phone present: +15
- website present: +15
- email present: +15
- full address: +15
- industry classified: +10
- SSL valid: +10
- tech stack detected: +10
- base score: +10

Respond with ONLY: {{"health_score": NUMBER, "reasoning": "brief explanation"}}"""


def build_summary_prompt(business: Dict) -> str:
    """Focused prompt for business summary generation."""
    return f"""Write a 2-sentence summary for this business.

Name: {business.get('name', 'Unknown')}
Industry: {business.get('industry', 'unknown')}
City: {business.get('city', '')}, {business.get('state', '')}
Website: {business.get('website_url', 'none')}

Respond with ONLY: {{"summary": "Your 2-sentence summary here"}}

Rules: factual only, 10-500 characters, describe what they do and who they serve."""
