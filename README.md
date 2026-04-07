# FORGE

![CI](https://github.com/ghealysr/forge/actions/workflows/ci.yml/badge.svg)

The open-source alternative to Apollo, ZoomInfo, and Clearbit. Enriches business databases using free data sources and local AI.

> Coverage is measured with branch coverage enabled (every conditional path counts). 46% branch-aware is roughly equivalent to 67% statement-only on the same 600-test suite.

```bash
pip install forge-enrichment

# enrich a CSV (no config needed)
forge enrich --file leads.csv --output enriched.csv

# or start the dashboard
forge dashboard

# or discover businesses by ZIP
forge discover --zip 33602 --enrich --output tampa.csv
```

No API keys, no accounts, no credit cards. Runs on your machine.

---

## What it does

Point FORGE at a PostgreSQL table of businesses and it fills in the blanks: emails, tech stacks, SSL status, site speed, industry classification, health scores, and pain points. It scrapes websites, imports government records, verifies emails via SMTP, and optionally generates AI summaries with a local model.

## Why

Apollo, ZoomInfo, and Clearbit charge $10K-50K/year for data that's largely scraped from public sources. FORGE does the same thing for free.

---

## Features

**Email extraction** (6 layers): mailto links, regex, Cloudflare decode, JSON-LD, obfuscation decode, contact page crawl.

**SMTP verification**: generates candidates (info@, contact@, admin@), verifies via RCPT TO without sending mail.

**Tech detection** (30+): WordPress, Shopify, React, Next.js, Stripe, Intercom, Google Analytics, Tailwind, and more.

**Government data importers**:
- FCC ULS -- telecom license holders (~3M records)
- NPI Registry -- healthcare providers (~7M records, free API)
- SAM.gov -- federal contractors (~900K entities, free API key)

**Local AI enrichment** (via Ollama): business summaries, industry classification, health scores, pain points. No API costs.

**Quality checks**: Haiku audit agent for spot-checking, field validation, COALESCE writes (never clobber existing data), rollback support.

**Ops**: checkpoint-based resume, hourly watchdog, graceful degradation when individual layers fail.

---

## Architecture

```
forge/
  core/           Agent loop, tool registry, context management, output parsing
  adapters/       LLM backends (Ollama local, Claude for audit)
  tools/          Database pool + async web scraper
  enrichment/     Pipeline orchestration + prompt templates
  safety/         Audit agent + error recovery + rollback
  importers/      FCC ULS, NPI Registry, SAM.gov, SMTP verifier
  monitor.py      Process health monitoring + auto-restart
```

Two parallel tracks write to the same table:

1. **Web scraping track** -- emails, tech stack, SSL, CMS, site speed. No LLM needed. ~16K records/hr with 4 workers.
2. **AI track** -- summaries, industry, health score, pain points. Runs Gemma via Ollama. ~7-20K/day depending on hardware.

Both use COALESCE writes (only update NULL fields) and checkpoint after every batch.

---

## Setup

```bash
git clone https://github.com/ghealysr/forge.git && cd forge
pip install -e .
```

Create a PostgreSQL database and table:

```sql
CREATE TABLE businesses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT, phone TEXT, email TEXT, website_url TEXT,
    city TEXT, state TEXT, zip TEXT,
    industry TEXT, sub_industry TEXT,
    tech_stack TEXT[], cms_detected TEXT,
    ssl_valid BOOLEAN, site_speed_ms INTEGER,
    ai_summary TEXT, health_score SMALLINT, pain_points TEXT[],
    npi_number TEXT, email_source TEXT,
    last_enriched_at TIMESTAMPTZ,
    enrichment_attempts INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

```bash
cp .env.example .env
# fill in your DB credentials
```

Optional: install Ollama for local AI.

```bash
brew install ollama && ollama pull gemma4:26b   # macOS
```

Run:

```bash
forge enrich --mode email --workers 50 --resume   # web scraping
forge enrich --mode ai                             # AI enrichment
forge enrich --mode both --workers 50              # both
```

---

## Config

| Variable | Required | Default | What it does |
|---|---|---|---|
| `FORGE_DB_HOST` | Yes | `localhost` | PostgreSQL host |
| `FORGE_DB_PORT` | Yes | `5432` | PostgreSQL port |
| `FORGE_DB_USER` | Yes | `forge` | PostgreSQL user |
| `FORGE_DB_PASSWORD` | Yes | -- | PostgreSQL password |
| `FORGE_DB_NAME` | Yes | `forge` | Database name |
| `ANTHROPIC_API_KEY` | No | -- | For Haiku audit agent only |
| `SAM_GOV_API_KEY` | No | -- | Free key from [api.data.gov](https://api.data.gov) |
| `FORGE_SMTP_FROM` | No | `verify@yourdomain.com` | SMTP verification identity |
| `FORGE_SMTP_EHLO` | No | `yourdomain.com` | EHLO domain |

---

## Usage

```bash
# web scraping only
forge enrich --mode email --workers 100

# AI only
forge enrich --mode ai

# both tracks
forge enrich --mode both --workers 50

# government data imports
python -m forge.importers.fcc_uls --data-dir data/fcc
python -m forge.importers.npi_registry --all-states
python -m forge.importers.sam_gov --api-key YOUR_KEY
python -m forge.importers.smtp_verifier --workers 10

# monitoring
python -m forge.monitor
```

---

## MCP integration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "forge": {
      "command": "forge",
      "args": ["mcp-server"]
    }
  }
}
```

Available tools:

| Tool | Description |
|------|-------------|
| `forge_discover` | Find businesses by ZIP via Overture Maps |
| `forge_enrich_record` | Add and enrich a single record |
| `forge_stats` | Database statistics |
| `forge_search` | Search by name, state, industry |
| `forge_export` | Export results to CSV |

---

## Comparison

| | FORGE | Apollo | ZoomInfo | Clearbit |
|---|---|---|---|---|
| Price | Free | $49-119/mo | $14,995/yr | Custom |
| Local AI | Yes | No | No | No |
| Government data | Yes | No | No | No |
| Self-hosted | Yes | No | No | No |
| Open source | Yes | No | No | No |
| SMTP verification | Yes | Yes | Yes | No |
| Tech detection | Yes | No | Yes | Yes |
| CSV in/out | Yes | Yes | Yes | No |
| Dashboard | Yes | Yes | Yes | No |
| MCP support | Yes | No | No | No |

---

## Contributing

1. Fork and branch
2. Make changes, add tests (`pytest forge/tests/`)
3. Lint (`ruff check .`)
4. PR

We could use help with: new tech detections, government data importers, alternative LLM backends, and scraping performance.

---

## License

MIT. See [LICENSE](LICENSE).

Built by [Nuclear Marmalade](https://nuclearmarmalade.com). If it saves you money, star the repo.
