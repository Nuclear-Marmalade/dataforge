# FORGE

<!-- TODO: Add logo -->

**Free Open-source Runtime for data enrichment.**
The open-source alternative to Apollo, ZoomInfo, and Clearbit.

---

## 5-Minute Quickstart

```bash
# Install
pip install forge-enrichment

# Enrich a CSV (zero config, works immediately)
forge enrich --file my_leads.csv --output enriched.csv

# Or start the dashboard
forge dashboard

# Or discover businesses by ZIP code
forge discover --zip 33602 --enrich --output tampa_businesses.csv
```

That's it. No API keys, no accounts, no credit cards. FORGE runs entirely on your machine.

---

## What It Does

FORGE enriches business databases at scale using free data sources and local AI. It extracts emails from websites, detects technology stacks, imports government records, and generates AI-powered business summaries -- all without paying a dime to data brokers. Point it at a PostgreSQL table of businesses and it fills in the blanks: emails, tech stacks, SSL status, site speed, industry classification, health scores, pain points, and more.

## Why It Exists

We got tired of paying $30K/year for business data enrichment. So we built our own. Then we open-sourced it.

Tools like Apollo, ZoomInfo, and Clearbit charge $10K-50K/year for data that's largely scraped from the same public sources you could access yourself. FORGE does exactly that -- systematically, reliably, and for free.

---

## Features

### Email Extraction (6-Layer Pipeline)
- **Mailto link parsing** -- pulls email addresses from `mailto:` href attributes
- **Regex extraction** -- pattern-matches email addresses across page content
- **Cloudflare email decode** -- decodes Cloudflare's obfuscated `data-cfemail` attributes
- **JSON-LD extraction** -- parses structured data for contact information
- **Obfuscation decode** -- handles common email obfuscation techniques (AT/DOT replacements, HTML entity encoding, etc.)
- **Contact page crawl** -- follows /contact, /about, /team links to find emails buried deeper in the site

### SMTP Email Verification
- Generates candidate emails (info@, contact@, admin@, sales@, support@) from domain
- Verifies deliverability via SMTP `RCPT TO` without sending actual mail
- Respects rate limits and backs off on greylisting

### Technology Detection (30+ Technologies)
- **CMS**: WordPress, Shopify, Squarespace, Wix, Webflow, Drupal, Joomla, Ghost, HubSpot CMS, BigCommerce
- **Analytics**: Google Analytics (GA4/UA), Google Tag Manager, Facebook Pixel, Hotjar, Segment, Mixpanel, Heap
- **Frameworks**: React, Vue.js, Angular, Next.js, Nuxt.js, Svelte, jQuery, Bootstrap, Tailwind CSS
- **Chat/Support**: Intercom, Zendesk, Drift, LiveChat, Crisp, Tidio, HubSpot Chat
- **Payments**: Stripe, Square, PayPal, Braintree
- **Email Marketing**: Mailchimp, Klaviyo, Constant Contact, SendGrid
- **Other**: Cloudflare, reCAPTCHA, Schema.org markup, ADA compliance tools

### Government Data Importers
- **FCC ULS** -- FCC Universal Licensing System (telecommunications license holders)
- **NPI Registry** -- National Provider Identifier database (healthcare providers)
- **SAM.gov** -- System for Award Management (government contractors, all businesses registered for federal work)

### AI Enrichment (Local, via Ollama/Gemma)
- Business summary generation from scraped website content
- Industry and sub-industry classification
- Health score computation (1-10 scale based on web presence signals)
- Pain point identification for sales targeting
- Runs entirely local -- no API costs, no data leaving your machine

### Quality Assurance
- **Claude Haiku audit agent** -- spot-checks AI-generated output, auto-pauses the pipeline if quality drops below threshold (requires PostgreSQL)
- **Field validation** -- type checking, format validation, and sanity checks before writes
- **COALESCE write pattern** -- never overwrites existing good data; only fills in NULL/empty fields
- **Rollback capability** -- tracks changes per-record for reversal if something goes wrong (requires PostgreSQL)

### Reliability
- **Checkpoint-based resume** -- crash-safe; restarts from last completed batch, not from scratch
- **Hourly self-monitoring** -- watchdog process detects stalls, OOM, or crashes and auto-restarts
- **Graceful degradation** -- if a single enrichment layer fails, the rest continue

---

## Architecture

```
forge/
  core/           Agent loop, tool registry, context management, output parsing
  adapters/       LLM backends (Ollama/Gemma local, with Anthropic for audit)
  tools/          Database connection pool + async web scraper
  enrichment/     Pipeline orchestration + AI prompt templates
  safety/         Audit agent + error recovery + rollback engine
  importers/      FCC ULS, NPI Registry, SAM.gov, SMTP verifier
  monitor.py      Process health monitoring + auto-restart
```

### Data Flow

```
                    +------------------+
                    |   PostgreSQL DB  |
                    |  (businesses)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Web Scraping   |          |  AI Enrichment  |
     |  Track          |          |  Track          |
     +--------+--------+          +--------+--------+
              |                             |
     +--------v--------+          +--------v--------+
     | 6-Layer Email   |          | Ollama/Gemma    |
     | Tech Detection  |          | Summary, Score  |
     | SSL + Speed     |          | Industry, Pain  |
     +---------+-------+          +--------+--------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |  COALESCE Write  |
                    |  (never clobber) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Haiku Audit     |
                    |  (spot-check)    |
                    +------------------+
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/ghealysr/forge.git
cd forge
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Set up PostgreSQL

Create your database and the businesses table:

```sql
CREATE TABLE businesses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    phone TEXT,
    email TEXT,
    website_url TEXT,
    city TEXT,
    state TEXT,
    zip TEXT,
    industry TEXT,
    sub_industry TEXT,
    tech_stack TEXT[],
    cms_detected TEXT,
    ssl_valid BOOLEAN,
    site_speed_ms INTEGER,
    ai_summary TEXT,
    health_score SMALLINT,
    pain_points TEXT[],
    npi_number TEXT,
    email_source TEXT,
    last_enriched_at TIMESTAMPTZ,
    enrichment_attempts INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_businesses_state ON businesses(state);
CREATE INDEX idx_businesses_industry ON businesses(industry);
CREATE INDEX idx_businesses_enriched ON businesses(last_enriched_at);
CREATE INDEX idx_businesses_website ON businesses(website_url) WHERE website_url IS NOT NULL;
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials
```

### 5. (Optional) Download FCC ULS data

Download bulk data files from the [FCC ULS License Search](https://www.fcc.gov/uls/transactions/daily-weekly):

```bash
mkdir -p data/fcc
# Download and extract the weekly dump files into data/fcc/
```

### 6. (Optional) Install Ollama for local AI

```bash
# macOS
brew install ollama
ollama pull gemma4:26b

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma4:26b
```

### 7. Run

```bash
# Email extraction + tech detection (web scraping track)
forge enrich --mode email --workers 50 --resume

# AI enrichment track
forge enrich --mode ai

# Both tracks
forge enrich --mode both --workers 50
```

---

## Configuration

| Environment Variable | Required | Default | Description |
|---|---|---|---|
| `FORGE_DB_HOST` | Yes | `localhost` | PostgreSQL host |
| `FORGE_DB_PORT` | Yes | `5432` | PostgreSQL port |
| `FORGE_DB_USER` | Yes | `forge` | PostgreSQL user |
| `FORGE_DB_PASSWORD` | Yes | -- | PostgreSQL password |
| `FORGE_DB_NAME` | Yes | `forge` | PostgreSQL database name |
| `ANTHROPIC_API_KEY` | No | -- | Anthropic API key (only for Haiku audit agent; enrichment uses local Ollama) |
| `SAM_GOV_API_KEY` | No | -- | SAM.gov API key (free; register at [api.data.gov](https://api.data.gov)) |
| `FORGE_SMTP_FROM` | No | `verify@yourdomain.com` | Identity used for SMTP verification |
| `FORGE_SMTP_EHLO` | No | `yourdomain.com` | EHLO domain for SMTP verification |
| `FORGE_SERVICE_PREFIX` | No | `com.forge` | Service prefix for launchd/systemd monitor |

---

## Usage Examples

### Web scraping only (email + tech detection)
```bash
forge enrich --mode email --workers 100
```

### AI enrichment only (summaries, scores, classification)
```bash
forge enrich --mode ai
```

### Both tracks in parallel
```bash
forge enrich --mode both --workers 50
```

### FCC Universal Licensing System import
```bash
python -m forge.importers.fcc_uls --data-dir data/fcc
```

### NPI Registry import (healthcare providers)
```bash
python -m forge.importers.npi_registry --all-states
```

### SMTP email verification
```bash
python -m forge.importers.smtp_verifier --workers 10
```

### SAM.gov import (government contractors)
```bash
python -m forge.importers.sam_gov --api-key YOUR_KEY
```

### Run the health monitor
```bash
python -m forge.monitor
```

---

## Government Data Sources

FORGE ships with importers for three free, public government databases:

### FCC Universal Licensing System (ULS)
The FCC publishes its entire licensing database as weekly bulk downloads. This includes every business holding an FCC license -- telecommunications companies, broadcasters, wireless carriers, satellite operators, and more. FORGE's importer parses the pipe-delimited dump files and matches records against your businesses table by name and address.

- **Source**: [FCC ULS Downloads](https://www.fcc.gov/uls/transactions/daily-weekly)
- **Coverage**: ~3 million active licenses
- **Cost**: Free

### NPI Registry
The National Plan and Provider Enumeration System (NPPES) maintains the NPI Registry -- a directory of every healthcare provider in the United States. FORGE queries the public API by state and matches providers to your business records.

- **Source**: [NPPES NPI Registry](https://npiregistry.cms.hhs.gov/)
- **Coverage**: ~7 million healthcare providers
- **Cost**: Free

### SAM.gov (System for Award Management)
Every business registered for federal contracts or grants appears in SAM.gov. The Entity Management API provides company details, NAICS codes, cage codes, and registration status. Requires a free API key from api.data.gov.

- **Source**: [SAM.gov Entity API](https://api.sam.gov/)
- **Coverage**: ~900K+ registered entities
- **Cost**: Free (API key required, no charge)

---

## How the Agent Loop Works

FORGE's `core/` module implements a lightweight agent loop designed for high-throughput data enrichment:

1. **Context Management** -- The agent maintains a sliding context window of the current batch (business records to enrich). It tracks which fields are populated, which are missing, and what enrichment has already been attempted.

2. **Tool Registry** -- Each enrichment capability (web scraper, email extractor, tech detector, AI summarizer) is registered as a tool. The agent loop selects which tools to invoke based on what data is missing for each record.

3. **Output Parsing** -- Raw tool outputs (HTML, JSON, SMTP responses) are parsed and normalized before writing. The parser handles malformed data, encoding issues, and edge cases.

4. **Orchestration** -- The pipeline processes records in configurable batches. Each batch goes through: fetch -> extract -> validate -> write -> checkpoint. Failed records are retried with exponential backoff before being marked as exhausted.

This is not an LLM-in-the-loop agent (except for the AI enrichment track). The "agent loop" is a deterministic pipeline with tool dispatch -- fast, predictable, and debuggable.

---

## Safety and Quality

### Audit Agent (requires PostgreSQL)
The optional Claude Haiku audit agent samples AI-generated output at configurable intervals (default: every 50 records). It evaluates summaries for accuracy, coherence, and hallucination. If quality drops below the threshold, the pipeline auto-pauses and alerts you. This is the only component that requires an API key -- everything else runs locally. Note: the audit agent requires a PostgreSQL backend; it is not available with SQLite.

### Field Validation
Every field write passes through validation: email format checks, URL normalization, score range enforcement (1-10), array deduplication for tech stacks, and type coercion. Invalid data is rejected, not written.

### COALESCE Write Pattern
FORGE never overwrites existing data. The write layer uses SQL COALESCE semantics: a field is only updated if the current value is NULL or empty. If you've manually corrected a record, FORGE won't clobber your work.

### Rollback (requires PostgreSQL)
Every enrichment batch logs the previous field values before writing. If a batch produces bad data, you can roll back to the pre-enrichment state for affected records. Note: rollback requires a PostgreSQL backend; it is not available with SQLite.

---

## Performance

- **Throughput**: 16,000+ records/hour with 4 parallel instances (web scraping track)
- **Scale**: Tested against databases with millions of records
- **Resource usage**: Moderate -- the bottleneck is network I/O (web scraping) and GPU/CPU (Ollama inference), not FORGE itself
- **Parallelism**: Configurable worker count; scales linearly up to network/database saturation
- **Resume**: Checkpoint-based; a crash at record 847,000 resumes at 847,000, not at 0

---

## Claude Code Integration

Add FORGE to your Claude Code MCP config (`~/.claude.json`):

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

Then ask Claude:

- "Find restaurants in Tampa and get their emails"
- "How many businesses do we have in Florida?"
- "Export all healthcare providers with emails to a CSV"
- "Enrich this business: Smith Family Dental in Tampa, FL"

FORGE exposes five tools to Claude Code:

| Tool | What it does |
|------|-------------|
| `forge_discover` | Search for businesses by ZIP code using Overture Maps |
| `forge_enrich_record` | Add and enrich a single business record |
| `forge_stats` | Get database statistics (total records, emails found, etc.) |
| `forge_search` | Search your enriched database by name, state, industry |
| `forge_export` | Export filtered results to CSV |

---

## How FORGE Compares

| Feature | FORGE | Apollo | ZoomInfo | Clearbit |
|---------|-------|--------|----------|----------|
| Price | Free | $49-119/mo | $14,995/yr | Custom |
| Local AI | Yes | No | No | No |
| Government data | Yes | No | No | No |
| Self-hosted | Yes | No | No | No |
| Open source | Yes | No | No | No |
| SMTP verification | Yes | Yes | Yes | No |
| Tech stack detection | Yes | No | Yes | Yes |
| CSV in/out | Yes | Yes | Yes | No |
| Dashboard | Yes | Yes | Yes | No |
| Claude Code MCP | Yes | No | No | No |

---

## Contributing

Contributions are welcome. Here's how:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests if applicable (`pytest tests/`)
5. Ensure code passes linting (`ruff check .`)
6. Submit a pull request

### Areas where help is appreciated
- Additional technology detections
- New government data source importers
- Alternative LLM backend adapters (vLLM, llama.cpp, etc.)
- Performance optimizations for the web scraping pipeline
- Documentation and examples

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

Built by [Nuclear Marmalade](https://nuclearmarmalade.com).

If FORGE saves you money, star the repo. If it saves you a lot of money, [tell us about it](mailto:hello@nuclearmarmalade.com).
