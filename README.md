
# Real Estate Assistant (batdongsan crawler)

This repository contains a Playwright-based crawler for batdongsan.com.vn listings.

**Quick TL;DR**: create a Python virtualenv, install dependencies, install Playwright browsers, then run `run.py` to crawl listings.

**Requirements**
- Python 3.10+ (3.11 recommended)
- Linux / macOS / Windows
- See `requirements.txt` for Python packages

**Setup (Linux / macOS)**

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install Playwright browsers (required for headless Chromium):

```bash
python -m playwright install chromium
```

Note: On CI or headless servers you may need additional system packages for Chromium (e.g. `libnss3`, `libatk1.0-0`, `libxss1`, etc.).

**Data directory**
- The crawler writes output JSON files to the `data/` directory. Existing example files:
	- `data/listings.json`
	- `data/listings_cho_thue.json`

**Running the crawler**

Primary runner: [run.py](run.py)

Basic usages:

```bash
# Crawl 3 pages of sale listings (default)
python run.py

# Crawl rental listings
python run.py --type cho-thue

# Crawl 5 pages of sale listings
python run.py --type ban --pages 5

# Crawl both sale and rental (3 pages each)
python run.py --type all --pages 3

# Skip visiting detail pages (faster, less data)
python run.py --no-details
```

Alternatively, you can call the crawler directly from Python:

```py
from crawler import crawl
import asyncio

asyncio.run(crawl(listing_type="ban", max_pages=3, visit_details=True))
```

**Output**
- By default the crawler saves results to `data/listings_<type>.json` (e.g. `data/listings_ban.json`).
- Files are UTF-8 encoded JSON with pretty indentation.

**Notes & tips**
- Be polite: the crawler includes a `REQUEST_DELAY` and uses stealth headers to reduce bot detection.
- If you see fewer results than expected, try increasing `--pages` or enabling `visit_details`.
- If Playwright Chromium fails on your machine, run `python -m playwright install --with-deps` or install system deps for Chromium.