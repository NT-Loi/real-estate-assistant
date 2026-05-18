"""
Crawler for batdongsan.com.vn — Real Estate Listings

Supports two listing types:
  - "ban"      → https://batdongsan.com.vn/nha-dat-ban      (For Sale)
  - "cho-thue" → https://batdongsan.com.vn/nha-dat-cho-thue  (For Rent)

Extracts listing data with fields matching the site's filter criteria:
- loai_hinh (listing type: ban / cho-thue)
- loai_nha_dat (property type)
- khu_vuc / dia_chi (location)
- gia (price)
- dien_tich (area m²)
- gia_per_m2 (price per m²)
- so_phong_ngu (bedrooms)
- so_phong_tam / so_toilet (bathrooms)
- huong_nha (house orientation)
- huong_ban_cong (balcony orientation)
- phap_ly (legal status)
- noi_that (furniture)

Plus additional detail fields:
- tieu_de (title)
- mo_ta (description)
- du_an (project name)
- url (detail page URL)
- hinh_anh (image URLs)
- ngay_dang (posted date)
- nguoi_dang (agent/poster name)
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
from playwright.async_api import async_playwright, Page, TimeoutError as PwTimeout

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://batdongsan.com.vn"
LISTING_URLS = {
    "ban": f"{BASE_URL}/nha-dat-ban",
    "cho-thue": f"{BASE_URL}/nha-dat-cho-thue",
}

# --- New section URLs ---
PROJECT_URL = f"{BASE_URL}/du-an-bat-dong-san"
NEWS_URL = f"{BASE_URL}/tin-tuc"
WIKI_URL = f"{BASE_URL}/wiki"
WIKI_CATEGORIES = {
    "mua-bds": "Mua BĐS",
    "ban-bds": "Bán BĐS",
    "thue-bds": "Thuê BĐS",
    "tai-chinh": "Tài chính BĐS",
    "quy-hoach-phap-ly": "Quy hoạch - Pháp lý",
    "noi-ngoai-that": "Nội - Ngoại thất",
    "phong-tuc": "Phong tục",
}

MAX_PAGES = 1  # Number of listing pages to crawl (change as needed)
DATA_DIR = Path(__file__).parent / "data"
REQUEST_DELAY = 2.0  # Seconds between page navigations (be polite)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bds_crawler")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(text: Optional[str]) -> Optional[str]:
    """Strip and normalize whitespace."""
    if not text:
        return None
    return re.sub(r"\s+", " ", text.strip()) or None


# ---------------------------------------------------------------------------
# Listing page scraper  (collects summary cards)
# ---------------------------------------------------------------------------
async def _wait_for_listings(page: Page) -> bool:
    """Wait until listing cards are visible. Returns True if found."""
    for selector in [".js__card", "[data-tracking-id]", ".re__card-full"]:
        try:
            await page.wait_for_selector(selector, timeout=10_000)
            return True
        except PwTimeout:
            continue
    return False


async def _extract_cards(page: Page) -> list[dict]:
    """Run JS extraction on the current page and return card dicts."""
    cards = await page.evaluate("""() => {
        const results = [];
        let cardEls = document.querySelectorAll('.js__card');
        if (cardEls.length === 0) {
            cardEls = document.querySelectorAll('[data-tracking-id]');
        }
        if (cardEls.length === 0) {
            cardEls = document.querySelectorAll('.re__card-full');
        }

        for (const card of cardEls) {
            const titleEl = card.querySelector('.js__card-title')
                || card.querySelector('[class*="card-title"]')
                || card.querySelector('h3 a, h2 a');
            const priceEl = card.querySelector('.re__card-config-price')
                || card.querySelector('[class*="config-price"]:not([class*="per"])');
            const areaEl = card.querySelector('.re__card-config-area')
                || card.querySelector('[class*="config-area"]');
            const ppmEl = card.querySelector('.re__card-config-price_per_m2')
                || card.querySelector('[class*="price_per_m2"]')
                || card.querySelector('[class*="price-per-m2"]');
            const locationEl = card.querySelector('.re__card-location')
                || card.querySelector('[class*="card-location"]');
            const bedroomEl = card.querySelector('.re__card-config-bedroom')
                || card.querySelector('[class*="config-bedroom"]');
            const toiletEl = card.querySelector('.re__card-config-toilet')
                || card.querySelector('[class*="config-toilet"]')
                || card.querySelector('[class*="config-bathroom"]');
            const dateEl = card.querySelector('.re__card-published-info-published-at')
                || card.querySelector('[class*="published-at"]');
            const descEl = card.querySelector('.js__card-description')
                || card.querySelector('.re__card-description')
                || card.querySelector('[class*="card-description"]');
            const linkEl = card.querySelector('a.js__card-title')
                || card.querySelector('a[href*="/ban-"]')
                || card.querySelector('a[href*="/nha-"]');
            const agentEl = card.querySelector('.re__card-published-info-agent-name')
                || card.querySelector('[class*="agent-name"]');
            const imgEl = card.querySelector('img[src*="batdongsan"], img[data-src]');

            if (titleEl || linkEl) {
                results.push({
                    tieu_de: titleEl ? titleEl.innerText.trim() : null,
                    gia: priceEl ? priceEl.innerText.trim() : null,
                    dien_tich: areaEl ? areaEl.innerText.trim() : null,
                    gia_per_m2: ppmEl ? ppmEl.innerText.trim() : null,
                    khu_vuc: locationEl ? locationEl.innerText.trim() : null,
                    so_phong_ngu: bedroomEl ? bedroomEl.innerText.trim() : null,
                    so_phong_tam: toiletEl ? toiletEl.innerText.trim() : null,
                    ngay_dang: dateEl ? dateEl.innerText.trim() : null,
                    mo_ta: descEl ? descEl.innerText.trim() : null,
                    url: linkEl ? linkEl.getAttribute('href') : null,
                    nguoi_dang: agentEl ? agentEl.innerText.trim() : null,
                    hinh_anh: imgEl ? (imgEl.getAttribute('src') || imgEl.getAttribute('data-src')) : null,
                });
            }
        }
        return results;
    }""")
    return cards


async def scrape_listing_page(page: Page, page_num: int, listing_url: str) -> list[dict]:
    """Scrape one page of listings, return a list of dicts with summary data."""
    url = listing_url if page_num == 1 else f"{listing_url}/p{page_num}"
    log.info(f"Navigating to listing page {page_num}: {url}")

    # Navigate with retry
    for attempt in range(3):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            break
        except Exception as e:
            log.warning(f"  Navigation attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                return []
            await asyncio.sleep(3)

    # Wait for page to fully render (site needs several seconds for JS)
    await asyncio.sleep(5)

    # Wait for listing cards
    found = await _wait_for_listings(page)
    if not found:
        log.warning(f"No listing cards found on page {page_num} (attempt via goto).")
        # Try scrolling down to trigger lazy load
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        await asyncio.sleep(2)
        found = await _wait_for_listings(page)
        if not found:
            log.warning(f"Still no listings after scroll. Skipping page {page_num}.")
            return []

    # Extra wait for dynamic content
    await asyncio.sleep(2)

    cards = await _extract_cards(page)

    # Normalize URLs
    for card in cards:
        if card.get("url") and not card["url"].startswith("http"):
            card["url"] = BASE_URL + card["url"]

    log.info(f"  Found {len(cards)} listings on page {page_num}")
    return cards


# ---------------------------------------------------------------------------
# Detail page scraper  (enriches with filter-matching fields)
# ---------------------------------------------------------------------------
async def scrape_detail_page(page: Page, url: str) -> dict:
    """Visit a detail page and extract all structured property specs."""
    detail = {}
    try:
        log.info(f"  Visiting detail: {url}")
        resp = await page.goto(url, wait_until="commit", timeout=30_000)
        
        # Wait for body to be ready
        await page.wait_for_load_state("domcontentloaded", timeout=15_000)
        
        # Give JS time to render the content
        await asyncio.sleep(3)

        # Extract structured specs from the detail page
        detail = await page.evaluate("""() => {
            const specs = {};

            // Method 1: Spec items (key-value pairs in property details section)
            const specItems = document.querySelectorAll('.re__pr-specs-content-item');
            for (const item of specItems) {
                const titleEl = item.querySelector('.re__pr-specs-content-item-title');
                const valueEl = item.querySelector('.re__pr-specs-content-item-value');
                if (titleEl && valueEl) {
                    specs[titleEl.innerText.trim()] = valueEl.innerText.trim();
                }
            }

            // Method 2: Short info section
            const shortInfoItems = document.querySelectorAll('.re__pr-short-info-item');
            for (const item of shortInfoItems) {
                const titleEl = item.querySelector('.title');
                const valueEl = item.querySelector('.value');
                if (titleEl && valueEl) {
                    specs[titleEl.innerText.trim()] = valueEl.innerText.trim();
                }
            }

            // Method 3: Try generic table-like layouts
            const tableRows = document.querySelectorAll('tr, .re__pr-attr-item');
            for (const row of tableRows) {
                const cells = row.querySelectorAll('td, th, span');
                if (cells.length >= 2) {
                    const key = cells[0].innerText.trim();
                    const value = cells[1].innerText.trim();
                    if (key && value && key.length < 50) {
                        specs[key] = value;
                    }
                }
            }
            
            // Method 4: Look for div pairs with label/value pattern
            const infoItems = document.querySelectorAll('[class*="info-item"], [class*="attr-item"]');
            for (const item of infoItems) {
                const children = item.children;
                if (children.length >= 2) {
                    const key = children[0].innerText.trim();
                    const value = children[1].innerText.trim();
                    if (key && value && key.length < 50) {
                        specs[key] = value;
                    }
                }
            }

            // Full description
            const descEl = document.querySelector('.re__detail-content .re__section-body')
                || document.querySelector('[class*="detail-content"] [class*="section-body"]')
                || document.querySelector('.re__detail-content');
            if (descEl) {
                specs['_mo_ta_chi_tiet'] = descEl.innerText.trim().substring(0, 2000);
            }

            // All images
            const images = [];
            const imgEls = document.querySelectorAll(
                '.re__media-thumb-item img, .slick-slide img, [class*="media"] img'
            );
            for (const img of imgEls) {
                const src = img.getAttribute('src') || img.getAttribute('data-src');
                if (src && !images.includes(src) && !src.includes('data:image')) {
                    images.push(src);
                }
            }
            if (images.length === 0) {
                // Fallback: grab any content images
                const allImgs = document.querySelectorAll('img[src*="batdongsan"]');
                for (const img of allImgs) {
                    const src = img.getAttribute('src');
                    if (src && !images.includes(src)) images.push(src);
                }
            }
            specs['_hinh_anh'] = images;

            // Address
            const addressEl = document.querySelector(
                '.re__pr-short-description--address, [class*="short-description--address"]'
            );
            if (addressEl) {
                specs['_dia_chi'] = addressEl.innerText.trim();
            }
            
            // Try to get address from breadcrumbs
            const breadcrumbs = [];
            const bcItems = document.querySelectorAll('.re__breadcrumb a, [class*="breadcrumb"] a');
            for (const bc of bcItems) {
                breadcrumbs.push(bc.innerText.trim());
            }
            if (breadcrumbs.length > 0) {
                specs['_breadcrumb'] = breadcrumbs;
            }

            return specs;
        }""")

    except PwTimeout:
        log.warning(f"  Timeout loading detail page: {url}")
    except Exception as e:
        log.warning(f"  Error loading detail page {url}: {e}")

    return detail


# ---------------------------------------------------------------------------
# Merge listing card data with detail page data
# ---------------------------------------------------------------------------
# Maps Vietnamese spec labels → standardized field names
SPEC_KEY_MAP = {
    "Mức giá": "gia",
    "Giá": "gia",
    "Diện tích": "dien_tich",
    "Hướng nhà": "huong_nha",
    "Hướng ban công": "huong_ban_cong",
    "Số phòng ngủ": "so_phong_ngu",
    "Số toilet": "so_phong_tam",
    "Số phòng tắm": "so_phong_tam",
    "Pháp lý": "phap_ly",
    "Nội thất": "noi_that",
    "Số tầng": "so_tang",
    "Đường vào": "duong_vao",
    "Mặt tiền": "mat_tien",
    "Loại hình": "loai_nha_dat",
    "Loại tin": "loai_tin",
    "Dự án": "du_an",
    "Chiều dài": "chieu_dai",
    "Chiều rộng": "chieu_rong",
}


def merge_listing(card: dict, detail_specs: dict, listing_type: str = "ban") -> dict:
    """Merge card summary data with detail page specs into a unified record."""
    record = {
        # --- Listing type ---
        "loai_hinh": listing_type,   # "ban" or "cho-thue"
        # --- Filter fields (primary) ---
        "loai_nha_dat": None,       # Property type
        "khu_vuc": None,            # Location / region
        "dia_chi": None,            # Full address
        "gia": None,                # Price
        "gia_per_m2": None,         # Price per m²
        "dien_tich": None,          # Area (m²)
        "so_phong_ngu": None,       # Bedrooms
        "so_phong_tam": None,       # Bathrooms / toilets
        "huong_nha": None,          # House orientation
        "huong_ban_cong": None,     # Balcony orientation
        "phap_ly": None,            # Legal status
        "noi_that": None,           # Furniture
        # --- Additional fields ---
        "tieu_de": None,            # Title
        "mo_ta": None,              # Description (short)
        "mo_ta_chi_tiet": None,     # Description (full)
        "du_an": None,              # Project name
        "so_tang": None,            # Number of floors
        "mat_tien": None,           # Frontage width
        "duong_vao": None,          # Access road width
        "chieu_dai": None,          # Length
        "chieu_rong": None,         # Width
        "url": None,                # Detail page URL
        "hinh_anh": [],             # Image URLs
        "ngay_dang": None,          # Posted date
        "nguoi_dang": None,         # Agent / poster name
    }

    # Fill from card data
    record["tieu_de"] = card.get("tieu_de")
    record["gia"] = card.get("gia")
    record["dien_tich"] = card.get("dien_tich")
    record["gia_per_m2"] = card.get("gia_per_m2")
    # Clean khu_vuc: strip leading "·\n" separator from card location text
    khu_vuc = card.get("khu_vuc")
    if khu_vuc:
        khu_vuc = re.sub(r"^[·\s]+", "", khu_vuc).strip()
    record["khu_vuc"] = khu_vuc or None
    record["so_phong_ngu"] = card.get("so_phong_ngu")
    record["so_phong_tam"] = card.get("so_phong_tam")
    record["mo_ta"] = card.get("mo_ta")
    record["url"] = card.get("url")
    record["ngay_dang"] = card.get("ngay_dang")
    record["nguoi_dang"] = card.get("nguoi_dang")
    if card.get("hinh_anh"):
        record["hinh_anh"] = [card["hinh_anh"]]

    # Overwrite / enrich from detail specs
    for vn_key, field_name in SPEC_KEY_MAP.items():
        if vn_key in detail_specs:
            record[field_name] = detail_specs[vn_key]

    # Special detail fields
    if detail_specs.get("_mo_ta_chi_tiet"):
        record["mo_ta_chi_tiet"] = detail_specs["_mo_ta_chi_tiet"]
    if detail_specs.get("_dia_chi"):
        record["dia_chi"] = detail_specs["_dia_chi"]
    if detail_specs.get("_hinh_anh"):
        record["hinh_anh"] = detail_specs["_hinh_anh"]

    # Infer loai_nha_dat from URL if not set
    if not record["loai_nha_dat"] and record.get("url"):
        url = record["url"]
        # Selling patterns
        if "ban-can-ho" in url:
            record["loai_nha_dat"] = "Căn hộ chung cư"
        elif "ban-nha-rieng" in url:
            record["loai_nha_dat"] = "Nhà riêng"
        elif "ban-nha-biet-thu" in url:
            record["loai_nha_dat"] = "Nhà biệt thự, liên kề"
        elif "ban-nha-mat-pho" in url:
            record["loai_nha_dat"] = "Nhà mặt phố"
        elif "ban-dat" in url:
            record["loai_nha_dat"] = "Đất"
        elif "ban-condotel" in url:
            record["loai_nha_dat"] = "Condotel"
        elif "ban-shophouse" in url:
            record["loai_nha_dat"] = "Shophouse"
        # Rental patterns
        elif "thue-can-ho" in url or "cho-thue-can-ho" in url:
            record["loai_nha_dat"] = "Căn hộ chung cư"
        elif "thue-nha-rieng" in url or "cho-thue-nha-rieng" in url:
            record["loai_nha_dat"] = "Nhà riêng"
        elif "thue-nha-biet-thu" in url or "cho-thue-nha-biet-thu" in url:
            record["loai_nha_dat"] = "Nhà biệt thự, liên kề"
        elif "thue-nha-mat-pho" in url or "cho-thue-nha-mat-pho" in url:
            record["loai_nha_dat"] = "Nhà mặt phố"
        elif "thue-mat-bang" in url or "cho-thue-mat-bang" in url:
            record["loai_nha_dat"] = "Mặt bằng"
        elif "thue-van-phong" in url or "cho-thue-van-phong" in url:
            record["loai_nha_dat"] = "Văn phòng"
        elif "thue-cua-hang" in url or "cho-thue-cua-hang" in url:
            record["loai_nha_dat"] = "Cửa hàng, ki ốt"
        elif "thue-kho-xuong" in url or "cho-thue-kho" in url:
            record["loai_nha_dat"] = "Kho, xưởng"
        elif "thue-phong-tro" in url or "cho-thue-phong-tro" in url:
            record["loai_nha_dat"] = "Phòng trọ"

    return record


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------
async def crawl(
    listing_type: str = "ban",
    max_pages: int = MAX_PAGES,
    visit_details: bool = True,
):
    """
    Main entry point.
    Args:
        listing_type: "ban" (for sale) or "cho-thue" (for rent)
        max_pages: number of listing pages to crawl
        visit_details: whether to visit each detail page for extra fields
    """
    if listing_type not in LISTING_URLS:
        raise ValueError(f"Invalid listing_type '{listing_type}'. Use 'ban' or 'cho-thue'.")

    listing_url = LISTING_URLS[listing_type]
    output_file = DATA_DIR / f"listings_{listing_type.replace('-', '_')}.json"
    log.info(f"Crawling {listing_type} listings from {listing_url}")

    all_listings: list[dict] = []

    STEALTH_SCRIPT = """
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.chrome = { runtime: {} };
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
    """

    CONTEXT_OPTS = dict(
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1440, "height": 900},
        locale="vi-VN",
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        for pg in range(1, max_pages + 1):
            # Fresh context per listing page to avoid anti-bot cookie buildup
            context = await browser.new_context(**CONTEXT_OPTS)
            await context.add_init_script(STEALTH_SCRIPT)
            page = await context.new_page()

            cards = await scrape_listing_page(page, pg, listing_url)
            if not cards:
                log.warning(f"No cards found on page {pg}, stopping.")
                await context.close()
                break

            if visit_details:
                for idx, card in enumerate(cards):
                    if not card.get("url"):
                        all_listings.append(merge_listing(card, {}, listing_type))
                        continue

                    detail = await scrape_detail_page(page, card["url"])
                    merged = merge_listing(card, detail, listing_type)
                    all_listings.append(merged)

                    # Log progress
                    if (idx + 1) % 5 == 0:
                        log.info(f"  Progress: {idx + 1}/{len(cards)} on page {pg}")

                    await asyncio.sleep(REQUEST_DELAY)
            else:
                for card in cards:
                    all_listings.append(merge_listing(card, {}, listing_type))

            log.info(f"  Total listings so far: {len(all_listings)}")
            await context.close()
            await asyncio.sleep(REQUEST_DELAY)

        await browser.close()

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_listings, f, ensure_ascii=False, indent=2)

    log.info(f"Saved {len(all_listings)} listings to {output_file}")
    return all_listings


# ---------------------------------------------------------------------------
# Shared browser setup
# ---------------------------------------------------------------------------
STEALTH_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    window.chrome = { runtime: {} };
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );
"""

CONTEXT_OPTS = dict(
    user_agent=(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    viewport={"width": 1440, "height": 900},
    locale="vi-VN",
)


async def _launch_browser(pw):
    return await pw.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )


async def _new_stealth_page(browser):
    context = await browser.new_context(**CONTEXT_OPTS)
    await context.add_init_script(STEALTH_SCRIPT)
    page = await context.new_page()
    return context, page


async def _goto_safe(page, url, retries=3):
    for attempt in range(retries):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            return True
        except Exception as e:
            log.warning(f"  Navigation attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                return False
            await asyncio.sleep(3)
    return False


# ---------------------------------------------------------------------------
# Dự án (Project) crawler
# ---------------------------------------------------------------------------
async def _extract_project_cards(page: Page) -> list[dict]:
    """Extract project cards from a project listing page."""
    return await page.evaluate("""() => {
        const results = [];
        // Project cards use .js__prj-card or similar selectors
        let cards = document.querySelectorAll('.js__card, .re__card-full, [class*="prj-card"], [data-tracking-id]');
        if (cards.length === 0) cards = document.querySelectorAll('a[href*="/du-an-"]');

        const seen = new Set();
        for (const card of cards) {
            const linkEl = card.querySelector('a[href*="/du-an-"], a[href*="-pj"]')
                || (card.tagName === 'A' && card.href && card.href.includes('du-an') ? card : null);
            const titleEl = card.querySelector('[class*="card-title"], h3, h2, .js__card-title')
                || card.querySelector('a[href*="-pj"]');
            const locEl = card.querySelector('[class*="card-location"], [class*="location"]');
            const priceEl = card.querySelector('[class*="config-price"], [class*="price"]');
            const areaEl = card.querySelector('[class*="config-area"], [class*="area"]');
            const imgEl = card.querySelector('img[src*="batdongsan"], img[data-src]');
            const statusEl = card.querySelector('[class*="status"], [class*="badge"]');

            const href = linkEl ? (linkEl.getAttribute('href') || '') : '';
            if (!href || seen.has(href) || !href.includes('pj') && !href.includes('du-an')) continue;
            seen.add(href);

            results.push({
                ten_du_an: titleEl ? titleEl.innerText.trim() : null,
                khu_vuc: locEl ? locEl.innerText.trim() : null,
                gia: priceEl ? priceEl.innerText.trim() : null,
                dien_tich: areaEl ? areaEl.innerText.trim() : null,
                trang_thai: statusEl ? statusEl.innerText.trim() : null,
                url: href,
                hinh_anh: imgEl ? (imgEl.getAttribute('src') || imgEl.getAttribute('data-src')) : null,
            });
        }
        return results;
    }""")


async def _scrape_project_detail(page: Page, url: str) -> dict:
    """Visit a project detail page and extract structured data."""
    detail = {}
    try:
        log.info(f"  Visiting project detail: {url}")
        await page.goto(url, wait_until="commit", timeout=30_000)
        await page.wait_for_load_state("domcontentloaded", timeout=15_000)
        await asyncio.sleep(3)

        detail = await page.evaluate("""() => {
            const d = {};
            // Project name — actual class is re__project-name
            const nameEl = document.querySelector('h1.re__project-name, .re__project-name, h1');
            if (nameEl) {
                const txt = nameEl.innerText.trim();
                // Skip if it's just the site name
                if (txt && txt !== 'batdongsan.com.vn') d.ten_du_an = txt;
            }

            // Status — actual class is re__prj-tag-info
            const statusEl = document.querySelector('.re__prj-tag-info');
            if (statusEl) d['Trạng thái'] = statusEl.innerText.trim();

            // Spec box items (key-value pairs as "title\\nvalue")
            const boxItems = document.querySelectorAll('.re__project-box-item');
            for (const item of boxItems) {
                const parts = item.innerText.trim().split('\\n');
                if (parts.length >= 2) d[parts[0].trim()] = parts[1].trim();
            }

            // Spec table rows
            const specRows = document.querySelectorAll('tbody.re__project-attr tr, .re__pr-specs-content-item');
            for (const row of specRows) {
                const cells = row.querySelectorAll('td, th, [class*="title"], [class*="value"]');
                if (cells.length >= 2) {
                    d[cells[0].innerText.trim()] = cells[1].innerText.trim();
                }
            }

            // Short info items
            const shortItems = document.querySelectorAll('.re__pr-short-info-item, [class*="short-info-item"]');
            for (const item of shortItems) {
                const k = item.querySelector('.title');
                const v = item.querySelector('.value');
                if (k && v) d[k.innerText.trim()] = v.innerText.trim();
            }

            // Config items
            const cfgItems = document.querySelectorAll('.re__prj-config-item, [class*="config-item"]');
            for (const item of cfgItems) {
                const k = item.querySelector('[class*="title"], .title, label');
                const v = item.querySelector('[class*="value"], .value');
                if (k && v) d[k.innerText.trim()] = v.innerText.trim();
            }

            // Description — actual class is js__prj-detail-content / re__project-editor
            const descEl = document.querySelector(
                '.js__prj-detail-content, .re__project-editor, .re__detail-content, .re__project-desc'
            );
            if (descEl) d._mo_ta_chi_tiet = descEl.innerText.trim().substring(0, 3000);

            // Address — actual class is re__project-address
            const addrEl = document.querySelector('.re__project-address, .re__pr-short-description--address');
            if (addrEl) d._dia_chi = addrEl.innerText.trim();

            // Images — filter out site chrome
            const imgs = [];
            const imgEls = document.querySelectorAll(
                '.re__project-album img, .re__media-thumb-item img, .slick-slide img, img[src*="file4.batdongsan"], img[src*="file1.batdongsan"]'
            );
            for (const img of imgEls) {
                const src = img.getAttribute('src') || img.getAttribute('data-src');
                if (src && !imgs.includes(src) && !src.includes('data:image')
                    && !src.includes('mobileSearch') && !src.includes('google-play')
                    && !src.includes('app_store') && !src.includes('footer')) imgs.push(src);
            }
            d._hinh_anh = imgs;

            // Amenities / utilities
            const utils = [];
            const utilEls = document.querySelectorAll(
                '.re__prj-facilities [class*="item"], [class*="utility"] [class*="item"], [class*="tien-ich"] li'
            );
            for (const u of utilEls) {
                const t = u.innerText.trim();
                if (t && !utils.includes(t)) utils.push(t);
            }
            if (utils.length) d._tien_ich = utils;

            return d;
        }""")
    except PwTimeout:
        log.warning(f"  Timeout loading project detail: {url}")
    except Exception as e:
        log.warning(f"  Error loading project detail {url}: {e}")
    return detail


PROJECT_SPEC_MAP = {
    "Chủ đầu tư": "chu_dau_tu",
    "Quy mô": "quy_mo",
    "Diện tích": "dien_tich",
    "Mức giá": "gia",
    "Giá": "gia",
    "Loại hình": "loai_du_an",
    "Pháp lý": "phap_ly",
    "Trạng thái": "trang_thai",
    "Số tòa": "so_toa",
    "Số căn hộ": "so_can_ho",
    "Năm bàn giao": "nam_ban_giao",
    "Năm khởi công": "nam_khoi_cong",
    "Mật độ xây dựng": "mat_do_xay_dung",
}


def _infer_project_name_from_url(url: str) -> Optional[str]:
    """Extract a readable project name from the URL slug (last path segment before pjNNNN)."""
    if not url:
        return None
    m = re.search(r'/([^/]+)-pj\d+', url)
    if m:
        slug = m.group(1)
        # Convert slug to title case: "fecon-ip-hiep-hoa" → "Fecon Ip Hiep Hoa"
        return slug.replace('-', ' ').title()
    return None


def _merge_project(card: dict, detail: dict) -> dict:
    # Determine project name: detail > card (if not junk) > infer from URL
    ten_du_an = detail.get("ten_du_an")
    if not ten_du_an:
        card_name = card.get("ten_du_an")
        if card_name and card_name != "batdongsan.com.vn":
            ten_du_an = card_name
    if not ten_du_an:
        ten_du_an = _infer_project_name_from_url(card.get("url"))

    record = {
        "ten_du_an": ten_du_an,
        "loai_du_an": None,
        "chu_dau_tu": None,
        "khu_vuc": card.get("khu_vuc"),
        "dia_chi": None,
        "quy_mo": None,
        "dien_tich": card.get("dien_tich"),
        "gia": card.get("gia"),
        "trang_thai": card.get("trang_thai"),
        "phap_ly": None,
        "so_toa": None,
        "so_can_ho": None,
        "nam_ban_giao": None,
        "nam_khoi_cong": None,
        "mat_do_xay_dung": None,
        "mo_ta_chi_tiet": None,
        "tien_ich": [],
        "url": card.get("url"),
        "hinh_anh": [card["hinh_anh"]] if card.get("hinh_anh") else [],
    }
    for vn_key, field in PROJECT_SPEC_MAP.items():
        if vn_key in detail:
            record[field] = detail[vn_key]
    if detail.get("_mo_ta_chi_tiet"):
        record["mo_ta_chi_tiet"] = detail["_mo_ta_chi_tiet"]
    if detail.get("_dia_chi"):
        record["dia_chi"] = detail["_dia_chi"]
    if detail.get("_hinh_anh"):
        record["hinh_anh"] = detail["_hinh_anh"]
    if detail.get("_tien_ich"):
        record["tien_ich"] = detail["_tien_ich"]
    # Infer loai_du_an from URL
    if not record["loai_du_an"] and record.get("url"):
        u = record["url"]
        type_map = {
            "can-ho-chung-cu": "Căn hộ chung cư", "cao-oc-van-phong": "Cao ốc văn phòng",
            "trung-tam-thuong-mai": "Trung tâm thương mại", "khu-do-thi-moi": "Khu đô thị mới",
            "khu-phuc-hop": "Khu phức hợp", "nha-o-xa-hoi": "Nhà ở xã hội",
            "khu-nghi-duong": "Khu nghỉ dưỡng", "khu-cong-nghiep": "Khu công nghiệp",
            "biet-thu-lien-ke": "Biệt thự, liền kề", "shophouse": "Shophouse",
            "nha-mat-pho": "Nhà mặt phố",
        }
        for slug, name in type_map.items():
            if slug in u:
                record["loai_du_an"] = name
                break
    return record


async def crawl_projects(max_pages: int = MAX_PAGES, visit_details: bool = True):
    """Crawl project listings from batdongsan.com.vn/du-an-bat-dong-san."""
    output_file = DATA_DIR / "projects.json"
    log.info(f"Crawling projects from {PROJECT_URL}")
    all_projects: list[dict] = []

    async with async_playwright() as pw:
        browser = await _launch_browser(pw)
        for pg in range(1, max_pages + 1):
            context, page = await _new_stealth_page(browser)
            url = PROJECT_URL if pg == 1 else f"{PROJECT_URL}/p{pg}"
            log.info(f"Navigating to project page {pg}: {url}")

            if not await _goto_safe(page, url):
                await context.close()
                break
            await asyncio.sleep(5)
            await _wait_for_listings(page)
            await asyncio.sleep(2)

            cards = await _extract_project_cards(page)
            if not cards:
                log.warning(f"No project cards on page {pg}, stopping.")
                await context.close()
                break

            for c in cards:
                if c.get("url") and not c["url"].startswith("http"):
                    c["url"] = BASE_URL + c["url"]

            log.info(f"  Found {len(cards)} projects on page {pg}")

            if visit_details:
                for idx, card in enumerate(cards):
                    if not card.get("url"):
                        all_projects.append(_merge_project(card, {}))
                        continue
                    detail = await _scrape_project_detail(page, card["url"])
                    all_projects.append(_merge_project(card, detail))
                    if (idx + 1) % 5 == 0:
                        log.info(f"  Progress: {idx+1}/{len(cards)} on page {pg}")
                    await asyncio.sleep(REQUEST_DELAY)
            else:
                for card in cards:
                    all_projects.append(_merge_project(card, {}))

            log.info(f"  Total projects so far: {len(all_projects)}")
            await context.close()
            await asyncio.sleep(REQUEST_DELAY)
        await browser.close()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_projects, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(all_projects)} projects to {output_file}")
    return all_projects


# ---------------------------------------------------------------------------
# Tin tức (News) & Wiki BĐS — article crawlers
# ---------------------------------------------------------------------------
async def _extract_article_cards(page: Page) -> list[dict]:
    """Extract article cards from news/wiki listing page."""
    return await page.evaluate("""() => {
        const results = [];
        const seen = new Set();
        // Try multiple selectors for article cards
        const cards = document.querySelectorAll(
            '[class*="news-item"], [class*="wiki-item"], article, '
            + '.re__card-full, .js__card, [data-tracking-id], '
            + 'a[href*="/tin-tuc/"][href*="-"], a[href*="/wiki/"][href*="-"]'
        );

        for (const card of cards) {
            let linkEl = card.querySelector('a[href*="/tin-tuc/"], a[href*="/wiki/"]');
            if (!linkEl && card.tagName === 'A') linkEl = card;
            if (!linkEl) continue;

            const href = linkEl.getAttribute('href') || '';
            // Filter: must be an article URL (contains digits at end = article ID)
            if (!href || seen.has(href)) continue;
            if (!(/\\d{4,}$/.test(href.replace(/\\/$/, '')))) continue;
            seen.add(href);

            const titleEl = card.querySelector('h2, h3, [class*="title"]') || linkEl;
            const descEl = card.querySelector('[class*="description"], [class*="summary"], p');
            const dateEl = card.querySelector('[class*="date"], [class*="time"], time');
            const catEl = card.querySelector('[class*="category"], [class*="tag"]');
            const imgEl = card.querySelector('img[src], img[data-src]');

            results.push({
                tieu_de: titleEl ? titleEl.innerText.trim() : null,
                mo_ta: descEl ? descEl.innerText.trim() : null,
                ngay_dang: dateEl ? dateEl.innerText.trim() : null,
                danh_muc: catEl ? catEl.innerText.trim() : null,
                url: href,
                hinh_anh: imgEl ? (imgEl.getAttribute('src') || imgEl.getAttribute('data-src')) : null,
            });
        }
        return results;
    }""")


async def _scrape_article_detail(page: Page, url: str) -> dict:
    """Visit an article detail page and extract content."""
    detail = {}
    try:
        log.info(f"  Visiting article: {url}")
        await page.goto(url, wait_until="commit", timeout=30_000)
        await page.wait_for_load_state("domcontentloaded", timeout=15_000)
        await asyncio.sleep(3)

        detail = await page.evaluate(r"""() => {
            const d = {};
            // Title
            const titleEl = document.querySelector('h1');
            if (titleEl) d.tieu_de = titleEl.innerText.trim();

            // Description from meta OG tag
            const ogDesc = document.querySelector('meta[property="og:description"]');
            if (ogDesc) d.mo_ta = ogDesc.getAttribute('content');

            // Author — try author link, but verify it has text
            const authorLink = document.querySelector('a[href*="/tac-gia/"]');
            const authorLinkText = authorLink ? authorLink.innerText.trim() : '';
            if (authorLinkText) {
                d.tac_gia = authorLinkText;
            }
            // Fallback: grab the full author area block (will be parsed in Python)
            if (!d.tac_gia) {
                const authorEl = document.querySelector('[class*="author"]');
                if (authorEl) d.tac_gia = authorEl.innerText.trim();
            }

            // Date — extract from author area text
            const authorArea = document.querySelector('[class*="author"], [class*="post-meta"]');
            if (authorArea) {
                const text = authorArea.innerText || '';
                const dateMatch = text.match(/(\d{2}\/\d{2}\/\d{4}(?:\s+\d{2}:\d{2})?)/);
                if (dateMatch) d.ngay_dang = dateMatch[1];
            }
            if (!d.ngay_dang) {
                const timeEl = document.querySelector('time[datetime]');
                if (timeEl) d.ngay_dang = timeEl.getAttribute('datetime') || timeEl.innerText.trim();
            }

            // Category from breadcrumb
            const bcLinks = document.querySelectorAll('.re__breadcrumb a, [class*="breadcrumb"] a');
            if (bcLinks.length >= 2) d.danh_muc = bcLinks[bcLinks.length - 1].innerText.trim();

            // Full content — try multiple selectors (article/main work on wiki subdomain)
            const contentSelectors = [
                '.re__detail-content .js__section-body',
                '.js__section-body',
                '.re__detail-content.re__project-editor',
                '.re__project-editor',
                '.re__detail-content',
                '.re__section-body',
                'article',
                'main',
                '[class*="article-body"]',
                '[class*="post-content"]',
            ];
            let bodyEl = null;
            for (const sel of contentSelectors) {
                bodyEl = document.querySelector(sel);
                if (bodyEl && bodyEl.innerText.trim().length > 50) break;
            }
            if (bodyEl) d.noi_dung = bodyEl.innerText.trim().substring(0, 5000);

            // Images — filter junk site-chrome images
            const junkPatterns = [
                'mobileSearch', 'authorDefault', 'google-play', 'app_store',
                'footer', 'logo', 'icon', 'avatar', 'placeholder',
                'staticfile.batdongsan.com.vn', 'cdn-assets-angel'
            ];
            const imgs = [];
            const imgArea = bodyEl || document;
            const imgEls = imgArea.querySelectorAll('img[src]');
            for (const img of imgEls) {
                const src = img.getAttribute('src') || img.getAttribute('data-src');
                if (!src || src.includes('data:image')) continue;
                const isJunk = junkPatterns.some(p => src.includes(p));
                if (!isJunk && !imgs.includes(src)) imgs.push(src);
            }
            d.hinh_anh = imgs;

            return d;
        }""")
    except PwTimeout:
        log.warning(f"  Timeout loading article: {url}")
    except Exception as e:
        log.warning(f"  Error loading article {url}: {e}")
    return detail


_IMAGE_JUNK_PATTERNS = [
    "mobileSearch", "authorDefault", "google-play", "app_store",
    "footer", "logo", "icon", "avatar", "placeholder",
    "staticfile.batdongsan.com.vn/images", "cdn-assets-angel",
]


def _filter_images(imgs: list) -> list:
    """Remove junk site-chrome images from a list of URLs."""
    filtered = []
    for src in imgs:
        if not src:
            continue
        if any(p in src for p in _IMAGE_JUNK_PATTERNS):
            continue
        if src not in filtered:
            filtered.append(src)
    return filtered


def _parse_author_block(raw: str):
    """Parse combined author block like 'Được đăng bởi X\\nCập nhật ... dd/mm/yyyy HH:MM ...'
    Returns (author_name, date_str)."""
    author = None
    date_str = None
    if not raw:
        return author, date_str

    # Extract author name: "Được đăng bởi <name>"
    m = re.search(r"Được đăng bởi\s+(.+?)(?:\n|$)", raw)
    if m:
        author = m.group(1).strip()
    else:
        # If the string is just a name (no "Được đăng bởi" prefix)
        lines = raw.strip().split("\n")
        if lines and len(lines[0]) < 50:
            author = lines[0].strip()

    # Extract date: dd/mm/yyyy or dd/mm/yyyy HH:MM
    m = re.search(r"(\d{2}/\d{2}/\d{4}(?:\s+\d{2}:\d{2})?)", raw)
    if m:
        date_str = m.group(1)

    return author, date_str


def _merge_article(card: dict, detail: dict, section: str = "tin-tuc", category: str = None) -> dict:
    # Get mo_ta: prefer detail's OG description, then card's
    mo_ta = detail.get("mo_ta") or card.get("mo_ta")

    # Get tac_gia and ngay_dang — parse from combined string if needed
    tac_gia = detail.get("tac_gia")
    ngay_dang = detail.get("ngay_dang") or card.get("ngay_dang")

    # If tac_gia looks like a combined block, parse it
    if tac_gia and ("Được đăng bởi" in tac_gia or "Cập nhật" in tac_gia):
        parsed_author, parsed_date = _parse_author_block(tac_gia)
        tac_gia = parsed_author
        if not ngay_dang and parsed_date:
            ngay_dang = parsed_date

    # Build images list and filter junk
    imgs = detail.get("hinh_anh") or ([card["hinh_anh"]] if card.get("hinh_anh") else [])
    imgs = _filter_images(imgs)

    record = {
        "loai": section,
        "danh_muc": category or card.get("danh_muc"),
        "tieu_de": detail.get("tieu_de") or card.get("tieu_de"),
        "mo_ta": mo_ta,
        "noi_dung": detail.get("noi_dung"),
        "tac_gia": tac_gia,
        "ngay_dang": ngay_dang,
        "url": card.get("url"),
        "hinh_anh": imgs,
    }
    if detail.get("danh_muc") and not record["danh_muc"]:
        record["danh_muc"] = detail["danh_muc"]
    return record


async def _crawl_articles(
    section_url: str,
    section_name: str,
    output_file: Path,
    max_pages: int = MAX_PAGES,
    visit_details: bool = True,
    category: str = None,
):
    """Generic article crawler used by both news and wiki."""
    log.info(f"Crawling {section_name} from {section_url}")
    all_articles: list[dict] = []

    async with async_playwright() as pw:
        browser = await _launch_browser(pw)
        for pg in range(1, max_pages + 1):
            context, page = await _new_stealth_page(browser)
            url = section_url if pg == 1 else f"{section_url}/p{pg}"
            log.info(f"Navigating to {section_name} page {pg}: {url}")

            if not await _goto_safe(page, url):
                await context.close()
                break
            await asyncio.sleep(5)

            # Scroll to trigger lazy load
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await asyncio.sleep(2)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

            cards = await _extract_article_cards(page)
            if not cards:
                log.warning(f"No article cards on {section_name} page {pg}, stopping.")
                await context.close()
                break

            for c in cards:
                if c.get("url") and not c["url"].startswith("http"):
                    c["url"] = BASE_URL + c["url"]

            log.info(f"  Found {len(cards)} articles on page {pg}")

            if visit_details:
                for idx, card in enumerate(cards):
                    if not card.get("url"):
                        all_articles.append(_merge_article(card, {}, section_name, category))
                        continue
                    detail = await _scrape_article_detail(page, card["url"])
                    all_articles.append(_merge_article(card, detail, section_name, category))
                    if (idx + 1) % 5 == 0:
                        log.info(f"  Progress: {idx+1}/{len(cards)} on page {pg}")
                    await asyncio.sleep(REQUEST_DELAY)
            else:
                for card in cards:
                    all_articles.append(_merge_article(card, {}, section_name, category))

            log.info(f"  Total articles so far: {len(all_articles)}")
            await context.close()
            await asyncio.sleep(REQUEST_DELAY)
        await browser.close()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(all_articles)} articles to {output_file}")
    return all_articles


async def crawl_news(max_pages: int = MAX_PAGES, visit_details: bool = True):
    """Crawl news articles from /tin-tuc."""
    return await _crawl_articles(
        section_url=NEWS_URL,
        section_name="tin-tuc",
        output_file=DATA_DIR / "news.json",
        max_pages=max_pages,
        visit_details=visit_details,
        category="Tin tức",
    )


async def crawl_wiki(
    max_pages: int = MAX_PAGES,
    visit_details: bool = True,
    wiki_category: str = None,
):
    """
    Crawl wiki articles from /wiki.
    If wiki_category is specified (e.g. 'mua-bds'), crawl only that sub-category.
    Otherwise crawl all sub-categories.
    """
    if wiki_category:
        cats = {wiki_category: WIKI_CATEGORIES.get(wiki_category, wiki_category)}
    else:
        cats = WIKI_CATEGORIES

    all_wiki: list[dict] = []
    for slug, name in cats.items():
        section_url = f"{WIKI_URL}/{slug}"
        out_file = DATA_DIR / f"wiki_{slug.replace('-', '_')}.json"
        articles = await _crawl_articles(
            section_url=section_url,
            section_name="wiki",
            output_file=out_file,
            max_pages=max_pages,
            visit_details=visit_details,
            category=name,
        )
        all_wiki.extend(articles)

    # Also save a combined file
    combined = DATA_DIR / "wiki_all.json"
    combined.parent.mkdir(parents=True, exist_ok=True)
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(all_wiki, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(all_wiki)} total wiki articles to {combined}")
    return all_wiki


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    lt = sys.argv[1] if len(sys.argv) > 1 else "ban"
    if lt == "du-an":
        asyncio.run(crawl_projects())
    elif lt == "tin-tuc":
        asyncio.run(crawl_news())
    elif lt == "wiki":
        asyncio.run(crawl_wiki(wiki_category="mua-bds"))
    else:
        asyncio.run(crawl(listing_type=lt))
