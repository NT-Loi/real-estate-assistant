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

from playwright.async_api import async_playwright, Page, TimeoutError as PwTimeout

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://batdongsan.com.vn"
LISTING_URLS = {
    "ban": f"{BASE_URL}/nha-dat-ban",
    "cho-thue": f"{BASE_URL}/nha-dat-cho-thue",
}
MAX_PAGES = 3  # Number of listing pages to crawl (change as needed)
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
if __name__ == "__main__":
    import sys
    lt = sys.argv[1] if len(sys.argv) > 1 else "ban"
    asyncio.run(crawl(listing_type=lt))
