import asyncio
import re
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright, Page

from crawlers.base import BaseCrawler
from crawlers.browser import launch_browser, new_stealth_page, goto_safe
from crawlers.config import LISTING_URLS, BASE_URL, REQUEST_DELAY

SPEC_KEY_MAP = {
    "Mức giá": "gia",
    "Giá": "gia",
    "Khoảng giá": "gia",
    "Giá thuê": "gia",
    "Giá cho thuê": "gia",
    "Diện tích": "dien_tich",
    "Diện tích sử dụng": "dien_tich",
    "Địa chỉ": "dia_chi",
    "Hướng nhà": "huong_nha",
    "Hướng": "huong_nha",
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
    "Vị trí": "dia_chi",
    "Đơn giá": "gia_per_m2",
    "Giá/m²": "gia_per_m2",
    "Giá/m2": "gia_per_m2",
    "Chiều dài": "chieu_dai",
    "Chiều rộng": "chieu_rong",
}

class ListingCrawler(BaseCrawler):
    """Crawler for property listings (For Sale and For Rent) on batdongsan.com.vn."""
    
    def __init__(self, listing_type: str = "ban", output_file = None):
        if listing_type not in LISTING_URLS:
            raise ValueError(f"Invalid listing_type '{listing_type}'. Must be 'ban' or 'cho-thue'")
        super().__init__(f"listings_{listing_type.replace('-', '_')}", output_file)
        self.listing_type = listing_type
        self.listing_url = LISTING_URLS[listing_type]

    async def _wait_for_listings(self, page: Page) -> bool:
        """Wait until listing cards are visible on the page."""
        for selector in [".js__card", "[data-tracking-id]", ".re__card-full"]:
            try:
                await page.wait_for_selector(selector, timeout=10_000)
                return True
            except Exception:
                continue
        return False

    async def _extract_cards(self, page: Page) -> List[Dict[str, Any]]:
        """Run Javascript extraction to collect all listing cards from current page."""
        cards = await page.evaluate(r"""() => {
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
                const addressEl = card.querySelector('.re__card-address')
                    || card.querySelector('.re__card-location')
                    || card.querySelector('[class*="card-address"]')
                    || card.querySelector('[class*="card-location"]')
                    || card.querySelector('[class*="address"]');
                const locationEl = addressEl || card.querySelector('.re__card-location')
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
                const linkEl = card.querySelector('a.js__product-link-for-product-id')
                    || card.querySelector('a[class*="product-link"]')
                    || card.querySelector('a[href*="/ban-"]')
                    || card.querySelector('a[href*="/cho-thue-"]')
                    || card.querySelector('a[href*="/nha-"]')
                    || card.querySelector('a');
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
                        dia_chi: addressEl ? addressEl.innerText.trim() : null,
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

    async def scrape_detail_page(self, page: Page, url: str) -> Dict[str, Any]:
        """Visit detail page of a listing to parse thorough specs and full content."""
        detail = {}
        try:
            self.log.info(f"  Visiting detail: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            
            # Check for Cloudflare / Security Verification wall
            title = await page.title()
            is_cf = any(kw in title for kw in ["Chờ một chút", "Xác minh bảo mật", "Just a moment", "Cloudflare"])
            self.log.info(f"  Page title: '{title}' (Security wall: {is_cf})")
            if is_cf:
                self.log.info("  Waiting 12 seconds for auto-redirect...")
                await asyncio.sleep(12)
            else:
                await asyncio.sleep(3)

            # Wait for content to load to ensure bypass succeeded
            try:
                await page.wait_for_selector(".re__detail-content, .re__pr-description, [class*='detail-content']", timeout=10000)
            except Exception:
                self.log.warning("  Timeout waiting for listing detail content selectors. Verification challenge may still be active.")

            detail = await page.evaluate(r"""() => {
                const specs = {};

                // Method 1: Spec items (generic)
                const specItems = document.querySelectorAll('.re__pr-specs-content-item');
                for (const item of specItems) {
                    const titleEl = item.querySelector('.re__pr-specs-content-item-title') || item.querySelector('.title') || item.querySelector('strong');
                    const valueEl = item.querySelector('.re__pr-specs-content-item-value') || item.querySelector('.value') || item.querySelector('span:last-child');
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

                // Method 3: Generic table-like layouts
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
                
                // Method 4: div pairs
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

                // Full description - Crawl full descriptions (up to 15,000 characters)
                const descEl = document.querySelector('.re__detail-content .re__section-body')
                    || document.querySelector('[class*="detail-content"] [class*="section-body"]')
                    || document.querySelector('.re__detail-content')
                    || document.querySelector('.re__pr-description .re__section-body');
                if (descEl) {
                    specs['_mo_ta_chi_tiet'] = descEl.innerText.trim().substring(0, 15000);
                }

                // Explicitly handle "Đặc điểm bất động sản" section: some pages list spec items here
                try {
                    const sections = document.querySelectorAll('.re__section');
                    for (const sec of sections) {
                        const title = sec.querySelector('.re__section-title');
                        if (title && /Đặc điểm/i.test(title.innerText)) {
                            const body = sec.querySelector('.re__section-body');
                            if (body) {
                                // extract structured items inside this section (if any)
                                const items = body.querySelectorAll('.re__pr-specs-content-item');
                                for (const item of items) {
                                    const t = item.querySelector('.re__pr-specs-content-item-title') || item.querySelector('.title') || item.querySelector('strong');
                                    const v = item.querySelector('.re__pr-specs-content-item-value') || item.querySelector('.value') || item.querySelector('span:last-child');
                                    if (t && v) {
                                        specs[t.innerText.trim()] = v.innerText.trim();
                                    }
                                }
                                // fallback: save raw section text if structured items aren't present
                                if (Object.keys(specs).length === 0) {
                                    specs['_dac_diem_bat_dong_san'] = body.innerText.trim().substring(0,15000);
                                }
                            }
                        }
                    }
                } catch (e) {
                    // ignore section-specific parsing errors
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
                specs['_hinh_anh'] = images;

                // Address - try multiple selectors, meta tags, then generic heuristics
                let addr = null;
                const addressSelectors = [
                    '.re__pr-short-description--address',
                    '.re__pr-address',
                    '[class*="short-description--address"]',
                    '[class*="address"]',
                    '[itemprop="address"]',
                    '.product-address',
                    '.re__pr-short-description--address .address'
                ];
                for (const sel of addressSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText && el.innerText.trim()) {
                        addr = el.innerText.trim();
                        break;
                    }
                }
                // meta tag fallback
                if (!addr) {
                    const meta = document.querySelector('meta[property="og:street-address"], meta[name="address"], meta[property="og:address"]');
                    if (meta && meta.getAttribute('content')) {
                        addr = meta.getAttribute('content').trim();
                    }
                }
                // Generic search for nodes containing the label "Địa chỉ" or "Address"
                if (!addr) {
                    const candidates = document.querySelectorAll('p, span, div, li');
                    for (const c of candidates) {
                        const txt = c.innerText || '';
                        if (txt && /địa chỉ|Địa chỉ|Address/i.test(txt) && txt.length < 300) {
                            addr = txt.replace(/(Địa chỉ|địa chỉ|Address)[:\s\-]*/i, '').trim();
                            if (addr) break;
                        }
                    }
                }
                if (addr) {
                    specs['_dia_chi'] = addr;
                }

                // Project Title/Name associated with listing
                const projectTitleEl = document.querySelector('.re__project-title')
                    || document.querySelector('[class*="project-title"]');
                if (projectTitleEl) {
                    specs['_du_an'] = projectTitleEl.innerText.trim();
                }

                return specs;
            }""")
        except Exception as e:
            self.log.warning(f"  Error loading detail page {url}: {e}")
        return detail

    def merge_listing(self, card: Dict[str, Any], detail_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge listing summaries with page detailed specifics."""
        record = {
            "loai_hinh": self.listing_type,
            "loai_nha_dat": None,
            "khu_vuc": None,
            "dia_chi": None,
            "gia": None,
            "gia_per_m2": None,
            "dien_tich": None,
            "so_phong_ngu": None,
            "so_phong_tam": None,
            "huong_nha": None,
            "huong_ban_cong": None,
            "phap_ly": None,
            "noi_that": None,
            "tieu_de": None,
            "mo_ta": None,
            "mo_ta_chi_tiet": None,
            "du_an": None,
            "so_tang": None,
            "mat_tien": None,
            "duong_vao": None,
            "chieu_dai": None,
            "chieu_rong": None,
            "url": None,
            "hinh_anh": [],
            "ngay_dang": None,
            "nguoi_dang": None,
        }

        # Cards summary
        record["tieu_de"] = card.get("tieu_de")
        record["gia"] = card.get("gia")
        record["dien_tich"] = card.get("dien_tich")
        record["gia_per_m2"] = card.get("gia_per_m2")
        
        khu_vuc = card.get("khu_vuc")
        if khu_vuc:
            khu_vuc = re.sub(r"^[·\s]+", "", khu_vuc).strip()
        record["khu_vuc"] = khu_vuc or None
        record["so_phong_ngu"] = card.get("so_phong_ngu")
        record["so_phong_tam"] = card.get("so_phong_tam")
        # Prefer a single description field: normalize to `mo_ta_chi_tiet`
        card_desc = card.get("mo_ta")
        record["url"] = card.get("url")
        record["ngay_dang"] = card.get("ngay_dang")
        record["nguoi_dang"] = card.get("nguoi_dang")
        if card.get("hinh_anh"):
            record["hinh_anh"] = [card["hinh_anh"]]

        # Enrich details
        # Map Vietnamese spec keys from detail page into our record fields.
        for vn_key, field_name in SPEC_KEY_MAP.items():
            for ds_key, ds_val in detail_specs.items():
                if not ds_key or ds_val is None:
                    continue
                norm_vn = str(vn_key).strip().lower()
                norm_ds = str(ds_key).strip().lower()
                if norm_vn == norm_ds or norm_vn in norm_ds or norm_ds in norm_vn:
                    record[field_name] = ds_val
                    break

        # Description: prefer detailed description from the detail page,
        # otherwise use the card-level short description. Store only `mo_ta_chi_tiet`.
        if detail_specs.get("_mo_ta_chi_tiet"):
            record["mo_ta_chi_tiet"] = detail_specs["_mo_ta_chi_tiet"]
        elif card_desc:
            record["mo_ta_chi_tiet"] = card_desc
        # Clear the short `mo_ta` to avoid duplicate fields — downstream expects single description.
        record["mo_ta"] = None

        # Address: priority — detail page, card-level address, then khu_vuc
        if detail_specs.get("_dia_chi"):
            record["dia_chi"] = detail_specs["_dia_chi"]
        elif card.get("dia_chi"):
            record["dia_chi"] = card.get("dia_chi")
        elif record.get("khu_vuc"):
            record["dia_chi"] = record.get("khu_vuc")
        if detail_specs.get("_hinh_anh") and len(detail_specs["_hinh_anh"]) > 0:
            record["hinh_anh"] = detail_specs["_hinh_anh"]
        if detail_specs.get("_du_an"):
            record["du_an"] = detail_specs["_du_an"]

        # Infer loai_nha_dat from URL slug
        if not record["loai_nha_dat"] and record.get("url"):
            url = record["url"]
            if "ban-can-ho" in url or "thue-can-ho" in url or "cho-thue-can-ho" in url:
                record["loai_nha_dat"] = "Căn hộ chung cư"
            elif "ban-nha-rieng" in url or "thue-nha-rieng" in url or "cho-thue-nha-rieng" in url:
                record["loai_nha_dat"] = "Nhà riêng"
            elif "ban-nha-biet-thu" in url or "thue-nha-biet-thu" in url or "cho-thue-nha-biet-thu" in url:
                record["loai_nha_dat"] = "Nhà biệt thự, liền kề"
            elif "ban-nha-mat-pho" in url or "thue-nha-mat-pho" in url or "cho-thue-nha-mat-pho" in url:
                record["loai_nha_dat"] = "Nhà mặt phố"
            elif "ban-dat" in url:
                record["loai_nha_dat"] = "Đất"
            elif "thue-mat-bang" in url or "cho-thue-mat-bang" in url:
                record["loai_nha_dat"] = "Mặt bằng"
            elif "thue-van-phong" in url or "cho-thue-van-phong" in url:
                record["loai_nha_dat"] = "Văn phòng"
            elif "thue-cua-hang" in url or "cho-thue-cua-hang" in url:
                record["loai_nha_dat"] = "Cửa hàng, ki ốt"
            elif "thue-phong-tro" in url or "cho-thue-phong-tro" in url:
                record["loai_nha_dat"] = "Phòng trọ"

        return record

    async def crawl(self, max_pages: int = 1, visit_details: bool = True, resume: bool = False) -> List[Dict[str, Any]]:
        """Main crawl flow executing the batdongsan crawler."""
        self.log.info(f"Starting {self.listing_type} crawl. Target pages: {max_pages}")
        
        all_listings: List[Dict[str, Any]] = []
        start_page = 1
        
        if resume:
            self.checkpoint_mgr.load()
            start_page = self.checkpoint_mgr.get_last_page() + 1
            all_listings = self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resuming {self.listing_type} crawl from page {start_page}. Items loaded: {len(all_listings)}")

        if start_page > max_pages:
            self.log.info(f"Target pages {max_pages} already reached/exceeded in loaded checkpoint.")
            return all_listings

        async with async_playwright() as pw:
            browser = await launch_browser(pw)
            
            for pg in range(start_page, max_pages + 1):
                # Fresh browser context per page to thwart session/cookie tracking
                context, page = await new_stealth_page(browser)
                
                url = self.listing_url if pg == 1 else f"{self.listing_url}/p{pg}"
                self.log.info(f"Navigating to page {pg}: {url}")
                
                if not await goto_safe(page, url):
                    self.log.warning(f"Failed to navigate to {url}. Skipping page {pg}")
                    await context.close()
                    continue

                await asyncio.sleep(5)
                found = await self.sleep_polite() or await self._wait_for_listings(page)
                if not found:
                    self.log.warning(f"No cards found on page {pg}. Scrolling...")
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await asyncio.sleep(2)
                    found = await self._wait_for_listings(page)
                
                if not found:
                    self.log.error(f"Still no listings on page {pg}. Stop crawler.")
                    await context.close()
                    break

                cards = await self._extract_cards(page)
                self.log.info(f"Extracted {len(cards)} listings on page {pg}")

                page_listings = []
                for idx, card in enumerate(cards):
                    card_url = card.get("url")
                    if card_url:
                        if not card_url.startswith("http"):
                            card["url"] = BASE_URL + card_url
                        
                        # Dedup check
                        if self.checkpoint_mgr.is_seen(card["url"]):
                            continue

                    detail_specs = {}
                    if visit_details and card.get("url"):
                        # Use a fresh stealth context/page for detail page to bypass Cloudflare completely
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            detail_context, detail_page = await new_stealth_page(browser)
                            detail_specs = await self.scrape_detail_page(detail_page, card["url"])
                            await detail_context.close()
                            # Consider detail retrieval successful if we got any non-empty data (not just _mo_ta_chi_tiet)
                            if detail_specs and any(v for v in detail_specs.values() if v):
                                break
                            if attempt < max_attempts - 1:
                                self.log.info(f"  Listing detail page returned empty. Retrying with new context (Attempt {attempt+2}/{max_attempts})...")
                                await asyncio.sleep(3)
                        await self.sleep_polite()

                    merged = self.merge_listing(card, detail_specs)
                    page_listings.append(merged)
                    all_listings.append(merged)
                    
                    if card.get("url"):
                        self.checkpoint_mgr.add_seen(card["url"])

                    if (idx + 1) % 5 == 0:
                        self.log.info(f"  Progress page {pg}: {idx+1}/{len(cards)}")

                await context.close()
                self.checkpoint_mgr.save(pg, page_listings)
                await self.sleep_polite(REQUEST_DELAY * 2)

            await browser.close()
            
        self.save_final_results(all_listings, resume)
        return all_listings
