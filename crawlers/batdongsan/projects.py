import asyncio
import re
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright, Page

from crawlers.base import BaseCrawler
from crawlers.browser import launch_browser, new_stealth_page, goto_safe
from crawlers.config import PROJECT_URL, BASE_URL, REQUEST_DELAY

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
    "Bàn giao": "nam_ban_giao",
    "Năm khởi công": "nam_khoi_cong",
    "Khởi công": "nam_khoi_cong",
    "Mật độ xây dựng": "mat_do_xay_dung",
}

class ProjectCrawler(BaseCrawler):
    """Crawler for real estate project listings on batdongsan.com.vn."""
    
    def __init__(self, output_file = None):
        super().__init__("projects", output_file)

    async def _wait_for_projects(self, page: Page) -> bool:
        """Wait until project listing elements appear on the page."""
        for selector in [".js__card", ".re__card-full", "[class*='prj-card']", "a[href*='/du-an-']"]:
            try:
                await page.wait_for_selector(selector, timeout=10_000)
                return True
            except Exception:
                continue
        return False

    async def _extract_project_cards(self, page: Page) -> List[Dict[str, Any]]:
        """Extract project summary card elements via browser evaluation."""
        return await page.evaluate("""() => {
            const results = [];
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
                if (!href || seen.has(href) || (!href.includes('pj') && !href.includes('du-an'))) continue;
                seen.add(href);

                results.push({
                    ten_du_an: titleEl ? titleEl.innerText.trim() : null,
                    khu_vuc: locEl ? locEl.innerText.trim() : null,
                    gia: priceEl ? priceEl.innerText.trim() : null,
                    dien_tich: areaEl ? areaEl.innerText.trim() : null,
                    trang_thai: statusEl ? statusEl.innerText.trim() : null,
                    url: href,
                });
            }
            return results;
        }""")

    async def _scrape_project_detail(self, page: Page, url: str) -> Dict[str, Any]:
        """Scrape structured specifications from a project detail page."""
        detail = {}
        try:
            self.log.info(f"  Visiting project detail: {url}")
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
                await page.wait_for_selector(".re__project-box-item, tr, .re__project-name, [class*='project-box-item']", timeout=10000)
            except Exception:
                self.log.warning("  Timeout waiting for project detail content selectors. Verification challenge may still be active.")

            # Try to expand collapsed sections and reveal hidden content (click "Xem thêm" / toggles)
            try:
                await page.evaluate("""() => {
                    const clickByText = (re) => {
                        const elements = Array.from(document.querySelectorAll('button, a, span, div, li'));
                        for (const el of elements) {
                            try {
                                const txt = (el.innerText || '').trim();
                                if (txt && re.test(txt)) el.click();
                            } catch (e) {}
                        }
                    };
                    // Click common 'Xem thêm' patterns and equivalents
                    clickByText(/xem thêm|xem tiếp|read more|view more|xem thêm chi tiết/i);

                    // Click known toggle/button selectors that expand content
                    const toggleSelectors = [
                        '.re__section .re__section-toggle', '.re__project-more', '.js__toggle', '[data-action="toggle"]',
                        '.re__toggle', '.icon-angle-down', '.icon-arrow-down', '.re__more', '.js__prj-more', '.re__view-more'
                    ];
                    for (const sel of toggleSelectors) {
                        const els = document.querySelectorAll(sel) || [];
                        for (const e of els) {
                            try { e.click(); } catch (ex) {}
                        }
                    }

                    // Click headings that likely toggle the sections
                    const headings = Array.from(document.querySelectorAll('h2, h3, .re__section-title, .title')).filter(el => /thông tin chi tiết|tiện ích|tien ich/i.test(el.innerText || ''));
                    for (const h of headings) { try { h.click(); } catch (e) {} }

                    // Trigger scroll to reveal lazy-loaded content
                    window.scrollTo(0, document.body.scrollHeight/2);
                    window.scrollTo(0, document.body.scrollHeight);
                }""")
                await asyncio.sleep(1)
            except Exception:
                self.log.warning("  Could not auto-expand project detail sections.")

            detail = await page.evaluate("""() => {
                const d = {};
                // Project Name
                const nameEl = document.querySelector('h1.re__project-name, .re__project-name, h1');
                if (nameEl) {
                    const txt = nameEl.innerText.trim();
                    if (txt && txt !== 'batdongsan.com.vn') d.ten_du_an = txt;
                }

                // Project Status
                const statusEl = document.querySelector('.re__prj-tag-info');
                if (statusEl) d['Trạng thái'] = statusEl.innerText.trim();

                // Spec items (Key-value formatted) - Updated to parse label/span pair reliably
                const boxItems = document.querySelectorAll('.re__project-box-item');
                for (const item of boxItems) {
                    const k = item.querySelector('label');
                    const v = item.querySelector('span');
                    if (k && v) {
                        d[k.innerText.trim()] = v.innerText.trim();
                    } else {
                        const parts = item.innerText.trim().split('\\n');
                        if (parts.length >= 2) d[parts[0].trim()] = parts[1].trim();
                    }
                }

                // Spec attribute tables - Updated to handle re__attr-item-label and re__attr-item-value
                const specRows = document.querySelectorAll('tr');
                for (const row of specRows) {
                    const k = row.querySelector('.re__attr-item-label');
                    const v = row.querySelector('.re__attr-item-value');
                    if (k && v) {
                        d[k.innerText.trim()] = v.innerText.trim();
                    }
                }

                // Configuration list items
                const cfgItems = document.querySelectorAll('.re__prj-config-item, [class*="config-item"]');
                for (const item of cfgItems) {
                    const k = item.querySelector('[class*="title"], .title, label');
                    const v = item.querySelector('[class*="value"], .value');
                    if (k && v) d[k.innerText.trim()] = v.innerText.trim();
                }

                // Detailed description content - Up to 15,000 characters
                const descEl = document.querySelector(
                    '.js__prj-detail-content, .re__project-editor, .re__detail-content, .re__project-desc'
                );
                if (descEl) d._mo_ta_chi_tiet = descEl.innerText.trim().substring(0, 15000);

                // Section-based extraction: try to capture 'Thông tin chi tiết' and 'Tiện ích' content
                try {
                    const sections = document.querySelectorAll('.re__section, .re__project-section, .project-section, .section');
                    for (const sec of sections) {
                        const titleEl = sec.querySelector('.re__section-title, h2, h3, .title');
                        const title = titleEl ? titleEl.innerText.trim() : '';
                        const body = sec.querySelector('.re__section-body, .editor, .content') || sec;
                        if (/Thông tin chi tiết/i.test(title)) {
                            if (body) d._thong_tin_chi_tiet = body.innerText.trim().substring(0,15000);
                        }
                        if (/Tiện ích|Tien ich/i.test(title)) {
                            if (body) d._tien_ich_section = body.innerText.trim().substring(0,15000);
                        }
                    }
                } catch (e) {}

                // Full address
                const addrEl = document.querySelector('.re__project-address, .re__pr-short-description--address');
                if (addrEl) d._dia_chi = addrEl.innerText.trim();

                // Project galleries
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

                // Facility indicators
                const utils = [];
                const utilEls = document.querySelectorAll(
                    '.re__prj-facilities li, [class*="facility"] li, [class*="tien-ich"] li, [class*="utility"] li'
                );
                for (const u of utilEls) {
                    const t = u.innerText.trim();
                    if (t && !utils.includes(t)) utils.push(t);
                }
                if (utils.length) d._tien_ich = utils;
                // if utilities weren't in list form, include the raw section text (if present)
                if (d._tien_ich_section && (!d._tien_ich || d._tien_ich.length === 0)) {
                    d._tien_ich_section = d._tien_ich_section.trim();
                }

                return d;
            }""")
        except Exception as e:
            self.log.warning(f"  Error loading project details {url}: {e}")
        return detail

    def _infer_project_name_from_url(self, url: str) -> Optional[str]:
        """Extract readable project name from the URL slug segment."""
        if not url:
            return None
        m = re.search(r'/([^/]+)-pj\d+', url)
        if m:
            slug = m.group(1)
            return slug.replace('-', ' ').title()
        return None

    def _merge_project(self, card: Dict[str, Any], detail: Dict[str, Any]) -> Dict[str, Any]:
        """Combine raw card attributes and detailed descriptions into standardized outputs."""
        ten_du_an = detail.get("ten_du_an")
        if not ten_du_an:
            card_name = card.get("ten_du_an")
            if card_name and card_name != "batdongsan.com.vn":
                ten_du_an = card_name
        if not ten_du_an:
            ten_du_an = self._infer_project_name_from_url(card.get("url"))

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

        # Spec mappings
        for vn_key, field in PROJECT_SPEC_MAP.items():
            if vn_key in detail:
                record[field] = detail[vn_key]

        # Prefer detailed description; fallback to 'Thông tin chi tiết' section if present
        if detail.get("_mo_ta_chi_tiet"):
            record["mo_ta_chi_tiet"] = detail["_mo_ta_chi_tiet"]
        elif detail.get("_thong_tin_chi_tiet"):
            record["mo_ta_chi_tiet"] = detail["_thong_tin_chi_tiet"]
        if detail.get("_dia_chi"):
            record["dia_chi"] = detail["_dia_chi"]
        if detail.get("_hinh_anh") and len(detail["_hinh_anh"]) > 0:
            record["hinh_anh"] = detail["_hinh_anh"]
        # Utilities: prefer structured list; otherwise parse the raw section text into lines
        if detail.get("_tien_ich"):
            record["tien_ich"] = detail["_tien_ich"]
        elif detail.get("_tien_ich_section"):
            raw = detail.get("_tien_ich_section") or ''
            items = [l.strip() for l in raw.split('\n') if l.strip()]
            record["tien_ich"] = items

        # Infer sub-category type from URL segment
        if not record["loai_du_an"] and record.get("url"):
            u = record["url"]
            type_map = {
                "can-ho-chung-cu": "Căn hộ chung cư",
                "cao-oc-van-phong": "Cao ốc văn phòng",
                "trung-tam-thuong-mai": "Trung tâm thương mại",
                "khu-do-thi-moi": "Khu đô thị mới",
                "khu-phuc-hop": "Khu phức hợp",
                "nha-o-xa-hoi": "Nhà ở xã hội",
                "khu-nghi-duong": "Khu nghỉ dưỡng",
                "khu-cong-nghiep": "Khu công nghiệp",
                "biet-thu-lien-ke": "Biệt thự, liền kề",
                "shophouse": "Shophouse",
                "nha-mat-pho": "Nhà mặt phố",
            }
            for slug, name in type_map.items():
                if slug in u:
                    record["loai_du_an"] = name
                    break
        return record

    async def crawl(self, max_pages: int = 1, visit_details: bool = True, resume: bool = False) -> List[Dict[str, Any]]:
        """Run complete crawl flow for estate projects."""
        self.log.info(f"Starting projects crawl. Target pages: {max_pages}")
        all_projects: List[Dict[str, Any]] = []
        start_page = 1
        
        if resume:
            self.checkpoint_mgr.load()
            start_page = self.checkpoint_mgr.get_last_page() + 1
            all_projects = self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resuming projects crawl from page {start_page}. Items loaded: {len(all_projects)}")

        if start_page > max_pages:
            self.log.info("Crawl already fulfilled by checkpoint.")
            return all_projects

        async with async_playwright() as pw:
            browser = await launch_browser(pw)
            
            for pg in range(start_page, max_pages + 1):
                context, page = await new_stealth_page(browser)
                url = PROJECT_URL if pg == 1 else f"{PROJECT_URL}/p{pg}"
                self.log.info(f"Navigating to projects page {pg}: {url}")
                
                if not await goto_safe(page, url):
                    self.log.warning(f"Failed to navigate to page {pg}. Skipping.")
                    await context.close()
                    continue

                await asyncio.sleep(5)
                found = await self._wait_for_projects(page)
                if not found:
                    self.log.warning("No project cards found. Scrolling half page...")
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await asyncio.sleep(2)
                    found = await self._wait_for_projects(page)

                if not found:
                    self.log.error(f"Failed to load project cards on page {pg}. Aborting page.")
                    await context.close()
                    break

                cards = await self._extract_project_cards(page)
                self.log.info(f"Extracted {len(cards)} project cards from page {pg}")

                page_projects = []
                for idx, card in enumerate(cards):
                    card_url = card.get("url")
                    if card_url:
                        if not card_url.startswith("http"):
                            card["url"] = BASE_URL + card_url
                        
                        # Dedup check
                        if self.checkpoint_mgr.is_seen(card["url"]):
                            continue

                    detail = {}
                    if visit_details and card.get("url"):
                        # Use a fresh stealth context/page for detail page to bypass Cloudflare completely
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            detail_context, detail_page = await new_stealth_page(browser)
                            detail = await self._scrape_project_detail(detail_page, card["url"])
                            await detail_context.close()
                            
                            # Bypassed successfully if we got description content
                            if detail and detail.get("_mo_ta_chi_tiet"):
                                break
                            
                            if attempt < max_attempts - 1:
                                self.log.info(f"  Project detail page returned empty. Retrying with new context (Attempt {attempt+2}/{max_attempts})...")
                                await asyncio.sleep(3)
                        await self.sleep_polite()

                    merged = self._merge_project(card, detail)
                    page_projects.append(merged)
                    all_projects.append(merged)
                    
                    if card.get("url"):
                        self.checkpoint_mgr.add_seen(card["url"])

                    if (idx + 1) % 5 == 0:
                        self.log.info(f"  Progress: {idx+1}/{len(cards)} on projects page {pg}")

                await context.close()
                self.checkpoint_mgr.save(pg, page_projects)
                await self.sleep_polite(REQUEST_DELAY * 2)

            await browser.close()

        self.save_final_results(all_projects, resume)
        return all_projects
