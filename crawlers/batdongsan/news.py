import asyncio
import re
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright, Page

from crawlers.base import BaseCrawler
from crawlers.browser import launch_browser, new_stealth_page, goto_safe
from crawlers.config import NEWS_URL, BASE_URL, REQUEST_DELAY

_IMAGE_JUNK_PATTERNS = [
    "mobileSearch", "authorDefault", "google-play", "app_store",
    "footer", "logo", "icon", "avatar", "placeholder",
    "staticfile.batdongsan.com.vn/images", "cdn-assets-angel",
]

class NewsCrawler(BaseCrawler):
    """Crawler for estate news articles on batdongsan.com.vn."""
    
    def __init__(self, output_file = None):
        super().__init__("news", output_file)

    async def _extract_article_cards(self, page: Page) -> List[Dict[str, Any]]:
        """Extract lists of news summary cards from listing elements."""
        return await page.evaluate("""() => {
            const results = [];
            const seen = new Set();
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

    async def _scrape_article_detail(self, page: Page, url: str) -> Dict[str, Any]:
        """Scrape full content and deep metadata inside the article detail view."""
        detail = {}
        try:
            self.log.info(f"  Visiting article: {url}")
            await page.goto(url, wait_until="commit", timeout=30_000)
            await page.wait_for_load_state("domcontentloaded", timeout=15_000)
            await asyncio.sleep(3)

            detail = await page.evaluate(r"""() => {
                const d = {};
                // Title
                const titleEl = document.querySelector('h1');
                if (titleEl) d.tieu_de = titleEl.innerText.trim();

                // Description from Meta Tags
                const ogDesc = document.querySelector('meta[property="og:description"]');
                if (ogDesc) d.mo_ta = ogDesc.getAttribute('content');

                // Author Link
                const authorLink = document.querySelector('a[href*="/tac-gia/"]');
                const authorLinkText = authorLink ? authorLink.innerText.trim() : '';
                if (authorLinkText) {
                    d.tac_gia = authorLinkText;
                }
                
                if (!d.tac_gia) {
                    const authorEl = document.querySelector('[class*="author"]');
                    if (authorEl) d.tac_gia = authorEl.innerText.trim();
                }

                // Date published
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

                // Category Breadcrumb
                const bcLinks = document.querySelectorAll('.re__breadcrumb a, [class*="breadcrumb"] a');
                if (bcLinks.length >= 2) d.danh_muc = bcLinks[bcLinks.length - 1].innerText.trim();

                // Article body selectors
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

                // Inline images
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
        except Exception as e:
            self.log.warning(f"  Error loading article detail {url}: {e}")
        return detail

    def _filter_images(self, imgs: List[str]) -> List[str]:
        """Eliminate system assets, logos, and placeholders from the output."""
        filtered = []
        for src in imgs:
            if not src:
                continue
            if any(p in src for p in _IMAGE_JUNK_PATTERNS):
                continue
            if src not in filtered:
                filtered.append(src)
        return filtered

    def _parse_author_block(self, raw: str) -> tuple:
        """Parse author name and dates from unstructured strings."""
        author = None
        date_str = None
        if not raw:
            return author, date_str

        m = re.search(r"Được đăng bởi\s+(.+?)(?:\n|$)", raw)
        if m:
            author = m.group(1).strip()
        else:
            lines = raw.strip().split("\n")
            if lines and len(lines[0]) < 50:
                author = lines[0].strip()

        m = re.search(r"(\d{2}/\d{2}/\d{4}(?:\s+\d{2}:\d{2})?)", raw)
        if m:
            date_str = m.group(1)

        return author, date_str

    def _merge_article(self, card: Dict[str, Any], detail: Dict[str, Any], category: Optional[str] = None) -> Dict[str, Any]:
        """Merge listing representations and detail specifics into standardized article records."""
        mo_ta = detail.get("mo_ta") or card.get("mo_ta")
        tac_gia = detail.get("tac_gia")
        ngay_dang = detail.get("ngay_dang") or card.get("ngay_dang")

        if tac_gia and ("Được đăng bởi" in tac_gia or "Cập nhật" in tac_gia):
            parsed_author, parsed_date = self._parse_author_block(tac_gia)
            tac_gia = parsed_author
            if not ngay_dang and parsed_date:
                ngay_dang = parsed_date

        imgs = detail.get("hinh_anh") or ([card["hinh_anh"]] if card.get("hinh_anh") else [])
        imgs = self._filter_images(imgs)

        record = {
            "loai": "tin-tuc",
            "danh_muc": category or card.get("danh_muc") or "Tin tức",
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

    async def crawl(self, max_pages: int = 1, visit_details: bool = True, resume: bool = False) -> List[Dict[str, Any]]:
        """Run complete page iteration and details scraping for news articles."""
        self.log.info(f"Starting news crawl. Target pages: {max_pages}")
        all_articles: List[Dict[str, Any]] = []
        start_page = 1
        
        if resume:
            self.checkpoint_mgr.load()
            start_page = self.checkpoint_mgr.get_last_page() + 1
            all_articles = self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resuming news crawl from page {start_page}. Items loaded: {len(all_articles)}")

        if start_page > max_pages:
            self.log.info("Crawl target already reached/exceeded by checkpoints.")
            return all_articles

        consecutive_empty_pages = 0

        async with async_playwright() as pw:
            browser = await launch_browser(pw)
            
            for pg in range(start_page, max_pages + 1):
                context, page = await new_stealth_page(browser)
                url = NEWS_URL if pg == 1 else f"{NEWS_URL}/p{pg}"
                self.log.info(f"Navigating to news page {pg}: {url}")
                
                if not await goto_safe(page, url):
                    self.log.warning(f"Failed to navigate to {url}. Skipping.")
                    await context.close()
                    continue

                await asyncio.sleep(5)
                # Lazy loading scroll logic
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(2)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)

                cards = await self._extract_article_cards(page)
                self.log.info(f"Extracted {len(cards)} articles from page {pg}")

                if not cards:
                    consecutive_empty_pages += 1
                    await context.close()
                    self.checkpoint_mgr.save(pg, [])
                    if consecutive_empty_pages >= 3:
                        self.log.info(
                            f"Stopping news crawl after {consecutive_empty_pages} consecutive empty pages."
                        )
                        break
                    await self.sleep_polite(REQUEST_DELAY * 2)
                    continue

                consecutive_empty_pages = 0
                page_articles = []
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
                        detail = await self._scrape_article_detail(page, card["url"])
                        await self.sleep_polite()

                    merged = self._merge_article(card, detail)
                    page_articles.append(merged)
                    all_articles.append(merged)

                    if card.get("url"):
                        self.checkpoint_mgr.add_seen(card["url"])

                    if (idx + 1) % 5 == 0:
                        self.log.info(f"  Progress: {idx+1}/{len(cards)} on news page {pg}")

                await context.close()
                self.checkpoint_mgr.save(pg, page_articles)
                await self.sleep_polite(REQUEST_DELAY * 2)

            await browser.close()

        self.save_final_results(all_articles, resume)
        return all_articles
