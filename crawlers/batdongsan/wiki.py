import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright

from crawlers.batdongsan.news import NewsCrawler
from crawlers.config import WIKI_URL, WIKI_CATEGORIES, DATA_DIR, REQUEST_DELAY
from crawlers.browser import launch_browser, new_stealth_page, goto_safe

class WikiCrawler(NewsCrawler):
    """Crawler for estate guide and wiki articles on batdongsan.com.vn."""
    
    def __init__(self, output_file = None):
        # We'll save category-specific files dynamically, but we define 'wiki_all' as the default unified file
        super().__init__(output_file or (DATA_DIR / "wiki_all.json"))
        self.name = "wiki"

    async def crawl_category(
        self,
        category_slug: str,
        category_name: str,
        max_pages: int = 1,
        visit_details: bool = True,
        resume: bool = False
    ) -> List[Dict[str, Any]]:
        """Crawl a specific wiki subcategory slug (e.g., 'mua-bds')."""
        self.log.info(f"Starting wiki category crawl: {category_name} ({category_slug})")
        
        # Instantiate sub-checkpoint and dynamic category-specific output file
        category_output = DATA_DIR / f"wiki_{category_slug.replace('-', '_')}.json"
        
        # Set instance attributes for the super class's checkpoint manager and output settings
        orig_output = self.output_file
        self.output_file = category_output
        self.checkpoint_mgr = self.checkpoint_mgr.__class__(f"wiki_{category_slug}", DATA_DIR / ".checkpoints")
        
        all_articles: List[Dict[str, Any]] = []
        start_page = 1
        
        if resume:
            self.checkpoint_mgr.load()
            start_page = self.checkpoint_mgr.get_last_page() + 1
            all_articles = self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resuming wiki category crawl from page {start_page}. Items loaded: {len(all_articles)}")

        if start_page <= max_pages:
            async with async_playwright() as pw:
                browser = await launch_browser(pw)
                
                for pg in range(start_page, max_pages + 1):
                    context, page = await new_stealth_page(browser)
                    url = f"{WIKI_URL}/{category_slug}"
                    if pg > 1:
                        url = f"{url}/p{pg}"
                    
                    self.log.info(f"Navigating to wiki page {pg}: {url}")
                    if not await goto_safe(page, url):
                        self.log.warning(f"Failed to navigate to {url}. Skipping.")
                        await context.close()
                        continue

                    await asyncio.sleep(5)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await asyncio.sleep(2)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(2)

                    cards = await self._extract_article_cards(page)
                    self.log.info(f"Extracted {len(cards)} wiki cards from page {pg}")

                    page_articles = []
                    for idx, card in enumerate(cards):
                        card_url = card.get("url")
                        if card_url:
                            if not card_url.startswith("http"):
                                card["url"] = WIKI_URL.replace("/wiki", "") + card_url
                            
                            if self.checkpoint_mgr.is_seen(card["url"]):
                                continue

                        detail = {}
                        if visit_details and card.get("url"):
                            detail = await self._scrape_article_detail(page, card["url"])
                            await self.sleep_polite()

                        # Overwrite sections
                        merged = self._merge_article(card, detail, category=category_name)
                        merged["loai"] = "wiki"
                        page_articles.append(merged)
                        all_articles.append(merged)

                        if card.get("url"):
                            self.checkpoint_mgr.add_seen(card["url"])

                        if (idx + 1) % 5 == 0:
                            self.log.info(f"  Progress: {idx+1}/{len(cards)} on wiki category page {pg}")

                    await context.close()
                    self.checkpoint_mgr.save(pg, page_articles)
                    await self.sleep_polite(REQUEST_DELAY * 2)

                await browser.close()

            self.save_final_results(all_articles, resume)
            
        # Revert changes to properties
        self.output_file = orig_output
        return all_articles

    async def crawl(
        self,
        max_pages: int = 1,
        visit_details: bool = True,
        resume: bool = False,
        wiki_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run complete crawl for all or selected wiki categories, and consolidate to a single file."""
        cats = {}
        if wiki_category:
            if wiki_category in WIKI_CATEGORIES:
                cats = {wiki_category: WIKI_CATEGORIES[wiki_category]}
            else:
                self.log.error(f"Unknown wiki category slug: {wiki_category}")
                return []
        else:
            cats = WIKI_CATEGORIES

        all_wiki_articles: List[Dict[str, Any]] = []
        for slug, name in cats.items():
            articles = await self.crawl_category(
                category_slug=slug,
                category_name=name,
                max_pages=max_pages,
                visit_details=visit_details,
                resume=resume
            )
            all_wiki_articles.extend(articles)

        # Consolidate all crawled wiki articles in a central wiki_all.json output
        if not wiki_category:
            self.log.info(f"Consolidating {len(all_wiki_articles)} wiki articles into unified database...")
            # We want to read all files in wiki_*.json to make sure they are up-to-date
            unified_wiki = []
            seen_urls = set()
            
            for slug in WIKI_CATEGORIES:
                cat_file = DATA_DIR / f"wiki_{slug.replace('-', '_')}.json"
                if cat_file.exists():
                    try:
                        with open(cat_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    url = item.get("url")
                                    if url and url not in seen_urls:
                                        seen_urls.add(url)
                                        unified_wiki.append(item)
                    except Exception as e:
                        self.log.warning(f"Failed to read wiki segment {cat_file}: {e}")

            temp_unified = self.output_file.with_suffix(".tmp")
            try:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_unified, "w", encoding="utf-8") as f:
                    json.dump(unified_wiki, f, ensure_ascii=False, indent=2)
                temp_unified.replace(self.output_file)
                self.log.info(f"Saved unified wiki dataset of {len(unified_wiki)} articles to {self.output_file}")
            except Exception as e:
                self.log.error(f"Unified wiki saving failure: {e}")
                if temp_unified.exists():
                    temp_unified.unlink()
                    
        return all_wiki_articles
