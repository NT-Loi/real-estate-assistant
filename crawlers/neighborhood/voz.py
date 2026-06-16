import asyncio
import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from crawlers.base import BaseCrawler
from crawlers.browser import launch_browser, new_stealth_page, goto_safe
from crawlers.config import DATA_DIR, REQUEST_DELAY

log = logging.getLogger("bds_crawler.voz")

class VozCrawler(BaseCrawler):
    """Crawler for extracting local discussions and neighborhood reviews from VOZ Forum (voz.vn)."""
    
    def __init__(self, output_file: Optional[Path] = None):
        super().__init__("voz_neighborhood", output_file or (DATA_DIR / "voz_discussions.json"))
        self.base_url = "https://voz.vn"
        self.forum_url = "https://voz.vn/f/bat-dong-san.79"



    def parse_threads_page(self, html: str) -> List[Dict[str, Any]]:
        """Parse list of threads from a forum subcategory page."""
        soup = BeautifulSoup(html, "html.parser")
        threads = []
        
        # Threads are inside a div.structItemContainer
        container = soup.find(class_="structItemContainer")
        if container:
            items = container.find_all(class_="structItem--thread")
        else:
            # Fallback: search the whole page directly
            items = soup.find_all(class_="structItem--thread")
        for item in items:
            title_el = item.find(class_="structItem-title")
            if not title_el:
                continue

            # The title element has two links: a labelLink prefix (SG/HN/khác) and the actual thread link.
            # Target the thread link specifically (href contains /t/).
            a_tag = title_el.find("a", href=lambda h: h and "/t/" in h)
            if not a_tag:
                continue

            title = a_tag.get_text(strip=True)
            href = a_tag["href"]
            url = href if href.startswith("http") else self.base_url + href

            # Extract label/prefix tag (e.g. "SG", "HN", "khác")
            label_tag = ""
            label_el = title_el.find("a", class_="labelLink")
            if label_el:
                label_tag = label_el.get_text(strip=True)
            
            # Stats (Replies & Views)
            stats_el = item.find(class_="structItem-cell--meta")
            replies = "0"
            views = "0"
            if stats_el:
                pairs = stats_el.find_all("dd")
                if len(pairs) >= 2:
                    replies = pairs[0].get_text(strip=True)
                    views = pairs[1].get_text(strip=True)

            # Author and date
            parts_el = item.find(class_="structItem-parts")
            author = "Unknown"
            if parts_el:
                author_el = parts_el.find("a", class_="username") or parts_el.find("span")
                if author_el:
                    author = author_el.get_text(strip=True)

            date_el = item.find(class_="structItem-startDate")
            date_str = None
            if date_el:
                time_el = date_el.find("time") or date_el.find("span")
                if time_el:
                    date_str = time_el.get("datetime") or time_el.get_text(strip=True)

            threads.append({
                "title": title,
                "url": url,
                "label": label_tag,
                "author": author,
                "published_at": date_str,
                "replies_count": replies,
                "views_count": views
            })
            
        return threads

    def parse_thread_posts(self, html: str) -> List[Dict[str, Any]]:
        """Parse detailed list of posts/messages inside an individual thread page."""
        soup = BeautifulSoup(html, "html.parser")
        posts = []
        
        message_elements = soup.find_all("article", class_="message--post")
        for idx, msg in enumerate(message_elements):
            author_el = msg.find(class_="message-name") or msg.find(class_="username")
            author = author_el.get_text(strip=True) if author_el else "Unknown"

            # Content body
            body_el = msg.find("div", class_="bbWrapper")
            content = ""
            if body_el:
                # Remove quotes to get clean post text if necessary, or keep them marked
                content = body_el.get_text(separator="\n", strip=True)

            # Date/Time
            date_el = msg.find("time", class_="u-dt") or msg.find("time")
            date_str = None
            if date_el:
                date_str = date_el.get("datetime") or date_el.get_text(strip=True)

            # Reactions
            reactions_count = 0
            react_el = msg.find(class_="reactionsBar-link")
            if react_el:
                text = react_el.get_text(strip=True)
                # Matches e.g. "You, Admin and 12 others" or "15 others"
                m = re.search(r"(\d+)", text)
                if m:
                    reactions_count = int(m.group(1))
                elif "others" in text or "and" in text:
                    reactions_count = 3  # rough fallback
                else:
                    reactions_count = 1

            posts.append({
                "author": author,
                "content": content,
                "published_at": date_str,
                "reactions_count": reactions_count,
                "post_index": idx
            })
            
        return posts

    def parse_search_results(self, html: str) -> List[Dict[str, Any]]:
        """Parse VOZ search results page. Search results use 'contentRow' containers
        instead of the 'structItem' containers used on forum listing pages."""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        rows = soup.find_all(class_="contentRow")
        for row in rows:
            title_el = row.find(class_="contentRow-title")
            if not title_el:
                continue

            a_tag = title_el.find("a", href=lambda h: h and "/t/" in h)
            if not a_tag:
                continue

            title = a_tag.get_text(strip=True)
            href = a_tag["href"]
            url = href if href.startswith("http") else self.base_url + href

            # Snippet of the matching post
            snippet_el = row.find(class_="contentRow-snippet")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            # Minor info (author, date)
            minor_el = row.find(class_="contentRow-minor")
            author = "Unknown"
            date_str = None
            if minor_el:
                author_el = minor_el.find("a", class_="username") or minor_el.find("a")
                if author_el:
                    author = author_el.get_text(strip=True)
                time_el = minor_el.find("time")
                if time_el:
                    date_str = time_el.get("datetime") or time_el.get_text(strip=True)

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "author": author,
                "published_at": date_str,
            })

        return results

    async def search_threads(
        self, keyword: str, page: object, max_results: int = 20, max_scan_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """Search VOZ forum for threads matching a keyword.
        
        Since VOZ's built-in search requires login, this method browses forum
        listing pages and filters threads whose titles contain keyword terms.
        Scans up to max_scan_pages pages of the BDS forum to find matches.
        """
        self.log.info(f"Searching VOZ for: '{keyword}' (scanning up to {max_scan_pages} pages)")
        
        # Build search terms (lowercase, split into words, ignore short/common words)
        search_terms = [w.lower() for w in keyword.split() if len(w) >= 2]
        if not search_terms:
            return []
        
        all_results = []
        seen_urls = set()

        for pg in range(1, max_scan_pages + 1):
            url = self.forum_url if pg == 1 else f"{self.forum_url}/page-{pg}"
            self.log.info(f"  Scanning page {pg}: {url}")

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                await asyncio.sleep(4)
                html = await page.content()
            except Exception as e:
                self.log.warning(f"  Failed to load page {pg}: {e}")
                break

            threads = self.parse_threads_page(html)
            if not threads:
                break

            for t in threads:
                title_lower = t["title"].lower()
                # Match if ANY search term appears in the title
                if any(term in title_lower for term in search_terms):
                    if t["url"] not in seen_urls:
                        seen_urls.add(t["url"])
                        all_results.append(t)
                        if len(all_results) >= max_results:
                            break

            if len(all_results) >= max_results:
                break
            
            await self.sleep_polite(REQUEST_DELAY)

        self.log.info(f"Found {len(all_results)} matching threads for keyword '{keyword}'")
        return all_results

    async def crawl(
        self,
        keywords: List[str] = None,
        max_pages: int = 1,
        max_threads_per_page: int = 10,
        max_threads_per_kw: int = 10,
        visit_posts: bool = True,
        resume: bool = False
    ) -> List[Dict[str, Any]]:
        """Run VOZ forum crawler.
        
        Supports two modes:
        - Keyword search: search VOZ for threads matching keywords (preferred)
        - Forum browse: paginate through the BDS forum listing pages (fallback)
        Both can be combined; results are deduplicated by URL.
        """
        self.log.info(f"Starting VOZ forum crawl. Keywords: {keywords}, Pages: {max_pages}")
        all_discussions: List[Dict[str, Any]] = []
        start_page = 1

        if resume:
            self.checkpoint_mgr.load()
            start_page = self.checkpoint_mgr.get_last_page() + 1
            all_discussions = self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resuming VOZ crawl. Discussions loaded: {len(all_discussions)}")

        processed_threads = {item["url"] for item in all_discussions if "url" in item}

        async with async_playwright() as pw:
            browser = await launch_browser(pw)
            context, page = await new_stealth_page(browser)

            # --- Mode 1: Keyword search ---
            if keywords:
                for kw in keywords:
                    search_results = await self.search_threads(kw, page, max_results=max_threads_per_kw)
                    threads_crawled = 0

                    for idx, sr in enumerate(search_results):
                        t_url = sr["url"]
                        if t_url in processed_threads:
                            continue

                        self.log.info(f"  [{kw}] Crawling thread {idx+1}/{len(search_results)}: {sr['title']}")
                        posts_data = []

                        if visit_posts:
                            try:
                                await page.goto(t_url, wait_until="domcontentloaded", timeout=60_000)
                                await asyncio.sleep(3)
                                t_html = await page.content()
                                posts_data = self.parse_thread_posts(t_html)
                            except Exception as e:
                                self.log.warning(f"    Failed to visit thread: {e}")
                            await self.sleep_polite(REQUEST_DELAY)

                        record = {
                            "keyword": kw,
                            "forum_section": "Bất động sản",
                            "title": sr["title"],
                            "url": t_url,
                            "snippet": sr.get("snippet", ""),
                            "author": sr["author"],
                            "published_at": sr.get("published_at"),
                            "posts": posts_data,
                            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        }

                        all_discussions.append(record)
                        processed_threads.add(t_url)
                        threads_crawled += 1

                        self.checkpoint_mgr.save(0, [record])

            # --- Mode 2: Forum page browsing (when no keywords or as additional source) ---
            if not keywords:
                if start_page > max_pages:
                    self.log.info("Crawl goal already achieved.")
                else:
                    consecutive_empty_pages = 0

                    for pg in range(start_page, max_pages + 1):
                        url = self.forum_url if pg == 1 else f"{self.forum_url}/page-{pg}"
                        self.log.info(f"Fetching forum list page {pg}: {url}")

                        try:
                            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                            await asyncio.sleep(5)
                            html = await page.content()
                        except Exception as e:
                            self.log.warning(f"Failed to navigate to VOZ listing page {pg}: {e}")
                            continue

                        threads = self.parse_threads_page(html)
                        self.log.info(f"Found {len(threads)} threads on page {pg}")

                        if not threads:
                            consecutive_empty_pages += 1
                            self.checkpoint_mgr.save(pg, [])
                            if consecutive_empty_pages >= 3:
                                self.log.info(
                                    f"Stopping VOZ crawl after {consecutive_empty_pages} consecutive empty pages."
                                )
                                break
                            await self.sleep_polite(REQUEST_DELAY * 2)
                            continue

                        consecutive_empty_pages = 0
                        page_discussions = []
                        threads_processed_count = 0

                        for idx, t in enumerate(threads):
                            if threads_processed_count >= max_threads_per_page:
                                break

                            t_url = t["url"]
                            if t_url in processed_threads:
                                continue

                            self.log.info(f"  Crawling thread {idx+1}/{len(threads)}: {t['title']}")
                            posts_data = []

                            if visit_posts:
                                try:
                                    await page.goto(t_url, wait_until="domcontentloaded", timeout=60_000)
                                    await asyncio.sleep(3)
                                    t_html = await page.content()
                                    posts_data = self.parse_thread_posts(t_html)
                                except Exception as e:
                                    self.log.warning(f"    Failed to navigate to thread post: {e}")
                                await self.sleep_polite(REQUEST_DELAY)

                            record = {
                                "keyword": None,
                                "forum_section": "Bất động sản",
                                "title": t["title"],
                                "url": t_url,
                                "label": t.get("label", ""),
                                "author": t["author"],
                                "published_at": t["published_at"],
                                "replies_count": t["replies_count"],
                                "views_count": t["views_count"],
                                "posts": posts_data,
                                "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            }

                            page_discussions.append(record)
                            all_discussions.append(record)
                            processed_threads.add(t_url)
                            threads_processed_count += 1

                        self.checkpoint_mgr.save(pg, page_discussions)
                        await self.sleep_polite(REQUEST_DELAY * 2)

            await browser.close()

        self.save_final_results(all_discussions, resume)
        return all_discussions
