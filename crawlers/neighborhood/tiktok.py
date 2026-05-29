import asyncio
import datetime
import json
import logging
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from playwright.async_api import async_playwright

from crawlers.base import BaseCrawler
from crawlers.browser import launch_browser, new_stealth_page
from crawlers.config import DATA_DIR, TIKTOK_COOKIES_FILE

log = logging.getLogger("bds_crawler.tiktok")

class TikTokCrawler(BaseCrawler):
    """Crawler for collecting qualitative comments and video insights from TikTok using the cookies approach."""
    
    def __init__(self, output_file: Optional[Path] = None):
        super().__init__("tiktok_neighborhood", output_file or (DATA_DIR / "tiktok_comments.json"))
        self.cookie_string = self._load_cookies()

    def _load_cookies(self) -> str:
        """Read cookies from cookies.txt (handles Netscape format and raw strings)."""
        if not TIKTOK_COOKIES_FILE.exists():
            self.log.warning(f"TikTok cookies file not found at {TIKTOK_COOKIES_FILE}. Comment crawling may be blocked.")
            return ""

        try:
            with open(TIKTOK_COOKIES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Try to see if it's a Netscape cookies format
            cookies_dict = {}
            raw_cookie_line = ""
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split("\t")
                if len(parts) >= 7:
                    # Netscape format: domain, flag, path, secure, expiration, name, value
                    name = parts[5]
                    value = parts[6]
                    cookies_dict[name] = value
                else:
                    # Plain cookie line or single string
                    raw_cookie_line = line

            if cookies_dict:
                # Format to a Cookie string: "name1=value1; name2=value2"
                cookie_str = "; ".join([f"{k}={v}" for k, v in cookies_dict.items()])
                self.log.info("Successfully parsed Netscape cookies format.")
                return cookie_str
            
            self.log.info("Using raw cookie string format from cookies.txt.")
            return raw_cookie_line
        except Exception as e:
            self.log.error(f"Error reading cookies from {TIKTOK_COOKIES_FILE}: {e}")
            return ""

    def _get_playwright_cookies(self) -> List[Dict[str, Any]]:
        """Parse Netscape cookies into list of dicts suitable for Playwright context.add_cookies()."""
        playwright_cookies = []
        if not TIKTOK_COOKIES_FILE.exists():
            return playwright_cookies
        try:
            with open(TIKTOK_COOKIES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 7:
                        domain = parts[0]
                        path = parts[2]
                        secure = parts[3].upper() == "TRUE"
                        expires = int(parts[4])
                        name = parts[5]
                        value = parts[6]
                        
                        cookie = {
                            "name": name,
                            "value": value,
                            "domain": domain,
                            "path": path,
                            "secure": secure,
                        }
                        if expires > 0:
                            cookie["expires"] = expires
                        playwright_cookies.append(cookie)
            self.log.info(f"Loaded {len(playwright_cookies)} Playwright cookies from cookies.txt")
        except Exception as e:
            self.log.error(f"Error parsing Netscape cookies for Playwright: {e}")
        return playwright_cookies

    def fetch_oembed(self, video_url: str) -> Optional[dict]:
        """Fetch video metadata using TikTok's public oEmbed service."""
        oembed_url = "https://www.tiktok.com/oembed"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(oembed_url, params={"url": video_url}, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            self.log.error(f"oEmbed fetch error for {video_url}: {e}")
        return None

    def _get_headers(self, aweme_id: str) -> dict:
        """Prepare authenticated headers for comments API calls."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"https://www.tiktok.com/@/video/{aweme_id}",
            "Cookie": self.cookie_string
        }

    def fetch_comments(self, aweme_id: str, max_comments: int = 100) -> List[dict]:
        """Fetch top-level comments using the internal TikTok API."""
        url = "https://www.tiktok.com/api/comment/list/"
        headers = self._get_headers(aweme_id)
        
        cursor = 0
        all_comments = []

        while len(all_comments) < max_comments:
            params = {
                "aid": 1988,
                "aweme_id": aweme_id,
                "cursor": cursor,
                "count": 20
            }
            try:
                r = requests.get(url, params=params, headers=headers, timeout=15)
                if r.status_code != 200:
                    self.log.warning(f"TikTok API returned status code {r.status_code}")
                    break

                data = r.json()
                comments = data.get("comments", [])
                if not comments:
                    break

                all_comments.extend(comments)
                self.log.info(f"  Fetched {len(all_comments)} top-level comments...")

                if not data.get("has_more") or len(comments) < 20:
                    break
                cursor = data.get("cursor", cursor + 20)
                
                # Politeness sleep to prevent IP bans
                time.sleep(1.5)
            except Exception as e:
                self.log.error(f"Error fetching TikTok comments for aweme {aweme_id}: {e}")
                break

        return all_comments[:max_comments]

    def fetch_replies(self, aweme_id: str, comment_id: str, max_replies: int = 50) -> List[dict]:
        """Fetch nested replies for a parent comment ID."""
        url = "https://www.tiktok.com/api/comment/list/reply/"
        headers = self._get_headers(aweme_id)
        
        cursor = 0
        all_replies = []

        while len(all_replies) < max_replies:
            params = {
                "aid": 1988,
                "aweme_id": aweme_id,
                "comment_id": comment_id,
                "cursor": cursor,
                "count": 20
            }
            try:
                r = requests.get(url, params=params, headers=headers, timeout=15)
                if r.status_code != 200:
                    break

                data = r.json()
                replies = data.get("comments", [])
                if not replies:
                    break

                all_replies.extend(replies)
                if not data.get("has_more") or len(replies) < 20:
                    break
                cursor = data.get("cursor", cursor + 20)
                
                time.sleep(1.0)
            except Exception as e:
                self.log.error(f"Error fetching TikTok replies for comment {comment_id}: {e}")
                break

        return all_replies[:max_replies]

    def fetch_comments_with_replies(self, aweme_id: str, max_comments: int = 50) -> List[dict]:
        """Extract top comments and fetch associated replies in a structured format."""
        comments = self.fetch_comments(aweme_id, max_comments=max_comments)
        results = []
        
        for idx, c in enumerate(comments):
            cid = c.get("cid")
            item = {
                "comment_id": cid,
                "comment_raw": c.get("text"),
                "author": c.get("user", {}).get("nickname") or c.get("user", {}).get("unique_id"),
                "like_count": c.get("digg_count", 0),
                "published_at": datetime.datetime.fromtimestamp(c.get("create_time", 0)).isoformat() + "Z" if c.get("create_time") else None,
                "replies": []
            }
            
            # Fetch replies if comment has reply counts
            reply_count = c.get("reply_comment_total", 0)
            if reply_count > 0 and cid:
                try:
                    self.log.info(f"    Fetching replies for comment {idx+1}/{len(comments)} (ID: {cid})")
                    replies = self.fetch_replies(aweme_id, cid, max_replies=30)
                    for r in replies:
                        item["replies"].append({
                            "comment_id": r.get("cid"),
                            "comment_raw": r.get("text"),
                            "author": r.get("user", {}).get("nickname") or r.get("user", {}).get("unique_id"),
                            "like_count": r.get("digg_count", 0),
                            "published_at": datetime.datetime.fromtimestamp(r.get("create_time", 0)).isoformat() + "Z" if r.get("create_time") else None,
                        })
                except Exception as e:
                    self.log.warning(f"Failed to fetch replies for comment {cid}: {e}")
            
            results.append(item)
        return results

    def fetch_video_metadata_with_yt_dlp(self, video_url: str) -> dict:
        """Download video metadata using yt-dlp library if available."""
        try:
            import yt_dlp
            ydl_opts = {
                "noplaylist": True,
                "quiet": True,
                "skip_download": True,
                "extract_flat": True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info or {}
        except Exception as e:
            self.log.warning(f"yt-dlp metadata extraction failed for {video_url}: {e}")
        return {}

    async def search_videos_by_keyword(self, keyword: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search TikTok for videos matching a keyword using Playwright stealth browser.
        
        Returns list of dicts with keys: url, title, author, keyword.
        """
        self.log.info(f"Searching TikTok for: '{keyword}'")
        encoded_kw = urllib.parse.quote(keyword)
        search_url = f"https://www.tiktok.com/search?q={encoded_kw}"
        found_videos = []

        try:
            async with async_playwright() as pw:
                browser = await launch_browser(pw)
                context, page = await new_stealth_page(browser)

                # Load cookies into context to avoid login walls / empty search results
                pw_cookies = self._get_playwright_cookies()
                if pw_cookies:
                    await context.add_cookies(pw_cookies)
                    self.log.info("Loaded cookies into Playwright search context.")

                await page.goto(search_url, wait_until="domcontentloaded", timeout=60_000)
                # Let search results render (TikTok is SPA-heavy)
                await asyncio.sleep(8)

                # Scroll down a couple times to load more results
                for _ in range(3):
                    await page.evaluate("window.scrollBy(0, 1000)")
                    await asyncio.sleep(2)

                html = await page.content()
                await browser.close()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Strategy 1: Find video links matching TikTok video URL pattern
            seen_urls = set()
            video_links = soup.find_all("a", href=re.compile(r"/@[^/]+/video/\d+"))
            for a in video_links:
                href = a["href"]
                # Normalize to full URL
                if href.startswith("/"):
                    href = "https://www.tiktok.com" + href
                # Deduplicate
                vid_match = re.search(r"/video/(\d+)", href)
                if not vid_match:
                    continue
                vid_id = vid_match.group(1)
                if vid_id in seen_urls:
                    continue
                seen_urls.add(vid_id)

                # Try to extract text context from nearby elements
                title_text = a.get_text(strip=True) or ""
                # Walk up to find a card container for more context
                parent = a.parent
                for _ in range(5):
                    if parent is None:
                        break
                    candidate = parent.get_text(separator=" ", strip=True)
                    if len(candidate) > len(title_text) and len(candidate) < 500:
                        title_text = candidate
                    parent = parent.parent

                found_videos.append({
                    "url": href,
                    "video_id": vid_id,
                    "title": title_text[:300] if title_text else None,
                    "author": None,  # Will be enriched via oEmbed later
                    "keyword": keyword,
                })

                if len(found_videos) >= max_results:
                    break

            self.log.info(f"Found {len(found_videos)} TikTok videos for keyword '{keyword}'")
        except Exception as e:
            self.log.warning(f"TikTok keyword search failed for '{keyword}': {e}")

        return found_videos

    async def crawl(
        self,
        urls: List[str] = None,
        keywords: List[str] = None,
        max_videos_per_kw: int = 5,
        max_comments_per_video: int = 50,
        resume: bool = False
    ) -> List[Dict[str, Any]]:
        """Run complete cookies-based comments ingestion workflow.
        
        Supports two modes:
        - Direct URLs: provide a list of TikTok video URLs
        - Keyword search: search TikTok by keywords and crawl the results
        Both can be combined; results are deduplicated by video ID.
        """
        if not self.cookie_string:
            self.log.error("TikTok cookies file cookies.txt is empty or invalid. Comment API calls may fail.")

        all_results = []
        if resume:
            self.checkpoint_mgr.load()
            all_results = self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resumed crawl. Loaded {len(all_results)} existing videos from checkpoints.")

        processed_ids = set()
        for item in all_results:
            vid_id = item.get("video_id")
            if vid_id:
                processed_ids.add(vid_id)

        # Build a unified work list: [{url, keyword, ...}]
        work_items = []
        seen_ids_in_work = set()

        # 1. Keyword search phase
        if keywords:
            self.log.info(f"Starting TikTok keyword search. Keywords: {keywords}")
            for kw in keywords:
                search_results = await self.search_videos_by_keyword(kw, max_results=max_videos_per_kw)
                for sr in search_results:
                    vid_id = sr.get("video_id") or ""
                    if vid_id not in seen_ids_in_work and vid_id not in processed_ids:
                        seen_ids_in_work.add(vid_id)
                        work_items.append(sr)

        # 2. Direct URLs
        if urls:
            for url in urls:
                m = re.search(r"/video/(\d+)", url)
                vid_id = m.group(1) if m else url
                if vid_id not in seen_ids_in_work and vid_id not in processed_ids:
                    seen_ids_in_work.add(vid_id)
                    work_items.append({"url": url, "video_id": vid_id, "keyword": None})

        self.log.info(f"Starting TikTok Crawl. Total videos to process: {len(work_items)}")

        for idx, item in enumerate(work_items):
            url = item["url"]
            kw = item.get("keyword")

            self.log.info(f"Processing TikTok video {idx+1}/{len(work_items)}: {url}")

            # Fetch oEmbed to get title and aweme ID
            oembed = self.fetch_oembed(url)
            aweme_id = None
            title = None
            author = None

            if oembed:
                aweme_id = oembed.get("embed_product_id")
                title = oembed.get("title")
                author = oembed.get("author_name")

            if not aweme_id:
                m = re.search(r"/video/(\d+)", url)
                if m:
                    aweme_id = m.group(1)

            if not aweme_id:
                self.log.warning(f"Could not extract video ID for URL: {url}. Skipping.")
                continue

            if aweme_id in processed_ids:
                continue

            # Fetch extra metadata via yt-dlp
            yt_info = self.fetch_video_metadata_with_yt_dlp(url)

            # Fetch comments using the cookie REST API
            comments_data = self.fetch_comments_with_replies(aweme_id, max_comments=max_comments_per_video)

            # Assemble record
            record = {
                "keyword": kw,
                "url": url,
                "video_id": aweme_id,
                "title": title or yt_info.get("title") or item.get("title"),
                "description": yt_info.get("description"),
                "author": author or yt_info.get("uploader"),
                "stats": {
                    "views": yt_info.get("view_count"),
                    "likes": yt_info.get("like_count"),
                    "comments_count": yt_info.get("comment_count"),
                },
                "comments": comments_data,
                "crawled_at": datetime.datetime.utcnow().isoformat() + "Z"
            }

            all_results.append(record)
            processed_ids.add(aweme_id)

            # Save checkpoint
            self.checkpoint_mgr.save(idx, [record])

            # Politeness delay
            await self.sleep_polite(2.0)

        self.save_final_results(all_results, resume)
        return all_results

