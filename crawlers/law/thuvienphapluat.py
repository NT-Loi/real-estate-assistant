import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from crawlers.base import BaseCrawler
from crawlers.browser import launch_browser, new_stealth_page
from crawlers.config import LAW_DIR, REQUEST_DELAY

log = logging.getLogger("bds_crawler.law")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DEFAULT_LAWS = [
    {
        "ten_luat": "Luật Đất Đai 2024",
        "so_hieu": "31/2024/QH15",
        "slug": "luat_dat_dai_2024",
        "url": "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Dat-dai-2024-31-2024-QH15-544065.aspx",
    },
    {
        "ten_luat": "Luật Nhà Ở 2023",
        "so_hieu": "27/2023/QH15",
        "slug": "luat_nha_o_2023",
        "url": "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-nha-o-2023-27-2023-QH15-542394.aspx",
    },
    {
        "ten_luat": "Luật Kinh Doanh Bất Động Sản 2023",
        "so_hieu": "29/2023/QH15",
        "slug": "luat_kinh_doanh_bds_2023",
        "url": "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-Kinh-doanh-bat-dong-san-2023-29-2023-QH15-542395.aspx",
    },
]

class LawCrawler(BaseCrawler):
    """Crawler for structured legal documents (laws and articles) from thuvienphapluat.vn."""
    
    def __init__(self, output_file: Optional[Path] = None):
        super().__init__("laws", output_file or (LAW_DIR / "laws_all.json"))
        self.laws_list = DEFAULT_LAWS

    def _fetch_page(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch raw HTML from thuvienphapluat.vn using standard requests."""
        for attempt in range(retries):
            try:
                self.log.info(f"Fetching law text: {url} (attempt {attempt + 1})")
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                resp.encoding = "utf-8"
                return resp.text
            except requests.RequestException as e:
                self.log.warning(f"  Fetch failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5.0)
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Parse the publication date from the raw text body."""
        patterns = [
            r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
        ]
        for pat in patterns:
            m = re.search(pat, text[:2000], re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 3:
                    return f"{groups[0]}/{groups[1]}/{groups[2]}"
        return None

    def _parse_chapters(self, content_div) -> List[Dict[str, Any]]:
        """Parse structured law contents into chapters, articles, and clean texts."""
        elements = content_div.find_all(["p", "div", "h1", "h2", "h3", "h4", "span"])
        blocks: List[tuple] = []

        for el in elements:
            if el.find_parent(["p"]) and el.name != "p":
                continue

            text = el.get_text(strip=True)
            if not text or len(text) < 2:
                continue

            # Chapter match patterns
            chapter_match = re.match(
                r"^Ch[uư][oơ]ng\s+([IVXLCDM\d]+)\s*[.\-:]?\s*(.*)",
                text,
                re.IGNORECASE,
            )
            if chapter_match:
                chapter_num = chapter_match.group(1).strip()
                chapter_title = chapter_match.group(2).strip()
                if not chapter_title:
                    next_sib = el.find_next_sibling()
                    if next_sib:
                        next_text = next_sib.get_text(strip=True)
                        if next_text and not re.match(r"^(Ch[uư]|Đi[eề]u)", next_text, re.IGNORECASE):
                            chapter_title = next_text
                blocks.append(("chapter", f"Chương {chapter_num}: {chapter_title}"))
                continue

            # Article match patterns
            article_match = re.match(
                r"^Đi[eề]u\s+(\d+)[.\s:]+\s*(.*)",
                text,
                re.IGNORECASE,
            )
            if article_match:
                article_num = article_match.group(1).strip()
                article_title = article_match.group(2).strip()
                if len(article_title) > 200:
                    parts = article_title.split(".", 1)
                    if len(parts) == 2 and len(parts[0]) < 200:
                        blocks.append(("article", f"Điều {article_num}. {parts[0]}"))
                        blocks.append(("content", parts[1].strip()))
                    else:
                        blocks.append(("article", f"Điều {article_num}. {article_title[:150]}"))
                        if len(article_title) > 150:
                            blocks.append(("content", article_title[150:]))
                else:
                    blocks.append(("article", f"Điều {article_num}. {article_title}"))
                continue

            blocks.append(("content", text))

        chapters: List[Dict[str, Any]] = []
        current_chapter: Dict[str, Any] = {"chuong": "Phần mở đầu", "dieu_list": []}
        current_article: Optional[Dict[str, Any]] = None

        for block_type, block_text in blocks:
            if block_type == "chapter":
                if current_article:
                    current_chapter["dieu_list"].append(current_article)
                    current_article = None
                if current_chapter["dieu_list"]:
                    chapters.append(current_chapter)
                current_chapter = {"chuong": block_text, "dieu_list": []}

            elif block_type == "article":
                if current_article:
                    current_chapter["dieu_list"].append(current_article)
                
                m = re.match(r"Điều\s+(\d+)[.\s:]+\s*(.*)", block_text)
                if m:
                    current_article = {
                        "dieu_so": m.group(1),
                        "tieu_de": m.group(2).strip(),
                        "noi_dung": "",
                    }
                else:
                    current_article = {
                        "dieu_so": "",
                        "tieu_de": block_text,
                        "noi_dung": "",
                    }

            elif block_type == "content":
                if current_article:
                    if current_article["noi_dung"]:
                        current_article["noi_dung"] += "\n" + block_text
                    else:
                        current_article["noi_dung"] = block_text

        if current_article:
            current_chapter["dieu_list"].append(current_article)
        if current_chapter["dieu_list"]:
            chapters.append(current_chapter)

        return chapters

    def parse_law_document(self, html: str, law_info: dict) -> dict:
        """Parse html string into rich structured dictionary contents."""
        soup = BeautifulSoup(html, "html.parser")
        
        content_div = (
            soup.find("div", class_="content1")
            or soup.find("div", {"class": re.compile(r"content")})
            or soup.find("div", {"id": "vanbanContent"})
        )
        if not content_div:
            content_div = soup.find("article") or soup.find("div", {"class": "toanvancontent"})
        if not content_div:
            content_div = soup.find("body")

        if not content_div:
            return {**law_info, "full_text": "", "chapters": []}

        full_text = content_div.get_text(separator="\n", strip=True)
        ngay_ban_hanh = self._extract_date(full_text)
        chapters = self._parse_chapters(content_div)

        return {
            "ten_luat": law_info["ten_luat"],
            "so_hieu": law_info["so_hieu"],
            "url": law_info["url"],
            "ngay_ban_hanh": ngay_ban_hanh,
            "co_quan_ban_hanh": "Quốc Hội",
            "full_text": full_text,
            "chapters": chapters,
        }

    async def crawl(self, resume: bool = False) -> List[Dict[str, Any]]:
        """Crawl all designated law files sequentially via Playwright stealth browser, write detailed segments, and save the central database."""
        LAW_DIR.mkdir(parents=True, exist_ok=True)
        all_laws = []

        async with async_playwright() as pw:
            browser = await launch_browser(pw)
            context, page = await new_stealth_page(browser)

            for idx, law_info in enumerate(self.laws_list):
                self.log.info(f"Processing structured law: {law_info['ten_luat']}")
                
                # Individual output path
                law_output_path = LAW_DIR / f"{law_info['slug']}.json"
                if resume and law_output_path.exists():
                    try:
                        with open(law_output_path, "r", encoding="utf-8") as f:
                            all_laws.append(json.load(f))
                            self.log.info(f"  Loaded existing law file {law_output_path} (resume mode)")
                            continue
                    except Exception:
                        pass

                self.log.info(f"  Navigating to: {law_info['url']}")
                try:
                    await page.goto(law_info["url"], wait_until="domcontentloaded", timeout=60_000)
                    await asyncio.sleep(5)
                    
                    # Scroll a bit to trigger any dynamic text injection
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 4)")
                    await asyncio.sleep(2)
                    
                    html = await page.content()
                except Exception as e:
                    self.log.error(f"  Failed to load document via browser: {e}")
                    continue

                parsed = self.parse_law_document(html, law_info)
                
                # Remove full_text in segment file to prevent file size blowup
                save_data = {k: v for k, v in parsed.items() if k != "full_text"}
                
                total_articles = sum(len(ch["dieu_list"]) for ch in parsed["chapters"])
                self.log.info(f"  Success: Parsed {len(parsed['chapters'])} chapters, {total_articles} articles")

                # Save segment file
                with open(law_output_path, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                self.log.info(f"  Saved segment: {law_output_path}")

                all_laws.append(save_data)
                await asyncio.sleep(REQUEST_DELAY)

            await browser.close()

        # Save unified combined law database
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(all_laws, f, ensure_ascii=False, indent=2)
        self.log.info(f"Successfully compiled all laws to unified database: {self.output_file}")
        
        return all_laws
