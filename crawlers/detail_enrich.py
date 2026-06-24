import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright

from crawlers.batdongsan.listings import ListingCrawler
from crawlers.batdongsan.news import NewsCrawler
from crawlers.batdongsan.projects import ProjectCrawler
from crawlers.browser import launch_browser, new_stealth_page
from crawlers.config import DATA_DIR, WIKI_CATEGORIES

log = logging.getLogger("bds_crawler.detail_enrich")


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save_records(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _merge_non_empty(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _missing_listing_detail(record: Dict[str, Any]) -> bool:
    return bool(record.get("url")) and not bool(record.get("mo_ta_chi_tiet"))


def _missing_project_detail(record: Dict[str, Any]) -> bool:
    return bool(record.get("url")) and (
        not bool(record.get("mo_ta_chi_tiet")) or not bool(record.get("dia_chi"))
    )


def _missing_article_detail(record: Dict[str, Any]) -> bool:
    return bool(record.get("url")) and not bool(record.get("noi_dung"))


async def enrich_listing_details(
    listing_type: str,
    data_dir: Path = DATA_DIR,
    limit: Optional[int] = None,
) -> int:
    crawler = ListingCrawler(listing_type=listing_type, output_file=data_dir / f"listings_{listing_type.replace('-', '_')}.json")
    path = crawler.output_file
    records = _load_records(path)
    targets = [idx for idx, record in enumerate(records) if _missing_listing_detail(record)]
    if limit is not None:
        targets = targets[:limit]

    log.info("Listing detail enrichment target=%s missing=%s limit=%s", listing_type, len(targets), limit)
    enriched = 0

    async with async_playwright() as pw:
        browser = await launch_browser(pw)
        for pos, idx in enumerate(targets, start=1):
            record = records[idx]
            url = record.get("url")
            detail: Dict[str, Any] = {}
            for attempt in range(3):
                context, page = await new_stealth_page(browser)
                detail = await crawler.scrape_detail_page(page, url)
                await context.close()
                if detail and any(v for v in detail.values() if v):
                    break
                if attempt < 2:
                    log.info("  Empty listing detail; retrying %s (%s/3)", url, attempt + 2)
                    await crawler.sleep_polite(3)

            if detail and any(v for v in detail.values() if v):
                updated = crawler.merge_listing(record, detail)
                records[idx] = _merge_non_empty(record, updated)
                enriched += 1
                _save_records(path, records)

            if pos % 10 == 0 or pos == len(targets):
                log.info("Listing enrichment progress %s/%s enriched=%s", pos, len(targets), enriched)
            await crawler.sleep_polite()

        await browser.close()

    return enriched


async def enrich_project_details(
    data_dir: Path = DATA_DIR,
    limit: Optional[int] = None,
) -> int:
    crawler = ProjectCrawler(output_file=data_dir / "projects.json")
    path = crawler.output_file
    records = _load_records(path)
    targets = [idx for idx, record in enumerate(records) if _missing_project_detail(record)]
    if limit is not None:
        targets = targets[:limit]

    log.info("Project detail enrichment missing=%s limit=%s", len(targets), limit)
    enriched = 0

    async with async_playwright() as pw:
        browser = await launch_browser(pw)
        for pos, idx in enumerate(targets, start=1):
            record = records[idx]
            url = record.get("url")
            detail: Dict[str, Any] = {}
            for attempt in range(3):
                context, page = await new_stealth_page(browser)
                detail = await crawler._scrape_project_detail(page, url)
                await context.close()
                if detail and any(v for v in detail.values() if v):
                    break
                if attempt < 2:
                    log.info("  Empty project detail; retrying %s (%s/3)", url, attempt + 2)
                    await crawler.sleep_polite(3)

            if detail and any(v for v in detail.values() if v):
                updated = crawler._merge_project(record, detail)
                records[idx] = _merge_non_empty(record, updated)
                enriched += 1
                _save_records(path, records)

            if pos % 10 == 0 or pos == len(targets):
                log.info("Project enrichment progress %s/%s enriched=%s", pos, len(targets), enriched)
            await crawler.sleep_polite()

        await browser.close()

    return enriched


async def enrich_article_details(
    filename: str,
    data_dir: Path = DATA_DIR,
    limit: Optional[int] = None,
    article_type: str = "tin-tuc",
    category: Optional[str] = None,
) -> int:
    crawler = NewsCrawler(output_file=data_dir / filename)
    path = crawler.output_file
    records = _load_records(path)
    targets = [idx for idx, record in enumerate(records) if _missing_article_detail(record)]
    if limit is not None:
        targets = targets[:limit]

    log.info("Article detail enrichment file=%s missing=%s limit=%s", filename, len(targets), limit)
    enriched = 0

    async with async_playwright() as pw:
        browser = await launch_browser(pw)
        for pos, idx in enumerate(targets, start=1):
            record = records[idx]
            url = record.get("url")
            detail: Dict[str, Any] = {}
            for attempt in range(3):
                context, page = await new_stealth_page(browser)
                detail = await crawler._scrape_article_detail(page, url)
                await context.close()
                if detail and detail.get("noi_dung"):
                    break
                if attempt < 2:
                    log.info("  Empty article detail; retrying %s (%s/3)", url, attempt + 2)
                    await crawler.sleep_polite(3)

            if detail and detail.get("noi_dung"):
                updated = crawler._merge_article(record, detail, category=category)
                updated["loai"] = article_type
                records[idx] = _merge_non_empty(record, updated)
                enriched += 1
                _save_records(path, records)

            if pos % 10 == 0 or pos == len(targets):
                log.info(
                    "Article enrichment progress file=%s %s/%s enriched=%s",
                    filename,
                    pos,
                    len(targets),
                    enriched,
                )
            await crawler.sleep_polite()

        await browser.close()

    return enriched


def rebuild_wiki_all(data_dir: Path = DATA_DIR) -> int:
    unified: List[Dict[str, Any]] = []
    seen_urls = set()
    for slug in WIKI_CATEGORIES:
        path = data_dir / f"wiki_{slug.replace('-', '_')}.json"
        for item in _load_records(path):
            url = item.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            unified.append(item)
    _save_records(data_dir / "wiki_all.json", unified)
    return len(unified)


async def enrich_wiki_details(
    data_dir: Path = DATA_DIR,
    limit: Optional[int] = None,
) -> int:
    total = 0
    remaining = limit
    for slug, category_name in WIKI_CATEGORIES.items():
        if remaining is not None and remaining <= 0:
            break
        filename = f"wiki_{slug.replace('-', '_')}.json"
        enriched = await enrich_article_details(
            filename=filename,
            data_dir=data_dir,
            limit=remaining,
            article_type="wiki",
            category=category_name,
        )
        total += enriched
        if remaining is not None:
            remaining -= enriched
    rebuild_wiki_all(data_dir)
    return total


async def enrich_details(
    target: str = "all",
    data_dir: Path = DATA_DIR,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}

    if target in ("ban", "all-listings", "all"):
        stats["ban"] = await enrich_listing_details("ban", data_dir=data_dir, limit=limit)
    if target in ("cho-thue", "all-listings", "all"):
        stats["cho-thue"] = await enrich_listing_details("cho-thue", data_dir=data_dir, limit=limit)
    if target in ("projects", "du-an", "all"):
        stats["projects"] = await enrich_project_details(data_dir=data_dir, limit=limit)
    if target in ("news", "tin-tuc", "all"):
        stats["news"] = await enrich_article_details(
            filename="news.json",
            data_dir=data_dir,
            limit=limit,
            article_type="tin-tuc",
        )
    if target in ("wiki", "all"):
        stats["wiki"] = await enrich_wiki_details(data_dir=data_dir, limit=limit)

    return stats
