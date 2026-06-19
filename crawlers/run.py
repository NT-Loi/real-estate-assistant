import sys
from pathlib import Path

# Add project root to sys.path to ensure absolute imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import logging

# Set up global logging format matching our package crawlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bds_crawler.runner")

# Import all crawlers from our new package
from crawlers import (
    ListingCrawler,
    ProjectCrawler,
    NewsCrawler,
    WikiCrawler,
    YouTubeCrawler,
    TikTokCrawler,
    VozCrawler,
)

def generate_keywords_from_listings(data_dir: Path, max_keywords: int = 15) -> list[str]:
    """
    Generate review search keywords from crawled listings and projects.

    Reads listings_ban.json, listings_cho_thue.json, and projects.json.
    - Listings: use dia_chi -> "review {dia_chi}"
    - Projects: use ten_du_an -> "review {ten_du_an}"
    """
    import json

    def clean_keyword_value(value: object) -> str:
        return " ".join(str(value or "").split())

    generated = []

    # 1. Read listings
    for filename in ["listings_ban.json", "listings_cho_thue.json"]:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for r in records:
                    dia_chi = clean_keyword_value(r.get("dia_chi"))
                    if len(dia_chi) > 3:
                        generated.append(f"review {dia_chi}")
            except Exception as e:
                log.warning(f"Error reading listing file {filename} for keyword generation: {e}")

    # 2. Read projects
    proj_filepath = data_dir / "projects.json"
    if proj_filepath.exists():
        try:
            with open(proj_filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                ten_du_an = clean_keyword_value(r.get("ten_du_an"))
                if len(ten_du_an) > 3:
                    generated.append(f"review {ten_du_an}")
        except Exception as e:
            log.warning(f"Error reading projects.json for keyword generation: {e}")

    # 3. Deduplicate and limit, preserving source file order.
    seen = set()
    final_keywords = []
    for g in generated:
        g_clean = g.strip()
        if g_clean.lower() not in seen:
            seen.add(g_clean.lower())
            final_keywords.append(g_clean)
            if len(final_keywords) >= max_keywords:
                break
                
    return final_keywords


def parse_args():
    parser = argparse.ArgumentParser(description="Real Estate Recommendation System - Consolidated Crawler Suite")
    parser.add_argument(
        "--type",
        choices=[
            "ban", "cho-thue", "all-listings", "du-an", "tin-tuc", "wiki", 
            "youtube", "tiktok", "voz", "all"
        ],
        default="ban",
        help=(
            "Type of crawl to run. "
            "Layer 1: 'ban' (sales), 'cho-thue' (rentals), 'all-listings', 'du-an' (projects), 'tin-tuc' (news), 'wiki'. "
            "Layer 2: 'youtube', 'tiktok', 'voz'. "
            "'all' crawls everything."
        ),
    )
    parser.add_argument(
        "--pages", type=int, default=3,
        help="Number of pages to crawl per section (default: 3)",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip visiting detail pages (faster but retrieves less structured fields)",
    )
    parser.add_argument(
        "--wiki-cat",
        default=None,
        help="Specific wiki category to crawl (e.g. 'mua-bds', default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load progress checkpoints and resume from last page (preventing duplicates)",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="Comma-separated query keywords for search-based crawlers. Defaults to generated 'review {dia_chi}' and 'review {ten_du_an}' keywords from crawled data.",
    )
    parser.add_argument(
        "--urls",
        type=str,
        default="",
        help="Comma-separated direct TikTok URLs to scrape. Defaults to keyword search results only.",
    )
    parser.add_argument(
        "--auto-keywords",
        action="store_true",
        help="Automatically generate location-corresponding review keywords from crawled listings & projects.",
    )
    return parser.parse_args()

async def run_crawlers_async(args):
    visit = not args.no_details
    
    def resolve_keywords():
        kw_list = [k.strip() for k in args.keywords.split(",") if k.strip()]
        if not kw_list or getattr(args, "auto_keywords", False):
            from crawlers.config import DATA_DIR
            gen_kws = generate_keywords_from_listings(DATA_DIR)
            if gen_kws:
                log.info(f"Automatically resolved location review keywords: {gen_kws}")
                kw_list = gen_kws + kw_list if kw_list else gen_kws
        return kw_list

    url_list = [u.strip() for u in args.urls.split(",") if u.strip()]

    # 1. Properties (For Sale / Rent)
    if args.type in ("ban", "cho-thue"):
        crawler = ListingCrawler(listing_type=args.type)
        await crawler.crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)

    elif args.type == "all-listings":
        for lt in ["ban", "cho-thue"]:
            crawler = ListingCrawler(listing_type=lt)
            await crawler.crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)

    # 2. Projects
    elif args.type == "du-an":
        crawler = ProjectCrawler()
        await crawler.crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)

    # 3. News
    elif args.type == "tin-tuc":
        crawler = NewsCrawler()
        await crawler.crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)

    # 4. Wiki
    elif args.type == "wiki":
        crawler = WikiCrawler()
        await crawler.crawl(max_pages=args.pages, visit_details=visit, resume=args.resume, wiki_category=args.wiki_cat)

    # 5. YouTube Comments & Transcripts
    elif args.type == "youtube":
        crawler = YouTubeCrawler()
        review_keywords = resolve_keywords()
        await crawler.crawl(keywords=review_keywords, max_videos_per_kw=20, max_comments_per_video=50, resume=args.resume)
 
    # 6. TikTok Comments (keyword search + direct URLs)
    elif args.type == "tiktok":
        crawler = TikTokCrawler()
        review_keywords = resolve_keywords()
        await crawler.crawl(
            urls=url_list if url_list else None,
            keywords=review_keywords if review_keywords else None,
            max_videos_per_kw=20,
            max_comments_per_video=50,
            resume=args.resume
        )
 
    # 7. VOZ Forum Discussions (keyword search or forum browse)
    elif args.type == "voz":
        crawler = VozCrawler()
        review_keywords = resolve_keywords()
        await crawler.crawl(
            keywords=review_keywords if review_keywords else None,
            max_pages=args.pages,
            max_threads_per_page=20,
            max_threads_per_kw=20,
            visit_posts=visit,
            resume=args.resume
        )

    # 8. Crawl Everything (Layer 1 + Layer 2)
    elif args.type == "all":
        log.info("Initiating full multi-source catalog collection...")
        
        log.info("--- 1. Property Sale Listings ---")
        await ListingCrawler(listing_type="ban").crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)
        
        log.info("--- 2. Property Rent Listings ---")
        await ListingCrawler(listing_type="cho-thue").crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)
        
        log.info("--- 3. Real Estate Projects ---")
        await ProjectCrawler().crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)
        
        log.info("--- 4. Market News ---")
        await NewsCrawler().crawl(max_pages=args.pages, visit_details=visit, resume=args.resume)
        
        log.info("--- 5. Wiki & Knowledge Base ---")
        await WikiCrawler().crawl(max_pages=args.pages, visit_details=visit, resume=args.resume, wiki_category=args.wiki_cat)
        
        log.info("--- 6. VOZ Neighborhood Forums ---")
        current_kws = resolve_keywords()
        await VozCrawler().crawl(keywords=current_kws if current_kws else None, max_pages=args.pages, max_threads_per_page=20, max_threads_per_kw=20, visit_posts=visit, resume=args.resume)
        
        log.info("--- 7. YouTube Neighborhood Reviews ---")
        await YouTubeCrawler().crawl(keywords=current_kws, max_videos_per_kw=20, max_comments_per_video=30, resume=args.resume)
        
        log.info("--- 8. TikTok Neighborhood Discussions ---")
        await TikTokCrawler().crawl(urls=url_list[:2] if url_list else None, keywords=current_kws if current_kws else None, max_videos_per_kw=20, max_comments_per_video=30, resume=args.resume)
        
        log.info("Consolidated catalog collection sequence completed.")

def main():
    args = parse_args()
    asyncio.run(run_crawlers_async(args))

if __name__ == "__main__":
    main()
