import argparse
import asyncio
import logging
from pathlib import Path

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
    GoogleReviewsCrawler,
    GoogleMapsPOI,
    LawCrawler,
)

def generate_keywords_from_listings(data_dir: Path, max_keywords: int = 15) -> list[str]:
    """
    Generate relevant review search keywords from crawled listings and projects.
    
    Reads listings_ban.json, listings_cho_thue.json, and projects.json, extracts 
    unique locations (quan_huyen, tinh_thanh) and project names, and formats them 
    into search terms like 'review chung cư [location]'.
    """
    import json
    from db.normalizer import split_location
    
    locations = set()
    projects = set()
    
    # 1. Read listings
    for filename in ["listings_ban.json", "listings_cho_thue.json"]:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for r in records:
                    # Extract project
                    proj = r.get("du_an")
                    if proj and len(proj.strip()) > 3:
                        projects.add(proj.strip())
                    
                    # Extract location
                    loc_str = r.get("khu_vuc") or r.get("dia_chi")
                    if loc_str:
                        tinh, quan = split_location(loc_str)
                        if quan and tinh:
                            locations.add((quan.strip(), tinh.strip()))
                        elif quan:
                            locations.add((quan.strip(), ""))
            except Exception as e:
                log.warning(f"Error reading listing file {filename} for keyword generation: {e}")
                
    # 2. Read projects
    proj_filepath = data_dir / "projects.json"
    if proj_filepath.exists():
        try:
            with open(proj_filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                title = r.get("tieu_de") or r.get("ten_du_an")
                if title and len(title.strip()) > 3:
                    projects.add(title.strip())
                loc_str = r.get("khu_vuc") or r.get("dia_chi")
                if loc_str:
                    tinh, quan = split_location(loc_str)
                    if quan and tinh:
                        locations.add((quan.strip(), tinh.strip()))
        except Exception as e:
            log.warning(f"Error reading projects.json for keyword generation: {e}")

    # 3. Generate review search terms
    generated = []
    
    # Project specific
    for p in sorted(list(projects)):
        generated.append(f"review {p}")
        generated.append(f"review chung cư {p}")
        generated.append(f"đánh giá {p}")
        
    # Location specific
    for quan, tinh in sorted(list(locations)):
        loc_suffix = f"{quan}, {tinh}" if tinh else quan
        generated.append(f"review chung cư {loc_suffix}")
        generated.append(f"review khu dân cư {loc_suffix}")
        generated.append(f"review nhà đất {loc_suffix}")
        generated.append(f"review cho o {loc_suffix}")
        
    # Deduplicate and limit
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
            "youtube", "tiktok", "voz", "google-reviews", "poi", "law", "all"
        ],
        default="ban",
        help=(
            "Type of crawl to run. "
            "Layer 1: 'ban' (sales), 'cho-thue' (rentals), 'all-listings', 'du-an' (projects), 'tin-tuc' (news), 'wiki'. "
            "Layer 2: 'youtube', 'tiktok', 'voz', 'google-reviews'. "
            "Layer 3 & Supp: 'poi', 'law'. "
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
        default="Vinhomes Grand Park, Masteri Thảo Điền, Sun Avenue Quận 2",
        help="Comma-separated query keywords for search-based crawlers (youtube, tiktok, voz, google-reviews). E.g. 'review chung cư Vinhomes, review khu dân cư Mega'",
    )
    parser.add_argument(
        "--urls",
        type=str,
        default="https://www.tiktok.com/@yeuvietnam633/video/7541610727442976002",
        help="Comma-separated URLs to scrape (primarily for tiktok crawler)",
    )
    parser.add_argument(
        "--coords",
        type=str,
        default="10.7769,106.7009",
        help="Target latitude,longitude pair to run POI search tests (default: Ben Thanh, HCMC)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1000,
        help="Search radius in meters for POIs (default: 1000)",
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
        if getattr(args, "auto_keywords", False):
            from crawlers.config import DATA_DIR
            gen_kws = generate_keywords_from_listings(DATA_DIR)
            if gen_kws:
                log.info(f"Automatically resolved location review keywords: {gen_kws}")
                kw_list = gen_kws + kw_list
        return kw_list

    kw_list = resolve_keywords()
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
        await crawler.crawl(keywords=resolve_keywords(), max_videos_per_kw=args.pages, max_comments_per_video=50, resume=args.resume)
 
    # 6. TikTok Comments (keyword search + direct URLs)
    elif args.type == "tiktok":
        crawler = TikTokCrawler()
        await crawler.crawl(
            urls=url_list if url_list else None,
            keywords=resolve_keywords() if resolve_keywords() else None,
            max_videos_per_kw=args.pages,
            max_comments_per_video=50,
            resume=args.resume
        )
 
    # 7. VOZ Forum Discussions (keyword search or forum browse)
    elif args.type == "voz":
        crawler = VozCrawler()
        await crawler.crawl(
            keywords=resolve_keywords() if resolve_keywords() else None,
            max_pages=args.pages,
            max_threads_per_page=args.pages * 3,
            max_threads_per_kw=args.pages * 3,
            visit_posts=visit,
            resume=args.resume
        )

    # 8. Google Maps reviews
    elif args.type == "google-reviews":
        crawler = GoogleReviewsCrawler()
        await crawler.crawl(queries=kw_list, resume=args.resume)

    # 9. Google Maps POIs (Dynamic Query-Time retrieval test)
    elif args.type == "poi":
        try:
            lat_str, lng_str = args.coords.split(",")
            lat, lng = float(lat_str), float(lng_str)
            log.info(f"Running query-time POI search test at coordinates: ({lat}, {lng}), Radius: {args.radius}m")
            poi_service = GoogleMapsPOI()
            pois = poi_service.search_nearby(lat, lng, radius=args.radius)
            for poi_type, poi_list in pois.items():
                log.info(f"  Category '{poi_type}': found {len(poi_list)} nearby places.")
        except ValueError:
            log.error("Invalid coordinate string. Please use format --coords 'lat,lng' (e.g. '10.7769,106.7009')")

    # 10. Laws
    elif args.type == "law":
        crawler = LawCrawler()
        await crawler.crawl(resume=args.resume)

    # 11. Crawl Everything (Layer 1 + Layer 2 + Supp)
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
        await VozCrawler().crawl(keywords=current_kws[:3] if current_kws else None, max_pages=args.pages, max_threads_per_page=5, max_threads_per_kw=5, visit_posts=visit, resume=args.resume)
        
        log.info("--- 7. YouTube Neighborhood Reviews ---")
        await YouTubeCrawler().crawl(keywords=current_kws[:3], max_videos_per_kw=2, max_comments_per_video=30, resume=args.resume)
        
        log.info("--- 8. TikTok Neighborhood Discussions ---")
        await TikTokCrawler().crawl(urls=url_list[:2] if url_list else None, keywords=current_kws[:3] if current_kws else None, max_videos_per_kw=3, max_comments_per_video=30, resume=args.resume)
        
        log.info("--- 9. Google Maps Places Reviews ---")
        await GoogleReviewsCrawler().crawl(queries=current_kws[:3], resume=args.resume)
        
        log.info("Consolidated catalog collection sequence completed.")

def main():
    args = parse_args()
    asyncio.run(run_crawlers_async(args))

if __name__ == "__main__":
    main()
