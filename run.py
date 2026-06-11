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
    OSMPOI,
)
from crawlers.config import GOOGLE_MAPS_API_KEY

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


def load_geocoded_dia_chi_coords(data_dir: Path) -> list[dict]:
    """Load default POI coordinates from every geocoded dia_chi record in crawled data."""
    import json

    targets = [
        ("listing_ban", "listings_ban.json", data_dir / "listings_ban.json"),
        ("listing_cho_thue", "listings_cho_thue.json", data_dir / "listings_cho_thue.json"),
        ("project", "projects.json", data_dir / "projects.json"),
    ]
    coords = []

    for source_type, source_file, path in targets:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            log.warning(f"Error reading {path.name} for default coordinates: {e}")
            continue

        if not isinstance(records, list):
            continue

        for idx, record in enumerate(records):
            dia_chi = " ".join(str(record.get("dia_chi") or "").split())
            lat = record.get("latitude")
            lng = record.get("longitude")
            if not dia_chi or lat is None or lng is None:
                continue
            try:
                lat_f, lng_f = float(lat), float(lng)
            except Exception:
                continue

            coords.append(
                {
                    "lat": lat_f,
                    "lng": lng_f,
                    "label": record.get("tieu_de") or record.get("ten_du_an") or dia_chi,
                    "dia_chi": dia_chi,
                    "source_type": source_type,
                    "source_file": source_file,
                    "source_index": idx,
                    "source_id": record.get("id") or record.get("url") or record.get("slug") or record.get("ten_du_an"),
                    "has_nearby_amenities": bool(record.get("nearby_amenities")),
                    "nearby_amenities_radius_m": record.get("nearby_amenities_radius_m"),
                }
            )

    return coords


def save_nearby_amenities_to_source_files(data_dir: Path, nearby_payloads: list[dict]) -> dict:
    """Attach nearby amenities to each source listing/project record."""
    import json

    by_file: dict[str, list[dict]] = {}
    for payload in nearby_payloads:
        source_file = payload.get("source_file")
        source_index = payload.get("source_index")
        if not source_file or source_index is None:
            continue
        by_file.setdefault(source_file, []).append(payload)

    summary = {}
    for source_file, payloads in by_file.items():
        path = data_dir / source_file
        if not path.exists():
            summary[source_file] = {"updated": 0, "missing": True}
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            log.warning(f"Error reading {source_file} to save nearby amenities: {e}")
            summary[source_file] = {"updated": 0, "missing": False, "error": str(e)}
            continue

        if not isinstance(records, list):
            summary[source_file] = {"updated": 0, "missing": False, "error": "not a list"}
            continue

        updated = 0
        for payload in payloads:
            idx = payload.get("source_index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(records):
                continue
            records[idx]["nearby_amenities"] = payload.get("amenities") or {}
            records[idx]["nearby_amenities_radius_m"] = payload.get("radius_m")
            records[idx]["nearby_amenities_source"] = "osm_overpass"
            records[idx]["nearby_amenities_target_latitude"] = payload.get("target_latitude")
            records[idx]["nearby_amenities_target_longitude"] = payload.get("target_longitude")
            updated += 1

        if updated:
            temp_file = path.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            temp_file.replace(path)

        summary[source_file] = {"updated": updated, "missing": False}
        log.info(f"Saved nearby amenities to {updated} records in {source_file}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Real Estate Recommendation System - Consolidated Crawler Suite")
    parser.add_argument(
        "--type",
        choices=[
            "ban", "cho-thue", "all-listings", "du-an", "tin-tuc", "wiki", 
            "youtube", "tiktok", "voz", "google-reviews", "poi", "osm-poi", "geocode", "all"
        ],
        default="ban",
        help=(
            "Type of crawl to run. "
            "Layer 1: 'ban' (sales), 'cho-thue' (rentals), 'all-listings', 'du-an' (projects), 'tin-tuc' (news), 'wiki'. "
            "Layer 2: 'youtube', 'tiktok', 'voz', 'google-reviews'. "
            "Layer 3 & Supp: 'poi', 'osm-poi', 'geocode'. "
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
        "--coords",
        type=str,
        default="",
        help="Target latitude,longitude pair for POI search. Defaults to all geocoded dia_chi coordinates from listings/projects.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2000,
        help="Search radius in meters for POIs (default: 2000)",
    )
    parser.add_argument(
        "--auto-keywords",
        action="store_true",
        help="Automatically generate location-corresponding review keywords from crawled listings & projects.",
    )
    parser.add_argument(
        "--geocode-limit",
        type=int,
        default=0,
        help="Maximum records per data file to geocode. Defaults to 0; for --type geocode, a non-default --pages value is used as the limit.",
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
        await crawler.crawl(keywords=review_keywords, max_videos_per_kw=args.pages, max_comments_per_video=50, resume=args.resume)
 
    # 6. TikTok Comments (keyword search + direct URLs)
    elif args.type == "tiktok":
        crawler = TikTokCrawler()
        review_keywords = resolve_keywords()
        await crawler.crawl(
            urls=url_list if url_list else None,
            keywords=review_keywords if review_keywords else None,
            max_videos_per_kw=args.pages,
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
            max_threads_per_page=args.pages * 3,
            max_threads_per_kw=args.pages * 3,
            visit_posts=visit,
            resume=args.resume
        )

    # 8. Google Maps reviews
    elif args.type == "google-reviews":
        if not GOOGLE_MAPS_API_KEY:
            log.error("google-reviews requires GOOGLE_MAPS_API_KEY. Use VOZ/TikTok/YouTube for key-free social signals.")
            return
        crawler = GoogleReviewsCrawler()
        await crawler.crawl(queries=resolve_keywords(), resume=args.resume)

    # 9. POIs (OSM by default, Google only when the key is explicitly configured)
    elif args.type in ("poi", "osm-poi"):
        try:
            if args.coords:
                lat_str, lng_str = args.coords.split(",")
                coordinate_targets = [{
                    "lat": float(lat_str),
                    "lng": float(lng_str),
                    "label": "manual --coords",
                    "dia_chi": "",
                    "source_type": "manual",
                    "source_index": 0,
                }]
            else:
                from crawlers.config import DATA_DIR
                coordinate_targets = load_geocoded_dia_chi_coords(DATA_DIR)
                if not coordinate_targets:
                    log.error("No geocoded dia_chi coordinates found. Run: python run.py --type geocode --geocode-limit 0")
                    return

            log.info(f"Running POI search for {len(coordinate_targets)} coordinate target(s), Radius: {args.radius}m")
            if args.type == "poi" and GOOGLE_MAPS_API_KEY:
                poi_service = GoogleMapsPOI()
                log.info("Using Google Maps POI because GOOGLE_MAPS_API_KEY is configured.")
            else:
                poi_service = OSMPOI()
                log.info("Using key-free OpenStreetMap POI retrieval.")

            nearby_payloads = []
            for target in coordinate_targets:
                lat, lng = target["lat"], target["lng"]
                log.info(f"POI target: {target['source_type']}#{target['source_index']} ({lat}, {lng}) - {target['label']}")
                pois = poi_service.search_nearby(lat, lng, radius=args.radius)
                if hasattr(poi_service, "amenities_for_target"):
                    payload = poi_service.amenities_for_target(target, pois, args.radius)
                    nearby_payloads.append(payload)
                    poi_service.save_nearby_amenities(nearby_payloads)
                    from crawlers.config import DATA_DIR
                    save_nearby_amenities_to_source_files(DATA_DIR, [payload])
                for poi_type, poi_list in pois.items():
                    log.info(f"  Category '{poi_type}': found {len(poi_list)} nearby places.")
            if hasattr(poi_service, "save_nearby_amenities") and nearby_payloads:
                poi_service.save_nearby_amenities(nearby_payloads)
                from crawlers.config import DATA_DIR
                save_summary = save_nearby_amenities_to_source_files(DATA_DIR, nearby_payloads)
                log.info(f"Nearby amenities source-file summary: {save_summary}")
        except ValueError:
            log.error("Invalid coordinate string. Please use format --coords 'lat,lng' (e.g. '10.7769,106.7009')")

    # 10. Geocoding enrichment for listing/project JSON files
    elif args.type == "geocode":
        limit = args.geocode_limit or (args.pages if args.pages != 3 else None)
        log.info(f"Starting key-free OSM geocoding. Per-file limit: {limit or 'all'}")
        summary = OSMPOI().enrich_data_files(max_records_per_file=limit)
        log.info(f"Geocoding summary: {summary}")

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
        
        if GOOGLE_MAPS_API_KEY:
            log.info("--- 9. Google Maps Places Reviews ---")
            await GoogleReviewsCrawler().crawl(queries=current_kws[:3], resume=args.resume)
        else:
            log.info("--- 9. Google Maps Places Reviews skipped: GOOGLE_MAPS_API_KEY is not configured ---")

        log.info("--- 10. Key-Free OSM Geocoding Enrichment ---")
        OSMPOI().enrich_data_files(max_records_per_file=args.geocode_limit or None)

        log.info("--- 11. Key-Free OSM Nearby Amenities ---")
        from crawlers.config import DATA_DIR
        coordinate_targets = load_geocoded_dia_chi_coords(DATA_DIR)
        poi_service = OSMPOI()
        nearby_payloads = []
        for target in coordinate_targets:
            pois = poi_service.search_nearby(target["lat"], target["lng"], radius=args.radius)
            payload = poi_service.amenities_for_target(target, pois, args.radius)
            nearby_payloads.append(payload)
            poi_service.save_nearby_amenities(nearby_payloads)
            save_nearby_amenities_to_source_files(DATA_DIR, [payload])
        if nearby_payloads:
            poi_service.save_nearby_amenities(nearby_payloads)
            save_nearby_amenities_to_source_files(DATA_DIR, nearby_payloads)
        
        log.info("Consolidated catalog collection sequence completed.")

def main():
    args = parse_args()
    asyncio.run(run_crawlers_async(args))

if __name__ == "__main__":
    main()
