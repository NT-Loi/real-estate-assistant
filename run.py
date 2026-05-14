"""
Run the batdongsan.com.vn crawler.

Usage:
    python run.py                          # Crawl 3 pages of sale listings
    python run.py --type cho-thue          # Crawl rental listings
    python run.py --type ban --pages 5     # Crawl 5 pages of sale listings
    python run.py --type all --pages 3     # Crawl both sale and rental
    python run.py --no-details             # Skip detail page visits (faster)
"""
import argparse
import asyncio

from crawler import crawl


def main():
    parser = argparse.ArgumentParser(description="Crawl batdongsan.com.vn listings")
    parser.add_argument(
        "--type",
        choices=["ban", "cho-thue", "all"],
        default="ban",
        help="Listing type: 'ban' (sale), 'cho-thue' (rent), or 'all' (default: ban)",
    )
    parser.add_argument(
        "--pages", type=int, default=3, help="Number of listing pages to crawl (default: 3)"
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip visiting detail pages (faster but less data)",
    )
    args = parser.parse_args()

    types = ["ban", "cho-thue"] if args.type == "all" else [args.type]

    for lt in types:
        asyncio.run(crawl(listing_type=lt, max_pages=args.pages, visit_details=not args.no_details))


if __name__ == "__main__":
    main()
