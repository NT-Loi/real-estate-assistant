"""
Run the batdongsan.com.vn crawler.

Usage:
    python run.py                              # Crawl 3 pages of sale listings
    python run.py --type cho-thue              # Crawl rental listings
    python run.py --type ban --pages 5         # Crawl 5 pages of sale listings
    python run.py --type all-listings --pages 3 # Crawl both sale and rental
    python run.py --type du-an                 # Crawl project listings
    python run.py --type tin-tuc               # Crawl news articles
    python run.py --type wiki                  # Crawl all wiki categories
    python run.py --type wiki --wiki-cat mua-bds  # Crawl specific wiki category
    python run.py --type all                   # Crawl everything
    python run.py --no-details                 # Skip detail page visits (faster)

Wiki categories:
    mua-bds, ban-bds, thue-bds, tai-chinh,
    quy-hoach-phap-ly, noi-ngoai-that, phong-tuc
"""
import argparse
import asyncio

from crawler import crawl, crawl_projects, crawl_news, crawl_wiki, WIKI_CATEGORIES


def main():
    parser = argparse.ArgumentParser(description="Crawl batdongsan.com.vn listings")
    parser.add_argument(
        "--type",
        choices=["ban", "cho-thue", "all-listings", "du-an", "tin-tuc", "wiki", "all"],
        default="ban",
        help=(
            "Crawl type: 'ban' (sale), 'cho-thue' (rent), 'all-listings' (sale+rent), "
            "'du-an' (projects), 'tin-tuc' (news), 'wiki' (wiki BDS), "
            "'all' (everything). Default: ban"
        ),
    )
    parser.add_argument(
        "--pages", type=int, default=3,
        help="Number of pages to crawl per section (default: 3)",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip visiting detail pages (faster but less data)",
    )
    parser.add_argument(
        "--wiki-cat",
        choices=list(WIKI_CATEGORIES.keys()),
        default=None,
        help="Specific wiki category to crawl (default: all categories)",
    )
    args = parser.parse_args()

    visit = not args.no_details

    if args.type in ("ban", "cho-thue"):
        asyncio.run(crawl(listing_type=args.type, max_pages=args.pages, visit_details=visit))

    elif args.type == "all-listings":
        for lt in ["ban", "cho-thue"]:
            asyncio.run(crawl(listing_type=lt, max_pages=args.pages, visit_details=visit))

    elif args.type == "du-an":
        asyncio.run(crawl_projects(max_pages=args.pages, visit_details=visit))

    elif args.type == "tin-tuc":
        asyncio.run(crawl_news(max_pages=args.pages, visit_details=visit))

    elif args.type == "wiki":
        asyncio.run(crawl_wiki(max_pages=args.pages, visit_details=visit, wiki_category=args.wiki_cat))

    elif args.type == "all":
        # Crawl everything
        for lt in ["ban", "cho-thue"]:
            asyncio.run(crawl(listing_type=lt, max_pages=args.pages, visit_details=visit))
        asyncio.run(crawl_projects(max_pages=args.pages, visit_details=visit))
        asyncio.run(crawl_news(max_pages=args.pages, visit_details=visit))
        asyncio.run(crawl_wiki(max_pages=args.pages, visit_details=visit))


if __name__ == "__main__":
    main()
