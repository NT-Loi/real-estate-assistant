import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    stream=sys.stdout
)

from crawlers.batdongsan.listings import ListingCrawler
from crawlers.batdongsan.projects import ProjectCrawler

async def main():
    print("=== TESTING LISTING DETAIL CRAWL ===")
    listing_crawler = ListingCrawler(listing_type="ban")
    # Run a small crawl of 1 page with details enabled
    listings = await listing_crawler.crawl(max_pages=1, visit_details=True, resume=False)
    print(f"Crawled {len(listings)} listings.")
    if listings:
        # Check first listing
        l = listings[0]
        print(f"Listing Title: {l.get('tieu_de')}")
        print(f"Listing Specs:")
        print(f"  Price: {l.get('gia')}")
        print(f"  Area: {l.get('dien_tich')}")
        print(f"  Legal: {l.get('phap_ly')}")
        print(f"  Orientation: {l.get('huong_nha')}")
        print(f"  Full Description Length: {len(l.get('mo_ta_chi_tiet', '')) if l.get('mo_ta_chi_tiet') else 0}")
        if l.get('mo_ta_chi_tiet'):
            print(f"  Description Preview: {l['mo_ta_chi_tiet'][:150]}...")
            
    print("\n=== TESTING PROJECT DETAIL CRAWL ===")
    project_crawler = ProjectCrawler()
    # Run project crawl of 1 page with details
    projects = await project_crawler.crawl(max_pages=1, visit_details=True, resume=False)
    print(f"Crawled {len(projects)} projects.")
    if projects:
        p = projects[0]
        print(f"Project Title: {p.get('ten_du_an')}")
        print(f"Project Specs:")
        print(f"  Developer: {p.get('chu_dau_tu')}")
        print(f"  Construction Density: {p.get('mat_do_xay_dung')}")
        print(f"  Scale: {p.get('quy_mo')}")
        print(f"  Area: {p.get('dien_tich')}")
        print(f"  Full Description Length: {len(p.get('mo_ta_chi_tiet', '')) if p.get('mo_ta_chi_tiet') else 0}")
        if p.get('mo_ta_chi_tiet'):
            print(f"  Description Preview: {p['mo_ta_chi_tiet'][:150]}...")

if __name__ == "__main__":
    asyncio.run(main())
