import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from crawlers.browser import launch_browser, new_stealth_page
from crawlers.batdongsan.listings import ListingCrawler

async def main():
    url = "https://batdongsan.com.vn/ban-can-ho-chung-cu-duong-nguyen-trai-phuong-thuong-dinh-hanoi-seasons-garden/cao-xa-la-ra-hang-l2-gio-oc-quyen-tro-lai-goc-3-nam-pr45811137"
    
    print(f"Crawling detailed specs and description for listing: {url}")
    
    async with async_playwright() as pw:
        browser = await launch_browser(pw, headless=True)
        context, page = await new_stealth_page(browser)
        
        crawler = ListingCrawler(listing_type="ban")
        
        # Directly call scrape_detail_page
        detail = await crawler.scrape_detail_page(page, url)
        
        await browser.close()
        
    print("\n--- RESULTS ---")
    for k, v in detail.items():
        if k == "_mo_ta_chi_tiet":
            print(f"{k}: Length {len(v)} characters")
            print(f"  Preview: {v[:200]}...")
        elif k == "_hinh_anh":
            print(f"{k}: {len(v)} images found.")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())
