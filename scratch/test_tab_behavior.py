import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright

sys.path.append(str(Path(__file__).parent.parent))

from crawlers.browser import launch_browser, new_stealth_page

async def main():
    search_url = "https://batdongsan.com.vn/nha-dat-ban"
    detail_url = "https://batdongsan.com.vn/ban-can-ho-chung-cu-duong-nguyen-trai-phuong-thuong-dinh-hanoi-seasons-garden/cao-xa-la-ra-hang-l2-gio-oc-quyen-tro-lai-goc-3-nam-pr45811137"
    
    async with async_playwright() as pw:
        browser = await launch_browser(pw, headless=True)
        
        print("=== CASE 1: Navigating to detail page on the SAME tab that loaded search ===")
        context1, page1 = await new_stealth_page(browser)
        print("Navigating to search page...")
        await page1.goto(search_url, wait_until="domcontentloaded", timeout=60_000)
        print(f"Search page title: {await page1.title()}")
        await asyncio.sleep(2)
        
        print("Navigating to detail page on same tab...")
        await page1.goto(detail_url, wait_until="domcontentloaded", timeout=60_000)
        print(f"Detail page title (Same tab): {await page1.title()}")
        await context1.close()
        
        print("\n=== CASE 2: Navigating to detail page on a FRESH tab in the SAME browser ===")
        context2, page2 = await new_stealth_page(browser)
        print("Navigating to detail page on fresh tab...")
        await page2.goto(detail_url, wait_until="domcontentloaded", timeout=60_000)
        print(f"Detail page title (Fresh tab): {await page2.title()}")
        await context2.close()
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
