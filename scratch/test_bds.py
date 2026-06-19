import asyncio
from playwright.async_api import async_playwright
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crawlers.browser import launch_browser, new_stealth_page

async def main():
    async with async_playwright() as pw:
        browser = await launch_browser(pw)
        context, page = await new_stealth_page(browser)
        print("Navigating...")
        await page.goto("https://batdongsan.com.vn/nha-dat-ban", wait_until="domcontentloaded")
        print("Waiting 10s...")
        await asyncio.sleep(10)
        title = await page.title()
        print(f"Title: {title}")
        
        # Check if Cloudflare
        is_cf = any(kw in title for kw in ["Chờ một chút", "Xác minh bảo mật", "Just a moment", "Cloudflare"])
        print(f"Is CF: {is_cf}")
        
        await page.screenshot(path="scratch/bds_test.png", full_page=True)
        print("Saved screenshot to scratch/bds_test.png")
        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
