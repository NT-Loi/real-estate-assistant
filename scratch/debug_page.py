import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright

sys.path.append(str(Path(__file__).parent.parent))

from crawlers.browser import launch_browser, new_stealth_page

async def main():
    url = "https://batdongsan.com.vn/ban-can-ho-chung-cu-duong-nguyen-trai-phuong-thuong-dinh-hanoi-seasons-garden/cao-xa-la-ra-hang-l2-gio-oc-quyen-tro-lai-goc-3-nam-pr45811137"
    
    print(f"Debugging detail page: {url}")
    
    async with async_playwright() as pw:
        browser = await launch_browser(pw, headless=True)
        context, page = await new_stealth_page(browser)
        
        print("Navigating to URL...")
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        
        print(f"Initial page title: {await page.title()}")
        
        print("Waiting 15 seconds for any verification redirects...")
        await asyncio.sleep(15)
        
        print(f"Page title after wait: {await page.title()}")
        print(f"Current page URL: {page.url}")
        
        # Save screenshot
        screenshot_path = Path("scratch/debug_screenshot.png")
        await page.screenshot(path=str(screenshot_path))
        print(f"Saved screenshot to: {screenshot_path}")
        
        # Save HTML
        html_path = Path("scratch/debug_page.html")
        html_path.write_text(await page.content(), encoding="utf-8")
        print(f"Saved HTML to: {html_path}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
