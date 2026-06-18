import asyncio
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from crawlers.browser import launch_browser, new_stealth_page

async def main():
    url = "https://batdongsan.com.vn/du-an-khu-phuc-hop-long-bien/d-diamant-bleu-pj6732"
    
    print(f"Navigating to project detail: {url}")
    async with async_playwright() as pw:
        browser = await launch_browser(pw, headless=True)
        context, page = await new_stealth_page(browser)
        
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        
        # Check title
        title = await page.title()
        print(f"Initial Title: {title}")
        
        if "Chờ một chút" in title or "Xác minh" in title or "Just a moment" in title:
            print("Security page detected! Waiting 12 seconds for auto-bypass...")
            await asyncio.sleep(12)
            title = await page.title()
            
        html = await page.content()
        await browser.close()
        
    out_path = Path("scratch/project_detail.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Correct HTML saved to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
