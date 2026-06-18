import asyncio
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from crawlers.browser import launch_browser, new_stealth_page

async def main():
    url = "https://batdongsan.com.vn/du-an-bat-dong-san"
    print(f"Navigating to: {url}")
    async with async_playwright() as pw:
        browser = await launch_browser(pw, headless=True)
        context, page = await new_stealth_page(browser)
        
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        await asyncio.sleep(8)
        
        html = await page.content()
        await browser.close()
        
    soup = BeautifulSoup(html, "html.parser")
    
    # Save a copy of HTML for safe inspection
    out_path = Path("scratch/project_list_page.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved to {out_path}")
    
    # Find card elements
    card_selectors = ['.js__card', '.re__card-full', '[class*="prj-card"]', '[data-tracking-id]']
    for sel in card_selectors:
        cards = soup.select(sel)
        print(f"Selector '{sel}' found {len(cards)} elements.")
        if cards:
            # Print the HTML of the first card
            print("\nFirst project card HTML:")
            print(cards[0].prettify()[:2000])
            break

if __name__ == "__main__":
    asyncio.run(main())
