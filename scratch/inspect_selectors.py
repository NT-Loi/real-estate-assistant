import asyncio
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from crawlers.browser import launch_browser, new_stealth_page

async def main():
    url = "https://batdongsan.com.vn/ban-nha-biet-thu-lien-ke-xa-nghia-tru-vinhomes-ocean-park-3/khep-kin-24-7-vip-nhat-ocp3-on-song-giai-toa-re-kinh-khung-126m2-13-9-ty-mua-la-thang-pr45801502"
    
    print(f"Navigating to listing detail: {url}")
    async with async_playwright() as pw:
        browser = await launch_browser(pw, headless=True)
        context, page = await new_stealth_page(browser)
        
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        await asyncio.sleep(8)
        
        html = await page.content()
        await browser.close()
        
    soup = BeautifulSoup(html, "html.parser")
    
    # Save a copy of HTML for safe inspection
    out_path = Path("scratch/listing_detail.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved to {out_path}")
    
    # Let's inspect potential description containers
    print("\n--- DESCRIPTION CONTAINERS ---")
    desc_containers = [
        ".re__detail-content", ".re__section-body", "[class*='detail-content']",
        "[class*='section-body']", "[itemprop='description']", ".re__pr-description"
    ]
    for c in desc_containers:
        el = soup.select_one(c)
        if el:
            print(f"Selector '{c}' found! Length: {len(el.get_text())}")
            # print preview
            print(f"Preview: {el.get_text(strip=True)[:150]}...")
            
    # Let's inspect specifications ("Đặc điểm bất động sản")
    print("\n--- SPECIFICATION CONTAINERS ---")
    specs_containers = [
        ".re__pr-specs-content-item", ".re__pr-short-info-item", ".re__pr-attr-item",
        "[class*='specs-content']", "[class*='pr-specs']", "[class*='short-info']"
    ]
    for c in specs_containers:
        els = soup.select(c)
        print(f"Selector '{c}' found {len(els)} elements.")
        for el in els[:3]:
            print(f"  Item text: {el.get_text(strip=True)}")

if __name__ == "__main__":
    asyncio.run(main())
