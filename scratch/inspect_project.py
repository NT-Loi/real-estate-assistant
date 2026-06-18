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
        await asyncio.sleep(8)
        
        html = await page.content()
        await browser.close()
        
    soup = BeautifulSoup(html, "html.parser")
    
    # Save a copy of HTML for safe inspection
    out_path = Path("scratch/project_detail.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved to {out_path}")
    
    # Let's inspect potential description containers
    print("\n--- DESCRIPTION CONTAINERS ---")
    desc_containers = [
        ".js__prj-detail-content", ".re__project-editor", ".re__detail-content",
        ".re__project-desc", "[class*='project-editor']", "[class*='project-desc']"
    ]
    for c in desc_containers:
        el = soup.select_one(c)
        if el:
            print(f"Selector '{c}' found! Length: {len(el.get_text())}")
            print(f"Preview: {el.get_text(strip=True)[:150]}...")
            
    # Let's inspect specifications ("Thông tin dự án")
    print("\n--- SPECIFICATION CONTAINERS ---")
    specs_containers = [
        ".re__project-box-item", ".re__prj-config-item", "tbody.re__project-attr tr",
        "[class*='project-box']", "[class*='config-item']", "[class*='project-attr']",
        ".re__project-info-item", ".re__prj-overview-item", ".re__prj-attribute-item"
    ]
    for c in specs_containers:
        els = soup.select(c)
        print(f"Selector '{c}' found {len(els)} elements.")
        for el in els[:5]:
            print(f"  Item text: {el.get_text(separator=' - ', strip=True)}")

if __name__ == "__main__":
    asyncio.run(main())
