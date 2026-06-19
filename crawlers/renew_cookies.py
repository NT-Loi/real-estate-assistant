import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from playwright.async_api import async_playwright

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("renew_cookies")

def _save_netscape_cookies(cookies: List[Dict[str, Any]], file_path: Path):
    """Write Playwright cookies to a Netscape format cookies.txt file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# This file is generated automatically\n\n")
        for c in cookies:
            domain = c.get("domain", "")
            # Domain starting with dot means it includes subdomains
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure", False) else "FALSE"
            expires = str(int(c.get("expires", 0)))
            name = c.get("name", "")
            value = c.get("value", "")
            f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")

async def auto_renew_bds_cookies():
    """Launch a visible browser to let user/auto-bypass CF, and save cookies_bds.txt"""
    log.info("Launching visible browser to bypass Cloudflare and renew cookies...")
    
    async with async_playwright() as pw:
        # We use a non-headless browser to bypass Cloudflare easier
        browser = await pw.chromium.launch(
            headless=False, 
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        page = await context.new_page()
        log.info("Navigating to batdongsan.com.vn...")
        await page.goto("https://batdongsan.com.vn/nha-dat-ban", wait_until="domcontentloaded")
        
        # Wait until the title is NOT Cloudflare
        cf_titles = ["Chờ một chút", "Xác minh bảo mật", "Just a moment", "Cloudflare"]
        
        # Give up to 60 seconds to pass it
        success = False
        log.info("Waiting up to 60 seconds for Cloudflare wall to clear. Please click the checkbox if required...")
        for i in range(30):
            await asyncio.sleep(2)
            title = await page.title()
            if not any(kw in title for kw in cf_titles):
                log.info(f"Cloudflare bypassed successfully! Real title: {title}")
                success = True
                break
            if i % 5 == 0 and i > 0:
                log.info("Still waiting for Cloudflare challenge to be solved...")

        if success:
            # We wait 2 more seconds just to make sure cookies are fully set
            await asyncio.sleep(2)
            cookies = await context.cookies()
            _save_netscape_cookies(cookies, Path("cookies_bds.txt"))
            log.info(f"Successfully downloaded {len(cookies)} cookies and saved to cookies_bds.txt!")
        else:
            log.warning("Timeout waiting to bypass Cloudflare. Cookies not renewed.")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(auto_renew_bds_cookies())
