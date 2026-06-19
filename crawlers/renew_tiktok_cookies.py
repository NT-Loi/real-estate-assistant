import asyncio
import logging
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crawlers.renew_cookies import _save_netscape_cookies
from crawlers.browser import _parse_netscape_cookies

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("renew_tiktok_cookies")

async def auto_renew_tiktok_cookies():
    log.info("Launching visible browser to renew TikTok cookies...")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False, 
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Load existing cookies to preserve login state if they are just slightly expired
        old_cookies = Path("cookies.txt")
        if old_cookies.exists():
            pw_cookies = _parse_netscape_cookies(old_cookies)
            if pw_cookies:
                await context.add_cookies(pw_cookies)

        page = await context.new_page()
        log.info("Navigating to tiktok.com...")
        await page.goto("https://www.tiktok.com/", wait_until="domcontentloaded")
        
        log.info("Please solve any captchas or log in if necessary.")
        log.info("We will wait up to 60 seconds. You can close the TAB (not the whole browser window) to finish early.")
        
        try:
            for i in range(60):
                await asyncio.sleep(1)
                if page.is_closed():
                    log.info("Tab closed. Grabbing cookies now...")
                    break
        except Exception:
            pass

        try:
            cookies = await context.cookies()
            _save_netscape_cookies(cookies, Path("cookies.txt"))
            log.info(f"Successfully downloaded {len(cookies)} cookies and saved to cookies.txt!")
        except Exception as e:
            log.error(f"Failed to extract cookies. Make sure you don't close the entire browser too early: {e}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(auto_renew_tiktok_cookies())
