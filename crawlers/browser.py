import os
import sys
import subprocess
import logging
from typing import Optional, Tuple, List, Dict, Any
from playwright.async_api import Browser, BrowserContext, Page, Playwright
from pathlib import Path

from crawlers.config import CONTEXT_OPTS, STEALTH_SCRIPT

log = logging.getLogger("bds_crawler.browser")

class CloudflareBlockedError(Exception):
    """Exception raised when Cloudflare blocks the crawler and cookies need renewal."""
    pass

def _parse_netscape_cookies(file_path: Path) -> List[Dict[str, Any]]:
    """Parse Netscape cookies into a list of dicts suitable for Playwright context.add_cookies()."""
    playwright_cookies = []
    if not file_path.exists():
        return playwright_cookies

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") and not line.startswith("#HttpOnly_"):
                    continue
                
                is_httponly = False
                if line.startswith("#HttpOnly_"):
                    is_httponly = True
                    line = line[10:]
                
                parts = line.split("\t")
                if len(parts) >= 7:
                    cookie = {
                        "domain": parts[0],
                        "path": parts[2],
                        "secure": parts[3].lower() == "true",
                        "name": parts[5],
                        "value": parts[6],
                        "httpOnly": is_httponly,
                    }
                    try:
                        expires = int(parts[4])
                        if expires > 0:
                            cookie["expires"] = expires
                    except ValueError:
                        pass
                    playwright_cookies.append(cookie)
    except Exception as e:
        log.error(f"Error parsing cookies from {file_path}: {e}")
        
    return playwright_cookies

async def launch_browser(pw: Playwright, headless: bool = True) -> Browser:
    """Launch chromium browser with automation control features disabled."""
    return await pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )

async def new_stealth_page(browser: Browser) -> Tuple[BrowserContext, Page]:
    """Create a new browser context with stealth injections, custom options, and optional cookies."""
    context = await browser.new_context(**CONTEXT_OPTS)
    
    # Attempt to load batdongsan.com.vn cookies if provided to bypass Cloudflare
    cookie_file = Path("cookies_bds.txt")
    if cookie_file.exists():
        cookies = _parse_netscape_cookies(cookie_file)
        if cookies:
            await context.add_cookies(cookies)
            
    await context.add_init_script(STEALTH_SCRIPT)
    page = await context.new_page()
    return context, page

async def goto_safe(page: Page, url: str, retries: int = 3, sleep_on_fail: float = 3.0) -> bool:
    """Navigate to a URL with multiple retry attempts."""
    for attempt in range(retries):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            return True
        except Exception as e:
            log.warning(f"  Navigation attempt {attempt + 1} to {url} failed: {e}")
            if attempt == retries - 1:
                return False
            await asyncio.sleep(sleep_on_fail)
    return False

async def check_and_renew_cloudflare(page: Page) -> bool:
    """Check if Cloudflare is blocking. If so, synchronously run renew_cookies.py and raise CloudflareBlockedError."""
    title = await page.title()
    cf_titles = ["Chờ một chút", "Xác minh bảo mật", "Just a moment", "Cloudflare"]
    if any(kw in title for kw in cf_titles):
        log.warning(f"Cloudflare block detected ('{title}'). Pausing to auto-renew cookies...")
        try:
            # sys.executable is the venv python
            subprocess.run([sys.executable, "crawlers/renew_cookies.py"], check=True)
            log.info("Cookie renewal script finished successfully.")
        except subprocess.CalledProcessError as e:
            log.error(f"Cookie renewal script failed with exit code {e.returncode}")
        
        raise CloudflareBlockedError("Cloudflare blocked the page. Cookies have been renewed, please retry the context.")
    return False
