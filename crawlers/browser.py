import asyncio
import logging
from typing import Optional, Tuple
from playwright.async_api import Browser, BrowserContext, Page, Playwright

from crawlers.config import CONTEXT_OPTS, STEALTH_SCRIPT

log = logging.getLogger("bds_crawler.browser")

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
    """Create a new browser context with stealth injections and custom options."""
    context = await browser.new_context(**CONTEXT_OPTS)
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
