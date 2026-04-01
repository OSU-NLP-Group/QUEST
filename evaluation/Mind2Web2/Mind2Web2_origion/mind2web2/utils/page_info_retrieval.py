# Standard library imports
import asyncio
import base64
import hashlib
import random
import time
from io import BytesIO
from logging import Logger
from pathlib import Path
from typing import Optional, Tuple, Union

# Third-party imports
from PIL import Image
from patchright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)


import html2text

def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown."""
    h = html2text.HTML2Text()
    h.ignore_links = True      # Ignore hyperlinks
    h.ignore_emphasis = True   # Ignore bold/italic emphasis
    h.images_to_alt = True     # Convert images to alt text
    h.body_width = 0
    return h.handle(html)



# ================================ Constants ================================

def make_blank_png_b64() -> str:
    # Create 1×1 RGBA fully transparent pixel
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    # Convert to base64 and remove line breaks
    return base64.b64encode(buf.getvalue()).decode()


# Error handling constants
BLANK_IMG_B64 = make_blank_png_b64()
ERROR_TEXT = "\u26A0\ufe0f This URL could not be loaded (navigation error)."


# User-agent pools
DEFAULT_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
]


class PageManager:
    """
    Manage active Page within a BrowserContext, handling new pages, closures, crashes, navigations.
    """

    def __init__(self, context: BrowserContext, logger: Logger):
        self.context = context
        self.logger = logger
        self.current: Optional[Page] = None
        self.closing = False
        self._handlers = []
        # Listen for new page events on context
        handler = lambda page: asyncio.create_task(self._on_new_page(page))
        context.on('page', handler)
        self._handlers.append((context, 'page', handler))
        for pg in context.pages:
            asyncio.create_task(self._on_new_page(pg))

    async def _on_new_page(self, page: Page):
        if self.closing:
            return
        self.logger.debug(f'New page opened: {page.url}')
        self.current = page
        self._attach_handlers(page)

    def _attach_handlers(self, page: Page):
        for event in ('close', 'crash', 'framenavigated'):
            if event == 'close':
                cb = lambda: asyncio.create_task(self._on_close(page))
            elif event == 'crash':
                cb = lambda: asyncio.create_task(self._on_crash(page))
            else:
                cb = lambda frame: asyncio.create_task(self._on_navigate(page, frame))
            page.on(event, cb)
            self._handlers.append((page, event, cb))

    async def _on_close(self, page: Page):
        if self.closing:
            return
        self.logger.warning(f'Page closed: {page.url}')
        pages = self.context.pages
        if pages:
            await self._on_new_page(pages[-1])
        else:
            try:
                new_pg = await self.context.new_page()
                await self._on_new_page(new_pg)
            except Exception as e:
                self.logger.error(f'Failed to reopen page after close: {e}')

    async def _on_crash(self, page: Page):
        if self.closing:
            return
        self.logger.error(f'Page crashed: {page.url}, refreshing...')
        try:
            await page.reload()
        except Exception as e:
            self.logger.error(f'Reload after crash failed: {e}')

    async def _on_navigate(self, page: Page, frame):
        if self.closing:
            return
        if frame == page.main_frame:
            self.logger.debug(f'Frame navigated: {page.url}')
            self.current = page

    async def get(self) -> Page:
        if self.closing:
            raise RuntimeError('Context is closing')
        if not self.current or self.current.is_closed():
            # self.logger.info('No active page, creating a new one')
            page = await self.context.new_page()
            await self._on_new_page(page)
        return self.current

    def dispose(self):
        """Stop listening and prevent new page opens."""
        self.closing = True
        for emitter, event, cb in self._handlers:
            try:
                emitter.off(event, cb)
            except Exception:
                pass
        self._handlers.clear()


class BatchBrowserManager:
    """Robust browser manager for both batch and single web content extraction.
    
    Integrates PageManager's stability features while maintaining efficiency for batch processing.
    Can be used as a drop-in replacement for capture_page_content_async.
    """
    
    def __init__(self, headless: bool = True, max_retries: int = 3, max_concurrent_pages: int = 10):
        self.headless = headless
        self.max_retries = max_retries
        self.max_concurrent_pages = max_concurrent_pages
        self.playwright = None
        self.browser = None
        self._browser_lock = asyncio.Lock()
        self._page_semaphore = asyncio.Semaphore(max_concurrent_pages)
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        
    async def start(self):
        """Initialize the browser instance."""
        if self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security",
                    "--disable-site-isolation-trials", 
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--ignore-certificate-errors",
                    "--safebrowsing-disable-auto-save",
                    "--safebrowsing-disable-download-protection",
                    "--password-store=basic",
                    "--use-mock-keychain",
                ]
            )
            
    async def stop(self):
        """Clean up browser resources."""
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
            
    async def _restart_browser(self):
        """Restart browser if it crashes."""
        await self.stop()
        await self.start()
        
    async def capture_page(
        self, 
        url: str, 
        logger: Logger,
        wait_until: str = "load",
        timeout: int = 30000,
        grant_permissions: bool = True,
        user_data_dir: Union[str, Path] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Robust page capture with PageManager integration for stability.
        
        Returns:
            Tuple of (screenshot_b64, text_content)
        """

        logger.info(f"Start collecting page {url}")
        # Use semaphore to limit concurrent pages
        async with self._page_semaphore:
            # Ensure browser is running
            if not self.browser:
                async with self._browser_lock:
                    if not self.browser:  # Double-check pattern
                        await self.start()
            
            browser = self.browser  # Cache reference
            
            for attempt in range(self.max_retries):
                context = None
                page_manager = None
                try:
                    # Create context with enhanced settings (similar to original capture_page_content_async)
                    user_agent = random.choice(DEFAULT_USER_AGENTS)
                    headers = {"user-agent": user_agent}
                    
                    if user_data_dir:
                        # Use persistent context if user_data_dir provided
                        context = await self.playwright.chromium.launch_persistent_context(
                            user_data_dir=user_data_dir,
                            locale='en-US',
                            headless=self.headless,
                            ignore_https_errors=True,
                            extra_http_headers=headers,
                            viewport={
                                "width": random.randint(1050, 1150),
                                "height": random.randint(700, 800),
                            },
                        )
                    else:
                        # Regular context
                        context = await browser.new_context(
                            locale='en-US',
                            ignore_https_errors=True,
                            extra_http_headers=headers,
                            viewport={
                                "width": random.randint(1050, 1150),
                                "height": random.randint(700, 800),
                            }
                        )
                    
                    # Grant permissions if requested
                    if grant_permissions:
                        try:
                            await context.grant_permissions(
                                [
                                    "geolocation",
                                    "notifications", 
                                    "camera",
                                    "microphone",
                                    "clipboard-read",
                                    "clipboard-write",
                                ],
                                origin=url,
                            )
                        except Exception as e:
                            logger.debug(f'Failed to grant permissions: {e}')
                    
                    # Use PageManager for robust page handling
                    page_manager = PageManager(context, logger)
                    
                    # Navigate with robust error handling
                    try:
                        page = await page_manager.get()
                        await page.goto(url, wait_until=wait_until, timeout=timeout)
                    except Exception as e:
                        logger.info(f"Navigation timeout/error (continuing): {e}")
                    
                    # Enhanced scrolling for content discovery (from original implementation)
                    page = await page_manager.get()
                    for _ in range(3):
                        await page.keyboard.press("End")
                        await asyncio.sleep(random.uniform(0.3, 0.8))  # Faster for batch
                    await page.keyboard.press("Home")
                    await asyncio.sleep(random.uniform(0.3, 0.8))
                    
                    # Use CDP for efficient and reliable capture
                    page = await page_manager.get()
                    cdp = await context.new_cdp_session(page)
                    await cdp.send("Page.enable")
                    await cdp.send("DOM.enable")
                    await cdp.send("Runtime.enable")
                    
                    # Get proper page metrics
                    metrics = await cdp.send("Page.getLayoutMetrics")
                    css_vp = metrics["cssVisualViewport"]
                    css_content = metrics["cssContentSize"]
                    width = round(css_vp["clientWidth"])
                    height = round(min(css_content["height"], 6000))
                    scale = round(metrics.get("visualViewport", {}).get("scale", 1))
                    
                    # Set device metrics
                    await cdp.send(
                        "Emulation.setDeviceMetricsOverride",
                        {
                            "mobile": False,
                            "width": width,
                            "height": height,
                            "deviceScaleFactor": scale,
                        },
                    )
                    
                    # Small delay for stability
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    
                    # Capture screenshot and text using CDP
                    screenshot_task = cdp.send(
                        "Page.captureScreenshot",
                        {"format": "png", "captureBeyondViewport": True},
                    )

                    html_task = cdp.send("Runtime.evaluate", {
                        "expression": "document.documentElement.outerHTML",
                        "returnByValue": True,
                    })



                    shot_result, html_result = await asyncio.gather(screenshot_task, html_task)
                    screenshot_b64 = shot_result.get("data")
                    page_html = html_result.get("result", {}).get("value", "")
                    page_text=html_to_markdown(page_html)



                    return screenshot_b64, page_text
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
                    
                    # Check if browser crashed
                    if ("Target page, context or browser has been closed" in str(e) or 
                        "Browser has been closed" in str(e) or
                        "browser.newContext" in str(e)):
                        # Browser crash - restart under lock
                        async with self._browser_lock:
                            if self.browser == browser:
                                logger.warning("Browser crashed, restarting...")
                                await self._restart_browser()
                                browser = self.browser
                        
                    if attempt == self.max_retries - 1:
                        # Last attempt failed
                        return make_blank_png_b64(), ERROR_TEXT
                        
                finally:
                    # Cleanup resources
                    if page_manager:
                        page_manager.dispose()
                    if context:
                        try:
                            await context.close()
                        except:
                            pass
                            
            return make_blank_png_b64(), ERROR_TEXT
