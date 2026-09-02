"""
DataHawk — Dynamic Browser Scraper

Uses Selenium WebDriver for pages that require JavaScript rendering.
Supports both local Chrome (via webdriver-manager) and remote BrightData
proxy.  This is the medium-cost strategy.
"""

import logging
import time
from typing import Optional

from config.settings import get_settings
from scrapers.base_scraper import BaseScraper, ScrapeResult, ScrapingStrategy

logger = logging.getLogger(__name__)


class DynamicScraper(BaseScraper):
    """Scrapes dynamic/JS-heavy pages using Selenium WebDriver."""

    strategy = ScrapingStrategy.DYNAMIC

    def __init__(self, settings=None, use_brightdata: Optional[bool] = None):
        self.settings = settings or get_settings()
        self._use_brightdata = (
            use_brightdata if use_brightdata is not None
            else self.settings.use_brightdata
        )

    def scrape(self, url: str) -> ScrapeResult:
        """
        Render the page in a headless browser and return the full HTML.

        Strategy:
        1. If BrightData is configured and enabled → remote browser
        2. Else → local Chrome via webdriver-manager
        """
        if self._use_brightdata and self.settings.is_brightdata_configured:
            return self._scrape_brightdata(url)
        return self._scrape_local(url)

    # ── Local Selenium ─────────────────────────────────────────────

    def _scrape_local(self, url: str) -> ScrapeResult:
        """Use local headless Chrome via webdriver-manager."""
        result = ScrapeResult(url=url, strategy=ScrapingStrategy.DYNAMIC)

        for attempt in range(self.settings.max_retries):
            driver = None
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service

                try:
                    from webdriver_manager.chrome import ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                except ImportError:
                    logger.info("webdriver-manager not installed; using system chromedriver")
                    service = Service()

                options = Options()
                options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument(f"--user-agent={self.settings.user_agent}")
                options.add_argument("--window-size=1920,1080")

                start = time.time()
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_page_load_timeout(self.settings.selenium_timeout)
                driver.get(url)

                # Wait for dynamic content to load
                time.sleep(2)

                result.html = driver.page_source
                result.final_url = driver.current_url
                result.execution_time_ms = (time.time() - start) * 1000
                result.html_size_bytes = len(result.html.encode("utf-8", errors="ignore"))
                result.success = True
                result.status_code = 200

                logger.info(
                    f"Dynamic (local) scrape OK: {url} — "
                    f"{result.html_size_bytes:,} bytes in {result.execution_time_ms:.0f}ms"
                )
                return result

            except Exception as e:
                result.error = f"Local Selenium error: {str(e)[:200]}"
                logger.warning(
                    f"Dynamic (local) scrape failed (attempt {attempt + 1}): {e}"
                )
                result.retries = attempt + 1
                if attempt < self.settings.max_retries - 1:
                    time.sleep(self.settings.retry_delay)
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass

        result.success = False
        return result

    # ── BrightData Remote ──────────────────────────────────────────

    def _scrape_brightdata(self, url: str) -> ScrapeResult:
        """Use BrightData SuperProxy remote browser."""
        result = ScrapeResult(url=url, strategy=ScrapingStrategy.BRIGHTDATA)

        for attempt in range(self.settings.max_retries):
            try:
                from selenium.webdriver import Remote, ChromeOptions
                from selenium.webdriver.chromium.remote_connection import (
                    ChromiumRemoteConnection,
                )

                sbr_url = self.settings.brightdata_webdriver_url
                sbr_conn = ChromiumRemoteConnection(sbr_url, "goog", "chrome")

                start = time.time()
                with Remote(sbr_conn, options=ChromeOptions()) as driver:
                    driver.get(url)

                    # Attempt CAPTCHA solve (BrightData-specific)
                    try:
                        solve_res = driver.execute(
                            "executeCdpCommand",
                            {
                                "cmd": "Captcha.waitForSolve",
                                "params": {"detectTimeout": 10000},
                            },
                        )
                        logger.debug(
                            f"CAPTCHA status: {solve_res['value']['status']}"
                        )
                    except Exception:
                        pass  # Not all pages have CAPTCHAs

                    result.html = driver.page_source
                    result.final_url = driver.current_url

                result.execution_time_ms = (time.time() - start) * 1000
                result.html_size_bytes = len(result.html.encode("utf-8", errors="ignore"))
                result.success = True
                result.status_code = 200

                logger.info(
                    f"Dynamic (BrightData) scrape OK: {url} — "
                    f"{result.html_size_bytes:,} bytes in {result.execution_time_ms:.0f}ms"
                )
                return result

            except Exception as e:
                result.error = f"BrightData error: {str(e)[:200]}"
                logger.warning(
                    f"Dynamic (BrightData) scrape failed (attempt {attempt + 1}): {e}"
                )
                result.retries = attempt + 1
                if attempt < self.settings.max_retries - 1:
                    time.sleep(self.settings.retry_delay)

        result.success = False
        return result
