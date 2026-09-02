"""
DataHawk — Static HTML Scraper

Uses plain HTTP requests + BeautifulSoup for pages that do not require
JavaScript rendering.  This is the cheapest and fastest scraping strategy.
"""

import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config.settings import get_settings
from scrapers.base_scraper import BaseScraper, ScrapeResult, ScrapingStrategy

logger = logging.getLogger(__name__)


class StaticScraper(BaseScraper):
    """Scrapes static HTML pages using requests + BeautifulSoup."""

    strategy = ScrapingStrategy.STATIC

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def scrape(self, url: str) -> ScrapeResult:
        """
        Fetch page via HTTP GET and return raw HTML.

        Handles:
        - Automatic redirect following
        - Timeout enforcement
        - Retry with back-off
        - robots.txt respect (logged, not enforced — caller decides)
        """
        result = ScrapeResult(url=url, strategy=self.strategy)
        headers = {
            "User-Agent": self.settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        for attempt in range(self.settings.max_retries):
            try:
                start = time.time()
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.settings.request_timeout,
                    allow_redirects=True,
                )
                elapsed = (time.time() - start) * 1000

                result.status_code = response.status_code
                result.final_url = response.url
                result.execution_time_ms = elapsed

                if response.status_code == 200:
                    result.html = response.text
                    result.html_size_bytes = len(response.content)
                    result.success = True
                    logger.info(
                        f"Static scrape OK: {url} — {result.html_size_bytes:,} bytes "
                        f"in {elapsed:.0f}ms"
                    )
                    return result
                else:
                    result.error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"Static scrape HTTP {response.status_code}: {url}"
                    )

            except requests.Timeout:
                result.error = f"Timeout after {self.settings.request_timeout}s"
                logger.warning(f"Static scrape timeout: {url} (attempt {attempt + 1})")
            except requests.ConnectionError as e:
                result.error = f"Connection error: {str(e)[:100]}"
                logger.warning(f"Static scrape connection error: {url}")
            except Exception as e:
                result.error = f"Unexpected error: {str(e)[:100]}"
                logger.error(f"Static scrape failed: {url} — {e}")

            result.retries = attempt + 1
            if attempt < self.settings.max_retries - 1:
                time.sleep(self.settings.retry_delay * (attempt + 1))

        result.success = False
        return result
