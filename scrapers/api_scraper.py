"""
DataHawk — API / Structured Data Scraper

Extracts structured data directly from HTML pages without full rendering:
- JSON-LD (schema.org)
- Microdata
- Inline JSON in <script> tags
- Open Graph / meta tags

This is the cheapest strategy when structured data is present.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from config.settings import get_settings
from scrapers.base_scraper import BaseScraper, ScrapeResult, ScrapingStrategy

logger = logging.getLogger(__name__)


class APIScraper(BaseScraper):
    """Extracts structured/API data from pages without JS rendering."""

    strategy = ScrapingStrategy.API

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def scrape(self, url: str) -> ScrapeResult:
        """
        Fetch page HTML and extract embedded structured data.
        Falls back to returning raw HTML if no structured data found.
        """
        result = ScrapeResult(url=url, strategy=self.strategy)

        try:
            import time
            start = time.time()

            response = requests.get(
                url,
                headers={
                    "User-Agent": self.settings.user_agent,
                    "Accept": "text/html,application/xhtml+xml,*/*",
                },
                timeout=self.settings.request_timeout,
            )
            result.status_code = response.status_code
            result.final_url = response.url

            if response.status_code != 200:
                result.error = f"HTTP {response.status_code}"
                result.success = False
                return result

            html = response.text
            result.html = html
            result.html_size_bytes = len(response.content)

            # Extract all structured data
            structured = self._extract_structured_data(html)
            if structured:
                result.structured_data = structured
                logger.info(
                    f"API scrape: found {len(structured)} structured data sources from {url}"
                )

            result.success = True
            result.execution_time_ms = (time.time() - start) * 1000
            return result

        except Exception as e:
            result.error = f"API scraper error: {str(e)[:200]}"
            result.success = False
            logger.warning(f"API scrape failed for {url}: {e}")
            return result

    def _extract_structured_data(self, html: str) -> Dict[str, Any]:
        """Extract all structured data from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        data = {}

        # 1. JSON-LD
        json_ld = self._extract_json_ld(soup)
        if json_ld:
            data["json_ld"] = json_ld

        # 2. Open Graph tags
        og_data = self._extract_opengraph(soup)
        if og_data:
            data["opengraph"] = og_data

        # 3. Meta tags
        meta_data = self._extract_meta(soup)
        if meta_data:
            data["meta"] = meta_data

        # 4. Inline JSON from script tags
        inline_json = self._extract_inline_json(soup)
        if inline_json:
            data["inline_json"] = inline_json

        return data

    @staticmethod
    def _extract_json_ld(soup: BeautifulSoup) -> List[Dict]:
        """Extract JSON-LD structured data."""
        results = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                text = script.string or script.get_text()
                if text:
                    parsed = json.loads(text)
                    results.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    @staticmethod
    def _extract_opengraph(soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Open Graph meta tags."""
        og = {}
        for tag in soup.find_all("meta", property=re.compile(r"^og:")):
            prop = tag.get("property", "")
            content = tag.get("content", "")
            if prop and content:
                og[prop.replace("og:", "")] = content
        return og

    @staticmethod
    def _extract_meta(soup: BeautifulSoup) -> Dict[str, str]:
        """Extract relevant meta tags."""
        meta = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name", "") or tag.get("property", "")
            content = tag.get("content", "")
            if name and content and name not in ("viewport", "charset"):
                meta[name] = content
        return meta

    @staticmethod
    def _extract_inline_json(soup: BeautifulSoup) -> List[Dict]:
        """Extract JSON objects from <script> tags that aren't JSON-LD."""
        results = []
        for script in soup.find_all("script"):
            if script.get("type") == "application/ld+json":
                continue  # Already handled
            text = script.string or ""
            # Look for variable assignments containing JSON
            json_patterns = re.findall(
                r'(?:var\s+\w+\s*=|window\.\w+\s*=)\s*({[\s\S]*?});',
                text,
            )
            for match in json_patterns[:5]:  # Limit to first 5
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 2:
                        results.append(parsed)
                except json.JSONDecodeError:
                    continue
        return results
