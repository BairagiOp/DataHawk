"""
DataHawk — Baseline Extraction Methods

Implements three baseline methods for fair scientific comparison
against the proposed DataHawk framework.

Baseline 1: BeautifulSoup + rule-based extraction
Baseline 2: Selenium + rule-based extraction
Baseline 3: Raw webpage → LLM (no cleaning, no schema, no validation)
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from core.llm_client import GeminiClient, LLMResponse
from core.schema_generator import ExtractionSchema
from scrapers.static_scraper import StaticScraper
from scrapers.dynamic_scraper import DynamicScraper

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a baseline extraction method."""
    method: str
    records: List[Dict[str, Any]]
    success: bool = False
    error: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    html_size: int = 0
    content_size: int = 0
    retry_count: int = 0


class BaselineRunner:
    """
    Runs baseline extraction methods for comparison against DataHawk.

    Methods:
    1. bs4_rule_based: Static scrape + regex/heuristic extraction
    2. selenium_rule_based: Dynamic scrape + regex/heuristic extraction
    3. raw_llm: Static scrape + raw content → LLM (no schema/validation)
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def run_baseline_1(
        self, url: str, schema: ExtractionSchema, html: Optional[str] = None,
    ) -> BaselineResult:
        """
        Baseline 1: BeautifulSoup + rule-based extraction.

        Uses simple regex/heuristic patterns to extract data from static HTML.
        No LLM involved.
        """
        result = BaselineResult(method="bs4_rule_based", records=[])
        start = time.time()

        try:
            # Scrape if no HTML provided
            if html is None:
                scraper = StaticScraper()
                scrape_result = scraper.scrape(url)
                if not scrape_result.success:
                    result.error = f"Scrape failed: {scrape_result.error}"
                    return result
                html = scrape_result.html
                result.html_size = scrape_result.html_size_bytes

            soup = BeautifulSoup(html, "html.parser")

            # Remove scripts/styles
            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            result.content_size = len(text)

            # Rule-based extraction using regex patterns
            record = {}
            for field_name, field_schema in schema.fields.items():
                value = self._rule_extract(text, soup, field_name, field_schema.type)
                if value is not None:
                    record[field_name] = value

            if record:
                result.records = [record]
                result.success = True

        except Exception as e:
            result.error = str(e)

        result.latency_ms = (time.time() - start) * 1000
        return result

    def run_baseline_2(
        self, url: str, schema: ExtractionSchema, html: Optional[str] = None,
    ) -> BaselineResult:
        """
        Baseline 2: Selenium + rule-based extraction.

        Same rule-based approach but with Selenium for JS rendering.
        """
        result = BaselineResult(method="selenium_rule_based", records=[])
        start = time.time()

        try:
            if html is None:
                scraper = DynamicScraper()
                scrape_result = scraper.scrape(url)
                if not scrape_result.success:
                    result.error = f"Scrape failed: {scrape_result.error}"
                    return result
                html = scrape_result.html
                result.html_size = scrape_result.html_size_bytes
                result.retry_count = scrape_result.retries

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            result.content_size = len(text)

            record = {}
            for field_name, field_schema in schema.fields.items():
                value = self._rule_extract(text, soup, field_name, field_schema.type)
                if value is not None:
                    record[field_name] = value

            if record:
                result.records = [record]
                result.success = True

        except Exception as e:
            result.error = str(e)

        result.latency_ms = (time.time() - start) * 1000
        return result

    def run_baseline_3(
        self, url: str, schema: ExtractionSchema, html: Optional[str] = None,
    ) -> BaselineResult:
        """
        Baseline 3: Raw webpage → LLM (no cleaning, no schema, no validation).

        Sends the full cleaned text to Gemini with a simple free-text prompt,
        similar to the original DataHawk behavior.
        """
        result = BaselineResult(method="raw_llm", records=[])
        start = time.time()

        try:
            if html is None:
                scraper = StaticScraper()
                scrape_result = scraper.scrape(url)
                if not scrape_result.success:
                    result.error = f"Scrape failed: {scrape_result.error}"
                    return result
                html = scrape_result.html
                result.html_size = scrape_result.html_size_bytes

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            result.content_size = len(text)

            # Truncate for LLM (no intelligent chunking)
            truncated = text[:15000]

            # Simple free-text prompt (mimics original DataHawk)
            field_list = ", ".join(schema.field_names)
            prompt = (
                f"Extract the following information from this webpage content: "
                f"{field_list}.\n\n"
                f"Return the results as a JSON array of objects.\n\n"
                f"Content:\n{truncated}\n\n"
                f"Return JSON only:"
            )

            parsed, llm_response = self.llm.generate_json(prompt=prompt)
            result.input_tokens = llm_response.input_tokens
            result.output_tokens = llm_response.output_tokens

            if parsed is not None:
                if isinstance(parsed, list):
                    result.records = parsed
                elif isinstance(parsed, dict):
                    if "records" in parsed:
                        result.records = parsed["records"]
                    else:
                        result.records = [parsed]
                result.success = True
            else:
                result.error = "LLM returned invalid JSON"

        except Exception as e:
            result.error = str(e)

        result.latency_ms = (time.time() - start) * 1000
        return result

    # ── Rule-Based Extraction Helpers ──────────────────────────────

    @staticmethod
    def _rule_extract(text: str, soup: BeautifulSoup, field_name: str, field_type: str) -> Optional[Any]:
        """
        Attempt to extract a field value using regex/heuristic rules.
        """
        field_lower = field_name.lower()

        # Price extraction
        if "price" in field_lower or "cost" in field_lower:
            patterns = [
                r'[₹$€£¥]\s?[\d,]+\.?\d*',
                r'[\d,]+\.?\d*\s?(?:USD|EUR|INR|GBP)',
                r'(?:price|cost|amount)[:\s]*[₹$€£¥]?\s?[\d,]+\.?\d*',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group().strip()

        # Rating extraction
        if "rating" in field_lower or "score" in field_lower:
            patterns = [
                r'(\d\.?\d?)\s*/\s*5',
                r'(\d\.?\d?)\s*out of\s*5',
                r'(?:rating|score)[:\s]*(\d\.?\d?)',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1) if match.groups() else match.group()

        # URL extraction
        if "url" in field_lower or "link" in field_lower:
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith("http"):
                    return href

        # Title/name from headings
        if "title" in field_lower or "name" in field_lower:
            for heading in soup.find_all(["h1", "h2"]):
                text_content = heading.get_text(strip=True)
                if text_content and len(text_content) > 3:
                    return text_content

        # Date extraction
        if "date" in field_lower or "time" in field_lower or "published" in field_lower:
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{1,2}/\d{1,2}/\d{4}',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}',
            ]
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group()

        # Email extraction
        if "email" in field_lower:
            match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
            if match:
                return match.group()

        return None
