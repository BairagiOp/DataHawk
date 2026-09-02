"""
DataHawk — Base Scraper Interface

Defines the abstract contract that all scraper backends must implement,
along with the ScrapeResult data class that carries scraping output and
metrics for every operation.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ScrapingStrategy(str, Enum):
    """Enum identifying which scraping strategy was used."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    API = "api"
    BRIGHTDATA = "brightdata"
    FALLBACK = "fallback"


@dataclass
class ScrapeResult:
    """Result of a scraping operation with full metrics."""
    # Content
    html: str = ""
    url: str = ""
    final_url: str = ""  # After redirects

    # Metadata
    strategy: ScrapingStrategy = ScrapingStrategy.STATIC
    success: bool = False
    error: str = ""
    status_code: int = 0

    # Metrics
    execution_time_ms: float = 0.0
    html_size_bytes: int = 0
    retries: int = 0

    # Structured data found (JSON-LD, microdata etc.)
    structured_data: Optional[Dict] = None

    def __post_init__(self):
        if self.html and not self.html_size_bytes:
            self.html_size_bytes = len(self.html.encode("utf-8", errors="ignore"))


class BaseScraper(ABC):
    """Abstract base class for all scraper implementations."""

    strategy: ScrapingStrategy = ScrapingStrategy.STATIC

    @abstractmethod
    def scrape(self, url: str) -> ScrapeResult:
        """
        Scrape the given URL and return a ScrapeResult.

        Implementations should:
        - Handle errors gracefully (never raise; populate result.error)
        - Measure execution time
        - Track retries
        - Set success/failure flag
        """
        ...

    def scrape_with_timing(self, url: str) -> ScrapeResult:
        """Wrapper that automatically measures execution time."""
        start = time.time()
        result = self.scrape(url)
        result.execution_time_ms = (time.time() - start) * 1000
        return result
