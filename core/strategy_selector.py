"""
DataHawk — Adaptive Scraping Strategy Selector

One of the major research components.  Chooses the cheapest adequate
scraping strategy based on page classification, then falls back to
more expensive strategies if the chosen one fails.

Decision logic:
  STATIC / STRUCTURED → StaticScraper  (cheapest)
  API_DRIVEN          → APIScraper      (cheapest for structured)
  DYNAMIC             → DynamicScraper  (medium cost)
  COMPLEX / UNKNOWN   → DynamicScraper  (most robust, highest cost)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from core.classifier import PageClassifier, PageType, PageFeatures
from scrapers.base_scraper import BaseScraper, ScrapeResult, ScrapingStrategy
from scrapers.static_scraper import StaticScraper
from scrapers.dynamic_scraper import DynamicScraper
from scrapers.api_scraper import APIScraper

logger = logging.getLogger(__name__)


@dataclass
class StrategyDecision:
    """Records why a particular strategy was selected."""
    page_type: PageType
    selected_strategy: ScrapingStrategy
    fallback_chain: List[ScrapingStrategy]
    reason: str
    features: Optional[PageFeatures] = None


class StrategySelector:
    """
    Adaptive strategy selector that picks the cheapest adequate
    scraping approach and falls back to more robust alternatives.
    """

    # Cost ranking (lower = cheaper)
    STRATEGY_COST = {
        ScrapingStrategy.API: 1,
        ScrapingStrategy.STATIC: 2,
        ScrapingStrategy.DYNAMIC: 3,
        ScrapingStrategy.BRIGHTDATA: 4,
    }

    def __init__(
        self,
        classifier: Optional[PageClassifier] = None,
        static_scraper: Optional[StaticScraper] = None,
        dynamic_scraper: Optional[DynamicScraper] = None,
        api_scraper: Optional[APIScraper] = None,
    ):
        self.classifier = classifier or PageClassifier()
        self._scrapers = {
            ScrapingStrategy.STATIC: static_scraper or StaticScraper(),
            ScrapingStrategy.DYNAMIC: dynamic_scraper or DynamicScraper(),
            ScrapingStrategy.API: api_scraper or APIScraper(),
        }

    def select_strategy(
        self, html_preview: str = "", page_type: Optional[PageType] = None,
        features: Optional[PageFeatures] = None,
    ) -> StrategyDecision:
        """
        Determine the optimal scraping strategy.

        If html_preview is provided (e.g., from a HEAD/quick GET),
        classify the page and select strategy.
        If page_type is provided directly, skip classification.
        """
        if page_type is None and html_preview:
            page_type, features = self.classifier.classify(html_preview)
        elif page_type is None:
            page_type = PageType.UNKNOWN

        strategy, reason, fallbacks = self._decide(page_type)

        decision = StrategyDecision(
            page_type=page_type,
            selected_strategy=strategy,
            fallback_chain=fallbacks,
            reason=reason,
            features=features,
        )

        logger.info(
            f"Strategy selected: {strategy.value} for {page_type.value} page. "
            f"Reason: {reason}. Fallbacks: {[f.value for f in fallbacks]}"
        )
        return decision

    def _decide(
        self, page_type: PageType
    ) -> tuple[ScrapingStrategy, str, List[ScrapingStrategy]]:
        """Core decision logic. Returns (strategy, reason, fallbacks)."""

        if page_type == PageType.STATIC:
            return (
                ScrapingStrategy.STATIC,
                "Page is mostly static HTML; requests+BS4 is sufficient",
                [ScrapingStrategy.DYNAMIC],
            )

        if page_type == PageType.STRUCTURED:
            return (
                ScrapingStrategy.API,
                "Page contains structured data (JSON-LD/tables); API extraction first",
                [ScrapingStrategy.STATIC, ScrapingStrategy.DYNAMIC],
            )

        if page_type == PageType.API_DRIVEN:
            return (
                ScrapingStrategy.API,
                "SPA/API-driven page; try structured data extraction first",
                [ScrapingStrategy.DYNAMIC],
            )

        if page_type == PageType.DYNAMIC:
            return (
                ScrapingStrategy.DYNAMIC,
                "JavaScript-heavy page; browser rendering required",
                [ScrapingStrategy.STATIC],  # Fallback: maybe JS isn't essential
            )

        if page_type == PageType.COMPLEX:
            return (
                ScrapingStrategy.DYNAMIC,
                "Complex page with iframes/deep DOM; full browser needed",
                [],
            )

        # UNKNOWN
        return (
            ScrapingStrategy.STATIC,
            "Unknown page type; try static first as cheapest option",
            [ScrapingStrategy.DYNAMIC],
        )

    def scrape(self, url: str, html_preview: str = "") -> tuple[ScrapeResult, StrategyDecision]:
        """
        Full adaptive scraping pipeline: classify → select → scrape → fallback.

        Returns:
            Tuple of (ScrapeResult, StrategyDecision).
        """
        # Step 1: Quick static fetch for classification (if no preview)
        if not html_preview:
            quick = StaticScraper().scrape(url)
            if quick.success:
                html_preview = quick.html

        # Step 2: Select strategy
        decision = self.select_strategy(html_preview)

        # Step 3: Scrape with selected strategy
        scraper = self._scrapers.get(decision.selected_strategy)
        if scraper:
            result = scraper.scrape_with_timing(url)
            if result.success:
                return result, decision

            logger.warning(
                f"Primary strategy {decision.selected_strategy.value} failed: "
                f"{result.error}. Trying fallbacks..."
            )

        # Step 4: Fallback chain
        for fallback_strategy in decision.fallback_chain:
            fallback_scraper = self._scrapers.get(fallback_strategy)
            if fallback_scraper:
                logger.info(f"Trying fallback: {fallback_strategy.value}")
                result = fallback_scraper.scrape_with_timing(url)
                if result.success:
                    result.strategy = fallback_strategy
                    return result, decision

        # Step 5: If we got a successful quick fetch earlier, use that
        if html_preview and (not scraper or not result.success):
            logger.info("All strategies failed; using initial static fetch")
            fallback_result = ScrapeResult(
                html=html_preview,
                url=url,
                strategy=ScrapingStrategy.FALLBACK,
                success=True,
                html_size_bytes=len(html_preview.encode("utf-8", errors="ignore")),
            )
            return fallback_result, decision

        # Final failure
        return ScrapeResult(
            url=url, strategy=ScrapingStrategy.FALLBACK,
            success=False, error="All scraping strategies exhausted"
        ), decision
