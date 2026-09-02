"""
DataHawk — Intelligent Page Classifier

Analyses webpage characteristics to classify pages into categories
that inform scraping strategy selection.  Uses a heuristic/rule-based
approach designed for easy replacement with an ML classifier.

Classification signals:
  - Script tag ratio
  - Noscript fallback presence
  - JSON-LD / structured data
  - Table structures
  - DOM complexity
  - iframe usage
  - Repeated DOM patterns
  - Framework indicators (React, Angular, Vue, etc.)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PageType(str, Enum):
    """Classification categories for web pages."""
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    API_DRIVEN = "API_DRIVEN"
    STRUCTURED = "STRUCTURED"
    COMPLEX = "COMPLEX"
    UNKNOWN = "UNKNOWN"


@dataclass
class PageFeatures:
    """Quantified features extracted from a page for classification."""
    total_html_size: int = 0
    script_count: int = 0
    script_bytes: int = 0
    script_ratio: float = 0.0  # script bytes / total HTML bytes
    noscript_present: bool = False
    noscript_has_content: bool = False
    json_ld_count: int = 0
    table_count: int = 0
    table_rows_total: int = 0
    iframe_count: int = 0
    form_count: int = 0
    dom_depth: int = 0
    unique_tags: int = 0
    repeated_class_patterns: int = 0  # Number of classes appearing ≥3 times
    has_react: bool = False
    has_angular: bool = False
    has_vue: bool = False
    has_next: bool = False
    has_spa_indicator: bool = False
    meta_tags_count: int = 0
    opengraph_present: bool = False
    microdata_present: bool = False
    link_count: int = 0
    image_count: int = 0
    text_content_length: int = 0
    text_to_html_ratio: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {k: v for k, v in self.__dict__.items()}


class ClassificationStrategy(Protocol):
    """Protocol for classification strategies (heuristic or ML)."""

    def classify(self, features: PageFeatures, html: str = "") -> PageType:
        ...


class HeuristicClassifier:
    """Rule-based page classifier using page feature signals."""

    def classify(self, features: PageFeatures, html: str = "") -> PageType:
        """
        Classify page type based on extracted features.

        Decision tree (priority order):
        1. If abundant JSON-LD / structured data → STRUCTURED
        2. If API-driven SPA indicators → API_DRIVEN
        3. If high script ratio + SPA framework → DYNAMIC
        4. If lots of tables with data → STRUCTURED
        5. If simple HTML with good text ratio → STATIC
        6. If complex DOM + iframes → COMPLEX
        7. Otherwise → UNKNOWN
        """
        # 1. Structured data
        if features.json_ld_count >= 1 and features.table_count >= 1:
            return PageType.STRUCTURED
        if features.json_ld_count >= 2:
            return PageType.STRUCTURED

        # 2. API/SPA driven
        if features.has_spa_indicator and features.script_ratio > 0.5:
            return PageType.API_DRIVEN
        if features.noscript_present and features.noscript_has_content and features.script_ratio > 0.4:
            return PageType.API_DRIVEN

        # 3. Dynamic (JS-heavy)
        if features.script_ratio > 0.3 and (
            features.has_react or features.has_angular or features.has_vue or features.has_next
        ):
            return PageType.DYNAMIC
        if features.script_ratio > 0.5:
            return PageType.DYNAMIC

        # 4. Table-heavy structured
        if features.table_count >= 2 and features.table_rows_total >= 5:
            return PageType.STRUCTURED

        # 5. Static
        if features.text_to_html_ratio > 0.2 and features.script_ratio < 0.2:
            return PageType.STATIC
        if features.text_content_length > 500 and features.script_count < 10:
            return PageType.STATIC

        # 6. Complex
        if features.iframe_count >= 2 or (features.script_ratio > 0.4 and features.dom_depth > 20):
            return PageType.COMPLEX

        # 7. Fallback
        if features.text_to_html_ratio > 0.1:
            return PageType.STATIC

        return PageType.UNKNOWN


class PageClassifier:
    """
    Analyses a page's HTML and classifies it.

    Uses feature extraction + a pluggable classification strategy
    (default: HeuristicClassifier).
    """

    def __init__(self, strategy: Optional[ClassificationStrategy] = None):
        self._strategy = strategy or HeuristicClassifier()

    def classify(self, html: str) -> tuple[PageType, PageFeatures]:
        """
        Classify a page from its HTML content.

        Returns:
            Tuple of (PageType, PageFeatures) for downstream use.
        """
        features = self.extract_features(html)
        page_type = self._strategy.classify(features, html)
        logger.info(
            f"Page classified as {page_type.value} "
            f"(scripts={features.script_count}, ratio={features.script_ratio:.2f}, "
            f"json_ld={features.json_ld_count}, tables={features.table_count})"
        )
        return page_type, features

    @staticmethod
    def extract_features(html: str) -> PageFeatures:
        """Extract quantified features from raw HTML."""
        features = PageFeatures()
        features.total_html_size = len(html.encode("utf-8", errors="ignore"))

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return features

        # Scripts
        scripts = soup.find_all("script")
        features.script_count = len(scripts)
        script_bytes = sum(len(str(s)) for s in scripts)
        features.script_bytes = script_bytes
        features.script_ratio = (
            script_bytes / features.total_html_size
            if features.total_html_size > 0 else 0.0
        )

        # Noscript
        noscripts = soup.find_all("noscript")
        features.noscript_present = len(noscripts) > 0
        features.noscript_has_content = any(
            len(ns.get_text(strip=True)) > 50 for ns in noscripts
        )

        # JSON-LD
        features.json_ld_count = len(
            soup.find_all("script", type="application/ld+json")
        )

        # Tables
        tables = soup.find_all("table")
        features.table_count = len(tables)
        features.table_rows_total = sum(len(t.find_all("tr")) for t in tables)

        # Iframes
        features.iframe_count = len(soup.find_all("iframe"))

        # Forms
        features.form_count = len(soup.find_all("form"))

        # DOM depth (approximate)
        features.dom_depth = PageClassifier._estimate_dom_depth(soup)

        # Unique tags
        all_tags = [tag.name for tag in soup.find_all(True)]
        features.unique_tags = len(set(all_tags))

        # Repeated class patterns
        class_counts: Dict[str, int] = {}
        for tag in soup.find_all(True, class_=True):
            classes = tag.get("class", [])
            for cls in classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        features.repeated_class_patterns = sum(
            1 for count in class_counts.values() if count >= 3
        )

        # Framework detection
        html_lower = html.lower()
        features.has_react = "__react" in html_lower or "react" in html_lower
        features.has_angular = "ng-" in html_lower or "angular" in html_lower
        features.has_vue = "__vue" in html_lower or "v-" in html_lower
        features.has_next = "__next" in html_lower or "_next" in html_lower
        features.has_spa_indicator = any([
            features.has_react, features.has_angular,
            features.has_vue, features.has_next,
            '<div id="root"' in html_lower,
            '<div id="app"' in html_lower,
            '<div id="__next"' in html_lower,
        ])

        # Meta tags
        features.meta_tags_count = len(soup.find_all("meta"))
        features.opengraph_present = bool(
            soup.find("meta", property=re.compile(r"^og:"))
        )
        features.microdata_present = bool(
            soup.find(attrs={"itemscope": True})
        )

        # Content metrics
        features.link_count = len(soup.find_all("a"))
        features.image_count = len(soup.find_all("img"))

        text = soup.get_text(separator=" ", strip=True)
        features.text_content_length = len(text)
        features.text_to_html_ratio = (
            len(text) / features.total_html_size
            if features.total_html_size > 0 else 0.0
        )

        return features

    @staticmethod
    def _estimate_dom_depth(soup: BeautifulSoup, max_samples: int = 10) -> int:
        """Estimate maximum DOM nesting depth by sampling leaf nodes."""
        max_depth = 0
        leaves = soup.find_all(string=True)[:max_samples]
        for leaf in leaves:
            depth = 0
            parent = leaf.parent
            while parent and depth < 50:
                depth += 1
                parent = parent.parent
            max_depth = max(max_depth, depth)
        return max_depth
