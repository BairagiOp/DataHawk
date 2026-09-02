"""
DataHawk — Advanced DOM Cleaner

Intelligently strips irrelevant content from HTML while preserving
extraction-relevant elements.  This is a key research component:
reducing LLM input size while maintaining extraction accuracy.

Removes:  nav, footer, header, aside, ads, cookie banners, scripts,
          styles, iframes, social widgets, comment sections, SVGs.
Preserves: main, article, tables, lists, headings, paragraphs,
           product cards, JSON-LD, meta tags.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Set

from bs4 import BeautifulSoup, Comment, Tag

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """Result of DOM cleaning with metrics for research evaluation."""
    cleaned_html: str = ""
    cleaned_text: str = ""
    original_size: int = 0
    cleaned_size: int = 0
    text_size: int = 0
    reduction_ratio: float = 0.0
    elements_removed: int = 0
    elements_kept: int = 0
    json_ld_data: List[str] = None  # Preserved JSON-LD blocks
    meta_info: dict = None  # Preserved meta tags

    def __post_init__(self):
        if self.json_ld_data is None:
            self.json_ld_data = []
        if self.meta_info is None:
            self.meta_info = {}


# Tags to completely remove
REMOVE_TAGS: Set[str] = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "video", "audio", "source", "track", "map", "area",
    "object", "embed", "applet", "param",
}

# Semantic tags to remove (navigation, headers, footers)
REMOVE_SEMANTIC: Set[str] = {
    "nav", "footer", "header", "aside",
}

# Class/id patterns indicating irrelevant content
IRRELEVANT_PATTERNS: List[str] = [
    r"cookie",
    r"consent",
    r"gdpr",
    r"banner",
    r"popup",
    r"modal",
    r"overlay",
    r"newsletter",
    r"subscribe",
    r"social[-_]?share",
    r"social[-_]?media",
    r"share[-_]?button",
    r"advertisement",
    r"advert",
    r"ad[-_]?container",
    r"ad[-_]?wrapper",
    r"ad[-_]?slot",
    r"sidebar[-_]?ad",
    r"sponsored",
    r"promo[-_]?banner",
    r"comment[-_]?section",
    r"comments",
    r"disqus",
    r"breadcrumb",
    r"pagination",
    r"page[-_]?nav",
    r"skip[-_]?link",
    r"sr[-_]?only",
    r"visually[-_]?hidden",
    r"footer[-_]?links",
    r"back[-_]?to[-_]?top",
    r"related[-_]?articles",
    r"recommended",
    r"trending",
    r"follow[-_]?us",
]

# Compiled pattern for matching class/id attributes
_irrelevant_re = re.compile(
    "|".join(IRRELEVANT_PATTERNS), re.IGNORECASE
)

# Tags whose content is valuable for extraction
PRESERVE_TAGS: Set[str] = {
    "main", "article", "section", "table", "thead", "tbody", "tr", "td", "th",
    "ul", "ol", "li", "dl", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "span", "a", "strong", "em", "b", "i",
    "blockquote", "pre", "code",
    "figure", "figcaption", "img",
    "time", "address", "mark", "data",
    "div",  # kept if it passes relevance check
}


class DOMCleaner:
    """
    Advanced DOM cleaning pipeline.

    Steps:
    1. Extract and preserve JSON-LD and meta information
    2. Remove script/style/iframe/svg elements
    3. Remove semantic nav/footer/header/aside
    4. Remove elements matching irrelevant class/id patterns
    5. Remove HTML comments
    6. Remove empty elements
    7. Extract clean text
    """

    def __init__(self, preserve_links: bool = False, preserve_images: bool = False):
        self.preserve_links = preserve_links
        self.preserve_images = preserve_images

    def clean(self, html: str) -> CleaningResult:
        """
        Clean HTML content, removing irrelevant elements.

        Args:
            html: Raw HTML string.

        Returns:
            CleaningResult with cleaned HTML, text, and metrics.
        """
        result = CleaningResult()
        result.original_size = len(html.encode("utf-8", errors="ignore"))

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            logger.error(f"HTML parse failed: {e}")
            result.cleaned_text = html[:10000]  # Fallback
            return result

        removed = 0

        # Step 1: Extract JSON-LD before removing scripts
        result.json_ld_data = self._extract_json_ld(soup)
        result.meta_info = self._extract_meta(soup)

        # Step 2: Remove unwanted tags
        for tag_name in REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
                removed += 1

        # Step 3: Remove semantic navigation/footer/header
        for tag_name in REMOVE_SEMANTIC:
            for tag in soup.find_all(tag_name):
                tag.decompose()
                removed += 1

        # Step 4: Remove elements with irrelevant class/id
        removed += self._remove_by_pattern(soup)

        # Step 5: Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
            removed += 1

        # Step 6: Remove empty elements (but not self-closing ones)
        removed += self._remove_empty(soup)

        # Step 7: Focus on body content
        body = soup.find("body")
        if body:
            soup = body

        # Get cleaned HTML and text
        result.cleaned_html = str(soup)
        result.cleaned_size = len(result.cleaned_html.encode("utf-8", errors="ignore"))

        # Clean text extraction
        result.cleaned_text = self._extract_text(soup)
        result.text_size = len(result.cleaned_text.encode("utf-8", errors="ignore"))

        result.elements_removed = removed
        result.elements_kept = len(soup.find_all(True))
        result.reduction_ratio = (
            1.0 - (result.cleaned_size / result.original_size)
            if result.original_size > 0 else 0.0
        )

        logger.info(
            f"DOM cleaning: {result.original_size:,} → {result.cleaned_size:,} bytes "
            f"({result.reduction_ratio:.1%} reduction), "
            f"{removed} elements removed, {result.elements_kept} kept"
        )

        return result

    def _remove_by_pattern(self, soup: BeautifulSoup) -> int:
        """Remove elements whose class or id match irrelevant patterns."""
        removed = 0
        for tag in soup.find_all(True):
            classes = " ".join(tag.get("class", []))
            tag_id = tag.get("id", "")
            combined = f"{classes} {tag_id}"

            if _irrelevant_re.search(combined):
                tag.decompose()
                removed += 1
        return removed

    @staticmethod
    def _remove_empty(soup: BeautifulSoup) -> int:
        """Remove elements with no meaningful content."""
        removed = 0
        # Multiple passes to catch nested empties
        for _ in range(3):
            for tag in soup.find_all(True):
                if tag.name in ("br", "hr", "img", "input", "meta", "link"):
                    continue
                if not tag.get_text(strip=True) and not tag.find("img"):
                    tag.decompose()
                    removed += 1
        return removed

    @staticmethod
    def _extract_json_ld(soup: BeautifulSoup) -> List[str]:
        """Extract JSON-LD blocks before script removal."""
        results = []
        for script in soup.find_all("script", type="application/ld+json"):
            text = script.string or script.get_text()
            if text and text.strip():
                results.append(text.strip())
        return results

    @staticmethod
    def _extract_meta(soup: BeautifulSoup) -> dict:
        """Extract useful meta tag information."""
        meta = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name", "") or tag.get("property", "")
            content = tag.get("content", "")
            if name and content:
                meta[name] = content
        return meta

    @staticmethod
    def _extract_text(soup) -> str:
        """Extract clean text content with structural preservation."""
        lines = []
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th",
             "span", "a", "blockquote", "pre", "div", "dt", "dd"]
        ):
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 1:
                # Prefix headings for structure
                if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    level = int(element.name[1])
                    text = f"{'#' * level} {text}"
                lines.append(text)

        # Deduplicate consecutive identical lines
        deduped = []
        for line in lines:
            if not deduped or line != deduped[-1]:
                deduped.append(line)

        return "\n".join(deduped)
