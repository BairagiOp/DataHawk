"""
DataHawk — Section Relevance Scoring

Assigns a 0–1 relevance score to each DOM section based on text density,
keyword overlap with the extraction schema, structural role, and tag
semantics.  Sections below a configurable threshold are filtered out
before LLM extraction to reduce token waste.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from bs4 import BeautifulSoup, Tag

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ScoredSection:
    """A DOM section with its relevance score and metadata."""
    html: str
    text: str
    score: float
    tag_name: str = ""
    section_id: str = ""
    word_count: int = 0
    keyword_matches: int = 0

    @property
    def is_relevant(self) -> bool:
        settings = get_settings()
        return self.score >= settings.min_relevance_score


# High-value tags that indicate content relevance
HIGH_VALUE_TAGS: Set[str] = {
    "article", "main", "table", "tbody", "thead",
    "h1", "h2", "h3", "h4",
    "p", "blockquote", "pre",
    "ul", "ol", "dl",
    "figure", "figcaption",
}

# Tags that typically contain data-rich content
DATA_TAGS: Set[str] = {
    "table", "tr", "td", "th", "ul", "ol", "li", "dl", "dt", "dd",
}

# Tags usually containing boilerplate
LOW_VALUE_TAGS: Set[str] = {
    "nav", "footer", "aside", "header",
}


class RelevanceScorer:
    """
    Scores DOM sections for extraction relevance.

    Scoring formula (0–1):
      score = w1 * text_density + w2 * tag_value + w3 * keyword_overlap
              + w4 * position_bonus

    Weights are configurable for experimentation.
    """

    def __init__(
        self,
        w_text_density: float = 0.3,
        w_tag_value: float = 0.25,
        w_keyword_overlap: float = 0.35,
        w_position: float = 0.1,
    ):
        self.w_text = w_text_density
        self.w_tag = w_tag_value
        self.w_keyword = w_keyword_overlap
        self.w_position = w_position

    def score_sections(
        self,
        html: str,
        keywords: Optional[List[str]] = None,
    ) -> List[ScoredSection]:
        """
        Split HTML into sections and score each for relevance.

        Args:
            html: Cleaned HTML content.
            keywords: Extraction-related keywords (field names, types, etc.).

        Returns:
            List of ScoredSection, sorted by score descending.
        """
        keywords = keywords or []
        keyword_set = {kw.lower() for kw in keywords}

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return [ScoredSection(html=html, text=html[:5000], score=0.5)]

        # Find top-level content sections
        sections = self._identify_sections(soup)

        scored = []
        total_sections = len(sections)

        for idx, section in enumerate(sections):
            text = section.get_text(separator=" ", strip=True)
            word_count = len(text.split())

            # Score components
            text_density = self._text_density_score(section, text)
            tag_value = self._tag_value_score(section)
            keyword_score, keyword_matches = self._keyword_overlap_score(
                text, keyword_set
            )
            position_score = self._position_score(idx, total_sections)

            # Weighted combination
            score = (
                self.w_text * text_density
                + self.w_tag * tag_value
                + self.w_keyword * keyword_score
                + self.w_position * position_score
            )
            score = max(0.0, min(1.0, score))

            scored.append(ScoredSection(
                html=str(section),
                text=text,
                score=score,
                tag_name=section.name if isinstance(section, Tag) else "text",
                section_id=section.get("id", "") if isinstance(section, Tag) else "",
                word_count=word_count,
                keyword_matches=keyword_matches,
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def filter_relevant(
        self,
        html: str,
        keywords: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> tuple[str, List[ScoredSection]]:
        """
        Filter HTML to only relevant sections.

        Returns:
            Tuple of (filtered_text, all_scored_sections).
        """
        settings = get_settings()
        threshold = min_score or settings.min_relevance_score

        sections = self.score_sections(html, keywords)
        relevant = [s for s in sections if s.score >= threshold]

        if not relevant:
            # If nothing passes threshold, keep top 3 sections
            relevant = sections[:3] if sections else []
            logger.warning(
                f"No sections above threshold {threshold}; keeping top {len(relevant)}"
            )

        filtered_text = "\n\n".join(s.text for s in relevant if s.text)

        logger.info(
            f"Relevance filter: {len(sections)} sections → "
            f"{len(relevant)} relevant (threshold={threshold})"
        )

        return filtered_text, sections

    @staticmethod
    def _identify_sections(soup: BeautifulSoup) -> list:
        """Identify top-level content sections in the DOM."""
        # Try semantic sections first
        sections = soup.find_all(
            ["article", "main", "section", "table", "div"],
            recursive=False,
        )

        if not sections:
            body = soup.find("body")
            if body:
                sections = body.find_all(
                    ["article", "main", "section", "table", "div"],
                    recursive=False,
                )

        if not sections:
            # Fall back to all direct children
            body = soup.find("body") or soup
            sections = [c for c in body.children if isinstance(c, Tag)]

        # If still too few, use body itself
        if len(sections) <= 1:
            body = soup.find("body") or soup
            sections = body.find_all(
                ["div", "section", "article", "table", "main"],
            )

        return sections[:50]  # Cap to avoid excessive processing

    @staticmethod
    def _text_density_score(element: Tag, text: str) -> float:
        """Score based on text-to-HTML ratio."""
        html_len = len(str(element))
        text_len = len(text)
        if html_len == 0:
            return 0.0
        ratio = text_len / html_len
        # Normalize: typical content pages have 0.1–0.5 ratio
        return min(1.0, ratio * 2)

    @staticmethod
    def _tag_value_score(element: Tag) -> float:
        """Score based on the semantic value of the tag."""
        if not isinstance(element, Tag):
            return 0.3

        tag = element.name
        if tag in HIGH_VALUE_TAGS:
            return 1.0
        if tag in DATA_TAGS:
            return 0.8
        if tag in LOW_VALUE_TAGS:
            return 0.1
        if tag == "div":
            # Check if div contains high-value children
            for child_tag in HIGH_VALUE_TAGS:
                if element.find(child_tag):
                    return 0.7
            return 0.4
        return 0.3

    @staticmethod
    def _keyword_overlap_score(
        text: str, keywords: Set[str]
    ) -> tuple[float, int]:
        """Score based on how many extraction keywords appear in text."""
        if not keywords:
            return 0.5, 0  # Neutral if no keywords given

        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        score = min(1.0, matches / max(len(keywords), 1))
        return score, matches

    @staticmethod
    def _position_score(index: int, total: int) -> float:
        """Score based on position in page (main content is usually central)."""
        if total <= 1:
            return 0.5
        # Middle sections get highest score
        relative_pos = index / total
        if 0.1 <= relative_pos <= 0.7:
            return 0.8
        if relative_pos < 0.1:
            return 0.5  # Could be title/header area
        return 0.3  # Bottom = likely footer
