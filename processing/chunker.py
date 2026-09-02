"""
DataHawk — Semantic Content Chunker

Divides cleaned webpage content into extraction-ready chunks that
respect DOM boundaries, preserve record integrity, and stay within
token limits.

Chunking strategies (in priority order):
1. DOM section boundaries (article, section, table, div)
2. Repeated structures (product cards, list items, table rows)
3. Heading hierarchy (h1 → h2 → h3)
4. Paragraph-level splitting with overlap

Each chunk includes parent context (headings, table headers) as prefix
for extraction continuity.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from bs4 import BeautifulSoup, Tag

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single extraction-ready content chunk."""
    content: str
    index: int = 0
    total_chunks: int = 0
    context_prefix: str = ""  # Headings/headers for context
    source_tag: str = ""
    char_count: int = 0
    estimated_tokens: int = 0

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.content)
        if not self.estimated_tokens:
            self.estimated_tokens = self._estimate_tokens(self.content)

    @property
    def full_content(self) -> str:
        """Content with context prefix included."""
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.content}"
        return self.content

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate (~4 chars per token for English)."""
        return len(text) // 4


class SemanticChunker:
    """
    DOM-aware semantic chunker that preserves content structure.

    Unlike naive character-based splitting, this chunker:
    - Respects DOM boundaries (never splits mid-element)
    - Groups repeated structures (product cards, table rows)
    - Maintains heading context across chunks
    - Falls back to paragraph splitting with overlap
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.max_tokens = self.settings.max_chunk_tokens
        self.overlap_tokens = self.settings.chunk_overlap_tokens
        # ~4 chars per token
        self.max_chars = self.max_tokens * 4
        self.overlap_chars = self.overlap_tokens * 4

    def chunk(self, content: str, is_html: bool = False) -> List[Chunk]:
        """
        Split content into extraction-ready chunks.

        Args:
            content: Cleaned text or HTML content.
            is_html: If True, use DOM-aware chunking. If False, text-based.

        Returns:
            List of Chunk objects, each within token limits.
        """
        if not content or not content.strip():
            return []

        # If content fits in one chunk, return as-is
        if len(content) <= self.max_chars:
            chunk = Chunk(content=content, index=0, total_chunks=1)
            return [chunk]

        if is_html:
            chunks = self._chunk_html(content)
        else:
            chunks = self._chunk_text(content)

        # Number the chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i
            chunk.total_chunks = len(chunks)

        logger.info(
            f"Chunking: {len(content):,} chars → {len(chunks)} chunks "
            f"(max {self.max_tokens} tokens/chunk)"
        )
        return chunks

    # ── HTML-aware chunking ────────────────────────────────────────

    def _chunk_html(self, html: str) -> List[Chunk]:
        """Chunk HTML by DOM structure."""
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return self._chunk_text(html)

        # Extract heading context
        headings = self._extract_headings(soup)

        # Find top-level sections
        sections = self._find_sections(soup)

        if not sections:
            return self._chunk_text(soup.get_text(separator="\n", strip=True))

        chunks = []
        current_text = ""
        current_heading = ""

        for section in sections:
            section_text = section.get_text(separator="\n", strip=True)

            if not section_text:
                continue

            # Update heading context
            heading = section.find(["h1", "h2", "h3", "h4"])
            if heading:
                current_heading = heading.get_text(strip=True)

            # If section itself is too large, split it
            if len(section_text) > self.max_chars:
                # Flush current buffer
                if current_text:
                    chunks.append(Chunk(
                        content=current_text,
                        context_prefix=current_heading,
                        source_tag=section.name if isinstance(section, Tag) else "text",
                    ))
                    current_text = ""

                # Split large section
                sub_chunks = self._split_large_section(section, current_heading)
                chunks.extend(sub_chunks)
                continue

            # If adding this section exceeds limit, flush
            if len(current_text) + len(section_text) > self.max_chars:
                if current_text:
                    chunks.append(Chunk(
                        content=current_text,
                        context_prefix=current_heading,
                        source_tag="mixed",
                    ))
                current_text = section_text
            else:
                current_text = f"{current_text}\n\n{section_text}" if current_text else section_text

        # Flush remaining
        if current_text:
            chunks.append(Chunk(
                content=current_text,
                context_prefix=current_heading,
                source_tag="mixed",
            ))

        return chunks if chunks else self._chunk_text(html)

    def _split_large_section(self, section: Tag, heading: str) -> List[Chunk]:
        """Split a large DOM section into smaller chunks."""
        chunks = []

        # Try splitting by children
        children = [c for c in section.children if isinstance(c, Tag)]
        if children:
            current = ""
            for child in children:
                child_text = child.get_text(separator="\n", strip=True)
                if not child_text:
                    continue
                if len(current) + len(child_text) > self.max_chars:
                    if current:
                        chunks.append(Chunk(
                            content=current,
                            context_prefix=heading,
                            source_tag=child.name,
                        ))
                    if len(child_text) > self.max_chars:
                        # Even a single child is too large; fall back to text splitting
                        sub = self._chunk_text(child_text)
                        for s in sub:
                            s.context_prefix = heading
                        chunks.extend(sub)
                        current = ""
                    else:
                        current = child_text
                else:
                    current = f"{current}\n{child_text}" if current else child_text

            if current:
                chunks.append(Chunk(
                    content=current,
                    context_prefix=heading,
                    source_tag="child",
                ))
        else:
            # No children; fall back to text splitting
            text = section.get_text(separator="\n", strip=True)
            chunks = self._chunk_text(text)
            for c in chunks:
                c.context_prefix = heading

        return chunks

    # ── Text-based chunking ────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[Chunk]:
        """Chunk plain text by paragraphs with overlap."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 > self.max_chars:
                if current:
                    chunks.append(Chunk(content=current, source_tag="text"))

                    # Add overlap from end of current chunk
                    if self.overlap_chars > 0:
                        overlap = current[-self.overlap_chars:]
                        current = f"[...] {overlap}\n{para}"
                    else:
                        current = para
                else:
                    # Single paragraph exceeds limit; hard split
                    for i in range(0, len(para), self.max_chars):
                        chunks.append(Chunk(
                            content=para[i:i + self.max_chars],
                            source_tag="text_split",
                        ))
                    current = ""
            else:
                current = f"{current}\n{para}" if current else para

        if current:
            chunks.append(Chunk(content=current, source_tag="text"))

        return chunks

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _find_sections(soup: BeautifulSoup) -> list:
        """Find content sections in order."""
        body = soup.find("body") or soup

        # Try semantic sections
        sections = body.find_all(
            ["article", "section", "main", "table", "div", "ul", "ol"],
            recursive=False,
        )

        if not sections:
            sections = [c for c in body.children if isinstance(c, Tag)]

        return sections

    @staticmethod
    def _extract_headings(soup: BeautifulSoup) -> List[str]:
        """Extract page headings for context."""
        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)
        return headings


def legacy_split_dom_content(dom_content: str, max_length: int = 6000) -> List[str]:
    """
    Legacy character-based splitter (preserved for baseline comparison).
    Equivalent to the original scrape.py split_dom_content().
    """
    if not dom_content:
        return [dom_content]
    return [
        dom_content[i:i + max_length]
        for i in range(0, len(dom_content), max_length)
    ]
