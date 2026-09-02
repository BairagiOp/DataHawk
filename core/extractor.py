"""
DataHawk — Schema-Guided LLM Extraction Engine

Extracts structured data from webpage content using Gemini, guided by
a typed extraction schema.  Forces JSON output, tracks field-level
extraction status (FOUND / NOT_FOUND / UNCERTAIN), and provides source
traceability for every extracted value.

This is the central extraction component of the research pipeline.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.llm_client import GeminiClient, LLMResponse
from core.schema_generator import ExtractionSchema

logger = logging.getLogger(__name__)


class FieldStatus(str, Enum):
    """Extraction status for each field."""
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ExtractedField:
    """A single extracted field with metadata."""
    name: str
    value: Any = None
    status: FieldStatus = FieldStatus.NOT_FOUND
    source_snippet: str = ""
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Result of extracting data from a single chunk or page."""
    records: List[Dict[str, ExtractedField]] = field(default_factory=list)
    raw_json: Optional[Dict] = None
    llm_response: Optional[LLMResponse] = None
    chunk_index: int = 0
    success: bool = False
    error: str = ""
    extraction_time_ms: float = 0.0

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def flat_records(self) -> List[Dict[str, Any]]:
        """Convert to simple dicts with just name→value pairs."""
        result = []
        for record in self.records:
            flat = {}
            for name, field_obj in record.items():
                flat[name] = field_obj.value
            result.append(flat)
        return result

    @property
    def field_statuses(self) -> Dict[str, FieldStatus]:
        """Get status of all fields across all records."""
        statuses = {}
        for record in self.records:
            for name, field_obj in record.items():
                if name not in statuses or field_obj.status == FieldStatus.FOUND:
                    statuses[name] = field_obj.status
        return statuses


# ── Extraction Prompt ──────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a precise structured data extraction engine.
Your ONLY job is to extract data from webpage content according to a given schema.

CRITICAL RULES:
1. Extract ONLY information explicitly present in the content.
2. NEVER invent, guess, or hallucinate data.
3. If a field value is not found in the content, set its status to "NOT_FOUND" and value to null.
4. If a field value is ambiguous or uncertain, set its status to "UNCERTAIN".
5. For each found value, include a short source_snippet showing the relevant text.
6. Return valid JSON only. No markdown, no explanations, no extra text.
"""

EXTRACTION_PROMPT_TEMPLATE = """EXTRACTION SCHEMA:
{schema_json}

WEBPAGE CONTENT:
{content}

INSTRUCTIONS:
Extract all records matching the schema from the content above.
Return a JSON object with this exact structure:

{{
  "records": [
    {{
      "<field_name>": {{
        "value": <extracted_value_or_null>,
        "status": "FOUND" | "NOT_FOUND" | "UNCERTAIN",
        "source_snippet": "<brief text from content supporting this value>"
      }}
    }}
  ]
}}

If the content contains MULTIPLE records (e.g. multiple products), extract ALL of them.
If only ONE record exists, return it as a single-element array.
If NO relevant data is found, return {{"records": []}}.

OUTPUT (valid JSON only):"""


class LLMExtractor:
    """
    Schema-guided LLM extraction engine.

    Sends cleaned content + schema to Gemini, parses structured JSON
    response, and returns typed ExtractionResult with per-field metadata.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def extract(
        self,
        content: str,
        schema: ExtractionSchema,
        chunk_index: int = 0,
    ) -> ExtractionResult:
        """
        Extract structured data from content using the given schema.

        Args:
            content: Cleaned webpage text/HTML.
            schema: The extraction schema defining fields and types.
            chunk_index: Index of this chunk (for multi-chunk extraction).

        Returns:
            ExtractionResult with records, metadata, and metrics.
        """
        result = ExtractionResult(chunk_index=chunk_index)

        if not content or not content.strip():
            result.error = "Empty content"
            return result

        # Build prompt
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            schema_json=schema.to_json(),
            content=content[:15000],  # Safety limit
        )

        start = time.time()
        parsed, llm_response = self.llm.generate_json(
            prompt=prompt,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
        )
        result.extraction_time_ms = (time.time() - start) * 1000
        result.llm_response = llm_response

        if parsed is None:
            result.error = f"LLM JSON parse failed: {llm_response.error}"
            logger.warning(f"Extraction failed for chunk {chunk_index}: {result.error}")
            return result

        result.raw_json = parsed

        # Parse records
        try:
            raw_records = parsed.get("records", [])
            if isinstance(raw_records, dict):
                raw_records = [raw_records]

            for raw_record in raw_records:
                record = self._parse_record(raw_record, schema)
                if record:
                    result.records.append(record)

            result.success = True
            logger.info(
                f"Extraction OK: chunk {chunk_index}, "
                f"{len(result.records)} records, "
                f"{llm_response.total_tokens} tokens, "
                f"{result.extraction_time_ms:.0f}ms"
            )

        except Exception as e:
            result.error = f"Record parsing error: {str(e)}"
            logger.error(f"Record parsing failed: {e}")

        return result

    def extract_multi_chunk(
        self,
        chunks: List[str],
        schema: ExtractionSchema,
    ) -> List[ExtractionResult]:
        """
        Extract from multiple content chunks and merge results.

        Args:
            chunks: List of content chunks.
            schema: The extraction schema.

        Returns:
            List of ExtractionResult (one per chunk).
        """
        results = []
        for i, chunk in enumerate(chunks):
            result = self.extract(chunk, schema, chunk_index=i)
            results.append(result)
        return results

    @staticmethod
    def merge_results(results: List[ExtractionResult]) -> ExtractionResult:
        """Merge extraction results from multiple chunks."""
        merged = ExtractionResult(success=True)
        total_tokens_in = 0
        total_tokens_out = 0
        total_time = 0.0

        for r in results:
            merged.records.extend(r.records)
            total_time += r.extraction_time_ms
            if r.llm_response:
                total_tokens_in += r.llm_response.input_tokens
                total_tokens_out += r.llm_response.output_tokens
            if not r.success:
                merged.success = False

        merged.extraction_time_ms = total_time
        merged.llm_response = LLMResponse(
            text="[merged]",
            input_tokens=total_tokens_in,
            output_tokens=total_tokens_out,
            latency_ms=total_time,
        )

        return merged

    @staticmethod
    def _parse_record(
        raw: Dict[str, Any], schema: ExtractionSchema
    ) -> Optional[Dict[str, ExtractedField]]:
        """Parse a single record from LLM output into typed fields."""
        if not raw or not isinstance(raw, dict):
            return None

        record = {}
        for field_name in schema.field_names:
            field_data = raw.get(field_name, {})

            if isinstance(field_data, dict):
                value = field_data.get("value")
                status_str = field_data.get("status", "NOT_FOUND").upper()
                snippet = field_data.get("source_snippet", "")
            else:
                # LLM returned simple value instead of structured
                value = field_data
                status_str = "FOUND" if field_data is not None else "NOT_FOUND"
                snippet = ""

            try:
                status = FieldStatus(status_str)
            except ValueError:
                status = FieldStatus.UNCERTAIN

            record[field_name] = ExtractedField(
                name=field_name,
                value=value,
                status=status,
                source_snippet=str(snippet)[:200],
            )

        return record
