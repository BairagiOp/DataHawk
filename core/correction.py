"""
DataHawk — Self-Correction Loop

Major research component: automatically detects extraction failures,
modifies the extraction prompt/context, and re-extracts to improve
results.

Pipeline:
  Extract → Validate → Valid? → YES → Done
                                 NO  → Identify failure
                                       → Build targeted prompt
                                       → Re-extract
                                       → Re-validate
                                       → Repeat (max N attempts)

Logs all correction metrics for experimental evaluation.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.extractor import LLMExtractor, ExtractionResult, FieldStatus
from core.llm_client import GeminiClient, LLMResponse
from core.schema_generator import ExtractionSchema
from core.validator import ValidationEngine, ValidationResult
from core.confidence import ConfidenceScorer, ConfidenceResult
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CorrectionAttempt:
    """Record of a single correction attempt."""
    attempt_number: int
    failure_type: str  # "missing_fields", "type_errors", "low_confidence"
    failed_fields: List[str]
    extraction_result: Optional[ExtractionResult] = None
    validation_result: Optional[ValidationResult] = None
    confidence_result: Optional[ConfidenceResult] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    improved: bool = False


@dataclass
class CorrectionResult:
    """Complete result of the self-correction pipeline."""
    # Initial results
    initial_extraction: Optional[ExtractionResult] = None
    initial_validation: Optional[ValidationResult] = None
    initial_confidence: Optional[ConfidenceResult] = None

    # Final results (after correction)
    final_extraction: Optional[ExtractionResult] = None
    final_validation: Optional[ValidationResult] = None
    final_confidence: Optional[ConfidenceResult] = None

    # Correction history
    attempts: List[CorrectionAttempt] = field(default_factory=list)
    total_attempts: int = 0
    correction_successful: bool = False
    total_additional_tokens: int = 0
    total_additional_latency_ms: float = 0.0

    @property
    def improved(self) -> bool:
        """Whether correction improved over initial results."""
        if not self.initial_confidence or not self.final_confidence:
            return False
        return (
            self.final_confidence.overall_confidence
            > self.initial_confidence.overall_confidence
        )

    @property
    def improvement_delta(self) -> float:
        """Confidence improvement from correction."""
        if not self.initial_confidence or not self.final_confidence:
            return 0.0
        return (
            self.final_confidence.overall_confidence
            - self.initial_confidence.overall_confidence
        )


CORRECTION_PROMPT_TEMPLATE = """PREVIOUS EXTRACTION FAILED VALIDATION.

EXTRACTION SCHEMA:
{schema_json}

VALIDATION ERRORS:
{validation_errors}

MISSING FIELDS: {missing_fields}
INVALID FIELDS: {invalid_fields}

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
Re-extract the data, specifically focusing on the fields that were missing
or invalid. Pay special attention to:
{focus_instructions}

CRITICAL RULES:
1. Extract ONLY from the provided content — do NOT invent data.
2. If a field truly does not exist in the content, report status as "NOT_FOUND".
3. Return valid JSON with the same structure as before.

Return JSON:
{{
  "records": [
    {{
      "<field_name>": {{
        "value": <value_or_null>,
        "status": "FOUND" | "NOT_FOUND" | "UNCERTAIN",
        "source_snippet": "<text from content>"
      }}
    }}
  ]
}}

OUTPUT (valid JSON only):"""


class SelfCorrectionLoop:
    """
    Research-oriented self-correction loop for extraction reliability.

    Iteratively re-extracts data focusing on failed fields, using
    targeted prompts and expanded context.
    """

    def __init__(
        self,
        extractor: Optional[LLMExtractor] = None,
        validator: Optional[ValidationEngine] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        llm_client: Optional[GeminiClient] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_client or GeminiClient()
        self.extractor = extractor or LLMExtractor(self.llm)
        self.validator = validator or ValidationEngine()
        self.scorer = confidence_scorer or ConfidenceScorer()
        self.max_attempts = self.settings.max_correction_attempts

    def extract_with_correction(
        self,
        content: str,
        schema: ExtractionSchema,
        initial_extraction: Optional[ExtractionResult] = None,
    ) -> CorrectionResult:
        """
        Full extraction pipeline with self-correction.

        Args:
            content: Cleaned webpage content.
            schema: Extraction schema.
            initial_extraction: Optional pre-computed extraction.

        Returns:
            CorrectionResult with full correction history.
        """
        result = CorrectionResult()

        # Step 1: Initial extraction
        if initial_extraction is None:
            initial_extraction = self.extractor.extract(content, schema)

        result.initial_extraction = initial_extraction

        if not initial_extraction.success:
            result.final_extraction = initial_extraction
            return result

        # Step 2: Validate
        validation = self.validator.validate(initial_extraction, schema)
        result.initial_validation = validation

        # Step 3: Score confidence
        confidence = self.scorer.score(initial_extraction, schema, validation)
        result.initial_confidence = confidence

        # Step 4: Check if correction needed
        current_extraction = initial_extraction
        current_validation = validation
        current_confidence = confidence

        if validation.valid and confidence.overall_confidence >= 0.7:
            # Good enough — no correction needed
            result.final_extraction = current_extraction
            result.final_validation = current_validation
            result.final_confidence = current_confidence
            result.correction_successful = True
            logger.info(
                "Extraction valid with high confidence; no correction needed."
            )
            return result

        # Step 5: Correction loop
        for attempt_num in range(1, self.max_attempts + 1):
            logger.info(
                f"Correction attempt {attempt_num}/{self.max_attempts}: "
                f"missing={current_validation.missing_fields}, "
                f"invalid={len(current_validation.invalid_fields)}"
            )

            attempt = self._correction_attempt(
                content, schema, current_extraction,
                current_validation, attempt_num,
            )
            result.attempts.append(attempt)
            result.total_attempts = attempt_num
            result.total_additional_tokens += attempt.input_tokens + attempt.output_tokens
            result.total_additional_latency_ms += attempt.latency_ms

            if attempt.extraction_result and attempt.extraction_result.success:
                current_extraction = attempt.extraction_result
                current_validation = attempt.validation_result
                current_confidence = attempt.confidence_result

                if current_validation and current_validation.valid:
                    logger.info(
                        f"Correction successful after {attempt_num} attempt(s)"
                    )
                    result.correction_successful = True
                    break

        result.final_extraction = current_extraction
        result.final_validation = current_validation
        result.final_confidence = current_confidence

        return result

    def _correction_attempt(
        self,
        content: str,
        schema: ExtractionSchema,
        prev_extraction: ExtractionResult,
        prev_validation: ValidationResult,
        attempt_number: int,
    ) -> CorrectionAttempt:
        """Execute a single correction attempt."""
        attempt = CorrectionAttempt(
            attempt_number=attempt_number,
            failure_type=self._classify_failure(prev_validation),
            failed_fields=(
                prev_validation.missing_fields
                + [f["field"] for f in prev_validation.invalid_fields]
            ),
        )

        # Build focused correction prompt
        focus_instructions = self._build_focus_instructions(
            prev_validation, schema
        )

        prompt = CORRECTION_PROMPT_TEMPLATE.format(
            schema_json=schema.to_json(),
            validation_errors=json.dumps(prev_validation.to_dict(), indent=2),
            missing_fields=", ".join(prev_validation.missing_fields) or "none",
            invalid_fields=json.dumps(prev_validation.invalid_fields) or "none",
            content=content[:12000],
            focus_instructions=focus_instructions,
        )

        start = time.time()
        parsed, llm_response = self.llm.generate_json(
            prompt=prompt,
            system_instruction=(
                "You are a precise data extraction correction engine. "
                "Fix previous extraction errors. Output ONLY valid JSON."
            ),
        )
        attempt.latency_ms = (time.time() - start) * 1000
        attempt.input_tokens = llm_response.input_tokens
        attempt.output_tokens = llm_response.output_tokens

        if parsed is None:
            attempt.improved = False
            return attempt

        # Build extraction result from correction
        from core.extractor import ExtractedField
        correction_extraction = ExtractionResult(success=True)
        correction_extraction.raw_json = parsed
        correction_extraction.llm_response = llm_response

        raw_records = parsed.get("records", [])
        if isinstance(raw_records, dict):
            raw_records = [raw_records]

        for raw_record in raw_records:
            record = {}
            for field_name in schema.field_names:
                field_data = raw_record.get(field_name, {})
                if isinstance(field_data, dict):
                    record[field_name] = ExtractedField(
                        name=field_name,
                        value=field_data.get("value"),
                        status=FieldStatus(
                            field_data.get("status", "NOT_FOUND").upper()
                        ),
                        source_snippet=str(
                            field_data.get("source_snippet", "")
                        )[:200],
                    )
                else:
                    record[field_name] = ExtractedField(
                        name=field_name,
                        value=field_data,
                        status=(
                            FieldStatus.FOUND
                            if field_data is not None
                            else FieldStatus.NOT_FOUND
                        ),
                    )
            if record:
                correction_extraction.records.append(record)

        # Merge with previous: keep better fields
        merged = self._merge_extractions(
            prev_extraction, correction_extraction, schema
        )
        attempt.extraction_result = merged

        # Validate merged result
        validation = self.validator.validate(merged, schema)
        attempt.validation_result = validation

        confidence = self.scorer.score(merged, schema, validation)
        attempt.confidence_result = confidence

        attempt.improved = (
            validation.completeness_score
            > prev_validation.completeness_score
            or validation.validity_score > prev_validation.validity_score
        )

        return attempt

    @staticmethod
    def _merge_extractions(
        prev: ExtractionResult,
        correction: ExtractionResult,
        schema: ExtractionSchema,
    ) -> ExtractionResult:
        """Merge previous and correction results, keeping best fields."""
        merged = ExtractionResult(success=True)

        # Align records by index
        max_records = max(len(prev.records), len(correction.records))
        for i in range(max_records):
            prev_record = prev.records[i] if i < len(prev.records) else {}
            corr_record = (
                correction.records[i] if i < len(correction.records) else {}
            )

            merged_record = {}
            for field_name in schema.field_names:
                prev_field = prev_record.get(field_name)
                corr_field = corr_record.get(field_name)

                # Pick the better field
                if corr_field and corr_field.status == FieldStatus.FOUND:
                    merged_record[field_name] = corr_field
                elif prev_field and prev_field.status == FieldStatus.FOUND:
                    merged_record[field_name] = prev_field
                elif corr_field:
                    merged_record[field_name] = corr_field
                elif prev_field:
                    merged_record[field_name] = prev_field

            merged.records.append(merged_record)

        return merged

    @staticmethod
    def _classify_failure(validation: ValidationResult) -> str:
        """Classify the type of extraction failure."""
        if validation.missing_fields:
            return "missing_fields"
        if validation.invalid_fields:
            return "type_errors"
        if validation.completeness_score < 0.5:
            return "incomplete"
        return "low_quality"

    @staticmethod
    def _build_focus_instructions(
        validation: ValidationResult, schema: ExtractionSchema
    ) -> str:
        """Build specific instructions for the correction prompt."""
        instructions = []

        for field_name in validation.missing_fields:
            field_schema = schema.fields.get(field_name)
            if field_schema:
                instructions.append(
                    f"- Field '{field_name}' (type: {field_schema.type}) was MISSING. "
                    f"Look carefully for this information."
                )

        for inv in validation.invalid_fields:
            instructions.append(
                f"- Field '{inv['field']}' had errors: {inv.get('reason', 'unknown')}. "
                f"Re-extract with correct format."
            )

        return "\n".join(instructions) if instructions else "Re-check all fields carefully."
