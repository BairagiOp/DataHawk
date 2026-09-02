"""
DataHawk — Field-Level Confidence Scoring

Provides a heuristic confidence estimate (0.0–1.0) for each extracted
field based on multiple signals.  Clearly documented as an estimate,
not a calibrated probability.

Signals used:
1. Schema/type validity (from validation)
2. LLM extraction status (FOUND/UNCERTAIN/NOT_FOUND)
3. Source snippet quality
4. Validation pass/fail
5. Consistency with schema constraints
6. Multi-chunk agreement (if applicable)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.extractor import ExtractedField, ExtractionResult, FieldStatus
from core.schema_generator import ExtractionSchema
from core.validator import ValidationResult, FieldValidation

logger = logging.getLogger(__name__)


@dataclass
class FieldConfidence:
    """Confidence breakdown for a single field."""
    field_name: str
    confidence: float = 0.0
    components: Dict[str, float] = None  # Individual signal contributions

    def __post_init__(self):
        if self.components is None:
            self.components = {}

    def to_dict(self) -> Dict:
        return {
            "field_name": self.field_name,
            "confidence": round(self.confidence, 3),
            "components": {k: round(v, 3) for k, v in self.components.items()},
        }


@dataclass
class ConfidenceResult:
    """Confidence scores for all fields in an extraction."""
    field_scores: Dict[str, FieldConfidence] = None
    overall_confidence: float = 0.0

    def __post_init__(self):
        if self.field_scores is None:
            self.field_scores = {}

    def to_dict(self) -> Dict:
        return {
            "overall_confidence": round(self.overall_confidence, 3),
            "fields": {k: v.to_dict() for k, v in self.field_scores.items()},
        }


class ConfidenceScorer:
    """
    Heuristic confidence estimator for extraction results.

    NOTE: Confidence scores are heuristic estimates based on observable
    signals. They are NOT calibrated probabilities and should not be
    interpreted as guaranteed accuracy measures.

    Weighting:
      status_weight    = 0.30  (LLM said FOUND/UNCERTAIN/NOT_FOUND)
      validation_weight = 0.25  (passed type/semantic validation)
      snippet_weight   = 0.20  (source snippet quality)
      type_weight      = 0.15  (type validity)
      constraint_weight = 0.10  (constraint satisfaction)
    """

    def __init__(
        self,
        w_status: float = 0.30,
        w_validation: float = 0.25,
        w_snippet: float = 0.20,
        w_type: float = 0.15,
        w_constraint: float = 0.10,
    ):
        self.w_status = w_status
        self.w_validation = w_validation
        self.w_snippet = w_snippet
        self.w_type = w_type
        self.w_constraint = w_constraint

    def score(
        self,
        extraction: ExtractionResult,
        schema: ExtractionSchema,
        validation: Optional[ValidationResult] = None,
    ) -> ConfidenceResult:
        """
        Compute confidence scores for all fields in the extraction.

        Args:
            extraction: The extraction result.
            schema: The extraction schema.
            validation: Optional validation result for additional signals.

        Returns:
            ConfidenceResult with per-field and overall scores.
        """
        result = ConfidenceResult()

        if not extraction.records:
            result.overall_confidence = 0.0
            return result

        all_scores = []

        for record in extraction.records:
            for field_name, field_schema in schema.fields.items():
                extracted = record.get(field_name)
                if extracted is None:
                    fc = FieldConfidence(
                        field_name=field_name,
                        confidence=0.0,
                        components={"status": 0.0},
                    )
                    result.field_scores[field_name] = fc
                    all_scores.append(0.0)
                    continue

                # Get validation result for this field
                fv = (
                    validation.field_results.get(field_name)
                    if validation else None
                )

                fc = self._score_field(extracted, field_schema, fv)
                result.field_scores[field_name] = fc
                all_scores.append(fc.confidence)

        result.overall_confidence = (
            sum(all_scores) / len(all_scores) if all_scores else 0.0
        )

        logger.info(
            f"Confidence scoring: overall={result.overall_confidence:.2f}, "
            f"fields={len(result.field_scores)}"
        )
        return result

    def _score_field(
        self,
        field: ExtractedField,
        schema,
        validation: Optional[FieldValidation] = None,
    ) -> FieldConfidence:
        """Compute confidence for a single field."""
        components = {}

        # 1. Status signal
        status_score = {
            FieldStatus.FOUND: 1.0,
            FieldStatus.UNCERTAIN: 0.5,
            FieldStatus.NOT_FOUND: 0.0,
        }.get(field.status, 0.0)
        components["status"] = status_score

        # 2. Validation signal
        if validation:
            val_score = 1.0 if validation.valid else 0.3
            if validation.errors:
                val_score = 0.1
            elif validation.warnings:
                val_score = 0.6
            components["validation"] = val_score
        else:
            components["validation"] = 0.5  # Unknown

        # 3. Source snippet quality
        snippet = field.source_snippet or ""
        snippet_score = 0.0
        if snippet:
            # Longer, more specific snippets → higher confidence
            if len(snippet) > 50:
                snippet_score = 0.9
            elif len(snippet) > 20:
                snippet_score = 0.7
            elif len(snippet) > 5:
                snippet_score = 0.5
            else:
                snippet_score = 0.3
        components["snippet"] = snippet_score

        # 4. Type validity
        type_score = 1.0
        if validation and not validation.type_valid:
            type_score = 0.2
        elif field.value is None:
            type_score = 0.0
        components["type"] = type_score

        # 5. Constraint satisfaction
        constraint_score = 1.0
        if validation:
            if validation.semantic_valid:
                constraint_score = 1.0
            else:
                constraint_score = 0.3
        components["constraint"] = constraint_score

        # Weighted combination
        confidence = (
            self.w_status * components["status"]
            + self.w_validation * components["validation"]
            + self.w_snippet * components["snippet"]
            + self.w_type * components["type"]
            + self.w_constraint * components["constraint"]
        )
        confidence = max(0.0, min(1.0, confidence))

        return FieldConfidence(
            field_name=field.name,
            confidence=confidence,
            components=components,
        )

    @staticmethod
    def aggregate_multi_extraction(
        confidence_results: List[ConfidenceResult],
    ) -> ConfidenceResult:
        """
        Aggregate confidence from multiple extraction attempts.
        Higher agreement → higher confidence.
        """
        if not confidence_results:
            return ConfidenceResult()

        if len(confidence_results) == 1:
            return confidence_results[0]

        merged = ConfidenceResult()
        field_scores_list: Dict[str, List[float]] = {}

        for cr in confidence_results:
            for name, fc in cr.field_scores.items():
                if name not in field_scores_list:
                    field_scores_list[name] = []
                field_scores_list[name].append(fc.confidence)

        for name, scores in field_scores_list.items():
            avg = sum(scores) / len(scores)
            # Boost if consistent, penalize if inconsistent
            spread = max(scores) - min(scores)
            agreement_bonus = 0.1 if spread < 0.2 else -0.1
            final = max(0.0, min(1.0, avg + agreement_bonus))

            merged.field_scores[name] = FieldConfidence(
                field_name=name,
                confidence=final,
                components={"average": avg, "agreement_bonus": agreement_bonus},
            )

        all_scores = [fc.confidence for fc in merged.field_scores.values()]
        merged.overall_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return merged
