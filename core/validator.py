"""
DataHawk — Multi-Layer Validation Engine

Independently validates extracted data through four layers:
1. Type validation (price → numeric, URL → valid URL, etc.)
2. Semantic validation (rating in range, price positive, etc.)
3. Completeness validation (all required fields present)
4. Consistency validation (cross-field checks)

Returns detailed ValidationResult with per-field diagnostics.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from core.extractor import ExtractedField, ExtractionResult, FieldStatus
from core.schema_generator import ExtractionSchema, FieldSchema

logger = logging.getLogger(__name__)


@dataclass
class FieldValidation:
    """Validation result for a single field."""
    field_name: str
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    type_valid: bool = True
    semantic_valid: bool = True


@dataclass
class ValidationResult:
    """Complete validation result for an extraction."""
    valid: bool = True
    field_results: Dict[str, FieldValidation] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    invalid_fields: List[Dict[str, str]] = field(default_factory=list)
    completeness_score: float = 1.0
    validity_score: float = 1.0
    total_errors: int = 0
    total_warnings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "missing_fields": self.missing_fields,
            "invalid_fields": self.invalid_fields,
            "completeness_score": round(self.completeness_score, 3),
            "validity_score": round(self.validity_score, 3),
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
        }


class ValidationEngine:
    """
    Multi-layer validation engine for extracted data.

    Validates extraction results against the schema through
    type checking, semantic rules, completeness, and consistency.
    """

    def validate(
        self,
        extraction: ExtractionResult,
        schema: ExtractionSchema,
    ) -> ValidationResult:
        """
        Validate all records in an extraction result.

        Args:
            extraction: The extraction result to validate.
            schema: The schema defining expected fields and types.

        Returns:
            ValidationResult with per-field diagnostics.
        """
        result = ValidationResult()

        if not extraction.records:
            result.valid = False
            result.missing_fields = schema.required_fields
            result.completeness_score = 0.0
            result.validity_score = 0.0
            return result

        all_field_results = {}
        all_missing = set()
        all_invalid = []

        for record_idx, record in enumerate(extraction.records):
            for field_name, field_schema in schema.fields.items():
                extracted = record.get(field_name)

                fv = FieldValidation(field_name=field_name)

                if extracted is None or extracted.status == FieldStatus.NOT_FOUND:
                    if field_schema.required:
                        fv.valid = False
                        fv.errors.append(f"Required field missing")
                        all_missing.add(field_name)
                    continue

                # Layer 1: Type validation
                self._validate_type(extracted, field_schema, fv)

                # Layer 2: Semantic validation
                self._validate_semantics(extracted, field_schema, fv)

                if not fv.valid:
                    all_invalid.append({
                        "field": field_name,
                        "record": record_idx,
                        "reason": "; ".join(fv.errors),
                    })

                all_field_results[field_name] = fv

        # Layer 3: Completeness
        result.missing_fields = list(all_missing)
        required_count = len(schema.required_fields)
        found_required = required_count - len(all_missing)
        result.completeness_score = (
            found_required / required_count if required_count > 0 else 1.0
        )

        # Layer 4: Consistency (cross-field)
        for record in extraction.records:
            self._validate_consistency(record, schema, all_field_results)

        # Aggregate
        result.field_results = all_field_results
        result.invalid_fields = all_invalid
        result.total_errors = sum(
            len(fv.errors) for fv in all_field_results.values()
        )
        result.total_warnings = sum(
            len(fv.warnings) for fv in all_field_results.values()
        )

        total_fields = len(all_field_results)
        valid_fields = sum(1 for fv in all_field_results.values() if fv.valid)
        result.validity_score = (
            valid_fields / total_fields if total_fields > 0 else 1.0
        )

        result.valid = (
            result.completeness_score >= 0.8
            and result.validity_score >= 0.8
            and len(all_invalid) == 0
        )

        logger.info(
            f"Validation: valid={result.valid}, "
            f"completeness={result.completeness_score:.2f}, "
            f"validity={result.validity_score:.2f}, "
            f"errors={result.total_errors}, warnings={result.total_warnings}"
        )

        return result

    def validate_record(
        self,
        record: Dict[str, ExtractedField],
        schema: ExtractionSchema,
    ) -> ValidationResult:
        """Validate a single record (convenience wrapper)."""
        extraction = ExtractionResult(records=[record], success=True)
        return self.validate(extraction, schema)

    # ── Type Validation ────────────────────────────────────────────

    def _validate_type(
        self, field: ExtractedField, schema: FieldSchema, result: FieldValidation
    ) -> None:
        """Validate field value against its declared type."""
        value = field.value
        if value is None:
            return

        ftype = schema.type.lower()

        if ftype == "number":
            if not self._is_numeric(value):
                result.type_valid = False
                result.valid = False
                result.errors.append(
                    f"Expected number, got '{type(value).__name__}': {str(value)[:50]}"
                )

        elif ftype == "url":
            if not self._is_valid_url(str(value)):
                result.type_valid = False
                result.valid = False
                result.errors.append(f"Invalid URL: {str(value)[:100]}")

        elif ftype == "date":
            if not self._is_valid_date(str(value)):
                result.type_valid = False
                result.warnings.append(f"Could not parse as date: {str(value)[:50]}")

        elif ftype == "boolean":
            if not isinstance(value, bool) and str(value).lower() not in (
                "true", "false", "yes", "no", "1", "0"
            ):
                result.type_valid = False
                result.warnings.append(f"Not a clear boolean: {value}")

        elif ftype == "categorical":
            enum_values = schema.constraints.get("enum", [])
            if enum_values and str(value) not in enum_values:
                result.warnings.append(
                    f"Value '{value}' not in expected categories: {enum_values}"
                )

        elif ftype == "list":
            if not isinstance(value, list):
                result.warnings.append(f"Expected list, got {type(value).__name__}")

    # ── Semantic Validation ────────────────────────────────────────

    def _validate_semantics(
        self, field: ExtractedField, schema: FieldSchema, result: FieldValidation
    ) -> None:
        """Validate field value against semantic constraints."""
        value = field.value
        if value is None:
            return

        constraints = schema.constraints

        # Min/Max for numbers
        if schema.type == "number" and self._is_numeric(value):
            num_val = self._to_number(value)
            if num_val is not None:
                min_val = constraints.get("min")
                max_val = constraints.get("max")
                if min_val is not None and num_val < min_val:
                    result.semantic_valid = False
                    result.valid = False
                    result.errors.append(
                        f"Value {num_val} below minimum {min_val}"
                    )
                if max_val is not None and num_val > max_val:
                    result.semantic_valid = False
                    result.valid = False
                    result.errors.append(
                        f"Value {num_val} above maximum {max_val}"
                    )

        # Common field-name semantic checks
        field_lower = field.name.lower()
        if "price" in field_lower and self._is_numeric(value):
            num_val = self._to_number(value)
            if num_val is not None and num_val < 0:
                result.semantic_valid = False
                result.valid = False
                result.errors.append("Price should not be negative")

        if "rating" in field_lower and self._is_numeric(value):
            num_val = self._to_number(value)
            if num_val is not None and (num_val < 0 or num_val > 10):
                result.warnings.append(f"Unusual rating value: {num_val}")

    # ── Consistency Validation ─────────────────────────────────────

    def _validate_consistency(
        self,
        record: Dict[str, ExtractedField],
        schema: ExtractionSchema,
        field_results: Dict[str, FieldValidation],
    ) -> None:
        """Cross-field consistency checks."""
        # Example: if both price and discounted_price exist,
        # discounted should be less
        price = record.get("price")
        discount = record.get("discounted_price") or record.get("sale_price")

        if (
            price and discount
            and price.status == FieldStatus.FOUND
            and discount.status == FieldStatus.FOUND
        ):
            p = self._to_number(price.value)
            d = self._to_number(discount.value)
            if p is not None and d is not None and d > p:
                fv = field_results.get("discounted_price") or field_results.get("sale_price")
                if fv:
                    fv.warnings.append(
                        f"Discounted price ({d}) exceeds original price ({p})"
                    )

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Check if value is or can be converted to a number."""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            cleaned = re.sub(r"[₹$€£¥,\s]", "", value)
            try:
                float(cleaned)
                return True
            except (ValueError, TypeError):
                return False
        return False

    @staticmethod
    def _to_number(value: Any) -> Optional[float]:
        """Convert value to float, handling currency symbols."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[₹$€£¥,\s]", "", value)
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _is_valid_url(value: str) -> bool:
        """Check if value is a valid URL."""
        try:
            parsed = urlparse(value)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    @staticmethod
    def _is_valid_date(value: str) -> bool:
        """Try common date formats."""
        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S", "%B %d, %Y",
            "%b %d, %Y", "%d %B %Y", "%d %b %Y",
        ]
        for fmt in formats:
            try:
                datetime.strptime(value.strip(), fmt)
                return True
            except ValueError:
                continue
        return False
