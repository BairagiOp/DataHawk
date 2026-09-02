"""
DataHawk — Natural-Language to Extraction Schema Generator

Converts a free-text extraction requirement (e.g. "Extract product name,
price, rating, availability and product URL") into a typed JSON schema
that drives the entire extraction + validation pipeline.

Supported field types: string, number, date, url, boolean, categorical,
list, object.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.llm_client import GeminiClient, LLMResponse

logger = logging.getLogger(__name__)

# ── Data Structures ────────────────────────────────────────────────

@dataclass
class FieldSchema:
    """Schema definition for a single extraction field."""
    name: str
    type: str  # string | number | date | url | boolean | categorical | list | object
    required: bool = True
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"min": 0, "max": 5} for rating, {"enum": [...]} for categorical

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type, "required": self.required}
        if self.description:
            result["description"] = self.description
        if self.constraints:
            result["constraints"] = self.constraints
        return result


@dataclass
class ExtractionSchema:
    """Complete extraction schema generated from natural language."""
    fields: Dict[str, FieldSchema] = field(default_factory=dict)
    description: str = ""
    raw_requirement: str = ""

    @property
    def required_fields(self) -> List[str]:
        return [name for name, f in self.fields.items() if f.required]

    @property
    def field_names(self) -> List[str]:
        return list(self.fields.keys())

    @property
    def field_types(self) -> Dict[str, str]:
        return {name: f.type for name, f in self.fields.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
            "required": self.required_fields,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_requirement: str = "") -> "ExtractionSchema":
        """Construct schema from a dictionary (e.g. parsed LLM output)."""
        fields = {}
        fields_data = data.get("fields", {})
        for name, fdef in fields_data.items():
            if isinstance(fdef, str):
                # Simple format: {"product_name": "string"}
                fields[name] = FieldSchema(name=name, type=fdef)
            elif isinstance(fdef, dict):
                fields[name] = FieldSchema(
                    name=name,
                    type=fdef.get("type", "string"),
                    required=fdef.get("required", True),
                    description=fdef.get("description", ""),
                    constraints=fdef.get("constraints", {}),
                )
        required = data.get("required", [])
        for name in required:
            if name in fields:
                fields[name].required = True

        return cls(
            fields=fields,
            description=data.get("description", ""),
            raw_requirement=raw_requirement,
        )


# ── Schema Generator ──────────────────────────────────────────────

SCHEMA_GENERATION_PROMPT = """You are a data extraction schema designer.

Given a natural-language extraction requirement, produce a JSON schema that
specifies exactly what fields to extract, their data types, whether they
are required, and any constraints.

SUPPORTED TYPES:
- "string"      — free text
- "number"      — integer or decimal
- "date"        — a date or datetime
- "url"         — a valid URL
- "boolean"     — true / false
- "categorical" — one of a fixed set (provide "enum" in constraints)
- "list"        — a list of values (provide "item_type" in constraints)
- "object"      — a nested object (provide "properties" in constraints)

RULES:
1. Normalise field names to snake_case.
2. Infer types from context (e.g. "price" → number, "rating" → number,
   "product URL" → url, "available" → boolean or categorical).
3. Mark all explicitly mentioned fields as required unless context
   suggests otherwise.
4. Add sensible constraints where obvious (e.g. rating 0-5, price >= 0).
5. Include a short "description" for each field.

OUTPUT FORMAT (strict JSON, no markdown fences):
{
  "description": "<one-line summary of the extraction task>",
  "fields": {
    "<field_name>": {
      "type": "<type>",
      "required": true|false,
      "description": "<what this field contains>",
      "constraints": { ... }
    }
  },
  "required": ["<field_name>", ...]
}

USER REQUIREMENT:
"""


class SchemaGenerator:
    """Converts natural-language extraction requirements into typed schemas."""

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def generate(self, requirement: str) -> tuple[ExtractionSchema, LLMResponse]:
        """
        Generate an ExtractionSchema from a natural-language requirement.

        Args:
            requirement: Free-text description of what to extract.

        Returns:
            Tuple of (ExtractionSchema, LLMResponse with token/timing info).
        """
        logger.info(f"Generating schema for: {requirement[:80]}...")

        parsed, llm_response = self.llm.generate_json(
            prompt=SCHEMA_GENERATION_PROMPT + requirement,
            system_instruction="You are a precise JSON schema generator. Output ONLY valid JSON.",
        )

        if parsed is None:
            logger.warning("LLM returned invalid JSON; attempting fallback parse.")
            schema = self._fallback_parse(requirement)
            return schema, llm_response

        try:
            schema = ExtractionSchema.from_dict(parsed, raw_requirement=requirement)
            logger.info(f"Schema generated: {len(schema.fields)} fields — {schema.field_names}")
            return schema, llm_response
        except Exception as e:
            logger.error(f"Schema construction failed: {e}")
            schema = self._fallback_parse(requirement)
            return schema, llm_response

    def _fallback_parse(self, requirement: str) -> ExtractionSchema:
        """
        Heuristic fallback: extract field names from the requirement text
        and assign default types.
        """
        import re

        # Common type indicators
        type_hints = {
            "price": "number",
            "cost": "number",
            "rating": "number",
            "score": "number",
            "count": "number",
            "quantity": "number",
            "number": "number",
            "amount": "number",
            "url": "url",
            "link": "url",
            "href": "url",
            "date": "date",
            "time": "date",
            "published": "date",
            "available": "categorical",
            "availability": "categorical",
            "in_stock": "boolean",
            "active": "boolean",
        }

        # Extract potential field names: words between commas/and
        text = requirement.lower()
        text = re.sub(r"\b(extract|get|find|list|show|give|fetch|scrape)\b", "", text)
        text = re.sub(r"\b(all|the|each|every|and|or|with|from|of|for|in|a|an)\b", " ", text)

        # Split by commas and "and"
        parts = re.split(r"[,\n]|\band\b", text)
        fields = {}
        for part in parts:
            part = part.strip().strip(".")
            if not part:
                continue
            name = "_".join(part.split())
            name = re.sub(r"[^a-z0-9_]", "", name)
            if name and len(name) > 1:
                # Guess type
                ftype = "string"
                for hint_key, hint_type in type_hints.items():
                    if hint_key in name:
                        ftype = hint_type
                        break
                fields[name] = FieldSchema(
                    name=name, type=ftype, required=True
                )

        return ExtractionSchema(
            fields=fields,
            description=f"Auto-parsed: {requirement[:60]}",
            raw_requirement=requirement,
        )
