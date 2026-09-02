"""Tests for schema generator — NL to typed schema conversion."""

import pytest
from core.schema_generator import SchemaGenerator, ExtractionSchema, FieldSchema


class TestExtractionSchema:
    """Test ExtractionSchema data structure."""

    def test_from_dict_simple(self, sample_schema_dict):
        schema = ExtractionSchema.from_dict(sample_schema_dict)
        assert len(schema.fields) == 4
        assert "product_name" in schema.fields
        assert schema.fields["price"].type == "number"
        assert schema.fields["product_name"].required is True

    def test_from_dict_string_types(self):
        data = {"fields": {"name": "string", "price": "number"}}
        schema = ExtractionSchema.from_dict(data)
        assert schema.fields["name"].type == "string"
        assert schema.fields["price"].type == "number"

    def test_required_fields(self, sample_schema_dict):
        schema = ExtractionSchema.from_dict(sample_schema_dict)
        required = schema.required_fields
        assert "product_name" in required
        assert "price" in required
        assert "availability" in required

    def test_to_json(self, sample_schema_dict):
        schema = ExtractionSchema.from_dict(sample_schema_dict)
        json_str = schema.to_json()
        assert "product_name" in json_str
        assert "number" in json_str

    def test_field_names(self, sample_schema_dict):
        schema = ExtractionSchema.from_dict(sample_schema_dict)
        names = schema.field_names
        assert set(names) == {"product_name", "price", "rating", "availability"}

    def test_field_types(self, sample_schema_dict):
        schema = ExtractionSchema.from_dict(sample_schema_dict)
        types = schema.field_types
        assert types["price"] == "number"
        assert types["product_name"] == "string"


class TestSchemaGeneratorFallback:
    """Test the fallback (non-LLM) schema generation."""

    def test_fallback_basic(self):
        gen = SchemaGenerator.__new__(SchemaGenerator)
        schema = gen._fallback_parse("Extract product name, price, and rating")
        assert len(schema.fields) >= 2
        field_names = set(schema.field_names)
        # Should find at least price and rating as numbers
        has_price = any("price" in name for name in field_names)
        assert has_price

    def test_fallback_url_type(self):
        gen = SchemaGenerator.__new__(SchemaGenerator)
        schema = gen._fallback_parse("Get the product url and product name")
        has_url_type = any(
            f.type == "url" for f in schema.fields.values()
        )
        assert has_url_type

    def test_fallback_empty_input(self):
        gen = SchemaGenerator.__new__(SchemaGenerator)
        schema = gen._fallback_parse("")
        assert len(schema.fields) == 0


class TestFieldSchema:
    """Test FieldSchema data class."""

    def test_to_dict(self):
        field = FieldSchema(
            name="price", type="number", required=True,
            constraints={"min": 0}
        )
        d = field.to_dict()
        assert d["type"] == "number"
        assert d["required"] is True
        assert d["constraints"]["min"] == 0

    def test_to_dict_no_constraints(self):
        field = FieldSchema(name="name", type="string")
        d = field.to_dict()
        assert "constraints" not in d
