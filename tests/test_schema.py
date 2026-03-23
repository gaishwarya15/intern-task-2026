"""Schema validation tests -- verify models match JSON schemas."""

import json
from pathlib import Path

import jsonschema
import pytest
from app.models import FeedbackRequest, FeedbackResponse

SCHEMA_DIR = Path(__file__).parent.parent / "schema"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def load_examples() -> list[dict]:
    return json.loads((EXAMPLES_DIR / "sample_inputs.json").read_text())


class TestRequestSchema:
    def test_valid_request(self):
        schema = load_schema("request.schema.json")
        valid = {
            "sentence": "Hola mundo",
            "target_language": "Spanish",
            "native_language": "English",
        }
        jsonschema.validate(valid, schema)

    def test_missing_sentence_fails(self):
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"target_language": "Spanish", "native_language": "English"}, schema
            )

    def test_empty_sentence_fails(self):
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "sentence": "",
                    "target_language": "Spanish",
                    "native_language": "English",
                },
                schema,
            )


class TestResponseSchema:
    def test_correct_response(self):
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "Hola mundo",
            "is_correct": True,
            "errors": [],
            "difficulty": "A1",
        }
        jsonschema.validate(valid, schema)

    def test_response_with_errors(self):
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "Le chat noir",
            "is_correct": False,
            "errors": [
                {
                    "original": "La chat",
                    "correction": "Le chat",
                    "error_type": "gender_agreement",
                    "explanation": "Chat is masculine",
                }
            ],
            "difficulty": "A1",
        }
        jsonschema.validate(valid, schema)

    def test_invalid_difficulty_fails(self):
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": True,
                    "errors": [],
                    "difficulty": "Z9",
                },
                schema,
            )

    def test_invalid_error_type_fails(self):
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": False,
                    "errors": [
                        {
                            "original": "x",
                            "correction": "y",
                            "error_type": "not_a_real_type",
                            "explanation": "test",
                        }
                    ],
                    "difficulty": "A1",
                },
                schema,
            )


class TestRequestSchemaEdgeCases:
    def test_missing_target_language_fails(self):
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"sentence": "Hola", "native_language": "English"}, schema
            )

    def test_missing_native_language_fails(self):
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"sentence": "Hola", "target_language": "Spanish"}, schema
            )

    def test_extra_fields_rejected(self):
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "sentence": "Hola",
                    "target_language": "Spanish",
                    "native_language": "English",
                    "unexpected_field": "should fail",
                },
                schema,
            )


class TestResponseSchemaEdgeCases:
    def test_missing_errors_field_fails(self):
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": True,
                    "difficulty": "A1",
                },
                schema,
            )

    def test_missing_error_subfield_fails(self):
        """Error object missing 'explanation' should fail."""
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": False,
                    "errors": [
                        {
                            "original": "x",
                            "correction": "y",
                            "error_type": "grammar",
                        }
                    ],
                    "difficulty": "A1",
                },
                schema,
            )

    def test_extra_fields_in_response_rejected(self):
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": True,
                    "errors": [],
                    "difficulty": "A1",
                    "extra": "not allowed",
                },
                schema,
            )

    @pytest.mark.parametrize(
        "error_type",
        [
            "grammar", "spelling", "word_choice", "punctuation",
            "word_order", "missing_word", "extra_word", "conjugation",
            "gender_agreement", "number_agreement", "tone_register", "other",
        ],
    )
    def test_all_error_types_accepted(self, error_type):
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "test",
            "is_correct": False,
            "errors": [
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": error_type,
                    "explanation": "test",
                }
            ],
            "difficulty": "A1",
        }
        jsonschema.validate(valid, schema)

    @pytest.mark.parametrize("level", ["A1", "A2", "B1", "B2", "C1", "C2"])
    def test_all_difficulty_levels_accepted(self, level):
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "test",
            "is_correct": True,
            "errors": [],
            "difficulty": level,
        }
        jsonschema.validate(valid, schema)


class TestExamplesMatchSchemas:
    """Verify that all example inputs/outputs conform to the schemas."""

    def test_all_example_requests_valid(self):
        schema = load_schema("request.schema.json")
        for example in load_examples():
            jsonschema.validate(example["request"], schema)

    def test_all_example_responses_valid(self):
        schema = load_schema("response.schema.json")
        for example in load_examples():
            jsonschema.validate(example["expected_response"], schema)
