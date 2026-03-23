"""Integration tests -- require ANTHROPIC_API_KEY or OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them when no key is available.
"""

import os

import pytest
from app.feedback import get_feedback, reset_for_tests
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
    reason="No API key set -- skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _assert_common_response(result) -> None:
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_spanish_error():
    """Spanish: mixed verb forms should be corrected to a past-tense form."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_common_response(result)
    assert "fui" in result.corrected_sentence.lower()


@pytest.mark.asyncio
async def test_correct_german():
    """German: a correct sentence should return no errors and stay unchanged."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    _assert_common_response(result)
    assert result.corrected_sentence == "Ich habe gestern einen interessanten Film gesehen."


@pytest.mark.asyncio
async def test_french_gender_errors():
    """French: article-noun gender mismatches should be detected."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_common_response(result)
    assert "le chat" in result.corrected_sentence.lower()
    assert "la table" in result.corrected_sentence.lower()
    assert any(error.error_type == "gender_agreement" for error in result.errors)


@pytest.mark.asyncio
async def test_japanese_particle():
    """Japanese: incorrect location particle should be corrected."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    _assert_common_response(result)
    assert "東京に住んでいます" in result.corrected_sentence
    assert any(e.original == "を" and e.correction == "に" for e in result.errors)


@pytest.mark.asyncio
async def test_portuguese_spelling_and_preposition():
    """Portuguese: spelling and verb-preposition patterns should be handled."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_common_response(result)
    assert "presente" in result.corrected_sentence.lower()
    assert "do" in result.corrected_sentence.lower()
    assert any(error.error_type in {"spelling", "grammar"} for error in result.errors)


@pytest.mark.asyncio
async def test_korean_word_order():
    """Korean: time expression and verb order should be improved."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="저는 갔어요 어제 학교에.",
            target_language="Korean",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_common_response(result)
    assert result.corrected_sentence != "저는 갔어요 어제 학교에."
    assert "어제" in result.corrected_sentence


@pytest.mark.asyncio
async def test_arabic_conjugation():
    """Arabic: verb form should agree with the subject."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="أنا ذهب إلى المدرسة أمس.",
            target_language="Arabic",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_common_response(result)
    assert result.corrected_sentence != "أنا ذهب إلى المدرسة أمس."
    assert any(error.error_type == "conjugation" for error in result.errors)


@pytest.mark.asyncio
async def test_complex_french_subjunctive_and_agreement():
    """French: a complex sentence should still surface multiple advanced issues."""
    reset_for_tests()
    result = await get_feedback(
        FeedbackRequest(
            sentence="Bien qu'il soit probable que les études linguistiques permettront d'élucider les phénomènes pragmatiques, il n'est pas certain que les méthodologies actuelles seraient suffisant pour résoudre les ambiguïtés.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_common_response(result)
    assert result.difficulty == "C1"
    assert "ambigu" in result.corrected_sentence.lower()
    assert any(
        token in result.corrected_sentence.lower()
        for token in {"suffisantes", "soient"}
    )