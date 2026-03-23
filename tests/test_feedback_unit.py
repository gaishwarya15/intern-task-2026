"""Unit tests -- run without an API key using mocked LLM responses.

Covers multiple languages (Latin and non-Latin scripts), error types, and edge cases.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback, reset_for_tests
from app.models import FeedbackRequest


def _mock_openai_completion(response_data: dict) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _mock_anthropic_message(response_data: dict) -> MagicMock:
    """Build a mock Anthropic Message response with a tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = response_data
    message = MagicMock()
    message.content = [block]
    message.stop_reason = "end_turn"
    return message


# ─────────────────────────────────────────────────────────────────────────────
# SPANISH 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    """Spanish: verb conjugation error."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms here. 'Soy' is the present tense of 'ser' (to be), and 'fue' is the third-person past tense of 'ir' (to go). Since you're talking about going to the market yesterday, you only need 'fui' — the first-person preterite of 'ir' meaning 'I went'.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Yo soy fue al mercado ayer.",
                target_language="Spanish",
                native_language="English",
            )
        )

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_spanish_missing_word():
    """Spanish: missing preposition."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Ella va a la tienda.",
        "is_correct": False,
        "errors": [
            {
                "original": "va tienda",
                "correction": "va a la tienda",
                "error_type": "missing_word",
                "explanation": "In Spanish, the verb 'ir' (to go) requires the preposition 'a' before the destination. You also need the definite article 'la' before 'tienda' (store), since you're referring to a specific place. So the correct phrase is 'va a la tienda' (she goes to the store)."
            }
        ],
        "difficulty": "A1"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Ella va tienda.",
                target_language="Spanish",
                native_language="English",
            )
        )

    assert result.is_correct is False
    assert result.errors[0].error_type == "missing_word"


@pytest.mark.asyncio
async def test_spanish_word_order_error():
    """Spanish: adjective placed incorrectly."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Quiero un café con leche caliente.",
        "is_correct": False,
        "errors": [
            {
                "original": "caliente leche",
                "correction": "leche caliente",
                "error_type": "word_order",
                "explanation": "In Spanish, adjectives typically come after the noun they describe, not before it. So 'hot milk' is 'leche caliente' (noun first, then adjective), unlike in English where the adjective comes first."
            }
        ],
        "difficulty": "A1"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Quiero un café con caliente leche.",
                target_language="Spanish",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "word_order"


# ─────────────────────────────────────────────────────────────────────────────
# GERMAN 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_german_correct_sentence():
    """German: sentence with no errors."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Ich habe gestern einen interessanten Film gesehen.",
                target_language="German",
                native_language="English",
            )
        )

    assert result.is_correct is True
    assert result.errors == []


@pytest.mark.asyncio
async def test_german_extra_word():
    """German: redundant pronoun."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Ich gehe morgen ins Kino.",
        "is_correct": False,
        "errors": [
            {
                "original": "gehe ich",
                "correction": "gehe",
                "error_type": "extra_word",
                "explanation": "You wrote 'ich' twice in the same sentence. In German, the subject pronoun should only appear once. Remove the second 'ich' after the verb to make the sentence correct."
            }
        ],
        "difficulty": "A1"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Ich gehe ich morgen ins Kino.",
                target_language="German",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "extra_word"


# ─────────────────────────────────────────────────────────────────────────────
# FRENCH 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_french_multiple_gender_errors():
    """French: multiple gender agreement errors."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine.",
            },
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="La chat noir est sur le table.",
                target_language="French",
                native_language="English",
            )
        )

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# PORTUGUESE 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_portuguese_spelling_error():
    """Portuguese: spelling mistake."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Eu quero comprar um presente.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "'Gift/present' in Portuguese is spelled 'presente' with an 's', not a 'z'. The 'z' sound in this word is actually represented by the letter 's' in Portuguese spelling.",
            }
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Eu quero comprar um prezente.",
                target_language="Portuguese",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "spelling"


# ─────────────────────────────────────────────────────────────────────────────
# ITALIAN 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_italian_word_choice():
    """Italian: wrong word choice."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Vado al cinema ogni giorno.",
        "is_correct": False,
        "errors": [
            {
                "original": "at il",
                "correction": "al",
                "error_type": "grammar",
                "explanation": "In Italian, you don't use the English preposition 'at' — instead, you use 'a' (to/at). Also, when 'a' is followed by the masculine article 'il', they must contract into a single word: 'al'. So 'a + il = al cinema' is the correct form."
            }
        ],
        "difficulty": "A1"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Vado at il cinema ogni giorno.",
                target_language="Italian",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "grammar"


# ─────────────────────────────────────────────────────────────────────────────
# JAPANESE 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_japanese_particle_error():
    """Japanese: particle error (non-Latin script)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "The verb 住む (to live/reside) requires the particle に to mark the location of residence, not を. The particle を typically marks the direct object of an action or movement through a place, while に marks a static location where something exists or where you live. Think of に as 'at/in' for places you inhabit."
            }
        ],
        "difficulty": "A2"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="私は東京を住んでいます。",
                target_language="Japanese",
                native_language="English",
            )
        )

    assert result.is_correct is False
    assert result.errors[0].error_type == "grammar"
    assert "に" in result.corrected_sentence


@pytest.mark.asyncio
async def test_japanese_correct_sentence():
    """Japanese: correct sentence (non-Latin script)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "私は毎日学校に行きます。",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="私は毎日学校に行きます。",
                target_language="Japanese",
                native_language="English",
            )
        )

    assert result.is_correct is True
    assert result.errors == []


# ─────────────────────────────────────────────────────────────────────────────
# KOREAN 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_korean_word_order():
    """Korean: word order error (non-Latin script)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "저는 어제 학교에 갔어요.",
        "is_correct": False,
        "errors": [
            {
                "original": "갔어요 어제 학교에",
                "correction": "어제 학교에 갔어요",
                "error_type": "word_order",
                "explanation": "In Korean, the standard sentence order is Subject → Time → Place → Verb. The verb always comes at the end of the sentence. You placed the verb '갔어요' (went) before the time word '어제' (yesterday) and the location '학교에' (to school), which is incorrect. The correct order is: 저는 (subject) → 어제 (time) → 학교에 (place) → 갔어요 (verb)."
            }
        ],
        "difficulty": "A2"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="저는 갔어요 어제 학교에.",
                target_language="Korean",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "word_order"


# ─────────────────────────────────────────────────────────────────────────────
# RUSSIAN 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_russian_correct_sentence():
    """Russian: correct sentence (Cyrillic script)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Я каждый день читаю книги.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A2",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Я каждый день читаю книги.",
                target_language="Russian",
                native_language="English",
            )
        )

    assert result.is_correct is True


# ─────────────────────────────────────────────────────────────────────────────
# CHINESE 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chinese_word_choice():
    """Chinese: word choice error (logographic script)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "我喝咖啡。",
        "is_correct": False,
        "errors": [
            {
                "original": "吃",
                "correction": "喝",
                "error_type": "word_choice",
                "explanation": "Use '喝' (to drink) for beverages, not '吃' (to eat).",
            }
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="我吃咖啡。",
                target_language="Chinese",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "word_choice"


# ─────────────────────────────────────────────────────────────────────────────
# ARABIC
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_arabic_conjugation():
    """Arabic: verb conjugation error (RTL script)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "أنا ذهبتُ إلى المدرسة أمس.",
        "is_correct": False,
        "errors": [
            {
                "original": "ذهب",
                "correction": "ذهبتُ",
                "error_type": "conjugation",
                "explanation": "The verb 'ذهب' is in the third-person masculine past tense (he went), but your subject is 'أنا' (I). In Arabic, verbs must agree with their subject in person and number. For first-person singular past tense, you need 'ذهبتُ' (I went)."
            }
        ],
        "difficulty": "A2"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="أنا ذهب إلى المدرسة أمس.",
                target_language="Arabic",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "conjugation"


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES & SPECIAL SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_is_correct_guard():
    """Guard: if is_correct=true but errors exist, flip to false."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "She goes to school.",
        "is_correct": True,  # Model says correct
        "errors": [
            {
                "original": "go",
                "correction": "goes",
                "error_type": "conjugation",
                "explanation": "En inglés, cuando el sujeto es una tercera persona del singular (he, she, it), el verbo en presente simple debe llevar una '-s' al final. Por eso, 'go' se convierte en 'goes' cuando el sujeto es 'she'.",
            }
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="She go to school.",
                target_language="English",
                native_language="Spanish",
            )
        )

    assert result.is_correct is False
    assert len(result.errors) == 1


@pytest.mark.asyncio
async def test_punctuation_error():
    """Punctuation: missing or incorrect punctuation."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Hola, ¿cómo estás?",
        "is_correct": False,
        "errors": [
            {
                "original": "estás",
                "correction": "estás?",
                "error_type": "punctuation",
                "explanation": "In Spanish, questions must end with a closing question mark '?'. You already used the opening '¿' at the beginning, so don't forget to close it with '?' at the end of the question."
            }
        ],
        "difficulty": "A1"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Hola, ¿cómo estás",
                target_language="Spanish",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == "punctuation"


@pytest.mark.asyncio
async def test_anthropic_path():
    """Anthropic provider path (mocked)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Je suis content.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=True), \
         patch("app.feedback.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_message(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Je suis content.",
                target_language="French",
                native_language="English",
            )
        )

    assert result.is_correct is True
    assert result.errors == []


@pytest.mark.asyncio
async def test_anthropic_fallback_to_openai():
    """Anthropic fails, falls back to OpenAI."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Yo fui al mercado.",
        "is_correct": False,
        "errors": [
            {
                "original": "fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "The verb 'ir' (to go) must agree with the subject 'Yo' (I). 'Fue' is the third-person singular preterite form (he/she/it went), but for 'Yo' you need 'fui' (I went). Always match the verb ending to the subject pronoun.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback._use_anthropic", return_value=True), \
         patch("app.feedback.AsyncAnthropic") as MockAnthropic, \
         patch("app.feedback.AsyncOpenAI") as MockOpenAI:
        anthropic_instance = MockAnthropic.return_value
        anthropic_instance.messages.create = AsyncMock(side_effect=Exception("400 Bad Request"))
        openai_instance = MockOpenAI.return_value
        openai_instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Yo fue al mercado.",
                target_language="Spanish",
                native_language="English",
            )
        )

    assert result.is_correct is False
    assert result.errors[0].error_type == "conjugation"


@pytest.mark.asyncio
async def test_is_correct_guard_sentence_changed():
    """Guard: model says is_correct=True but corrected_sentence differs from input."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Elle est allée au marché.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A2",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Elle est allé au marché.",
                target_language="French",
                native_language="English",
            )
        )

    assert result.is_correct is False


@pytest.mark.asyncio
async def test_empty_llm_response():
    """Empty LLM response raises ValueError (caught as 502 by main.py)."""
    reset_for_tests()

    choice = MagicMock()
    choice.message.content = None
    choice.finish_reason = "length"
    completion = MagicMock()
    completion.choices = [choice]

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=completion)

        with pytest.raises(ValueError, match="Empty response"):
            await get_feedback(
                FeedbackRequest(
                    sentence="Hola mundo.",
                    target_language="Spanish",
                    native_language="English",
                )
            )


@pytest.mark.asyncio
async def test_cache_hit():
    """Second identical request returns cached result without calling LLM again."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Je suis content.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        create_mock = AsyncMock(return_value=_mock_openai_completion(mock_response))
        instance.chat.completions.create = create_mock

        request = FeedbackRequest(
            sentence="Je suis content.",
            target_language="French",
            native_language="English",
        )

        result1 = await get_feedback(request)
        result2 = await get_feedback(request)

    assert result1.corrected_sentence == result2.corrected_sentence
    assert create_mock.call_count == 1


@pytest.mark.asyncio
async def test_multiple_different_error_types():
    """Multiple different error types in one sentence (spelling + grammar)."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Eu quero comprar um presente para minha irmã, mas não sei do que ela gosta.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "'Gift' in Portuguese is spelled 'presente' with an 's', not a 'z'. The 'z' sound in this word is actually represented by the letter 's' in Portuguese."
            },
            {
                "original": "o que ela gosta",
                "correction": "do que ela gosta",
                "error_type": "grammar",
                "explanation": "The verb 'gostar' (to like) always requires the preposition 'de'. When 'de' combines with the article 'o', it becomes 'do', so 'what she likes' must be 'do que ela gosta' (de + o que), not just 'o que ela gosta'."
            }
        ],
        "difficulty": "B1"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
                target_language="Portuguese",
                native_language="English",
            )
        )

    assert result.is_correct is False
    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert "spelling" in error_types
    assert "grammar" in error_types


@pytest.mark.asyncio
async def test_complex_c1_sentence():
    """C1-level complex sentence with nested clauses and advanced vocabulary."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "Bien qu'il soit probable que les études linguistiques permettront d'élucider les phénomènes pragmatiques, il n'est pas certain que les méthodologies actuelles soient suffisantes pour résoudre les ambiguïtés sémantiques.",
        "is_correct": False,
        "errors": [
            {
                "original": "seraient suffisant",
                "correction": "soient suffisantes",
                "error_type": "conjugation",
                "explanation": "After expressions of doubt or uncertainty like 'il n'est pas certain que', French requires the subjunctive mood, not the conditional. So 'seraient' (conditional) must be replaced with 'soient' (present subjunctive of 'être')."
            },
            {
                "original": "suffisant",
                "correction": "suffisantes",
                "error_type": "gender_agreement",
                "explanation": "The adjective 'suffisant' must agree in gender and number with the noun it describes. Since 'méthodologies' is feminine plural, the correct form is 'suffisantes', not 'suffisant'."
            }
        ],
        "difficulty": "C1"
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="Bien qu'il soit probable que les études linguistiques permettront d'élucider les phénomènes pragmatiques, il n'est pas certain que les méthodologies actuelles seraient suffisant pour résoudre les ambiguïtés sémantiques.",
                target_language="French",
                native_language="English",
            )
        )

    assert result.difficulty == "C1"
    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert "conjugation" in error_types
    assert "gender_agreement" in error_types


@pytest.mark.asyncio
async def test_non_english_native_language():
    """Non-English native language: explanation should be in Spanish."""
    reset_for_tests()
    mock_response = {
        "corrected_sentence": "I went to the store yesterday.",
        "is_correct": False,
        "errors": [
            {
                "original": "goed",
                "correction": "went",
                "error_type": "conjugation",
                "explanation": "El verbo 'go' (ir) es irregular en inglés. No se forma el pasado simple añadiendo '-ed'. La forma correcta del pasado simple de 'go' es 'went'. Los verbos irregulares tienen formas únicas que debes memorizar.",
            }
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback._use_anthropic", return_value=False), \
         patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_openai_completion(mock_response)
        )
        result = await get_feedback(
            FeedbackRequest(
                sentence="I goed to the store yesterday.",
                target_language="English",
                native_language="Spanish",
            )
        )

    assert result.is_correct is False
    assert result.errors[0].error_type == "conjugation"
    assert "inglés" in result.errors[0].explanation.lower()
    assert "went" in result.errors[0].explanation.lower()


@pytest.mark.asyncio
async def test_health_returns_200():
    """Health endpoint returns 200 OK."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
