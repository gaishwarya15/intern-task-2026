"""System prompt and LLM interaction for language feedback.

Tries Anthropic (Claude) first. Falls back to OpenAI (GPT) when
ANTHROPIC_API_KEY is not set.
"""

import os
import json
import hashlib
import logging

from pathlib import Path
from typing import Optional
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from app.models import FeedbackRequest, FeedbackResponse
from cachetools import TTLCache

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "response.schema.json"
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "examples" / "sample_inputs.json"
EXAMPLES_DATA = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

def reset_for_tests() -> None:
    """To clear cache."""
    _cache.clear()

def _cache_key(req: FeedbackRequest) -> str:
    raw = f"{req.sentence}\x00{req.target_language.lower()}\x00{req.native_language.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _use_anthropic() -> bool:
    """True when an Anthropic key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """\
You are an expert computational linguist and a language teaching assistant, fluent in every world language and writing system. A language learner has written a sentence. Your job is to identify errors, produce a minimally corrected version and return structured JSON feedback that helps the learner understand what went wrong.

INPUT:
- sentence: The learner's sentence in the target language.
- target_language: The language the learner is studying.
- native_language: The learner's own language (used ONLY for explanation field).

GOALS:
1. Decide whether the sentence is fully correct in the target language.
2. If not correct, phrase the minimally corrected sentence.
3. Identify the smallest useful set of learner-facing errors needed to transform the original sentence into corrected_sentence.
4. If multiple error_types in a phrase, then all the errors must be addressed in the errors array with its own error_type.
5. Error_type MUST be selected based on the linguistic rules of the target_language.
6. Explain each error in the learner's native language.
7. Assign CEFR based on the ORIGINAL sentence's intended linguistic complexity, not on correctness.

RULES FOR OUTPUT FIELDS:
1. corrected_sentence: 
   - Must be the final corrected sentence in the target language. 
   - Make minimal corrections needed to preserve the learner's meaning, style and tone as much as possible. 
   - If the sentence is already correct, return it unchanged.
2. is_correct: 
   - true only if there are zero errors of any kind, false otherwise. 
   - CONSTRAINTS: if is_correct is true, the errors array must be empty array [].
3. errors: one object per distinct error. 
   - IMPORTANT: MUST address all the errors in the sentence. If multiple error_types in a phrase, then all the errors must be addressed in the errors array with its own error_type.
   - original: should be the shortest relevant exact substring from the learner sentence. Make sure it is same. The exact verbatim span that is wrong can be one word, a phrase, or a particle.
   - correction: should be the replacement text for that specificsubstring.
   - error_type: MUST be one of these 12 strings: grammar | spelling | word_choice | punctuation |  word_order | missing_word | extra_word | conjugation | gender_agreement | number_agreement | tone_register | other.
                 MUST Understand Morphology, Derivatives, Syntax, Semantics, Vocabulary, Pragmatics, and other linguistic features of the sentence in the target_language to determine the error type.
   - explanation: 1-2 short, concise, friendly, educational sentences written in the given learner's native language ONLY. Do NOT use the target language for explanations. Address the learner directly ("You"). Explain the underlying rule not just the fix. 
                  First see what is the native language of the learner and then write the explanation in the native language.
4. difficulty: 
   - Rate the difficulty of the CEFR level of the sentence's linguistic complexity. Apply standard CEFR descriptors precisely.
   - Common European Framework of Reference for Languages (CEFR) standardizes language proficiency into six level, these levels define abilities in reading, writing, listening, and speaking, ranging from basic phrases (A1) to near-native fluency (C2).
   - Not based on error count, but based on the ORIGINAL sentence itself ONLY.
   - Assess: vocabulary frequency and register, tense/mood/aspect range, depth of clause structure and subordination, level of abstraction, etc of the ORIGINAL sentence for CEFR level calculation.
   - The highest-scoring dimension sets the ceiling; pull back one level if only one dimension is elevated. 
   - CEFR difficulty levels: A1 | A2 | B1 | B2 | C1 | C2 .    

ERROR TYPE GUIDELINES:
Choose exactly one error_type per error. Use the most specific applicable type based on the rules below.
1. grammar: Use when the error is due to incorrect grammatical form or function that does not fit a more specific category. This includes syntax, punctuation, parts-of-speech,incorrect use of pronouns, articles, prepositions, or other function words required by sentence structure.
2. spelling: Use when a word is misspelled but the intended word is clear.
3. word_choice: Use when an incorrect vocabulary word is used but the grammatical structure is otherwise valid. Apply primarily to content words (nouns, verbs, adjectives, adverbs).
4. punctuation: Use when punctuation marks are missing, incorrect, or misused.
5. word_order: Use when words are arranged in an incorrect syntactic order.
6. missing_word: Use when a required word is missing for grammatical completeness or correctness.
7. extra_word: Use when an unnecessary word is present and should be removed.
8. conjugation: It is a specific part of grammar and applies only to changing verb forms. Use when a verb form is incorrect for tense, aspect, mood, voice, person, number, or auxiliary construction. Conjugation is the process of altering a verb from its base (infinitive) form to reflect grammatical context and ensure agreement with the subject.
9. gender_agreement: Use when grammatical gender does not match between related words.
10. number_agreement: Use when singular/plural agreement is incorrect between related words.
11. tone_register: Use when the sentence is inappropriate in level of formality or register.
12. other: Use only if no other category applies.

CEFR LEVEL GUIDELINES:
   A1 : Basic words, simple present tense, one clause.       
   A2 : Everyday words, past/future tense, simple connectors. 
   B1 : Opinions, multiple tenses, simple relative clauses.   
   B2 : Abstract ideas, conditionals, complex clauses.        
   C1 : Academic vocabulary, inversion, dense subordination.  
   C2 : Rare vocabulary, literary or rhetorical structures.

IMPORTANT CONSTRAINTS:
- Explanations MUST be written in the given native_language, NOT in the target_language.
- error_type MUST be one of the following: grammar | spelling | word_choice | punctuation |  word_order | missing_word | extra_word | conjugation | gender_agreement | number_agreement | tone_register | other. No new error types are allowed.
- Make sure to address all the errors in the sentence. 
- Your response MUST conform EXACTLY to the JSON schema structure provided. Do not add extra fields.

REAL EXAMPLES FROM TEST SUITE (Follow this format exactly):
{examples_json}

Learn from the examples and the rules to produce the best feedback for below.
"""

_examples_formatted = json.dumps(EXAMPLES_DATA, ensure_ascii=False, indent=2)
SYSTEM_PROMPT = SYSTEM_PROMPT.format(examples_json=_examples_formatted)

async def _call_anthropic(user_message: str) -> dict:
    """Call Anthropic Claude with tool use for structured output."""
    client = AsyncAnthropic(timeout=25.0)
    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        tools=[{
            "name": "feedback_response",
            "description": "Structured language feedback",
            "input_schema": RESPONSE_SCHEMA,
        }],
        tool_choice={"type": "tool", "name": "feedback_response"},
        temperature=0.2,
    )

    for block in message.content:
        if block.type == "tool_use":
            return block.input

    raise ValueError(f"No tool_use block in response (stop_reason={message.stop_reason})")


async def _call_openai(user_message: str) -> dict:
    """Call OpenAI GPT with json_schema response format."""
    client = AsyncOpenAI(timeout=30.0)
    response = await client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "feedback_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        reason = response.choices[0].finish_reason
        raise ValueError(f"Empty response from model (finish_reason={reason})")

    return json.loads(content)


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:

    key = _cache_key(request)
    if key in _cache:
        logger.info("cache hit key=%.12s", key)
        return _cache[key]

    user_message = json.dumps(
        {
            "sentence": request.sentence,
            "target_language": request.target_language,
            "native_language": request.native_language,
        },
        ensure_ascii=False,
    )

    if _use_anthropic():
        try:
            logger.info("Using Anthropic (Claude)")
            data = await _call_anthropic(user_message)
        except Exception as exc:
            logger.warning("Anthropic failed (%s), falling back to OpenAI", exc)
            data = await _call_openai(user_message)
    else:
        logger.info("Using OpenAI (GPT)")
        data = await _call_openai(user_message)

    if data.get("is_correct") is True:
        if data.get("errors"):
            data["is_correct"] = False
        if data.get("corrected_sentence") != request.sentence:
            data["is_correct"] = False

    if data.get("is_correct") is False and data.get("corrected_sentence") == request.sentence and not data.get("errors"):
        logger.warning("Model marked sentence incorrect but provided no errors and no corrections")

    result = FeedbackResponse(**data)
    _cache[key] = result

    return result
