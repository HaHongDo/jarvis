import json
from pathlib import Path
from typing import Iterator, Optional

import pytest

from app.llm.base import LLM, LLMResponse, StreamEvent
from app.speech import LLMCorrectionFallback, SpeechNormalizer, context_for_domain
from app.speech.evaluate import evaluate, load_cases

CASES_PATH = Path(__file__).parent / "speech" / "terminology.json"


def _load_raw_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())


class _FakeLLM(LLM):
    """A minimal LLM double that always returns a fixed response, so the LLM
    fallback can be tested without a live Ollama server."""

    def __init__(self, content: str):
        self._content = content

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        return LLMResponse(content=self._content)

    def stream_chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> Iterator[StreamEvent]:
        raise NotImplementedError


@pytest.fixture(scope="module")
def normalizer() -> SpeechNormalizer:
    return SpeechNormalizer()


# ---------------------------------------------------------------------------
# Corpus-driven tests (plan items 18-19)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _load_raw_cases(), ids=lambda c: c["input"][:40])
def test_terminology_corpus(normalizer, case):
    context = context_for_domain(case["context_domain"]) if case.get("context_domain") else None

    result = normalizer.normalize(case["input"], context=context)

    assert result.normalized == case["expected"]
    assert result.original == case["input"]


def test_evaluation_precision_and_recall_are_high(normalizer):
    cases = load_cases(CASES_PATH)

    report = evaluate(normalizer, cases)

    assert report.incorrectly_changed == 0
    assert report.precision == 100.0
    assert report.recall == 100.0


# ---------------------------------------------------------------------------
# Behavior of the confidence tiers (plan item 10)
# ---------------------------------------------------------------------------


def test_high_confidence_correction_needs_no_context(normalizer):
    result = normalizer.normalize("how does a gore teen communicate")

    assert result.normalized == "how does a goroutine communicate"
    [change] = result.changes
    assert change.original == "gore teen"
    assert change.replacement == "goroutine"
    assert change.confidence >= 0.95


def test_negative_case_is_never_rewritten(normalizer):
    result = normalizer.normalize("I wrote a routine in Go.")

    assert result.normalized == "I wrote a routine in Go."
    assert result.changes == []


def test_medium_confidence_term_requires_matching_context(normalizer):
    without_context = normalizer.normalize("we use red is for caching")
    with_context = normalizer.normalize("we use red is for caching", context=context_for_domain("database"))

    assert without_context.normalized == "we use red is for caching"
    assert with_context.normalized == "we use Redis for caching"


def test_low_confidence_term_is_never_auto_corrected_even_with_context(normalizer):
    result = normalizer.normalize("we call g c manually", context=context_for_domain("java"))

    assert result.normalized == "we call g c manually"
    assert result.changes == []


def test_raw_transcript_is_always_preserved(normalizer):
    result = normalizer.normalize("gore teen scheduler")

    assert result.original == "gore teen scheduler"
    assert result.normalized == "goroutine scheduler"


# ---------------------------------------------------------------------------
# LLM fallback (plan items 12-15)
# ---------------------------------------------------------------------------


def test_llm_fallback_confirms_low_confidence_match():
    fallback = LLMCorrectionFallback(_FakeLLM('[{"index": 1, "confirmed": true, "confidence": 0.9}]'))
    normalizer = SpeechNormalizer(llm_fallback=fallback)

    result = normalizer.normalize("we call g c manually")

    assert result.normalized == "we call GC manually"
    [change] = result.changes
    assert change.reason.startswith("llm fallback")


def test_llm_fallback_leaves_text_unchanged_when_not_confirmed():
    fallback = LLMCorrectionFallback(_FakeLLM('[{"index": 1, "confirmed": false, "confidence": 0.9}]'))
    normalizer = SpeechNormalizer(llm_fallback=fallback)

    result = normalizer.normalize("we call g c manually")

    assert result.normalized == "we call g c manually"
    assert result.changes == []


def test_llm_fallback_discards_malformed_response():
    fallback = LLMCorrectionFallback(_FakeLLM("not json"))
    normalizer = SpeechNormalizer(llm_fallback=fallback)

    result = normalizer.normalize("we call g c manually")

    assert result.normalized == "we call g c manually"
    assert result.changes == []


def test_llm_fallback_discards_response_with_wrong_candidate_count():
    fallback = LLMCorrectionFallback(
        _FakeLLM(
            '[{"index": 1, "confirmed": true, "confidence": 0.9}, '
            '{"index": 2, "confirmed": true, "confidence": 0.9}]'
        )
    )
    normalizer = SpeechNormalizer(llm_fallback=fallback)

    result = normalizer.normalize("we call g c manually")

    assert result.normalized == "we call g c manually"
    assert result.changes == []


def test_llm_fallback_never_runs_without_uncertain_matches():
    class _ExplodingLLM(LLM):
        def chat(self, messages, tools=None):
            raise AssertionError("LLM fallback should not be called when nothing is uncertain")

        def stream_chat(self, messages, tools=None):
            raise NotImplementedError

    fallback = LLMCorrectionFallback(_ExplodingLLM())
    normalizer = SpeechNormalizer(llm_fallback=fallback)

    result = normalizer.normalize("how does a gore teen communicate")

    assert result.normalized == "how does a goroutine communicate"
