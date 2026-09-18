import json
from pathlib import Path
from typing import Iterator, Optional

import pytest

from app.llm.base import LLM, LLMResponse, StreamEvent
from app.speech import (
    SOURCE_CONTEXT_MATCH,
    SOURCE_EXACT_MATCH,
    SOURCE_LLM_FALLBACK,
    ConversationContext,
    LLMCorrectionFallback,
    SpeechNormalizer,
    VocabularyManager,
    context_for_domain,
)
from app.speech.evaluate import evaluate, evaluate_by_category, load_cases

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


def test_corpus_covers_every_day_14_domain():
    """Day 14 item 16: the corpus spans the technical domains, not just Go."""
    categories = {case.category for case in load_cases(CASES_PATH)}

    assert len(load_cases(CASES_PATH)) >= 100
    assert {"golang", "java", "python", "database", "infrastructure", "distributed_systems"} <= categories


def test_no_domain_regresses(normalizer):
    for category, report in evaluate_by_category(normalizer, load_cases(CASES_PATH)).items():
        assert report.incorrectly_changed == 0, category
        assert report.missed_corrections == 0, category


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


# ---------------------------------------------------------------------------
# Correction provenance (day 14 item 15)
# ---------------------------------------------------------------------------


def test_high_confidence_correction_is_tagged_as_an_exact_match(normalizer):
    [change] = normalizer.normalize("how does a gore teen work").changes

    assert change.source == SOURCE_EXACT_MATCH


def test_context_gated_correction_is_tagged_as_a_context_match(normalizer):
    [change] = normalizer.normalize("we use red is for caching", context=context_for_domain("database")).changes

    assert change.source == SOURCE_CONTEXT_MATCH


def test_llm_confirmed_correction_is_tagged_as_an_llm_fallback():
    fallback = LLMCorrectionFallback(_FakeLLM('[{"index": 1, "confirmed": true, "confidence": 0.9}]'))
    normalizer = SpeechNormalizer(llm_fallback=fallback)

    [change] = normalizer.normalize("we call g c manually").changes

    assert change.source == SOURCE_LLM_FALLBACK


# ---------------------------------------------------------------------------
# Metrics (day 14 item 20)
# ---------------------------------------------------------------------------


def test_metrics_count_corrected_unchanged_and_uncertain():
    normalizer = SpeechNormalizer()

    normalizer.normalize("how does a gore teen work")  # corrected
    normalizer.normalize("the weather is nice today")  # unchanged
    normalizer.normalize("we call g c manually")  # uncertain, no fallback configured

    metrics = normalizer.metrics
    assert metrics.total == 3
    assert metrics.corrected == 1
    assert metrics.unchanged == 2
    assert metrics.uncertain == 1
    assert metrics.llm_fallback == 0
    assert metrics.corrections == 1


def test_llm_fallback_metric_counts_only_calls_that_needed_it():
    fallback = LLMCorrectionFallback(_FakeLLM('[{"index": 1, "confirmed": false, "confidence": 0.9}]'))
    normalizer = SpeechNormalizer(llm_fallback=fallback)

    normalizer.normalize("how does a gore teen work")  # resolved deterministically
    normalizer.normalize("we call g c manually")  # needed the fallback

    assert normalizer.metrics.llm_fallback == 1


# ---------------------------------------------------------------------------
# Conversation context (day 14 item 18)
# ---------------------------------------------------------------------------


def test_conversation_context_supplies_the_vocabulary(normalizer):
    manager = VocabularyManager()
    manager.observe("let's talk about databases")

    result = normalizer.normalize("we use red is for caching", conversation_context=manager.context)

    assert result.normalized == "we use Redis for caching"


def test_explicit_context_wins_over_conversation_context(normalizer):
    manager = VocabularyManager()
    manager.observe("let's talk about go concurrency")

    result = normalizer.normalize(
        "we use red is for caching",
        context=context_for_domain("database"),
        conversation_context=manager.context,
    )

    assert result.normalized == "we use Redis for caching"


def test_empty_conversation_context_corrects_nothing_ambiguous(normalizer):
    result = normalizer.normalize("we use red is for caching", conversation_context=ConversationContext())

    assert result.normalized == "we use red is for caching"
    assert result.changes == []
