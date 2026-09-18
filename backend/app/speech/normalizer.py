"""Deterministic + optionally LLM-assisted post-STT normalization (Day 13 plan
items 6-15, Day 14 items 14-15, 18-20).

Safe correction, not maximum correction: a term only gets replaced automatically
when it is either high-confidence on its own, or medium-confidence with the
active vocabulary confirming the topic. Anything less confident is left alone,
or handed to the optional LLM fallback to confirm/deny - never rewritten freely.

The normalizer stays after STT even once STT gets vocabulary hints, because those
hints are probabilistic: Whisper can still mishear a term it was told to expect
(day 14 item 14). It also stays stateless - conversation state belongs to the
VocabularyManager, which passes in a `ConversationContext` (day 14 item 18).
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..config import (
    SPEECH_CONFIDENCE_HIGH,
    SPEECH_CONFIDENCE_MEDIUM,
    SPEECH_LLM_FALLBACK_CONFIDENCE,
)
from .matching import normalize_whitespace, phrase_pattern, phrase_tokens
from .models import (
    SOURCE_CONTEXT_MATCH,
    SOURCE_EXACT_MATCH,
    SOURCE_LLM_FALLBACK,
    ConversationContext,
    Correction,
    NormalizationResult,
    TechnicalTerm,
    UncertainMatch,
    VocabularyContext,
)
from .vocabulary import TECH_TERMS

logger = logging.getLogger(__name__)


@dataclass
class NormalizationMetrics:
    """Per-call counters for the Day 14 item 20 observability list. Counted once
    per `normalize()` call, not once per correction - `corrections` is the
    exception, since "how many words did we rewrite" is a different question from
    "how many turns did we touch".

    There is deliberately no false-correction counter here: the normalizer cannot
    know it was wrong. That number comes from the evaluation corpus instead
    (`incorrectly_changed` in app/speech/evaluate.py)."""

    total: int = 0
    corrected: int = 0
    unchanged: int = 0
    uncertain: int = 0
    llm_fallback: int = 0
    corrections: int = 0


def _compile_terms(terms: list[TechnicalTerm]):
    compiled = []
    for term in terms:
        for alias in term.aliases:
            tokens = phrase_tokens(alias)
            if not tokens:
                continue
            compiled.append((term, len(tokens), phrase_pattern(alias)))
    # Longest phrase (most tokens) first, so a multi-word alias is matched before a
    # shorter alias could carve up part of it.
    compiled.sort(key=lambda item: (-item[1], -len(item[2].pattern)))
    return compiled


class SpeechNormalizer:
    """See Day 13 plan item 23. `normalize()` runs whitespace normalization, then
    exact vocabulary matching gated by confidence/context, then an optional LLM
    fallback for whatever the deterministic pass couldn't confidently resolve."""

    def __init__(
        self,
        terms: Optional[list[TechnicalTerm]] = None,
        llm_fallback=None,
        debug: bool = False,
        confidence_high: float = SPEECH_CONFIDENCE_HIGH,
        confidence_medium: float = SPEECH_CONFIDENCE_MEDIUM,
        llm_confirm_threshold: float = SPEECH_LLM_FALLBACK_CONFIDENCE,
    ):
        self.terms = terms if terms is not None else TECH_TERMS
        self._compiled = _compile_terms(self.terms)
        self.llm_fallback = llm_fallback
        self.debug = debug
        self.confidence_high = confidence_high
        self.confidence_medium = confidence_medium
        self.llm_confirm_threshold = llm_confirm_threshold
        self.metrics = NormalizationMetrics()

    def normalize(
        self,
        text: str,
        context: Optional[VocabularyContext] = None,
        conversation_context: Optional[ConversationContext] = None,
    ) -> NormalizationResult:
        """`context` pins the vocabulary explicitly (handy in tests and the
        evaluation corpus); `conversation_context` is what the VocabularyManager
        passes in at runtime. An explicit `context` wins if both are given."""
        original = text
        working = normalize_whitespace(text)

        if context is None and conversation_context is not None:
            context = conversation_context.active_vocabulary.to_vocabulary_context()

        changes: list[Correction] = []
        uncertain: list[UncertainMatch] = []
        working = self._match_vocabulary(working, changes, uncertain, context)

        used_llm = bool(uncertain) and self.llm_fallback is not None
        if used_llm:
            working = self._resolve_uncertain(working, changes, uncertain, context)

        self._record_metrics(changes, uncertain, used_llm)

        if self.debug:
            self._log(original, context, changes, uncertain, working)

        return NormalizationResult(original=original, normalized=working, changes=changes)

    def _record_metrics(self, changes: list[Correction], uncertain: list[UncertainMatch], used_llm: bool) -> None:
        self.metrics.total += 1
        if changes:
            self.metrics.corrected += 1
            self.metrics.corrections += len(changes)
        else:
            self.metrics.unchanged += 1
        if uncertain:
            self.metrics.uncertain += 1
        if used_llm:
            self.metrics.llm_fallback += 1

    def _supported_canonicals(self, context: Optional[VocabularyContext]) -> set:
        if context is None:
            return set()
        return {term.canonical.lower() for term in context.terms}

    def _context_supports(self, term: TechnicalTerm, context: Optional[VocabularyContext], supported: set) -> bool:
        """A medium-confidence term may only be corrected when the conversation is
        actually about it: either it is in the active vocabulary, or its whole
        category is the active domain (day 14 items 6, 17)."""
        if context is None:
            return False
        if context.domain is not None and term.category == context.domain:
            return True
        return term.canonical.lower() in supported

    def _match_vocabulary(
        self,
        text: str,
        changes: list[Correction],
        uncertain: list[UncertainMatch],
        context: Optional[VocabularyContext],
    ) -> str:
        supported = self._supported_canonicals(context)

        for term, _token_count, pattern in self._compiled:

            def _replace(match, term: TechnicalTerm = term) -> str:
                matched = match.group(0)
                if matched == term.canonical:
                    return matched  # already correct, nothing to do

                if term.confidence >= self.confidence_high:
                    changes.append(
                        Correction(
                            original=matched,
                            replacement=term.canonical,
                            reason=f"technical vocabulary ({term.category})",
                            confidence=term.confidence,
                            source=SOURCE_EXACT_MATCH,
                        )
                    )
                    return term.canonical

                if term.confidence >= self.confidence_medium and self._context_supports(term, context, supported):
                    changes.append(
                        Correction(
                            original=matched,
                            replacement=term.canonical,
                            reason=f"technical vocabulary ({term.category}, context-confirmed)",
                            confidence=term.confidence,
                            source=SOURCE_CONTEXT_MATCH,
                        )
                    )
                    return term.canonical

                uncertain.append(UncertainMatch(term=term, matched_text=matched))
                return matched

            text = pattern.sub(_replace, text)

        return text

    def _resolve_uncertain(
        self,
        text: str,
        changes: list[Correction],
        uncertain: list[UncertainMatch],
        context: Optional[VocabularyContext],
    ) -> str:
        decisions = self.llm_fallback.confirm(text, uncertain, context)
        if decisions is None or len(decisions) != len(uncertain):
            logger.warning("[SpeechNormalizer] LLM fallback returned no usable decision, preserving original")
            return text

        for candidate, decision in zip(uncertain, decisions):
            if not decision.confirmed or decision.confidence < self.llm_confirm_threshold:
                continue
            pattern = phrase_pattern(candidate.matched_text)
            if not pattern.search(text):
                continue  # already handled by an earlier candidate/decision
            text = pattern.sub(candidate.term.canonical, text, count=1)
            changes.append(
                Correction(
                    original=candidate.matched_text,
                    replacement=candidate.term.canonical,
                    reason=f"llm fallback ({candidate.term.category})",
                    confidence=decision.confidence,
                    source=SOURCE_LLM_FALLBACK,
                )
            )

        return text

    def _log(
        self,
        original: str,
        context: Optional[VocabularyContext],
        changes: list[Correction],
        uncertain: list[UncertainMatch],
        normalized: str,
    ) -> None:
        """Debug log in the day 14 item 20 shape, so a bad turn can be blamed on
        STT, the topic, or the correction itself.

        Unresolved matches are logged too: a phrase that shows up as UNCERTAIN
        over and over is exactly the kind of recurring STT error that should be
        verified by hand and promoted into the vocabulary (day 14 item 21)."""
        logger.info("[SpeechNormalizer]")
        logger.info("RAW: %s", original)
        logger.info("DOMAIN: %s", context.domain if context and context.domain else "-")
        for change in changes:
            logger.info('CORRECTION: "%s" -> "%s"', change.original, change.replacement)
            logger.info("SOURCE: %s", change.source)
            logger.info("CONFIDENCE: %.2f", change.confidence)
        for candidate in uncertain:
            logger.info(
                'UNCERTAIN: "%s" (suggested "%s", confidence %.2f) - preserved',
                candidate.matched_text,
                candidate.term.canonical,
                candidate.term.confidence,
            )
        logger.info("OUTPUT: %s", normalized)
