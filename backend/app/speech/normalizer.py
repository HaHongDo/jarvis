"""Deterministic + optionally LLM-assisted post-STT normalization (see
day_13_speech_normalizer_implementation_plan.md items 6-15).

Safe correction, not maximum correction: a term only gets replaced automatically
when it is either high-confidence on its own, or medium-confidence with a
`VocabularyContext` confirming the topic. Anything less confident is left alone,
or handed to the optional LLM fallback to confirm/deny - never rewritten freely.
"""

import logging
import re
from typing import Optional

from ..config import (
    SPEECH_CONFIDENCE_HIGH,
    SPEECH_CONFIDENCE_MEDIUM,
    SPEECH_LLM_FALLBACK_CONFIDENCE,
)
from .models import Correction, NormalizationResult, TechnicalTerm, UncertainMatch, VocabularyContext
from .vocabulary import TECH_TERMS

logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _alias_pattern(alias: str) -> re.Pattern:
    """Builds a case-insensitive, word-boundary regex for `alias`, treating any run
    of whitespace/hyphens between its tokens as interchangeable (plan item 7), so
    "go routine" and "go-routine" share one pattern and "routine written in Go"
    never matches."""
    tokens = [t for t in re.split(r"[\s\-]+", alias.strip()) if t]
    body = r"[\s-]+".join(re.escape(t) for t in tokens)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


def _compile_terms(terms: list[TechnicalTerm]):
    compiled = []
    for term in terms:
        for alias in term.aliases:
            tokens = [t for t in re.split(r"[\s\-]+", alias.strip()) if t]
            if not tokens:
                continue
            compiled.append((term, len(tokens), _alias_pattern(alias)))
    # Longest phrase (most tokens) first, so a multi-word alias is matched before a
    # shorter alias could carve up part of it.
    compiled.sort(key=lambda item: (-item[1], -len(item[2].pattern)))
    return compiled


class SpeechNormalizer:
    """See plan item 23. `normalize()` runs whitespace normalization, then exact
    vocabulary matching gated by confidence/context, then an optional LLM fallback
    for whatever the deterministic pass couldn't confidently resolve."""

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

    def normalize(self, text: str, context: Optional[VocabularyContext] = None) -> NormalizationResult:
        original = text
        working = _normalize_whitespace(text)

        changes: list[Correction] = []
        uncertain: list[UncertainMatch] = []
        working = self._match_vocabulary(working, changes, uncertain, context)

        if uncertain and self.llm_fallback is not None:
            working = self._resolve_uncertain(working, changes, uncertain, context)

        if self.debug:
            self._log(original, changes, working)

        return NormalizationResult(original=original, normalized=working, changes=changes)

    def _active_categories(self, context: Optional[VocabularyContext]) -> set:
        if context is None:
            return set()
        categories = {t.category for t in context.terms}
        if context.domain:
            categories.add(context.domain)
        return categories

    def _match_vocabulary(
        self,
        text: str,
        changes: list[Correction],
        uncertain: list[UncertainMatch],
        context: Optional[VocabularyContext],
    ) -> str:
        active = self._active_categories(context)

        for term, _token_count, pattern in self._compiled:

            def _replace(match: re.Match, term: TechnicalTerm = term) -> str:
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
                        )
                    )
                    return term.canonical

                if term.confidence >= self.confidence_medium and term.category in active:
                    changes.append(
                        Correction(
                            original=matched,
                            replacement=term.canonical,
                            reason=f"technical vocabulary ({term.category}, context-confirmed)",
                            confidence=term.confidence,
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
            pattern = _alias_pattern(candidate.matched_text)
            if not pattern.search(text):
                continue  # already handled by an earlier candidate/decision
            text = pattern.sub(candidate.term.canonical, text, count=1)
            changes.append(
                Correction(
                    original=candidate.matched_text,
                    replacement=candidate.term.canonical,
                    reason=f"llm fallback ({candidate.term.category})",
                    confidence=decision.confidence,
                )
            )

        return text

    def _log(self, original: str, changes: list[Correction], normalized: str) -> None:
        logger.info("[SpeechNormalizer]")
        logger.info("RAW: %s", original)
        for change in changes:
            logger.info('MATCH: "%s" -> "%s"', change.original, change.replacement)
        logger.info("OUTPUT: %s", normalized)
