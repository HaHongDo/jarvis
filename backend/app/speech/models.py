"""Data types for the Speech Normalizer (Day 13 plan item 3) and the Day 14
context-aware vocabulary layer (day 14 items 6, 8, 15)."""

from dataclasses import dataclass, field
from typing import Optional

# Where a correction came from (day 14 item 15). Recording this on every
# `Correction` makes it possible to tell "the dictionary fixed it", "the
# conversation topic made it safe to fix" and "the LLM confirmed it" apart when
# debugging a bad turn.
SOURCE_EXACT_MATCH = "exact_match"  # high-confidence alias, no context needed
SOURCE_CONTEXT_MATCH = "context_match"  # medium confidence, confirmed by the active vocabulary
SOURCE_LLM_FALLBACK = "llm_fallback"  # low confidence, confirmed by the LLM fallback
# Reserved for the STT-side comparison harness (app/speech/evaluate_stt.py):
# the term was never wrong in the first place because the vocabulary hint
# reached Whisper before transcription. The normalizer never emits it.
SOURCE_STT_HINT = "stt_hint"


@dataclass
class Correction:
    """A single change the normalizer made to the raw transcript."""

    original: str
    replacement: str
    reason: str
    confidence: float
    source: str = SOURCE_EXACT_MATCH


@dataclass
class NormalizationResult:
    """Output of `SpeechNormalizer.normalize()`. Always keeps the raw STT text
    alongside the normalized one, so a bad answer can be traced back to STT vs. the
    normalizer vs. the LLM fallback (see plan item 16)."""

    original: str
    normalized: str
    changes: list[Correction] = field(default_factory=list)


@dataclass
class TechnicalTerm:
    """A canonical technical term plus the STT mishearings/spellings that should be
    corrected to it (see plan item 5). Terms used only to describe a domain for
    `VocabularyContext` (e.g. "channel", "heap") carry no aliases and are therefore
    never matched deterministically - they exist to name the topic and to seed the
    STT vocabulary hint."""

    canonical: str
    aliases: list[str]
    category: str
    related_terms: list[str] = field(default_factory=list)
    confidence: float = 0.95


@dataclass
class VocabularyContext:
    """The vocabulary relevant to the current conversation topic. Lets a
    medium-confidence correction (see plan item 10) apply only when the topic
    actually supports it, per plan item 9.

    A term is "supported" when it is listed in `terms`, or when its category is
    `domain` - the latter keeps `context_for_domain("java")` meaning "all of Java"
    without having to enumerate it."""

    domain: Optional[str] = None
    terms: list[TechnicalTerm] = field(default_factory=list)


@dataclass
class UncertainMatch:
    """A vocabulary match the deterministic pass couldn't confidently resolve on
    its own (below the high-confidence threshold, or medium-confidence without a
    matching `VocabularyContext`). Handed to the optional LLM fallback (plan item
    12) instead of being silently corrected."""

    term: TechnicalTerm
    matched_text: str


@dataclass
class ActiveVocabulary:
    """The slice of the vocabulary database that is relevant right now (day 14
    item 6) - a handful of domains and their terms, not the whole dictionary.
    Produced by `VocabularyManager`, consumed by both STT (as a transcription
    hint) and the normalizer (as correction context)."""

    domains: list[str] = field(default_factory=list)
    terms: list[TechnicalTerm] = field(default_factory=list)

    def canonicals(self) -> list[str]:
        """Canonical terms, deduplicated, in domain-relevance order."""
        seen: set[str] = set()
        ordered: list[str] = []
        for term in self.terms:
            if term.canonical not in seen:
                seen.add(term.canonical)
                ordered.append(term.canonical)
        return ordered

    def to_vocabulary_context(self) -> VocabularyContext:
        """Bridge to the Day 13 normalizer API. The primary (highest-scoring)
        domain becomes `VocabularyContext.domain`; every active term is listed
        explicitly so cross-domain terms (Kafka under "distributed_systems",
        say) are supported without dragging in their whole home category."""
        return VocabularyContext(domain=self.domains[0] if self.domains else None, terms=list(self.terms))


@dataclass
class ConversationContext:
    """What the conversation is currently about (day 14 item 8). Owned by the
    conversation/vocabulary layer and passed *into* the normalizer, which stays
    stateless."""

    active_domain: Optional[str] = None
    active_vocabulary: ActiveVocabulary = field(default_factory=ActiveVocabulary)
    recent_terms: list[str] = field(default_factory=list)
