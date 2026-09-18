"""Data types for the Day 13 Speech Normalizer (see
day_13_speech_normalizer_implementation_plan.md item 3)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Correction:
    """A single change the normalizer made to the raw transcript."""

    original: str
    replacement: str
    reason: str
    confidence: float


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
    never matched deterministically."""

    canonical: str
    aliases: list[str]
    category: str
    related_terms: list[str] = field(default_factory=list)
    confidence: float = 0.95


@dataclass
class VocabularyContext:
    """The vocabulary relevant to the current conversation topic. Lets a
    medium-confidence correction (see plan item 10) apply only when the topic
    actually supports it, per plan item 9."""

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
