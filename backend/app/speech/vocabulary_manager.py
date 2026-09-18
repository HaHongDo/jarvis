"""Context-aware vocabulary selection (day 14 items 2, 6-11).

Day 13 gave the normalizer one flat dictionary of corrections. This module answers
a narrower question:

    what technical vocabulary is relevant to the user's current conversation?

and hands the answer to both sides of the STT stage:

    Conversation
         |
         v
    VocabularyManager  --> active vocabulary
         |                      |
         |  initial_prompt /    |  VocabularyContext
         |  hotwords            |
         v                      v
        STT               SpeechNormalizer

Topic detection is deliberately dumb: keyword scoring over the conversation, an
explicit "let's switch to Java" override, and timestamp-based decay so a topic
from an hour ago stops steering transcription (day 14 items 7-10). No LLM, no
ontology.

The manager owns conversation state; the normalizer does not (day 14 item 18).
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import (
    SPEECH_VOCAB_DECAY_SECONDS,
    SPEECH_VOCAB_MAX_ACTIVE_DOMAINS,
    SPEECH_VOCAB_MAX_HOTWORDS,
    SPEECH_VOCAB_MAX_PROMPT_TERMS,
    SPEECH_VOCAB_MAX_RECENT_TERMS,
    SPEECH_VOCAB_MIN_DOMAIN_SCORE,
    SPEECH_VOCAB_RECENT_SECONDS,
    SPEECH_VOCAB_STALE_SECONDS,
)
from .matching import phrase_pattern, phrase_tokens
from .models import ActiveVocabulary, ConversationContext, NormalizationResult, TechnicalTerm
from .vocabulary import (
    AMBIGUOUS_KEYWORDS,
    DOMAIN_ALIASES,
    DOMAIN_KEYWORDS,
    DOMAIN_LABELS,
    TECH_TERMS,
    VOCABULARY_DOMAINS,
    terms_for_domain,
)

logger = logging.getLogger(__name__)

# Scoring weights (day 14 item 7). A curated keyword ("goroutine", "spring boot")
# is decisive on its own; a keyword derived from a domain's own vocabulary is
# half a vote, so two of them are needed to pick up a topic nobody named.
STRONG_KEYWORD_WEIGHT = 1.0
DERIVED_KEYWORD_WEIGHT = 0.5

# An explicit "let's switch to Java" is worth several mentions - enough to stay
# active through the 50% decay tier even if the user says nothing Java-ish next.
TOPIC_SWITCH_SCORE = 4.0

# Ceilings, so a long monologue about Go can't pin the topic there forever and a
# single utterance can't max out a domain by repeating itself.
MAX_DOMAIN_SCORE = 5.0
MAX_SCORE_PER_UTTERANCE = 3.0

# Phrases that introduce a new topic, e.g. "let's switch to Java", "moving on to
# Kubernetes", "back to Go". Deliberately explicit: inferring every topic
# transition is out of scope for V1 (day 14 item 9).
_SWITCH_CUES = [
    "let's switch to",
    "lets switch to",
    "let's move on to",
    "lets move on to",
    "let's talk about",
    "lets talk about",
    "let's discuss",
    "lets discuss",
    "switching to",
    "switch to",
    "moving on to",
    "move on to",
    "changing topic to",
    "change topic to",
    "going back to",
    "back to",
    "new topic",
    "now let's do",
    "now lets do",
    "i want to talk about",
    "can we talk about",
    "tell me about",
]

# How far past a switch cue we still accept a domain name. Long enough for "let's
# switch to the Java side of things", short enough that a domain mentioned in the
# next clause doesn't count.
_SWITCH_LOOKAHEAD_CHARS = 40


def domain_keywords(domain: str) -> list[tuple[str, float]]:
    """Scoring keywords for `domain`: the curated decisive phrases, plus every
    canonical/alias in the domain that isn't ordinary English."""
    return _DOMAIN_KEYWORD_CACHE[domain]


def _build_domain_keywords(domain: str) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    seen: set[str] = set()

    for keyword in DOMAIN_KEYWORDS.get(domain, []):
        key = keyword.lower()
        if key not in seen:
            seen.add(key)
            entries.append((keyword, STRONG_KEYWORD_WEIGHT))

    for term in terms_for_domain(domain):
        for phrase in (term.canonical, *term.aliases):
            key = phrase.lower()
            if key in seen or key in AMBIGUOUS_KEYWORDS:
                continue
            seen.add(key)
            entries.append((phrase, DERIVED_KEYWORD_WEIGHT))

    return entries


_DOMAIN_KEYWORD_CACHE: dict[str, list[tuple[str, float]]] = {
    domain: _build_domain_keywords(domain) for domain in VOCABULARY_DOMAINS
}


def score_domains(text: str) -> dict[str, float]:
    """Keyword score per domain for one utterance (day 14 item 7). Each distinct
    keyword counts once, so repetition doesn't inflate a topic."""
    scores: dict[str, float] = {}
    for domain, keywords in _DOMAIN_KEYWORD_CACHE.items():
        score = sum(weight for keyword, weight in keywords if phrase_pattern(keyword).search(text))
        if score > 0:
            scores[domain] = min(score, MAX_SCORE_PER_UTTERANCE)
    return scores


def detect_domain(text: str, min_score: float = SPEECH_VOCAB_MIN_DOMAIN_SCORE) -> Optional[str]:
    """The single most likely domain for `text`, or None when nothing scores high
    enough. Stateless - `VocabularyManager` is what accumulates across turns."""
    scores = score_domains(text)
    if not scores:
        return None
    domain, score = max(scores.items(), key=lambda item: (item[1], -_domain_rank(item[0])))
    return domain if score >= min_score else None


def _domain_rank(domain: str) -> int:
    """Declaration order, used only as a deterministic tie-break."""
    try:
        return list(VOCABULARY_DOMAINS).index(domain)
    except ValueError:  # pragma: no cover - domain always comes from the table
        return len(VOCABULARY_DOMAINS)


def detect_topic_switch(text: str) -> Optional[str]:
    """The domain the user explicitly switched to, e.g. "let's switch to Java"
    (day 14 item 9), or None."""
    lowered = text.lower()

    best: Optional[tuple[int, int, str]] = None  # (cue position, -alias length, domain)
    for cue in _SWITCH_CUES:
        cue_at = lowered.find(cue)
        if cue_at == -1:
            continue
        window = lowered[cue_at + len(cue) : cue_at + len(cue) + _SWITCH_LOOKAHEAD_CHARS]
        for domain, aliases in DOMAIN_ALIASES.items():
            for alias in aliases:
                if phrase_pattern(alias).search(window):
                    # Prefer the earliest cue, and within one cue the longest
                    # alias ("distributed systems" over "systems").
                    candidate = (cue_at, -len(alias), domain)
                    if best is None or candidate < best:
                        best = candidate

    return best[2] if best else None


def build_stt_prompt(vocabulary: ActiveVocabulary, max_terms: int = SPEECH_VOCAB_MAX_PROMPT_TERMS) -> Optional[str]:
    """Whisper `initial_prompt` for the active vocabulary (day 14 item 11).

    Whisper conditions on this text as if it were the transcript so far, so it
    reads as a normal sentence naming the topic and the words to expect. Returns
    None when no topic is active, so STT runs unhinted rather than being nudged
    toward an arbitrary domain."""
    if not vocabulary.domains:
        return None

    terms = vocabulary.canonicals()[:max_terms]
    if not terms:
        return None

    labels = [DOMAIN_LABELS.get(domain, domain.replace("_", " ")) for domain in vocabulary.domains]
    if len(labels) == 1:
        topic = labels[0]
    else:
        topic = f"{', '.join(labels[:-1])} and {labels[-1]}"

    return f"The conversation is about {topic}. Technical terms: {', '.join(terms)}."


# Lowercase canonicals that exist only to describe a domain ("channel", "heap",
# "quorum"). Whisper already spells these correctly; hinting them wastes budget.
_PLAIN_WORDS = {term.canonical for term in TECH_TERMS if term.canonical.islower() and not term.aliases}


def build_hotwords(vocabulary: ActiveVocabulary, max_hotwords: int = SPEECH_VOCAB_MAX_HOTWORDS) -> Optional[str]:
    """faster-whisper `hotwords` string (day 14 item 12): only the terms Whisper
    is unlikely to produce on its own. Ordinary English words ("channel", "heap")
    are dropped - hinting them costs prompt budget and biases nothing useful."""
    selected = [term for term in vocabulary.canonicals() if _is_distinctive(term)][:max_hotwords]
    return " ".join(selected) if selected else None


def _is_distinctive(canonical: str) -> bool:
    if canonical.lower() in AMBIGUOUS_KEYWORDS:
        return False
    if len(phrase_tokens(canonical)) > 1:  # multi-word phrases are descriptions, not hotwords
        return False
    # Anything the speaker would spell out or capitalize: GOMAXPROCS, gRPC,
    # pgvector, G1. A plain lowercase dictionary-looking word is not worth a hint.
    return any(c.isupper() or c.isdigit() for c in canonical) or canonical not in _PLAIN_WORDS


@dataclass
class _DomainActivation:
    """How strongly a domain is in play, and when it was last mentioned."""

    score: float
    last_seen: float


class VocabularyManager:
    """Tracks what the conversation is about and exposes the matching slice of the
    vocabulary (day 14 item 2).

    Usage per turn:

        context = manager.observe(raw_transcript)           # topic detection
        result = normalizer.normalize(raw, conversation_context=context)
        manager.observe_result(result)                      # remember the terms used

    and, before the *next* transcription:

        stt.transcribe(audio, initial_prompt=manager.stt_prompt(), hotwords=manager.hotwords())
    """

    def __init__(
        self,
        terms: Optional[list[TechnicalTerm]] = None,
        *,
        max_active_domains: int = SPEECH_VOCAB_MAX_ACTIVE_DOMAINS,
        max_prompt_terms: int = SPEECH_VOCAB_MAX_PROMPT_TERMS,
        max_hotwords: int = SPEECH_VOCAB_MAX_HOTWORDS,
        max_recent_terms: int = SPEECH_VOCAB_MAX_RECENT_TERMS,
        min_domain_score: float = SPEECH_VOCAB_MIN_DOMAIN_SCORE,
        recent_seconds: float = SPEECH_VOCAB_RECENT_SECONDS,
        decay_seconds: float = SPEECH_VOCAB_DECAY_SECONDS,
        stale_seconds: float = SPEECH_VOCAB_STALE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.terms = terms if terms is not None else TECH_TERMS
        self.max_active_domains = max_active_domains
        self.max_prompt_terms = max_prompt_terms
        self.max_hotwords = max_hotwords
        self.max_recent_terms = max_recent_terms
        self.min_domain_score = min_domain_score
        self.recent_seconds = recent_seconds
        self.decay_seconds = decay_seconds
        self.stale_seconds = stale_seconds
        self._clock = clock

        self._activations: dict[str, _DomainActivation] = {}
        self._recent_terms: list[str] = []

    # -- observation -------------------------------------------------------

    def observe(self, text: str) -> ConversationContext:
        """Fold one utterance into the conversation topic and return the context
        the normalizer should use for *this* utterance - so "let's switch to Java"
        and the Java term in the same breath both land on Java."""
        if not text or not text.strip():
            return self.context

        switched_to = detect_topic_switch(text)
        if switched_to is not None:
            self.switch_to(switched_to)
            logger.debug("[VocabularyManager] explicit topic switch -> %s", switched_to)
        else:
            self._apply_scores(score_domains(text))

        self._record_terms_in(text)
        return self.context

    def observe_result(self, result: NormalizationResult) -> None:
        """Remember the canonical terms the normalizer actually produced, so a
        corrected term steers the next transcription even though the raw
        transcript never contained it."""
        for change in result.changes:
            self._push_recent_term(change.replacement)

    def switch_to(self, domain: Optional[str]) -> None:
        """Make `domain` the topic, dropping the previous one outright (day 14
        item 9). `None` clears the topic without clearing recent terms."""
        now = self._clock()
        self._activations.clear()
        if domain is not None and domain in VOCABULARY_DOMAINS:
            self._activations[domain] = _DomainActivation(score=TOPIC_SWITCH_SCORE, last_seen=now)

    def reset(self) -> None:
        self._activations.clear()
        self._recent_terms.clear()

    def _apply_scores(self, scores: dict[str, float]) -> None:
        now = self._clock()
        for domain, points in scores.items():
            current = self._decayed_score(domain, now)
            self._activations[domain] = _DomainActivation(
                score=min(current + points, MAX_DOMAIN_SCORE), last_seen=now
            )

    def _record_terms_in(self, text: str) -> None:
        matched: list[tuple[int, str]] = []
        for term in self.terms:
            positions = [
                match.start()
                for match in (phrase_pattern(phrase).search(text) for phrase in (term.canonical, *term.aliases))
                if match is not None
            ]
            if positions:
                matched.append((min(positions), term.canonical))

        # Pushed in reverse order of appearance, so the sentence's first term ends
        # up first in `recent_terms` after each push-to-front.
        matched.sort()
        for _position, canonical in reversed(matched):
            self._push_recent_term(canonical)

    def _push_recent_term(self, canonical: str) -> None:
        if canonical in self._recent_terms:
            self._recent_terms.remove(canonical)
        self._recent_terms.insert(0, canonical)
        del self._recent_terms[self.max_recent_terms :]

    # -- decay -------------------------------------------------------------

    def _relevance(self, age: float) -> float:
        """Timestamp-based decay (day 14 item 10): current topic 100%, recent 50%,
        old 20%, very old 0%. No ML, just how long ago it was mentioned."""
        if age < self.recent_seconds:
            return 1.0
        if age < self.decay_seconds:
            return 0.5
        if age < self.stale_seconds:
            return 0.2
        return 0.0

    def _decayed_score(self, domain: str, now: float) -> float:
        activation = self._activations.get(domain)
        if activation is None:
            return 0.0
        return activation.score * self._relevance(now - activation.last_seen)

    # -- current state -----------------------------------------------------

    def active_domains(self) -> list[str]:
        """Domains still relevant after decay, strongest first, capped at
        `max_active_domains` so the active vocabulary stays small."""
        now = self._clock()
        scored = [
            (domain, self._decayed_score(domain, now))
            for domain in self._activations
        ]
        relevant = [(d, s) for d, s in scored if s >= self.min_domain_score]
        relevant.sort(key=lambda item: (-item[1], _domain_rank(item[0])))
        return [domain for domain, _ in relevant[: self.max_active_domains]]

    def active_vocabulary(self) -> ActiveVocabulary:
        """The active domains' terms, with terms mentioned recently pulled to the
        front so they survive the STT prompt's term budget."""
        domains = self.active_domains()

        terms: list[TechnicalTerm] = []
        seen: set[int] = set()
        for domain in domains:
            for term in terms_for_domain(domain):
                if id(term) not in seen:
                    seen.add(id(term))
                    terms.append(term)

        recent_rank = {canonical: i for i, canonical in enumerate(self._recent_terms)}
        terms.sort(key=lambda term: recent_rank.get(term.canonical, len(recent_rank)))

        return ActiveVocabulary(domains=domains, terms=terms)

    @property
    def context(self) -> ConversationContext:
        vocabulary = self.active_vocabulary()
        return ConversationContext(
            active_domain=vocabulary.domains[0] if vocabulary.domains else None,
            active_vocabulary=vocabulary,
            recent_terms=list(self._recent_terms),
        )

    # -- STT hints (day 14 items 11-12) ------------------------------------

    def stt_prompt(self) -> Optional[str]:
        return build_stt_prompt(self.active_vocabulary(), max_terms=self.max_prompt_terms)

    def hotwords(self) -> Optional[str]:
        return build_hotwords(self.active_vocabulary(), max_hotwords=self.max_hotwords)
