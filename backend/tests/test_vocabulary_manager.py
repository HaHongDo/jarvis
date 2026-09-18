"""Day 14: context-aware vocabulary selection (see day_14_context_aware_speech_normalization.md)."""

import pytest

from app.speech import SpeechNormalizer, VocabularyManager, detect_domain, detect_topic_switch, score_domains
from app.speech.vocabulary_manager import MAX_SCORE_PER_UTTERANCE


@pytest.fixture
def clock():
    """A fake monotonic clock, so decay can be tested without waiting."""

    class _Clock:
        def __init__(self):
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    return _Clock()


@pytest.fixture
def manager(clock) -> VocabularyManager:
    return VocabularyManager(clock=clock)


# ---------------------------------------------------------------------------
# Keyword scoring (day 14 item 7)
# ---------------------------------------------------------------------------


def test_detect_domain_picks_the_topic_from_keywords():
    assert detect_domain("how does a goroutine talk to another goroutine") == "golang"
    assert detect_domain("the jvm warms up the jit compiler") == "java"
    assert detect_domain("asyncio runs the event loop") == "python"
    assert detect_domain("coopernetes reschedules the pod") == "infrastructure"


def test_detect_domain_returns_none_for_ordinary_speech():
    assert detect_domain("what time does the meeting start tomorrow") is None
    assert detect_domain("I need to buy groceries after work") is None


def test_single_ambiguous_word_does_not_pick_a_topic():
    """"channel" and "go" are technical terms *and* ordinary English, so one of
    them alone must not make Go the active topic."""
    assert detect_domain("please change the channel") is None
    assert detect_domain("I need to go home") is None


def test_score_is_capped_per_utterance():
    scores = score_domains("goroutine waitgroup gomaxprocs go scheduler go runtime golang")

    assert scores["golang"] == MAX_SCORE_PER_UTTERANCE


# ---------------------------------------------------------------------------
# Explicit topic switching (day 14 item 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("let's switch to Java", "java"),
        ("lets switch to java now", "java"),
        ("let's talk about Go concurrency", "golang"),
        ("moving on to Kubernetes", "infrastructure"),
        ("can we talk about distributed systems", "distributed_systems"),
        ("going back to python", "python"),
        ("tell me about oauth", "auth"),
    ],
)
def test_detect_topic_switch(text, expected):
    assert detect_topic_switch(text) == expected


def test_no_topic_switch_without_a_cue():
    assert detect_topic_switch("the goroutine blocks on a channel") is None
    assert detect_topic_switch("what time is it") is None


def test_switching_topic_replaces_the_active_vocabulary(manager):
    manager.observe("Let's talk about Go concurrency.")
    go_terms = manager.active_vocabulary().canonicals()

    manager.observe("Let's switch to Java.")
    java_terms = manager.active_vocabulary().canonicals()

    assert manager.context.active_domain == "java"
    assert "JVM" in java_terms and "heap" in java_terms
    assert "goroutine" in go_terms and "goroutine" not in java_terms


# ---------------------------------------------------------------------------
# Decay (day 14 item 10)
# ---------------------------------------------------------------------------


def test_topic_is_dropped_once_it_goes_stale(manager, clock):
    manager.observe("Let's talk about Go concurrency.")
    assert manager.active_domains() == ["golang"]

    clock.advance(130)  # past the 100% window, still relevant at 50%
    assert manager.active_domains() == ["golang"]

    clock.advance(600)  # down to 20% - a single switch no longer clears the bar
    assert manager.active_domains() == []


def test_a_heavily_discussed_topic_survives_longer(manager, clock):
    for _ in range(2):
        manager.observe("the goroutine and the waitgroup and gomaxprocs")

    clock.advance(700)  # 20% relevance, but off a much higher score

    assert manager.active_domains() == ["golang"]


def test_stale_topic_stops_hinting_stt(manager, clock):
    manager.observe("Let's talk about Go concurrency.")
    assert manager.stt_prompt() is not None

    clock.advance(2000)

    assert manager.stt_prompt() is None
    assert manager.hotwords() is None


# ---------------------------------------------------------------------------
# Active vocabulary (day 14 item 6)
# ---------------------------------------------------------------------------


def test_active_vocabulary_is_a_slice_not_the_whole_database(manager):
    manager.observe("Let's talk about Go concurrency.")

    vocabulary = manager.active_vocabulary()

    assert vocabulary.domains == ["golang"]
    assert "goroutine" in vocabulary.canonicals()
    assert "Kubernetes" not in vocabulary.canonicals()


def test_active_vocabulary_includes_cross_domain_terms(manager):
    """Kafka lives in "backend" but belongs in a distributed-systems conversation."""
    manager.observe("let's talk about distributed systems")

    canonicals = manager.active_vocabulary().canonicals()

    assert "Raft" in canonicals
    assert "Kafka" in canonicals


def test_active_domains_are_capped(clock):
    manager = VocabularyManager(max_active_domains=2, clock=clock)

    manager.observe("goroutine gomaxprocs")
    manager.observe("the jvm and the jit compiler")
    manager.observe("coopernetes and dockerfile and terraform")

    assert len(manager.active_domains()) == 2


def test_recent_terms_are_remembered_in_order(manager):
    manager.observe("the goroutine writes to a channel")

    assert manager.context.recent_terms[:2] == ["goroutine", "channel"]


def test_corrected_terms_become_recent_terms(manager):
    normalizer = SpeechNormalizer()
    context = manager.observe("how does a gore teen work")

    result = normalizer.normalize("how does a gore teen work", conversation_context=context)
    manager.observe_result(result)

    assert manager.context.recent_terms[0] == "goroutine"


# ---------------------------------------------------------------------------
# STT hints (day 14 items 11-12)
# ---------------------------------------------------------------------------


def test_stt_prompt_names_the_topic_and_the_vocabulary(manager):
    manager.observe("Let's talk about Go concurrency.")

    prompt = manager.stt_prompt()

    assert prompt.startswith("The conversation is about Go programming.")
    assert "goroutine" in prompt
    assert "GOMAXPROCS" in prompt


def test_stt_prompt_is_none_without_a_topic(manager):
    assert manager.stt_prompt() is None
    assert manager.hotwords() is None

    manager.observe("what time does the meeting start")

    assert manager.stt_prompt() is None


def test_hotwords_only_include_distinctive_terms(manager):
    manager.observe("Let's talk about Go concurrency.")

    hotwords = manager.hotwords().split()

    assert "goroutine" in hotwords
    assert "GOMAXPROCS" in hotwords
    # Ordinary English words Whisper already spells correctly are not worth a hint.
    assert "channel" not in hotwords
    assert "select" not in hotwords


def test_prompt_term_budget_is_respected(clock):
    manager = VocabularyManager(max_prompt_terms=5, clock=clock)
    manager.observe("Let's talk about Go concurrency.")

    terms = manager.stt_prompt().split("Technical terms: ")[1].rstrip(".").split(", ")

    assert len(terms) == 5


# ---------------------------------------------------------------------------
# Day 14 Definition of Done: conversation context drives the normalizer
# ---------------------------------------------------------------------------


def test_conversation_context_enables_a_context_dependent_correction(manager):
    normalizer = SpeechNormalizer()
    sentence = "how does the wait group know when to stop"

    without_context = normalizer.normalize(sentence)

    manager.observe("Let's talk about Go concurrency.")
    with_context = normalizer.normalize(sentence, conversation_context=manager.context)

    assert without_context.normalized == sentence
    assert with_context.normalized == "how does the WaitGroup know when to stop"


def test_switching_topic_switches_which_corrections_apply(manager):
    normalizer = SpeechNormalizer()
    sentence = "the wait group blocks until done"

    manager.observe("Let's talk about Go concurrency.")
    assert normalizer.normalize(sentence, conversation_context=manager.context).normalized == (
        "the WaitGroup blocks until done"
    )

    manager.observe("Let's switch to Java.")
    assert normalizer.normalize(sentence, conversation_context=manager.context).normalized == sentence


def test_same_mishearing_resolves_differently_per_topic(manager):
    """"co routine" is a Python coroutine; in a Go conversation it is not a term
    the active vocabulary supports, so it is preserved rather than guessed at
    (day 14 item 17)."""
    normalizer = SpeechNormalizer()
    sentence = "the co routine yields control"

    manager.observe("let's talk about python")
    assert normalizer.normalize(sentence, conversation_context=manager.context).normalized == (
        "the coroutine yields control"
    )

    manager.observe("let's switch to go")
    assert normalizer.normalize(sentence, conversation_context=manager.context).normalized == sentence


def test_full_turn_matches_the_day_14_definition_of_done(manager):
    """Go topic established in one turn, mishearing corrected in the next."""
    normalizer = SpeechNormalizer()

    manager.observe("Let's talk about Go concurrency.")
    assert manager.context.active_domain == "golang"

    raw = "How does a gore teen communicate with another gore teen?"
    context = manager.observe(raw)
    result = normalizer.normalize(raw, conversation_context=context)

    assert result.normalized == "How does a goroutine communicate with another goroutine?"
    assert result.original == raw
