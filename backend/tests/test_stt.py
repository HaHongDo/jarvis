from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.stt import FasterWhisperSTT

AUDIO_DIR = Path(__file__).parent / "audio"


@pytest.fixture(scope="module")
def stt():
    return FasterWhisperSTT()


def _load_fixture(name):
    path = AUDIO_DIR / name
    if not path.exists():
        pytest.skip(f"missing fixture: {path}")
    audio, _sample_rate = sf.read(path, dtype="float32")
    return audio


def test_hello_wav_contains_hello(stt):
    audio = _load_fixture("hello.wav")

    text = stt.transcribe(audio)

    assert "hello" in text.lower()


def test_goroutine_wav_contains_goroutine(stt):
    audio = _load_fixture("goroutine.wav")

    text = stt.transcribe(audio)

    assert "goroutine" in text.lower()


def test_empty_wav_returns_no_text(stt):
    audio = _load_fixture("empty.wav")

    text = stt.transcribe(audio)

    assert text.strip() == ""


# ---------------------------------------------------------------------------
# Vocabulary hints (day 14 items 11-12). A fake model is used so the wiring can
# be tested without loading Whisper.
# ---------------------------------------------------------------------------


class _Segment:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    """Records the options it was called with. Its `transcribe` signature mirrors
    faster-whisper >= 1.0.2, i.e. it accepts `hotwords`."""

    def __init__(self, text="goroutine"):
        self.text = text
        self.calls = []

    def transcribe(self, audio, hotwords=None, **options):
        self.calls.append({"hotwords": hotwords, **options})
        return [_Segment(self.text)], None


class _OldFakeWhisperModel(_FakeWhisperModel):
    """faster-whisper before hotwords existed."""

    def transcribe(self, audio, **options):
        self.calls.append(options)
        return [_Segment(self.text)], None


def test_vocabulary_hints_are_passed_to_whisper():
    model = _FakeWhisperModel()
    stt = FasterWhisperSTT(model=model)

    stt.transcribe(
        np.zeros(16000, dtype="float32"),
        initial_prompt="The conversation is about Go programming.",
        hotwords="goroutine GOMAXPROCS",
    )

    [call] = model.calls
    assert call["initial_prompt"] == "The conversation is about Go programming."
    assert call["hotwords"] == "goroutine GOMAXPROCS"


def test_transcription_works_without_hints():
    model = _FakeWhisperModel()
    stt = FasterWhisperSTT(model=model)

    text = stt.transcribe(np.zeros(16000, dtype="float32"))

    [call] = model.calls
    assert text == "goroutine"
    assert call["initial_prompt"] is None
    assert "hotwords" not in call or call["hotwords"] is None


def test_hotwords_are_dropped_when_the_backend_does_not_support_them():
    model = _OldFakeWhisperModel()
    stt = FasterWhisperSTT(model=model)

    stt.transcribe(np.zeros(16000, dtype="float32"), initial_prompt="prompt", hotwords="goroutine")

    [call] = model.calls
    assert "hotwords" not in call
    assert call["initial_prompt"] == "prompt"
