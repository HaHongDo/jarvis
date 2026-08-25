from pathlib import Path

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
