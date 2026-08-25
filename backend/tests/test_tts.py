import pytest

from app.tts.preprocessing import preprocess_for_speech


def test_preprocessing_strips_inline_code():
    text = preprocess_for_speech("Use `fmt.Println()` to print a value.")

    assert text == "Use fmt.Println() to print a value."


def test_preprocessing_strips_bold_and_italic():
    text = preprocess_for_speech("This is **bold** and this is *italic*.")

    assert text == "This is bold and this is italic."


def test_preprocessing_strips_headers_and_links():
    text = preprocess_for_speech("# Title\nSee [the docs](https://example.com) for more.")

    assert text == "Title See the docs for more."


@pytest.fixture(scope="module")
def tts():
    pytest.importorskip("kokoro")
    from app.tts import KokoroTTS

    return KokoroTTS()


def test_synthesize_returns_nonempty_audio(tts):
    audio, sample_rate = tts.synthesize("Hello, I am Jarvis.")

    assert sample_rate == 24000
    assert len(audio) > 0
