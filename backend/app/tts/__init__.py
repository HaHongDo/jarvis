from .base import TextToSpeech

__all__ = ["TextToSpeech"]

try:
    from .kokoro import KokoroTTS

    __all__.append("KokoroTTS")
except ImportError:
    pass
