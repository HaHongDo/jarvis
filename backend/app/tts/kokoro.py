import logging
import time
from typing import Optional

import numpy as np
from kokoro import KPipeline

from ..audio import AudioPlayer
from ..config import TTS_LANGUAGE, TTS_SPEED, TTS_VOICE
from .base import TextToSpeech
from .preprocessing import preprocess_for_speech

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# Kokoro's KPipeline takes a single-letter lang_code; config.yaml uses
# human-readable language tags instead.
_LANG_CODE_MAP = {
    "en-us": "a",
    "en-gb": "b",
}


def _resolve_lang_code(language: str) -> str:
    return _LANG_CODE_MAP.get(language.lower(), language)


class KokoroTTS(TextToSpeech):
    def __init__(
        self,
        language: str = TTS_LANGUAGE,
        voice: str = TTS_VOICE,
        speed: float = TTS_SPEED,
        player: Optional[AudioPlayer] = None,
    ):
        super().__init__(player)
        lang_code = _resolve_lang_code(language)
        logger.info("Loading Kokoro pipeline (lang_code=%s, voice=%s)", lang_code, voice)
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.speed = speed
        self.sample_rate = SAMPLE_RATE
        logger.info("Kokoro pipeline loaded")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        clean_text = preprocess_for_speech(text)
        if not clean_text:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        start = time.perf_counter()
        chunks = []
        for _graphemes, _phonemes, audio in self.pipeline(clean_text, voice=self.voice, speed=self.speed):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        elapsed = time.perf_counter() - start

        duration = len(audio) / self.sample_rate
        logger.info("[TTS] response: %d chars", len(clean_text))
        logger.info("[TTS] generation: %.2fs", elapsed)
        logger.info("[TTS] audio duration: %.2fs", duration)

        return audio, self.sample_rate


if __name__ == "__main__":
    import soundfile as sf

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tts = KokoroTTS()
    demo_audio, demo_sample_rate = tts.synthesize("Hello, I am Jarvis.")
    sf.write("hello.wav", demo_audio, demo_sample_rate)
    tts.player.play(demo_audio, demo_sample_rate)
