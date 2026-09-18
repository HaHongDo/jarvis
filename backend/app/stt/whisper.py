import inspect
import logging
import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from ..config import (
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_LANGUAGE,
    STT_MODEL,
    STT_VAD_FILTER,
)
from .base import SpeechToText

logger = logging.getLogger(__name__)


def _accepts_hotwords(transcribe) -> bool:
    """faster-whisper only grew `hotwords` in 1.0.2; older builds raise TypeError
    if it is passed. Checked once per instance rather than assumed."""
    try:
        return "hotwords" in inspect.signature(transcribe).parameters
    except (TypeError, ValueError):  # pragma: no cover - C extensions without signatures
        return False


class FasterWhisperSTT(SpeechToText):
    def __init__(
        self,
        model_size: str = STT_MODEL,
        device: str = STT_DEVICE,
        compute_type: str = STT_COMPUTE_TYPE,
        language: str = STT_LANGUAGE,
        vad_filter: bool = STT_VAD_FILTER,
        model=None,
    ):
        if model is None:
            logger.info("Loading Whisper model '%s' (device=%s, compute_type=%s)", model_size, device, compute_type)
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("Whisper model loaded")
        self.model = model
        self.language = language
        self.vad_filter = vad_filter
        self._supports_hotwords = _accepts_hotwords(self.model.transcribe)
        self._warned_about_hotwords = False

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> str:
        """Transcribe `audio`, optionally biased toward the conversation's active
        technical vocabulary (Day 14 items 11-12).

        Whisper conditions on `initial_prompt` as if it were the preceding
        transcript, which is what makes it a place to name custom vocabulary and
        proper nouns. `hotwords` is faster-whisper's own hint mechanism, treated
        here as an experiment rather than a guarantee - measure it with
        `python -m app.speech.evaluate_stt` before trusting it.
        """
        start = time.perf_counter()

        options = dict(
            language=self.language,
            vad_filter=self.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500) if self.vad_filter else None,
            initial_prompt=initial_prompt,
        )
        if hotwords:
            if self._supports_hotwords:
                options["hotwords"] = hotwords
            elif not self._warned_about_hotwords:
                logger.warning("[STT] installed faster-whisper does not support hotwords, ignoring them")
                self._warned_about_hotwords = True

        segments, _info = self.model.transcribe(audio, **options)
        text = " ".join(segment.text for segment in segments).strip()

        elapsed = time.perf_counter() - start
        logger.info("[STT] processing: %.2fs", elapsed)
        if initial_prompt:
            logger.debug('[STT] vocabulary prompt: "%s"', initial_prompt)
        if hotwords and self._supports_hotwords:
            logger.debug('[STT] hotwords: "%s"', hotwords)
        logger.info('[STT] text: "%s"', text)

        return text
