import logging
import time

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


class FasterWhisperSTT(SpeechToText):
    def __init__(
        self,
        model_size: str = STT_MODEL,
        device: str = STT_DEVICE,
        compute_type: str = STT_COMPUTE_TYPE,
        language: str = STT_LANGUAGE,
        vad_filter: bool = STT_VAD_FILTER,
    ):
        logger.info("Loading Whisper model '%s' (device=%s, compute_type=%s)", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        self.vad_filter = vad_filter
        logger.info("Whisper model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        start = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=self.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500) if self.vad_filter else None,
        )
        text = " ".join(segment.text for segment in segments).strip()

        elapsed = time.perf_counter() - start
        logger.info("[STT] processing: %.2fs", elapsed)
        logger.info('[STT] text: "%s"', text)

        return text
