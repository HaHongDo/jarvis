import logging
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from ..config import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Captures mono audio from the default (or a selected) input device."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device if device is not None else self._default_input_device()

    @staticmethod
    def list_devices():
        return sd.query_devices()

    @staticmethod
    def _default_input_device() -> Optional[int]:
        try:
            device = sd.default.device[0]
        except Exception:
            logger.warning("Could not determine default input device")
            return None
        return device if device != -1 else None

    def record(self, duration_seconds: float) -> np.ndarray:
        """Record up to `duration_seconds` of mono audio and return it as float32 samples."""
        logger.info("[STT] recording started (max %.1fs, device=%s)", duration_seconds, self.device)
        frames = int(duration_seconds * self.sample_rate)

        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )
        sd.wait()

        audio = np.squeeze(audio)
        logger.info("[STT] recording: %.2fs", duration_seconds)
        return audio

    def save_wav(self, audio: np.ndarray, path: str) -> None:
        sf.write(path, audio, self.sample_rate)
