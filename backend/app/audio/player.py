import logging
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Plays mono audio samples through the default (or a selected) output device."""

    def __init__(self, device: Optional[int] = None):
        self.device = device

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio.size == 0:
            logger.warning("Nothing to play: empty audio")
            return
        sd.play(audio, samplerate=sample_rate, device=self.device)
        sd.wait()
