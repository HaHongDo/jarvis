from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..audio import AudioPlayer


class TextToSpeech(ABC):
    """Abstraction over a text-to-speech backend."""

    def __init__(self, player: Optional[AudioPlayer] = None):
        self.player = player or AudioPlayer()

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert text to speech and return (audio_samples, sample_rate)."""
        raise NotImplementedError

    def speak(self, text: str) -> None:
        """Synthesize text and play it through the configured audio player."""
        audio, sample_rate = self.synthesize(text)
        self.player.play(audio, sample_rate)
