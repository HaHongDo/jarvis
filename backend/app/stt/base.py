from abc import ABC, abstractmethod

import numpy as np


class SpeechToText(ABC):
    """Abstraction over a speech-to-text backend."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe mono audio samples and return plain text."""
        raise NotImplementedError
