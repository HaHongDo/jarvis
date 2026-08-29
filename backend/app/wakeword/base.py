from abc import ABC, abstractmethod

import numpy as np


class WakeWordDetector(ABC):
    """Abstraction over a wake-word detection backend."""

    @abstractmethod
    def process(self, audio_frame: np.ndarray) -> bool:
        """Process one int16 PCM audio frame and return True if the wake word was detected."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal state so audio from a previous activation doesn't linger."""
