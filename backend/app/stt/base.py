from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SpeechToText(ABC):
    """Abstraction over a speech-to-text backend."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> str:
        """Transcribe mono audio samples and return plain text.

        `initial_prompt` and `hotwords` carry the conversation's active technical
        vocabulary into the model (Day 14) so terms like "goroutine" or
        "GOMAXPROCS" are more likely to be predicted correctly. Both are hints: a
        backend that can't use them should ignore them, and callers must not
        assume the terms will come back spelled right - the Speech Normalizer
        still runs afterwards.
        """
        raise NotImplementedError
