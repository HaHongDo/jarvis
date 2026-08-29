import logging

import numpy as np
from openwakeword.model import Model

from ..config import WAKEWORD_MODEL, WAKEWORD_THRESHOLD
from .base import WakeWordDetector

logger = logging.getLogger(__name__)


class OpenWakeWordDetector(WakeWordDetector):
    def __init__(self, model_name: str = WAKEWORD_MODEL, threshold: float = WAKEWORD_THRESHOLD):
        logger.info("Loading openWakeWord model '%s' (threshold=%.2f)", model_name, threshold)
        self.model = Model(wakeword_models=[model_name])
        self.model_name = model_name
        self.threshold = threshold
        logger.info("openWakeWord model loaded")

    def process(self, audio_frame: np.ndarray) -> bool:
        predictions = self.model.predict(audio_frame)
        score = max(predictions.values()) if predictions else 0.0
        detected = score >= self.threshold

        if detected:
            logger.info("[WAKE] '%s' detected (score=%.2f)", self.model_name, score)

        return detected

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()
