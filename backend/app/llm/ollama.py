import logging
import time

import ollama

from ..config import OLLAMA_HOST, OLLAMA_MODEL
from .base import LLM

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama cannot be reached."""


class OllamaModelNotFoundError(Exception):
    """Raised when the configured model is not available in Ollama."""


class OllamaLLM(LLM):
    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)

    def chat(self, messages: list[dict]) -> str:
        logger.info("LLM request started (model=%s)", self.model)
        start = time.perf_counter()
        try:
            response = self.client.chat(model=self.model, messages=messages)
        except ollama.ResponseError as exc:
            elapsed = time.perf_counter() - start
            if exc.status_code == 404:
                logger.error("LLM request failed: model not found (%.2fs)", elapsed)
                raise OllamaModelNotFoundError(
                    f"Model '{self.model}' is not available. Run `ollama pull {self.model}`."
                ) from exc
            logger.error("LLM request failed: %s (%.2fs)", exc, elapsed)
            raise
        except ConnectionError as exc:
            elapsed = time.perf_counter() - start
            logger.error("LLM request failed: connection error (%.2fs)", elapsed)
            raise OllamaConnectionError(
                "Unable to connect to Ollama. Make sure Ollama is running."
            ) from exc

        elapsed = time.perf_counter() - start
        logger.info("LLM request completed")
        logger.info("LLM latency: %.2fs", elapsed)

        return response["message"]["content"]
