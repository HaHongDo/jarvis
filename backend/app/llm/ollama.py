import logging
import time
from typing import Iterator, Optional

import ollama

from ..config import OLLAMA_HOST, OLLAMA_MODEL
from .base import LLM, LLMResponse, StreamEvent, ToolCall

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama cannot be reached."""


class OllamaModelNotFoundError(Exception):
    """Raised when the configured model is not available in Ollama."""


def _to_tool_calls(message) -> list[ToolCall]:
    raw_calls = getattr(message, "tool_calls", None) or []
    return [ToolCall(name=call.function.name, arguments=dict(call.function.arguments)) for call in raw_calls]


class OllamaLLM(LLM):
    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        logger.info("LLM request started (model=%s)", self.model)
        start = time.perf_counter()
        try:
            response = self.client.chat(model=self.model, messages=messages, tools=tools)
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

        message = response["message"]
        return LLMResponse(content=message["content"] or "", tool_calls=_to_tool_calls(message))

    def stream_chat(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> Iterator[StreamEvent]:
        logger.info("LLM streaming request started (model=%s)", self.model)
        start = time.perf_counter()
        try:
            for chunk in self.client.chat(model=self.model, messages=messages, tools=tools, stream=True):
                content = chunk.message.content
                tool_calls = _to_tool_calls(chunk.message)
                if content or tool_calls:
                    yield StreamEvent(content=content or "", tool_calls=tool_calls)
        except ollama.ResponseError as exc:
            elapsed = time.perf_counter() - start
            if exc.status_code == 404:
                logger.error("LLM streaming failed: model not found (%.2fs)", elapsed)
                raise OllamaModelNotFoundError(
                    f"Model '{self.model}' is not available. Run `ollama pull {self.model}`."
                ) from exc
            logger.error("LLM streaming failed: %s (%.2fs)", exc, elapsed)
            raise
        except ConnectionError as exc:
            elapsed = time.perf_counter() - start
            logger.error("LLM streaming failed: connection error (%.2fs)", elapsed)
            raise OllamaConnectionError(
                "Unable to connect to Ollama. Make sure Ollama is running."
            ) from exc
        else:
            elapsed = time.perf_counter() - start
            logger.info("LLM streaming completed")
            logger.info("LLM latency: %.2fs", elapsed)
