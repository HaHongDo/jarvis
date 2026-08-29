from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """A complete (non-streamed) chat response: text content and/or requested tool calls."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class StreamEvent:
    """One increment of a streamed chat response.

    Either a text delta (content) or a batch of completed tool calls, never both -
    the underlying providers only emit tool calls once fully formed, so there is no
    partial tool-call fragment to accumulate.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLM(ABC):
    """Abstraction over a chat-capable language model backend."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        """Send messages (and optional tool definitions) and return the assistant's reply."""
        raise NotImplementedError

    @abstractmethod
    def stream_chat(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> Iterator[StreamEvent]:
        """Send messages and yield the assistant's reply as text deltas and/or tool calls."""
        raise NotImplementedError
