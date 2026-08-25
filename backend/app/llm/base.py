from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstraction over a chat-capable language model backend."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send a list of {"role", "content"} messages and return the assistant's reply."""
        raise NotImplementedError
