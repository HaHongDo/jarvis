from abc import ABC, abstractmethod

import ollama

from ..config import KNOWLEDGE_EMBEDDING_MODEL, OLLAMA_HOST

__all__ = ["Embedder", "OllamaEmbedder", "EmbeddingError"]


class EmbeddingError(Exception):
    """Raised when the embedding backend cannot be reached or fails."""


class Embedder(ABC):
    """Hides the specific embedding model behind a single interface (see Day 11 plan
    item 5: "Do not scatter model-specific code throughout the application")."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of document chunks."""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query (see Day 11 plan item 6: document vs query
        embeddings)."""
        return self.embed_documents([text])[0]


class OllamaEmbedder(Embedder):
    """Embeds text using a local Ollama embedding model (e.g. `nomic-embed-text`),
    keeping Jarvis fully local-first with no external embedding API."""

    def __init__(self, model: str = KNOWLEDGE_EMBEDDING_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embed(model=self.model, input=texts)
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise EmbeddingError(
                    f"Embedding model '{self.model}' is not available. Run `ollama pull {self.model}`."
                ) from exc
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc
        except ConnectionError as exc:
            raise EmbeddingError("Unable to connect to Ollama for embeddings.") from exc
        return [list(vector) for vector in response.embeddings]
