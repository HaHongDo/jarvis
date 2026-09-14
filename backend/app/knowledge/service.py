import logging
from typing import Optional

from ..config import KNOWLEDGE_DB_PATH, KNOWLEDGE_MAX_CONTEXT_CHARS, KNOWLEDGE_TOP_K
from .context import KnowledgeContextBuilder
from .embedder import Embedder, EmbeddingError, OllamaEmbedder
from .models import KnowledgeMatch
from .similarity import cosine_similarity
from .store import VectorStore

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeService"]


class KnowledgeService:
    """Orchestrates semantic search over the local knowledge base (see Day 11 plan):

        embed query -> brute-force cosine similarity over stored chunk embeddings
        -> top-K matches -> context builder

    This is the only entry point the `search_knowledge` tool calls - the LLM never
    sees embeddings or the vector store directly (see Day 11 plan item 21: protect
    private knowledge).
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        context_builder: Optional[KnowledgeContextBuilder] = None,
        top_k: int = KNOWLEDGE_TOP_K,
    ):
        self.store = store or VectorStore(KNOWLEDGE_DB_PATH)
        self.embedder = embedder or OllamaEmbedder()
        self.context_builder = context_builder or KnowledgeContextBuilder(KNOWLEDGE_MAX_CONTEXT_CHARS)
        self.top_k = top_k

    def search(self, query: str, top_k: Optional[int] = None) -> list[KnowledgeMatch]:
        top_k = top_k or self.top_k
        rows = self.store.all_chunks_with_documents()
        if not rows:
            return []

        try:
            query_embedding = self.embedder.embed_query(query)
        except EmbeddingError as exc:
            logger.warning("KNOWLEDGE embedding failed query=%r error=%s", query, exc)
            return []

        scored = [
            KnowledgeMatch(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                title=title,
                path=path,
                text=chunk.text,
                position=chunk.position,
                score=cosine_similarity(query_embedding, chunk.embedding),
            )
            for chunk, title, path in rows
        ]
        scored.sort(key=lambda match: match.score, reverse=True)
        top_matches = scored[:top_k]

        logger.info(
            "KNOWLEDGE query=%r matches=%d top_score=%.3f",
            query,
            len(top_matches),
            top_matches[0].score if top_matches else 0.0,
        )
        return top_matches

    def build_context(self, query: str, matches: list[KnowledgeMatch]) -> str:
        return self.context_builder.build(query, matches)
