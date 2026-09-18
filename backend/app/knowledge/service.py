import logging
from typing import Optional

from ..config import (
    KNOWLEDGE_CANDIDATE_LIMIT,
    KNOWLEDGE_DB_PATH,
    KNOWLEDGE_MAX_CONTEXT_CHARS,
    KNOWLEDGE_RRF_K,
    KNOWLEDGE_TOP_K,
)
from .context import KnowledgeContextBuilder
from .embedder import Embedder, OllamaEmbedder
from .models import SearchResult
from .retrieval import HybridRetriever, KeywordRetriever, VectorRetriever
from .store import VectorStore

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeService"]


class KnowledgeService:
    """Orchestrates hybrid retrieval over the local knowledge base (see Day 12 plan):

        keyword search ---+
                           +--> Reciprocal Rank Fusion --> top-K matches --> context builder
        vector search  ---+

    This is the only entry point the `search_knowledge` tool calls - the LLM never
    sees embeddings, FTS, or the vector store directly (see Day 11 plan item 21:
    protect private knowledge; Day 12 plan item 22: keep retrieval implementation
    details out of the LLM/router).
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        context_builder: Optional[KnowledgeContextBuilder] = None,
        top_k: int = KNOWLEDGE_TOP_K,
        rrf_k: int = KNOWLEDGE_RRF_K,
        candidate_limit: int = KNOWLEDGE_CANDIDATE_LIMIT,
    ):
        self.store = store or VectorStore(KNOWLEDGE_DB_PATH)
        self.embedder = embedder or OllamaEmbedder()
        self.context_builder = context_builder or KnowledgeContextBuilder(KNOWLEDGE_MAX_CONTEXT_CHARS)
        self.top_k = top_k

        self.vector_retriever = VectorRetriever(self.store, self.embedder)
        self.keyword_retriever = KeywordRetriever(self.store)
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever, self.keyword_retriever, k=rrf_k, candidate_limit=candidate_limit
        )

    def search(
        self, query: str, top_k: Optional[int] = None, filters: Optional[dict] = None
    ) -> list[SearchResult]:
        top_k = top_k or self.top_k
        matches = self.hybrid_retriever.search(query, limit=top_k, filters=filters)

        logger.info(
            "KNOWLEDGE query=%r matches=%d top_score=%.3f",
            query,
            len(matches),
            matches[0].score if matches else 0.0,
        )
        return matches

    def build_context(self, query: str, matches: list[SearchResult]) -> str:
        return self.context_builder.build(query, matches)
