import logging
from typing import Optional

from .embedder import Embedder, EmbeddingError
from .models import SearchResult
from .similarity import cosine_similarity
from .store import VectorStore

logger = logging.getLogger(__name__)

__all__ = ["VectorRetriever", "KeywordRetriever", "HybridRetriever", "reciprocal_rank_fusion"]


class VectorRetriever:
    """Semantic search: embed the query, brute-force cosine similarity over stored
    chunk embeddings (see Day 11 plan)."""

    source = "vector"

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def search(self, query: str, limit: int, filters: Optional[dict] = None) -> list[SearchResult]:
        rows = self.store.all_chunks_with_documents(filters)
        if not rows:
            return []

        try:
            query_embedding = self.embedder.embed_query(query)
        except EmbeddingError as exc:
            logger.warning("KNOWLEDGE vector search embedding failed query=%r error=%s", query, exc)
            return []

        scored = sorted(
            rows,
            key=lambda row: cosine_similarity(query_embedding, row[0].embedding),
            reverse=True,
        )[:limit]

        return [
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                title=title,
                path=path,
                content=chunk.text,
                position=chunk.position,
                score=cosine_similarity(query_embedding, chunk.embedding),
                rank=rank,
                source=self.source,
            )
            for rank, (chunk, title, path) in enumerate(scored, start=1)
        ]


class KeywordRetriever:
    """Exact-terminology search over chunk text via SQLite FTS5 + `bm25()` - the
    SQLite equivalent of PostgreSQL full-text search (see Day 12 plan items 2-4)."""

    source = "keyword"

    def __init__(self, store: VectorStore):
        self.store = store

    def search(self, query: str, limit: int, filters: Optional[dict] = None) -> list[SearchResult]:
        rows = self.store.search_keyword(query, limit=limit, filters=filters)
        return [
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                title=title,
                path=path,
                content=chunk.text,
                position=chunk.position,
                score=score,
                rank=rank,
                source=self.source,
            )
            for rank, (chunk, title, path, score) in enumerate(rows, start=1)
        ]


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Combines multiple ranked result lists into one, per Day 12 plan items 7-8:

        RRF score = sum over lists of 1 / (k + rank)

    Ranks (not raw scores) are combined since vector cosine similarity and keyword
    bm25 scores use different scales (see Day 12 plan item 6). Keying by `chunk_id`
    also deduplicates: a chunk returned by multiple lists is merged into one result
    with a summed score (see Day 12 plan item 20), using the first list's content/
    provenance as the merged result's content/provenance.
    """
    scores: dict[str, float] = {}
    merged: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (k + rank)
            merged.setdefault(result.chunk_id, result)

    ordered_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)

    return [
        SearchResult(
            chunk_id=merged[chunk_id].chunk_id,
            document_id=merged[chunk_id].document_id,
            title=merged[chunk_id].title,
            path=merged[chunk_id].path,
            content=merged[chunk_id].content,
            position=merged[chunk_id].position,
            score=scores[chunk_id],
            rank=rank,
            source="hybrid",
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


class HybridRetriever:
    """Combines `VectorRetriever` and `KeywordRetriever` via Reciprocal Rank Fusion
    (see Day 12 plan item 9). Retrieves `candidate_limit` candidates from each
    retriever - more than `limit` - so the fusion stage has enough overlap to work
    with."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        k: int = 60,
        candidate_limit: int = 20,
    ):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.k = k
        self.candidate_limit = candidate_limit

    def search(self, query: str, limit: int, filters: Optional[dict] = None) -> list[SearchResult]:
        vector_results = self.vector_retriever.search(query, limit=self.candidate_limit, filters=filters)
        keyword_results = self.keyword_retriever.search(query, limit=self.candidate_limit, filters=filters)
        fused = reciprocal_rank_fusion([vector_results, keyword_results], k=self.k)
        return fused[:limit]
