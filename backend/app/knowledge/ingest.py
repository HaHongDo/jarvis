import logging
from pathlib import Path
from typing import Optional

from ..config import KNOWLEDGE_CHUNK_OVERLAP, KNOWLEDGE_CHUNK_SIZE, KNOWLEDGE_DB_PATH
from .chunking import chunk_document
from .documents import load_documents
from .embedder import Embedder, EmbeddingError, OllamaEmbedder
from .models import IngestionReport
from .store import VectorStore

logger = logging.getLogger(__name__)

__all__ = ["ingest_directory"]


def ingest_directory(
    directory: Path,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    chunk_size: int = KNOWLEDGE_CHUNK_SIZE,
    chunk_overlap: int = KNOWLEDGE_CHUNK_OVERLAP,
) -> IngestionReport:
    """Runs the full ingestion pipeline for `directory` (see Day 11 plan items 9-11):

        find documents -> hash -> skip unchanged -> chunk -> embed -> store
                                                                          |
                                        delete stale documents no longer on disk
    """
    store = store or VectorStore(KNOWLEDGE_DB_PATH)
    embedder = embedder or OllamaEmbedder()
    report = IngestionReport()

    documents = load_documents(directory)
    seen_ids: set[str] = set()

    for document in documents:
        seen_ids.add(document.id)
        existing_hash = store.get_document_hash(document.id)
        if existing_hash == document.content_hash:
            report.documents_skipped += 1
            continue

        store.delete_document_chunks(document.id)
        chunks = chunk_document(document, chunk_size=chunk_size, overlap=chunk_overlap)

        if chunks:
            try:
                embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
            except EmbeddingError as exc:
                logger.warning("KNOWLEDGE ingest failed document=%r error=%s", document.path, exc)
                continue
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            store.insert_chunks(chunks)

        store.upsert_document(document)
        report.documents_indexed += 1
        report.chunks_indexed += len(chunks)
        report.per_document_chunks[document.path] = len(chunks)

    report.documents_removed = store.delete_documents_not_in(seen_ids)
    return report
