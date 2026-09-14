from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """A single ingested local knowledge document (Markdown/plain text), identified
    by its path relative to the knowledge directory. `content_hash` drives
    incremental ingestion: an unchanged hash means skip re-embedding (see Day 11
    plan: document ingestion pipeline + incremental ingestion)."""

    id: str
    path: str
    title: str
    content: str
    content_hash: str
    created_at: datetime
    updated_at: datetime


@dataclass
class Chunk:
    """A slice of a `Document`'s content. Generalizes Day 10's `SourceChunk` so the
    same chunk shape covers both web sources and local documents (see Day 11 plan
    item 4)."""

    id: str
    document_id: str
    text: str
    position: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class KnowledgeMatch:
    """A single scored retrieval result: a chunk plus its parent document's
    provenance (title/path) and a similarity score, so answers can cite "your
    notes" with a source path (see Day 11 plan items 13/20)."""

    chunk_id: str
    document_id: str
    title: str
    path: str
    text: str
    position: int
    score: float


@dataclass
class IngestionReport:
    """Summary of one `ingest_directory()` run (see Day 11 plan item 9)."""

    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_removed: int = 0
    chunks_indexed: int = 0
    per_document_chunks: dict = field(default_factory=dict)
