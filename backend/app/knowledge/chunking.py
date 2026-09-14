from ..config import KNOWLEDGE_CHUNK_OVERLAP, KNOWLEDGE_CHUNK_SIZE
from ..research.chunking import chunk_text
from .models import Chunk, Document

__all__ = ["chunk_document"]


def chunk_document(
    document: Document, chunk_size: int = KNOWLEDGE_CHUNK_SIZE, overlap: int = KNOWLEDGE_CHUNK_OVERLAP
) -> list[Chunk]:
    """Chunks a `Document`'s content using the same character-window strategy as Day
    10's web-source chunking (see Day 11 plan item 4: generalize the chunk model -
    reuse the same abstraction for web content and local documents)."""
    pieces = chunk_text(document.content, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(id=f"{document.id}_chunk_{i}", document_id=document.id, text=piece, position=i)
        for i, piece in enumerate(pieces)
    ]
