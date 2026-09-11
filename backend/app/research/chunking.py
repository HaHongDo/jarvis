from ..config import RESEARCH_CHUNK_OVERLAP, RESEARCH_CHUNK_SIZE
from .models import Source, SourceChunk


def chunk_text(text: str, chunk_size: int = RESEARCH_CHUNK_SIZE, overlap: int = RESEARCH_CHUNK_OVERLAP) -> list[str]:
    """Splits `text` into overlapping character windows. Simple character-based
    chunking is sufficient for V1 (see Day 10 plan: no embeddings/semantic chunking yet)."""
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= chunk_size:
        return [stripped]

    step = max(chunk_size - overlap, 1)
    chunks = []
    start = 0
    length = len(stripped)
    while start < length:
        end = min(start + chunk_size, length)
        piece = stripped[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start += step
    return chunks


def chunk_source(
    source: Source, chunk_size: int = RESEARCH_CHUNK_SIZE, overlap: int = RESEARCH_CHUNK_OVERLAP
) -> list[SourceChunk]:
    """Chunks a `Source`'s content and tags each piece with source/position metadata
    so it can be traced back and re-ordered after context-budget selection."""
    pieces = chunk_text(source.content, chunk_size=chunk_size, overlap=overlap)
    return [
        SourceChunk(source_id=source.id, chunk_id=f"{source.id}_chunk_{i}", text=piece, position=i)
        for i, piece in enumerate(pieces)
    ]
