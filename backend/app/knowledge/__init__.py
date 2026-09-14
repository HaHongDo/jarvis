from .context import KnowledgeContextBuilder
from .embedder import Embedder, EmbeddingError, OllamaEmbedder
from .ingest import ingest_directory
from .models import Chunk, Document, IngestionReport, KnowledgeMatch
from .service import KnowledgeService
from .store import VectorStore

__all__ = [
    "KnowledgeContextBuilder",
    "Embedder",
    "EmbeddingError",
    "OllamaEmbedder",
    "ingest_directory",
    "Chunk",
    "Document",
    "IngestionReport",
    "KnowledgeMatch",
    "KnowledgeService",
    "VectorStore",
]
