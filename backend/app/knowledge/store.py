import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from .models import Chunk, Document

__all__ = ["VectorStore"]


class VectorStore:
    """SQLite-backed store for documents, chunks, and embeddings.

    Day 11 plan: "For V1, don't deploy a distributed vector database." A local
    single-file store is plenty for a personal knowledge base, and similarity
    search here is a brute-force scan over `all_chunks_with_documents()` - which
    pgvector's own docs note is fine for small tables.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
                """
            )

    def get_document_hash(self, document_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT content_hash FROM documents WHERE id = ?", (document_id,)).fetchone()
        return row[0] if row else None

    def upsert_document(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, path, title, content_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    title = excluded.title,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    document.id,
                    document.path,
                    document.title,
                    document.content_hash,
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                ),
            )

    def delete_document_chunks(self, document_id: str) -> None:
        """Removes all chunks for `document_id` so it can be re-chunked/re-embedded
        from scratch (see Day 11 plan item 11: delete stale chunks)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chunks (id, document_id, position, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [
                    (chunk.id, chunk.document_id, chunk.position, chunk.text, json.dumps(chunk.embedding))
                    for chunk in chunks
                ],
            )

    def delete_documents_not_in(self, document_ids: set[str]) -> int:
        """Removes documents (and their chunks) no longer present on disk, so
        deleted/moved files stop being searchable (see Day 11 plan item 11)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM documents").fetchall()
            stale_ids = [row[0] for row in rows if row[0] not in document_ids]
            for stale_id in stale_ids:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (stale_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (stale_id,))
        return len(stale_ids)

    def all_chunks_with_documents(self) -> list[tuple[Chunk, str, str]]:
        """Returns every chunk joined with its parent document's title/path, for
        brute-force similarity search plus provenance (see Day 11 plan items 8, 20)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.document_id, c.position, c.text, c.embedding, d.title, d.path
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                """
            ).fetchall()

        results = []
        for chunk_id, document_id, position, text, embedding_json, title, path in rows:
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=text,
                position=position,
                embedding=json.loads(embedding_json),
            )
            results.append((chunk, title, path))
        return results

    def document_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def chunk_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0
