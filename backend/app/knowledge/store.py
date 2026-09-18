import json
import re
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from .models import Chunk, Document

__all__ = ["VectorStore"]

# Document columns a caller may filter retrieval on (see Day 12 plan items 13-14).
_FILTERABLE_COLUMNS = {"source_type", "document_type", "project", "owner_id"}


def _filter_clause(filters: Optional[dict], alias: str) -> tuple[str, list]:
    """Builds a parameterized `AND alias.col = ?` fragment from a filters dict, shared
    by the vector and keyword query paths so filters are applied in SQL before ranking
    rather than discarded after (see Day 12 plan item 14)."""
    if not filters:
        return "", []
    clauses = []
    params = []
    for key, value in filters.items():
        if key not in _FILTERABLE_COLUMNS:
            raise ValueError(f"Unknown filter field: {key!r}")
        clauses.append(f"{alias}.{key} = ?")
        params.append(value)
    return " AND " + " AND ".join(clauses), params


def _fts_query(query: str) -> str:
    """Turns free text into an FTS5 MATCH expression: each word-token is quoted (so
    punctuation in the query can't break FTS5's query syntax) and OR-ed together
    rather than the FTS5-default AND, since a natural-language question ("How does
    authz_version work?") would otherwise require every stopword to also appear in the
    chunk. `bm25()` still ranks chunks matching more/rarer terms higher."""
    tokens = re.findall(r"\w+", query)
    return " OR ".join(f'"{token}"' for token in tokens)


class VectorStore:
    """SQLite-backed store for documents, chunks, and embeddings, plus (Day 12) an
    FTS5 keyword index kept in sync alongside the chunks table.

    Day 11 plan: "For V1, don't deploy a distributed vector database." A local
    single-file store is plenty for a personal knowledge base, and similarity
    search here is a brute-force scan over `all_chunks_with_documents()` - which
    pgvector's own docs note is fine for small tables. Day 12 plan: SQLite has no
    tsvector/GIN index, so `chunks_fts` (FTS5 + `bm25()`) is the direct equivalent.
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
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text)"
            )
            self._migrate_documents_columns(conn)

    def _migrate_documents_columns(self, conn: sqlite3.Connection) -> None:
        """Adds Day 12's metadata columns to `documents` if they're missing, so an
        existing Day-11 `knowledge.db` keeps working without a manual migration step."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        for column in sorted(_FILTERABLE_COLUMNS - existing):
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT")

    def get_document_hash(self, document_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT content_hash FROM documents WHERE id = ?", (document_id,)).fetchone()
        return row[0] if row else None

    def upsert_document(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, path, title, content_hash, created_at, updated_at,
                    source_type, document_type, project, owner_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    title = excluded.title,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at,
                    source_type = excluded.source_type,
                    document_type = excluded.document_type,
                    project = excluded.project,
                    owner_id = excluded.owner_id
                """,
                (
                    document.id,
                    document.path,
                    document.title,
                    document.content_hash,
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                    document.source_type,
                    document.document_type,
                    document.project,
                    document.owner_id,
                ),
            )

    def delete_document_chunks(self, document_id: str) -> None:
        """Removes all chunks (and their FTS entries) for `document_id` so it can be
        re-chunked/re-embedded from scratch (see Day 11 plan item 11: delete stale
        chunks)."""
        with self._connect() as conn:
            chunk_ids = [
                row[0] for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
            ]
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(cid,) for cid in chunk_ids])

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
            conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                [(chunk.id, chunk.text) for chunk in chunks],
            )

    def delete_documents_not_in(self, document_ids: set[str]) -> int:
        """Removes documents (and their chunks/FTS entries) no longer present on disk,
        so deleted/moved files stop being searchable (see Day 11 plan item 11)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM documents").fetchall()
            stale_ids = [row[0] for row in rows if row[0] not in document_ids]
            for stale_id in stale_ids:
                chunk_ids = [
                    row[0] for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (stale_id,))
                ]
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (stale_id,))
                conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(cid,) for cid in chunk_ids])
                conn.execute("DELETE FROM documents WHERE id = ?", (stale_id,))
        return len(stale_ids)

    def all_chunks_with_documents(self, filters: Optional[dict] = None) -> list[tuple[Chunk, str, str]]:
        """Returns every chunk joined with its parent document's title/path, for
        brute-force similarity search plus provenance (see Day 11 plan items 8, 20).
        `filters` restricts to documents matching the given metadata columns,
        applied in SQL before the caller scores anything (see Day 12 plan item 14)."""
        clause, params = _filter_clause(filters, alias="d")
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.document_id, c.position, c.text, c.embedding, d.title, d.path
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE 1=1{clause}
                """,
                params,
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

    def search_keyword(
        self, query: str, limit: int, filters: Optional[dict] = None
    ) -> list[tuple[Chunk, str, str, float]]:
        """Full-text search over chunk text via SQLite FTS5 + `bm25()` - the SQLite
        equivalent of PostgreSQL `tsvector`/GIN + `ts_rank_cd` (see Day 12 plan items
        2-4). Returns `(chunk, title, path, score)` with higher score = better match
        (SQLite's raw `bm25()` is negative-is-better, so it's negated here to match
        cosine similarity's higher-is-better convention)."""
        fts_query = _fts_query(query)
        if not fts_query:
            return []

        clause, params = _filter_clause(filters, alias="d")
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.document_id, c.position, c.text, c.embedding, d.title, d.path,
                       bm25(chunks_fts) AS rank_score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ?{clause}
                ORDER BY rank_score
                LIMIT ?
                """,
                [fts_query, *params, limit],
            ).fetchall()

        results = []
        for chunk_id, document_id, position, text, embedding_json, title, path, rank_score in rows:
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=text,
                position=position,
                embedding=json.loads(embedding_json),
            )
            results.append((chunk, title, path, -rank_score))
        return results

    def document_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def chunk_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0
