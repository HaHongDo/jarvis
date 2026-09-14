from datetime import datetime, timezone
from pathlib import Path

from app.knowledge.chunking import chunk_document
from app.knowledge.context import KnowledgeContextBuilder
from app.knowledge.documents import load_documents
from app.knowledge.ingest import ingest_directory
from app.knowledge.models import Chunk, Document, KnowledgeMatch
from app.knowledge.service import KnowledgeService
from app.knowledge.similarity import cosine_similarity
from app.knowledge.store import VectorStore

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def _make_doc(**overrides) -> Document:
    now = datetime.now(timezone.utc)
    defaults = dict(
        id="notes/a.md",
        path="notes/a.md",
        title="A",
        content="hello world",
        content_hash="hash-a",
        created_at=now,
        updated_at=now,
    )
    defaults.update(overrides)
    return Document(**defaults)


def test_load_documents_reads_markdown_and_txt(tmp_path: Path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "redis.md").write_text("# Redis Notes\n\nSome content here.")
    (tmp_path / "plain.txt").write_text("plain text content")
    (tmp_path / "ignored.json").write_text("{}")

    documents = load_documents(tmp_path)

    paths = sorted(d.path for d in documents)
    assert paths == ["notes/redis.md", "plain.txt"]


def test_load_documents_extracts_title_from_heading(tmp_path: Path):
    (tmp_path / "a.md").write_text("# My Title\n\nBody text.")

    [document] = load_documents(tmp_path)

    assert document.title == "My Title"


def test_load_documents_falls_back_to_filename_when_no_heading(tmp_path: Path):
    (tmp_path / "no_heading.md").write_text("just some text, no heading")

    [document] = load_documents(tmp_path)

    assert document.title == "no_heading"


def test_load_documents_returns_empty_list_for_missing_directory(tmp_path: Path):
    assert load_documents(tmp_path / "does-not-exist") == []


def test_load_documents_hash_changes_when_content_changes(tmp_path: Path):
    file_path = tmp_path / "a.md"
    file_path.write_text("version one")
    [doc_v1] = load_documents(tmp_path)

    file_path.write_text("version two")
    [doc_v2] = load_documents(tmp_path)

    assert doc_v1.content_hash != doc_v2.content_hash


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_document_tags_chunks_with_document_metadata():
    document = _make_doc(id="doc_1", content="x" * 90)

    chunks = chunk_document(document, chunk_size=40, overlap=10)

    assert all(c.document_id == "doc_1" for c in chunks)
    assert [c.id for c in chunks] == [f"doc_1_chunk_{i}" for i in range(len(chunks))]
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunk_document_returns_single_chunk_when_under_size():
    document = _make_doc(content="hello world")

    chunks = chunk_document(document, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_handles_empty_or_mismatched_vectors():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def test_vector_store_upsert_and_get_document_hash(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    document = _make_doc()

    assert store.get_document_hash(document.id) is None

    store.upsert_document(document)

    assert store.get_document_hash(document.id) == "hash-a"


def test_vector_store_insert_and_query_chunks_with_documents(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    document = _make_doc()
    store.upsert_document(document)
    chunk = Chunk(id="doc_1_chunk_0", document_id=document.id, text="hello", position=0, embedding=[0.1, 0.2])
    store.insert_chunks([chunk])

    rows = store.all_chunks_with_documents()

    assert len(rows) == 1
    stored_chunk, title, path = rows[0]
    assert stored_chunk.text == "hello"
    assert stored_chunk.embedding == [0.1, 0.2]
    assert title == document.title
    assert path == document.path


def test_vector_store_delete_document_chunks(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    document = _make_doc()
    store.upsert_document(document)
    store.insert_chunks([Chunk(id="c0", document_id=document.id, text="x", position=0, embedding=[1.0])])

    store.delete_document_chunks(document.id)

    assert store.all_chunks_with_documents() == []


def test_vector_store_delete_documents_not_in_removes_stale_entries(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    keep = _make_doc(id="keep.md", path="keep.md")
    stale = _make_doc(id="stale.md", path="stale.md")
    store.upsert_document(keep)
    store.upsert_document(stale)
    store.insert_chunks([Chunk(id="stale_chunk_0", document_id="stale.md", text="x", position=0, embedding=[1.0])])

    removed = store.delete_documents_not_in({"keep.md"})

    assert removed == 1
    assert store.get_document_hash("stale.md") is None
    assert store.get_document_hash("keep.md") == "hash-a"


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _make_match(**overrides) -> KnowledgeMatch:
    defaults = dict(
        chunk_id="c0", document_id="d0", title="Redis Notes", path="notes/redis.md", text="x" * 50, position=0,
        score=0.9,
    )
    defaults.update(overrides)
    return KnowledgeMatch(**defaults)


def test_context_builder_empty_matches_returns_no_results_message():
    text = KnowledgeContextBuilder(max_context_chars=1000).build("rate limiting", [])

    assert "No relevant information" in text


def test_context_builder_includes_title_path_and_score():
    match = _make_match()

    text = KnowledgeContextBuilder(max_context_chars=1000).build("rate limiting", [match])

    assert "Redis Notes" in text
    assert "notes/redis.md" in text
    assert "0.90" in text


def test_context_builder_respects_budget():
    matches = [_make_match(chunk_id="c0", text="x" * 50), _make_match(chunk_id="c1", text="y" * 50)]

    text = KnowledgeContextBuilder(max_context_chars=60).build("q", matches)

    assert "x" * 50 in text
    assert "y" * 50 not in text


# ---------------------------------------------------------------------------
# KnowledgeService (stubbed embedder/store)
# ---------------------------------------------------------------------------


class StubEmbedder:
    def __init__(self, query_vector):
        self.query_vector = query_vector

    def embed_query(self, text):
        return self.query_vector


def test_knowledge_service_ranks_by_similarity(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    doc = _make_doc()
    store.upsert_document(doc)
    store.insert_chunks(
        [
            Chunk(id="c_near", document_id=doc.id, text="near match", position=0, embedding=[1.0, 0.0]),
            Chunk(id="c_far", document_id=doc.id, text="far match", position=1, embedding=[0.0, 1.0]),
        ]
    )
    service = KnowledgeService(store=store, embedder=StubEmbedder([1.0, 0.0]), top_k=5)

    matches = service.search("query")

    assert [m.chunk_id for m in matches] == ["c_near", "c_far"]
    assert matches[0].score > matches[1].score


def test_knowledge_service_returns_empty_when_no_chunks_indexed(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    service = KnowledgeService(store=store, embedder=StubEmbedder([1.0, 0.0]))

    assert service.search("anything") == []


# ---------------------------------------------------------------------------
# Ingestion pipeline (stubbed embedder)
# ---------------------------------------------------------------------------


class StubIngestEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t))] for t in texts]


def test_ingest_directory_indexes_new_documents(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "a.md").write_text("Some content about rate limiting.")
    store = VectorStore(str(tmp_path / "kb.db"))
    embedder = StubIngestEmbedder()

    report = ingest_directory(knowledge_dir, store=store, embedder=embedder)

    assert report.documents_indexed == 1
    assert report.documents_skipped == 0
    assert report.chunks_indexed >= 1
    assert embedder.calls == 1


def test_ingest_directory_skips_unchanged_documents(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "a.md").write_text("stable content")
    store = VectorStore(str(tmp_path / "kb.db"))
    embedder = StubIngestEmbedder()

    ingest_directory(knowledge_dir, store=store, embedder=embedder)
    report = ingest_directory(knowledge_dir, store=store, embedder=embedder)

    assert report.documents_indexed == 0
    assert report.documents_skipped == 1
    assert embedder.calls == 1  # only the first run embedded anything


def test_ingest_directory_reindexes_changed_documents(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    file_path = knowledge_dir / "a.md"
    file_path.write_text("version one")
    store = VectorStore(str(tmp_path / "kb.db"))
    embedder = StubIngestEmbedder()
    ingest_directory(knowledge_dir, store=store, embedder=embedder)

    file_path.write_text("version two, much longer content than before")
    report = ingest_directory(knowledge_dir, store=store, embedder=embedder)

    assert report.documents_indexed == 1
    assert report.documents_skipped == 0
    assert embedder.calls == 2


def test_ingest_directory_removes_stale_documents(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    keep_path = knowledge_dir / "keep.md"
    stale_path = knowledge_dir / "stale.md"
    keep_path.write_text("keep me")
    stale_path.write_text("delete me")
    store = VectorStore(str(tmp_path / "kb.db"))
    embedder = StubIngestEmbedder()
    ingest_directory(knowledge_dir, store=store, embedder=embedder)

    stale_path.unlink()
    report = ingest_directory(knowledge_dir, store=store, embedder=embedder)

    assert report.documents_removed == 1
    assert store.get_document_hash("stale.md") is None
    assert store.get_document_hash("keep.md") is not None
