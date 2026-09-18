import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.knowledge.chunking import chunk_document
from app.knowledge.context import KnowledgeContextBuilder
from app.knowledge.documents import load_documents
from app.knowledge.evaluation import EvalCase, evaluate_retriever
from app.knowledge.ingest import ingest_directory
from app.knowledge.models import Chunk, Document, SearchResult
from app.knowledge.retrieval import HybridRetriever, KeywordRetriever, VectorRetriever, reciprocal_rank_fusion
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


def test_load_documents_parses_frontmatter_metadata(tmp_path: Path):
    (tmp_path / "a.md").write_text(
        "---\nproject: jarvis\nsource_type: note\ndocument_type: architecture\nowner_id: u1\n"
        "---\n# My Title\n\nBody text."
    )

    [document] = load_documents(tmp_path)

    assert document.project == "jarvis"
    assert document.source_type == "note"
    assert document.document_type == "architecture"
    assert document.owner_id == "u1"
    assert document.title == "My Title"
    assert document.content == "# My Title\n\nBody text."


def test_load_documents_without_frontmatter_has_no_metadata(tmp_path: Path):
    (tmp_path / "a.md").write_text("# My Title\n\nBody text.")

    [document] = load_documents(tmp_path)

    assert document.project is None
    assert document.source_type is None
    assert document.content == "# My Title\n\nBody text."


def test_load_documents_ignores_malformed_frontmatter(tmp_path: Path):
    (tmp_path / "a.md").write_text("---\nthis: [is, not: valid\n---\nBody text.")

    [document] = load_documents(tmp_path)

    assert document.project is None
    assert document.content.startswith("---")


def test_load_documents_hash_changes_when_only_frontmatter_changes(tmp_path: Path):
    file_path = tmp_path / "a.md"
    file_path.write_text("---\nproject: a\n---\nBody text.")
    [doc_v1] = load_documents(tmp_path)

    file_path.write_text("---\nproject: b\n---\nBody text.")
    [doc_v2] = load_documents(tmp_path)

    assert doc_v1.content_hash != doc_v2.content_hash
    assert doc_v1.content == doc_v2.content == "Body text."


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
    assert store.search_keyword("x", limit=5) == []


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


def test_vector_store_search_keyword_finds_exact_term(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    document = _make_doc()
    store.upsert_document(document)
    store.insert_chunks(
        [
            Chunk(
                id="c_exact",
                document_id=document.id,
                text="The payment service uses authz_version to invalidate stale tokens.",
                position=0,
                embedding=[1.0],
            ),
            Chunk(
                id="c_other",
                document_id=document.id,
                text="General overview of the API gateway.",
                position=1,
                embedding=[1.0],
            ),
        ]
    )

    results = store.search_keyword("authz_version", limit=5)

    assert [chunk.id for chunk, _, _, _ in results] == ["c_exact"]


def test_vector_store_search_keyword_returns_empty_for_no_matches(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    document = _make_doc()
    store.upsert_document(document)
    store.insert_chunks([Chunk(id="c0", document_id=document.id, text="hello world", position=0, embedding=[1.0])])

    assert store.search_keyword("nonexistent_term_xyz", limit=5) == []


def test_vector_store_all_chunks_with_documents_applies_filters(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    doc_a = _make_doc(id="a.md", path="a.md", project="jarvis")
    doc_b = _make_doc(id="b.md", path="b.md", project="other")
    store.upsert_document(doc_a)
    store.upsert_document(doc_b)
    store.insert_chunks(
        [
            Chunk(id="ca", document_id="a.md", text="x", position=0, embedding=[1.0]),
            Chunk(id="cb", document_id="b.md", text="x", position=0, embedding=[1.0]),
        ]
    )

    rows = store.all_chunks_with_documents(filters={"project": "jarvis"})

    assert [chunk.id for chunk, _, _ in rows] == ["ca"]


def test_vector_store_search_keyword_applies_filters(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    doc_a = _make_doc(id="a.md", path="a.md", project="jarvis")
    doc_b = _make_doc(id="b.md", path="b.md", project="other")
    store.upsert_document(doc_a)
    store.upsert_document(doc_b)
    store.insert_chunks(
        [
            Chunk(id="ca", document_id="a.md", text="rate limiting notes", position=0, embedding=[1.0]),
            Chunk(id="cb", document_id="b.md", text="rate limiting notes", position=0, embedding=[1.0]),
        ]
    )

    results = store.search_keyword("rate limiting", limit=5, filters={"project": "jarvis"})

    assert [chunk.id for chunk, _, _, _ in results] == ["ca"]


def test_vector_store_migrates_legacy_documents_table(tmp_path: Path):
    """Simulates opening a Day-11 `knowledge.db` that predates the Day 12 metadata
    columns - `VectorStore` should add them without erroring."""
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    store = VectorStore(db_path)
    document = _make_doc(project="jarvis")
    store.upsert_document(document)

    assert store.get_document_hash(document.id) == document.content_hash
    rows = store.all_chunks_with_documents(filters={"project": "jarvis"})
    assert rows == []  # no chunks inserted, but the filter query itself must not error


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> SearchResult:
    defaults = dict(
        chunk_id="c0", document_id="d0", title="Redis Notes", path="notes/redis.md", content="x" * 50,
        position=0, score=0.9, rank=1, source="hybrid",
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


def test_context_builder_empty_matches_returns_no_results_message():
    text = KnowledgeContextBuilder(max_context_chars=1000).build("rate limiting", [])

    assert "No relevant information" in text


def test_context_builder_includes_title_path_and_score():
    match = _make_result()

    text = KnowledgeContextBuilder(max_context_chars=1000).build("rate limiting", [match])

    assert "Redis Notes" in text
    assert "notes/redis.md" in text
    assert "0.90" in text


def test_context_builder_respects_budget():
    matches = [_make_result(chunk_id="c0", content="x" * 50), _make_result(chunk_id="c1", content="y" * 50)]

    text = KnowledgeContextBuilder(max_context_chars=60).build("q", matches)

    assert "x" * 50 in text
    assert "y" * 50 not in text


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _sr(chunk_id: str, rank: int, source: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, document_id=chunk_id, title="t", path=f"{chunk_id}.md", content="c",
        position=0, score=0.0, rank=rank, source=source,
    )


def test_reciprocal_rank_fusion_combines_ranks_not_raw_scores():
    vector_results = [_sr("A", 1, "vector"), _sr("B", 2, "vector"), _sr("C", 3, "vector")]
    keyword_results = [_sr("C", 1, "keyword"), _sr("A", 2, "keyword"), _sr("D", 3, "keyword")]

    fused = reciprocal_rank_fusion([vector_results, keyword_results], k=60)

    # A and C each rank in both lists, so they outrank B and D (which rank in only one).
    assert [r.chunk_id for r in fused] == ["A", "C", "B", "D"]
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-9
    assert [r.rank for r in fused] == [1, 2, 3, 4]
    assert all(r.source == "hybrid" for r in fused)


def test_reciprocal_rank_fusion_dedupes_by_chunk_id():
    fused = reciprocal_rank_fusion([[_sr("A", 1, "vector")], [_sr("A", 1, "keyword")]], k=60)

    assert len(fused) == 1
    assert fused[0].chunk_id == "A"


def test_reciprocal_rank_fusion_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


# ---------------------------------------------------------------------------
# KnowledgeService / HybridRetriever (stubbed embedder/store)
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


def test_knowledge_service_search_applies_filters(tmp_path: Path):
    store = VectorStore(str(tmp_path / "kb.db"))
    doc_a = _make_doc(id="a.md", path="a.md", project="jarvis")
    doc_b = _make_doc(id="b.md", path="b.md", project="other")
    store.upsert_document(doc_a)
    store.upsert_document(doc_b)
    store.insert_chunks(
        [
            Chunk(id="ca", document_id="a.md", text="match text", position=0, embedding=[1.0, 0.0]),
            Chunk(id="cb", document_id="b.md", text="match text", position=0, embedding=[1.0, 0.0]),
        ]
    )
    service = KnowledgeService(store=store, embedder=StubEmbedder([1.0, 0.0]))

    matches = service.search("match text", filters={"project": "jarvis"})

    assert [m.document_id for m in matches] == ["a.md"]


def test_hybrid_search_exact_identifier_beats_misleading_vector_match(tmp_path: Path):
    """Day 12 plan item 10: keyword search should rescue an exact-identifier query
    even when the embedding happens to align better with an unrelated document."""
    store = VectorStore(str(tmp_path / "kb.db"))
    target = _make_doc(id="target.md", path="target.md")
    decoy = _make_doc(id="decoy.md", path="decoy.md")
    store.upsert_document(target)
    store.upsert_document(decoy)
    store.insert_chunks(
        [
            Chunk(
                id="c_target", document_id="target.md",
                text="The payment service uses authz_version to invalidate stale tokens.",
                position=0, embedding=[0.0, 1.0],
            ),
            Chunk(
                id="c_decoy", document_id="decoy.md",
                text="General overview of the authorization architecture for the platform.",
                position=0, embedding=[1.0, 0.0],
            ),
        ]
    )
    # The stub embedder aligns the query with the decoy, not the target - vector search
    # alone would rank decoy.md first.
    service = KnowledgeService(store=store, embedder=StubEmbedder([1.0, 0.0]))

    vector_only = service.vector_retriever.search("What is authz_version?", limit=5)
    assert vector_only[0].document_id == "decoy.md"

    hybrid = service.search("What is authz_version?")
    assert hybrid[0].document_id == "target.md"


def test_hybrid_search_preserves_paraphrase_with_no_lexical_overlap(tmp_path: Path):
    """Day 12 plan item 11: a paraphrased query with near-zero lexical overlap should
    still be found via vector search, and hybrid should retain that result."""
    store = VectorStore(str(tmp_path / "kb.db"))
    target = _make_doc(id="payment.md", path="payment.md")
    other = _make_doc(id="other.md", path="other.md")
    store.upsert_document(target)
    store.upsert_document(other)
    store.insert_chunks(
        [
            Chunk(
                id="c_target", document_id="payment.md",
                text="The system prevents duplicate payment processing using idempotency keys.",
                position=0, embedding=[1.0, 0.0],
            ),
            Chunk(
                id="c_other", document_id="other.md",
                text="Redis supports pub/sub messaging between services.",
                position=0, embedding=[0.0, 1.0],
            ),
        ]
    )
    service = KnowledgeService(store=store, embedder=StubEmbedder([1.0, 0.0]))
    query = "How do we avoid charging a customer more than once for one purchase?"

    keyword_only = service.keyword_retriever.search(query, limit=5)
    assert keyword_only == []  # no lexical overlap at all

    hybrid = service.search(query)
    assert hybrid[0].document_id == "payment.md"


def test_hybrid_search_mixed_query_favors_doc_strong_in_both_signals(tmp_path: Path):
    """Day 12 plan item 12: a doc that ranks well on both keyword and vector signals
    should beat docs that only rank well on one."""
    store = VectorStore(str(tmp_path / "kb.db"))
    both = _make_doc(id="authorization.md", path="authorization.md")
    vector_only = _make_doc(id="versioning.md", path="versioning.md")
    keyword_only = _make_doc(id="cache-notes.md", path="cache-notes.md")
    for doc in (both, vector_only, keyword_only):
        store.upsert_document(doc)
    store.insert_chunks(
        [
            Chunk(
                id="c_both", document_id="authorization.md",
                text="authz_version invalidates stale authorization tokens.",
                position=0, embedding=[0.5, 0.5],
            ),
            Chunk(
                id="c_vector", document_id="versioning.md",
                text="General API versioning conventions for the platform.",
                position=0, embedding=[1.0, 0.0],
            ),
            Chunk(
                id="c_keyword", document_id="cache-notes.md",
                text="authz_version keys are stored in the cache.",
                position=0, embedding=[0.0, 1.0],
            ),
        ]
    )
    service = KnowledgeService(store=store, embedder=StubEmbedder([0.7, 0.7]))

    hybrid = service.search("How does authz_version prevent stale authorization?")

    assert hybrid[0].document_id == "authorization.md"


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


def test_ingest_directory_indexes_frontmatter_metadata(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "a.md").write_text("---\nproject: jarvis\n---\nSome content.")
    store = VectorStore(str(tmp_path / "kb.db"))
    embedder = StubIngestEmbedder()

    ingest_directory(knowledge_dir, store=store, embedder=embedder)

    rows = store.all_chunks_with_documents(filters={"project": "jarvis"})
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def test_evaluate_retriever_computes_accuracy_percentages():
    ranked_by_query = {
        "q1": ["docA", "docB", "docC"],
        "q2": ["docX", "docB"],
    }

    def fake_search(query, limit):
        return [
            SearchResult(
                chunk_id=f"{doc_id}_c0", document_id=doc_id, title="t", path="p", content="c",
                position=0, score=1.0, rank=i + 1, source="hybrid",
            )
            for i, doc_id in enumerate(ranked_by_query[query])
        ]

    cases = [
        EvalCase(query="q1", expected_document_id="docB"),
        EvalCase(query="q2", expected_document_id="docB"),
    ]

    report = evaluate_retriever("test", cases, fake_search)

    assert report.cases == 2
    assert report.top1 == 0.0
    assert report.top3 == 100.0
    assert report.top5 == 100.0


def test_evaluate_retriever_handles_no_cases():
    report = evaluate_retriever("test", [], lambda q, limit: [])

    assert report.cases == 0
    assert report.top1 == report.top3 == report.top5 == 0.0


# ---------------------------------------------------------------------------
# Evaluation CLI (dataset loading)
# ---------------------------------------------------------------------------


def test_load_cases_reads_json_file(tmp_path: Path):
    import json

    from app.knowledge.evaluate import _load_cases

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps([{"query": "how does redis rate limit?", "expected_document_id": "notes/redis.md"}])
    )

    cases = _load_cases(cases_path)

    assert cases == [EvalCase(query="how does redis rate limit?", expected_document_id="notes/redis.md")]


def test_load_cases_returns_empty_for_missing_file(tmp_path: Path):
    from app.knowledge.evaluate import _load_cases

    assert _load_cases(tmp_path / "does-not-exist.json") == []


def test_shipped_eval_cases_file_is_valid():
    from app.knowledge.evaluate import _DEFAULT_CASES_PATH, _load_cases

    cases = _load_cases(_DEFAULT_CASES_PATH)

    assert len(cases) > 0
    assert all(case.query and case.expected_document_id for case in cases)
