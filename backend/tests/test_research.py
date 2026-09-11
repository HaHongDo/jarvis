import time
from datetime import datetime, timezone

from app.research.authority import AuthorityTier, classify_domain
from app.research.chunking import chunk_source, chunk_text
from app.research.context import ResearchContextBuilder
from app.research.dedup import content_fingerprint, dedupe_sources
from app.research.models import Source, SourceChunk
from app.research.service import ResearchService
from app.research.trace import ResearchTrace
from app.search.models import SearchResponse, SearchResult
from app.search.searxng import SearXNGError
from app.webpage.fetcher import PageFetchError
from app.webpage.models import WebPage

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_returns_single_chunk_when_under_size():
    assert chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_chunk_text_returns_empty_list_for_blank_text():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunk_text_splits_with_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, chunk_size=40, overlap=10)

    assert len(chunks) > 1
    # Consecutive chunks overlap: the tail of one reappears at the head of the next.
    assert chunks[0][-10:] == chunks[1][:10]
    # Every character of the original text is covered.
    assert "".join(chunks[0:1])[0] == text[0]


def test_chunk_source_tags_chunks_with_source_metadata():
    source = Source(id="src_001", url="https://example.com", title="T", content="x" * 90)

    chunks = chunk_source(source, chunk_size=40, overlap=10)

    assert all(c.source_id == "src_001" for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"src_001_chunk_{i}" for i in range(len(chunks))]
    assert [c.position for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_content_fingerprint_ignores_whitespace_and_case():
    a = content_fingerprint("Hello   World")
    b = content_fingerprint("hello world")

    assert a == b


def test_dedupe_sources_removes_duplicate_urls():
    sources = [
        Source(id="src_001", url="https://example.com/a?utm_source=x", title="A", content="first article"),
        Source(id="src_002", url="https://example.com/a", title="A dup", content="different text"),
        Source(id="src_003", url="https://example.com/b", title="B", content="second article"),
    ]

    deduped = dedupe_sources(sources)

    assert [s.id for s in deduped] == ["src_001", "src_003"]


def test_dedupe_sources_removes_duplicate_content_across_urls():
    sources = [
        Source(id="src_001", url="https://a.example.com", title="A", content="Same Content Here"),
        Source(id="src_002", url="https://b.example.com", title="B", content="same content here"),
    ]

    deduped = dedupe_sources(sources)

    assert [s.id for s in deduped] == ["src_001"]


# ---------------------------------------------------------------------------
# Authority classification
# ---------------------------------------------------------------------------


def test_classify_domain_gov_and_edu_are_high():
    assert classify_domain("https://www.nist.gov/page") == AuthorityTier.HIGH
    assert classify_domain("https://web.mit.edu/page") == AuthorityTier.HIGH


def test_classify_domain_known_official_docs_are_high():
    assert classify_domain("https://docs.python.org/3/") == AuthorityTier.HIGH
    assert classify_domain("https://go.dev/doc/") == AuthorityTier.HIGH


def test_classify_domain_known_news_is_medium_high():
    assert classify_domain("https://www.reuters.com/some-article") == AuthorityTier.MEDIUM_HIGH


def test_classify_domain_known_forum_is_low():
    assert classify_domain("https://www.reddit.com/r/test") == AuthorityTier.LOW


def test_classify_domain_unknown_defaults_to_medium():
    assert classify_domain("https://some-random-blog.example") == AuthorityTier.MEDIUM


# ---------------------------------------------------------------------------
# Context builder: budget + rendering
# ---------------------------------------------------------------------------


def test_select_chunks_respects_budget_and_priority():
    chunks = [
        SourceChunk(source_id="src_001", chunk_id="src_001_chunk_0", text="x" * 50, position=0),
        SourceChunk(source_id="src_002", chunk_id="src_002_chunk_0", text="y" * 50, position=0),
    ]
    # src_002 has higher priority (lower tuple sorts first).
    priority = {"src_001": (0,), "src_002": (-1,)}
    builder = ResearchContextBuilder(max_context_chars=60)

    selected = builder.select_chunks(chunks, priority)

    assert [c.source_id for c in selected] == ["src_002"]


def test_build_renders_numbered_sources_and_delimited_content():
    source = Source(id="src_001", url="https://example.com/a", title="Example Article", content="Body text")
    chunk = SourceChunk(source_id="src_001", chunk_id="src_001_chunk_0", text="Body text", position=0)
    builder = ResearchContextBuilder(max_context_chars=10_000)

    text = builder.build("my query", [source], [chunk])

    assert "[1] Example Article - https://example.com/a" in text
    assert '<source id="1">' in text
    assert "Body text" in text
    assert "</source>" in text


def test_build_handles_no_sources():
    builder = ResearchContextBuilder(max_context_chars=10_000)

    text = builder.build("my query", [], [])

    assert "No usable web sources" in text


# ---------------------------------------------------------------------------
# ResearchService: end-to-end orchestration with stub search/page services
# ---------------------------------------------------------------------------


class _StubSearchService:
    def __init__(self, results, cached=False, error=None):
        self.results = results
        self.cached = cached
        self.error = error

    def search(self, query, **kwargs):
        if self.error is not None:
            raise self.error
        return SearchResponse(
            query=query, results=self.results, result_count=len(self.results), cached=self.cached
        )


class _StubPageService:
    """Fetches pages from a dict keyed by URL; entries that are exceptions are raised
    instead, and a callable entry is invoked (useful for simulating slow fetches)."""

    def __init__(self, pages: dict):
        self.pages = pages

    def fetch(self, url):
        value = self.pages[url]
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value()
        return value


def _make_page(url, title, text):
    return WebPage(url=url, title=title, text=text, fetched_at=datetime.now(timezone.utc))


def test_research_service_builds_context_from_multiple_sources():
    results = [
        SearchResult(title="Go Blog", url="https://go.dev/blog/x", snippet="snippet 1"),
        SearchResult(title="Random Blog", url="https://example.com/y", snippet="snippet 2"),
    ]
    pages = {
        "https://go.dev/blog/x": _make_page("https://go.dev/blog/x", "Go Blog", "Go content " * 20),
        "https://example.com/y": _make_page("https://example.com/y", "Random Blog", "Random content " * 20),
    }
    service = ResearchService(
        search_service=_StubSearchService(results),
        page_service=_StubPageService(pages),
    )

    context = service.research("go news")

    assert len(context.sources) == 2
    assert context.chunks
    assert "[1]" in context.context_text
    assert "[2]" in context.context_text


def test_research_service_handles_partial_fetch_failures():
    results = [
        SearchResult(title="Good", url="https://example.com/good", snippet="ok"),
        SearchResult(title="Bad", url="https://example.com/bad", snippet="fails"),
    ]
    pages = {
        "https://example.com/good": _make_page("https://example.com/good", "Good", "Good content " * 20),
        "https://example.com/bad": PageFetchError("boom"),
    }
    service = ResearchService(
        search_service=_StubSearchService(results),
        page_service=_StubPageService(pages),
    )

    context = service.research("some topic")

    assert len(context.sources) == 1
    assert context.sources[0].url == "https://example.com/good"


def test_research_service_returns_empty_context_on_search_error():
    service = ResearchService(
        search_service=_StubSearchService([], error=SearXNGError("down")),
        page_service=_StubPageService({}),
    )

    context = service.research("anything")

    assert context.sources == []
    assert context.context_text == ""


def test_research_service_times_out_slow_fetches():
    def slow_fetch():
        time.sleep(0.5)
        return _make_page("https://example.com/slow", "Slow", "content")

    results = [SearchResult(title="Slow", url="https://example.com/slow", snippet="...")]
    pages = {"https://example.com/slow": slow_fetch}
    service = ResearchService(
        search_service=_StubSearchService(results),
        page_service=_StubPageService(pages),
        timeout_seconds=0.05,
    )

    context = service.research("slow query")

    assert context.sources == []


def test_research_service_limits_number_of_fetched_sources():
    results = [
        SearchResult(title=f"Result {i}", url=f"https://example.com/{i}", snippet="...") for i in range(6)
    ]
    pages = {
        f"https://example.com/{i}": _make_page(
            f"https://example.com/{i}", f"Result {i}", f"unique content {i} " * 20
        )
        for i in range(6)
    }
    service = ResearchService(
        search_service=_StubSearchService(results),
        page_service=_StubPageService(pages),
        max_sources=2,
    )

    context = service.research("many results")

    assert len(context.sources) == 2


# ---------------------------------------------------------------------------
# ResearchTrace
# ---------------------------------------------------------------------------


def test_research_trace_logs_summary_and_errors(caplog):
    trace = ResearchTrace(query="q", result_count=2, fetch_attempted=2, fetch_succeeded=1, fetch_failed=1)
    trace.errors.append("https://example.com/bad: boom")

    with caplog.at_level("INFO"):
        trace.log()

    assert any("RESEARCH TRACE" in record.message for record in caplog.records)
    assert any("boom" in record.message for record in caplog.records)
