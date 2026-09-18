from app.tools import (
    CalculatorTool,
    FetchPageTool,
    KnowledgeSearchTool,
    ResearchTool,
    SearchTool,
    TimeTool,
    ToolRegistry,
    default_registry,
)
from app.tools.base import ToolValidationError


def test_calculator_evaluates_expression():
    result = CalculatorTool().execute({"expression": "123 * 456"})

    assert result.success
    assert result.result == 123 * 456


def test_calculator_rejects_invalid_expression():
    result = CalculatorTool().execute({"expression": "import os"})

    assert not result.success
    assert result.error


def test_calculator_does_not_execute_arbitrary_code():
    # __import__ etc. must never be reachable through the expression evaluator.
    result = CalculatorTool().execute({"expression": "__import__('os').system('echo hi')"})

    assert not result.success


def test_calculator_validates_missing_argument():
    result = CalculatorTool().execute({})

    assert not result.success
    assert "expression" in result.error


def test_time_tool_defaults_to_utc():
    result = TimeTool().execute({})

    assert result.success
    assert result.result["timezone"] == "UTC"


def test_time_tool_supports_named_timezone():
    result = TimeTool().execute({"timezone": "Asia/Ho_Chi_Minh"})

    assert result.success
    assert result.result["timezone"] == "Asia/Ho_Chi_Minh"


def test_time_tool_rejects_unknown_timezone():
    result = TimeTool().execute({"timezone": "Not/AZone"})

    assert not result.success


def test_search_tool_returns_context_text(monkeypatch):
    from app.search import SearchContextBuilder, SearchResponse, SearchResult

    class StubService:
        def search(self, query, **kwargs):
            return SearchResponse(
                query=query,
                results=[SearchResult(title=f"About {query}", url="https://example.com", snippet="...")],
                result_count=1,
                searched_at="2026-08-30T00:00:00+00:00",
            )

        def build_context(self, response):
            return SearchContextBuilder().build(response.query, response.results)

    result = SearchTool(service=StubService()).execute({"query": "Redis"})

    assert result.success
    assert "Redis" in result.result


def test_search_tool_forwards_time_range_and_freshness():
    from app.search import SearchContextBuilder, SearchResponse, SearchResult

    class RecordingService:
        def __init__(self):
            self.calls = []

        def search(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return SearchResponse(
                query=query,
                results=[SearchResult(title="X", url="https://example.com", snippet="...")],
                result_count=1,
                searched_at="2026-08-30T00:00:00+00:00",
            )

        def build_context(self, response):
            return SearchContextBuilder().build(response.query, response.results)

    service = RecordingService()
    SearchTool(service=service).execute(
        {"query": "bitcoin price", "time_range": "day", "freshness": "realtime"}
    )

    assert service.calls == [("bitcoin price", {"time_range": "day", "freshness": "realtime"})]


def test_search_tool_defaults_freshness_to_normal():
    from app.search import SearchContextBuilder, SearchResponse, SearchResult

    class RecordingService:
        def __init__(self):
            self.calls = []

        def search(self, query, **kwargs):
            self.calls.append(kwargs)
            return SearchResponse(query=query, results=[], result_count=0, searched_at="")

        def build_context(self, response):
            return SearchContextBuilder().build(response.query, response.results)

    service = RecordingService()
    SearchTool(service=service).execute({"query": "anything"})

    assert service.calls == [{"time_range": None, "freshness": "normal"}]


def test_search_tool_handles_backend_errors_gracefully():
    from app.search import SearXNGError

    class FailingService:
        def search(self, query, **kwargs):
            raise SearXNGError("boom")

    result = SearchTool(service=FailingService()).execute({"query": "Redis"})

    assert result.success
    assert "couldn't reach" in result.result


def test_knowledge_search_tool_returns_context_text():
    from app.knowledge import KnowledgeContextBuilder, SearchResult

    match = SearchResult(
        chunk_id="c0", document_id="d0", title="Redis Notes", path="notes/redis.md",
        content="Redis can implement rate limiting.", position=0, score=0.9, rank=1, source="hybrid",
    )

    class StubService:
        def search(self, query, **kwargs):
            return [match]

        def build_context(self, query, matches):
            return KnowledgeContextBuilder(max_context_chars=1000).build(query, matches)

    result = KnowledgeSearchTool(service=StubService()).execute({"query": "rate limiting"})

    assert result.success
    assert "Redis Notes" in result.result


def test_knowledge_search_tool_handles_no_matches():
    class StubService:
        def search(self, query, **kwargs):
            return []

        def build_context(self, query, matches):
            raise AssertionError("build_context should not be called with no matches")

    result = KnowledgeSearchTool(service=StubService()).execute({"query": "anything"})

    assert result.success
    assert "No relevant information" in result.result


def test_knowledge_search_tool_validates_missing_query():
    result = KnowledgeSearchTool(service=object()).execute({})

    assert not result.success
    assert "query" in result.error


def test_fetch_page_tool_returns_context_text():
    from datetime import datetime, timezone

    from app.webpage import PageContextBuilder, WebPage

    class StubService:
        def fetch(self, url):
            return WebPage(
                url=url, title="Example", text="Some content.", fetched_at=datetime.now(timezone.utc)
            )

        def build_context(self, page):
            return PageContextBuilder().build(page)

    result = FetchPageTool(service=StubService()).execute({"url": "https://example.com"})

    assert result.success
    assert "Example" in result.result
    assert "Some content." in result.result


def test_fetch_page_tool_requires_url_argument():
    result = FetchPageTool(service=None).execute({})

    assert not result.success
    assert "url" in result.error


def test_research_tool_returns_context_text():
    from app.research.models import ResearchContext, Source

    class StubService:
        def research(self, query, **kwargs):
            source = Source(id="src_001", url="https://example.com", title="Redis 8", content="c")
            return ResearchContext(
                query=query, sources=[source], context_text=f"## Research Results\n\n[1] {source.title}"
            )

    result = ResearchTool(service=StubService()).execute({"query": "compare Redis and PostgreSQL"})

    assert result.success
    assert "Redis 8" in result.result


def test_research_tool_forwards_time_range_and_freshness():
    from app.research.models import ResearchContext

    class RecordingService:
        def __init__(self):
            self.calls = []

        def research(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return ResearchContext(query=query, sources=[], context_text="")

    service = RecordingService()
    ResearchTool(service=service).execute(
        {"query": "go 1.26 changes", "time_range": "month", "freshness": "recent"}
    )

    assert service.calls == [("go 1.26 changes", {"time_range": "month", "freshness": "recent"})]


def test_research_tool_reports_no_sources_gracefully():
    from app.research.models import ResearchContext

    class StubService:
        def research(self, query, **kwargs):
            return ResearchContext(query=query)

    result = ResearchTool(service=StubService()).execute({"query": "anything"})

    assert result.success
    assert "couldn't retrieve" in result.result


def test_fetch_page_tool_handles_ssrf_error():
    from app.webpage import URLSafetyError

    class FailingService:
        def fetch(self, url):
            raise URLSafetyError("unsafe address")

    result = FetchPageTool(service=FailingService()).execute({"url": "http://127.0.0.1"})

    assert result.success
    assert "unsafe" in result.result.lower()


def test_fetch_page_tool_handles_unsupported_content_type():
    from app.webpage import UnsupportedContentTypeError

    class FailingService:
        def fetch(self, url):
            raise UnsupportedContentTypeError("nope")

    result = FetchPageTool(service=FailingService()).execute({"url": "https://example.com/file.pdf"})

    assert result.success
    assert "content type" in result.result.lower()


def test_fetch_page_tool_handles_generic_fetch_error():
    from app.webpage import PageFetchError

    class FailingService:
        def fetch(self, url):
            raise PageFetchError("boom")

    result = FetchPageTool(service=FailingService()).execute({"url": "https://example.com"})

    assert result.success
    assert "couldn't fetch" in result.result.lower()


def test_tool_validate_arguments_checks_type():
    tool = CalculatorTool()

    try:
        tool.validate_arguments({"expression": 123})
        assert False, "expected ToolValidationError"
    except ToolValidationError:
        pass


def test_registry_executes_registered_tool():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    result = registry.execute("calculator", {"expression": "1 + 1"})

    assert result.success
    assert result.result == 2


def test_registry_reports_unknown_tool_without_raising():
    registry = ToolRegistry()

    result = registry.execute("does_not_exist", {})

    assert not result.success
    assert "Unknown tool" in result.error


def test_registry_schemas_expose_name_description_parameters():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    schemas = registry.schemas()

    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "calculator"
    assert "parameters" in schemas[0]["function"]


def test_default_registry_includes_all_day11_tools():
    registry = default_registry()

    names = {schema["function"]["name"] for schema in registry.schemas()}

    assert names == {"calculator", "get_time", "search", "fetch_page", "research", "search_knowledge"}
