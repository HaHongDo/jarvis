from app.tools import CalculatorTool, SearchTool, TimeTool, ToolRegistry, default_registry
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


def test_default_registry_includes_all_day6_tools():
    registry = default_registry()

    names = {schema["function"]["name"] for schema in registry.schemas()}

    assert names == {"calculator", "get_time", "search"}
