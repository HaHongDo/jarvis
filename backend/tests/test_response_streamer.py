import threading

import numpy as np

from app.llm.base import LLM, StreamEvent, ToolCall
from app.pipeline.response_streamer import ResponseStreamer, should_flush
from app.tools.base import Tool
from app.tools.registry import ToolRegistry
from app.tts.base import TextToSpeech


def test_should_flush_requires_sentence_end_and_min_length():
    assert not should_flush("Yes", min_chunk_characters=20)
    assert not should_flush("Yes.", min_chunk_characters=20)
    assert should_flush("This is a long enough sentence.", min_chunk_characters=20)


def test_should_flush_ignores_empty_buffer():
    assert not should_flush("   ", min_chunk_characters=20)


class _StubLLM(LLM):
    """Yields pre-baked text chunks instead of calling a real model."""

    def __init__(self, chunks):
        self.chunks = chunks

    def chat(self, messages, tools=None):
        raise NotImplementedError

    def stream_chat(self, messages, tools=None):
        yield from (StreamEvent(content=chunk) for chunk in self.chunks)


class _StubTTS(TextToSpeech):
    """Encodes each synthesized chunk's index into its audio so playback order is verifiable."""

    def __init__(self):
        super().__init__(player=_RecordingPlayer())
        self.synthesized_texts = []

    def synthesize(self, text):
        self.synthesized_texts.append(text)
        return np.array([len(self.synthesized_texts) - 1], dtype=np.float32), 16000


class _RecordingPlayer:
    def __init__(self):
        self.played = []

    def play(self, audio, sample_rate):
        self.played.append(int(audio[0]))


def test_stream_response_accumulates_full_reply_and_plays_chunks_in_order():
    llm = _StubLLM(["This is sentence one. ", "This is sentence two. ", "Short."])
    tts = _StubTTS()
    streamer = ResponseStreamer(llm, tts, min_chunk_characters=10)

    reply, timings = streamer.stream_response([{"role": "user", "content": "hi"}])

    assert reply == "This is sentence one. This is sentence two. Short."
    assert "first_token" in timings
    assert "first_tts_chunk" in timings
    assert "first_audio" in timings
    assert tts.synthesized_texts == ["This is sentence one.", "This is sentence two.", "Short."]
    # The player must consume queued chunks strictly in the order they were produced.
    assert tts.player.played == [0, 1, 2]


def test_stream_response_invokes_on_first_audio_callback():
    llm = _StubLLM(["A short reply."])
    tts = _StubTTS()
    streamer = ResponseStreamer(llm, tts, min_chunk_characters=5)

    called = []
    streamer.stream_response(
        [{"role": "user", "content": "hi"}],
        on_first_audio=lambda: called.append(True),
    )

    assert called == [True]


def test_stream_response_stops_early_when_cancelled():
    def slow_events():
        yield StreamEvent(content="First sentence. ")
        cancel_event.set()
        yield StreamEvent(content="Second sentence that should not be produced.")

    class _CancellableLLM(LLM):
        def chat(self, messages, tools=None):
            raise NotImplementedError

        def stream_chat(self, messages, tools=None):
            yield from slow_events()

    cancel_event = threading.Event()
    llm = _CancellableLLM()
    tts = _StubTTS()
    streamer = ResponseStreamer(llm, tts, min_chunk_characters=5)

    reply, _timings = streamer.stream_response(
        [{"role": "user", "content": "hi"}],
        cancel_event=cancel_event,
    )

    assert reply == "First sentence. "


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes back the given value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def run(self, arguments):
        return arguments["value"]


class _ToolCallingLLM(LLM):
    """First round requests a tool call; second round answers using the tool result
    found in the messages it was given."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None):
        raise NotImplementedError

    def stream_chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            yield StreamEvent(tool_calls=[ToolCall(name="echo", arguments={"value": "42"})])
            return
        tool_message = next(m for m in messages if m.get("role") == "tool")
        yield StreamEvent(content=f"The tool said: {tool_message['content']}.")


def test_stream_response_executes_tool_call_and_continues_with_result():
    llm = _ToolCallingLLM()
    tts = _StubTTS()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    streamer = ResponseStreamer(llm, tts, min_chunk_characters=1, tool_registry=registry)
    messages = [{"role": "user", "content": "echo 42"}]

    reply, _timings = streamer.stream_response(messages)

    assert llm.calls == 2
    assert '"result": "42"' in reply
    # The tool exchange must be recorded in the conversation the LLM sees.
    assert any(m.get("role") == "tool" and m.get("name") == "echo" for m in messages)


def test_stream_response_gives_up_after_max_tool_rounds():
    class _LoopingLLM(LLM):
        def chat(self, messages, tools=None):
            raise NotImplementedError

        def stream_chat(self, messages, tools=None):
            yield StreamEvent(tool_calls=[ToolCall(name="echo", arguments={"value": "x"})])

    llm = _LoopingLLM()
    tts = _StubTTS()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    streamer = ResponseStreamer(llm, tts, tool_registry=registry, max_tool_rounds=2)

    reply, _timings = streamer.stream_response([{"role": "user", "content": "loop"}])

    assert "couldn't finish" in reply


def test_stream_response_reports_tool_error_without_crashing():
    class _FailingTool(Tool):
        name = "boom"
        description = "Always fails."
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, arguments):
            raise RuntimeError("kaboom")

    class _CallsFailingToolLLM(LLM):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            raise NotImplementedError

        def stream_chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield StreamEvent(tool_calls=[ToolCall(name="boom", arguments={})])
                return
            tool_message = next(m for m in messages if m.get("role") == "tool")
            yield StreamEvent(content=f"It failed: {tool_message['content']}.")

    llm = _CallsFailingToolLLM()
    tts = _StubTTS()
    registry = ToolRegistry()
    registry.register(_FailingTool())
    streamer = ResponseStreamer(llm, tts, min_chunk_characters=1, tool_registry=registry)

    reply, _timings = streamer.stream_response([{"role": "user", "content": "trigger the failure"}])

    assert '"success": false' in reply
    assert "kaboom" in reply


class _SearchStubTool(Tool):
    """Fake `search` tool that records every query it actually receives, so tests can
    verify the streamer's per-request budget and duplicate-query dedupe."""

    name = "search"
    description = "Fake search."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "time_range": {"type": "string"},
        },
        "required": ["query"],
    }

    def __init__(self):
        self.received_queries = []

    def run(self, arguments):
        self.received_queries.append(arguments["query"])
        return f"result for {arguments['query']}"


def _search_calling_llm(queries, rounds=None):
    """Builds a stub LLM that requests a search for each query in `queries` (one per
    round), then answers in plain text once all queries have been requested."""

    class _SearchLLM(LLM):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            raise NotImplementedError

        def stream_chat(self, messages, tools=None):
            self.calls += 1
            if self.calls <= len(queries):
                query = queries[self.calls - 1]
                yield StreamEvent(tool_calls=[ToolCall(name="search", arguments={"query": query})])
                return
            yield StreamEvent(content="Done searching.")

    return _SearchLLM()


def test_stream_response_enforces_search_budget_per_request():
    llm = _search_calling_llm(["q1", "q2", "q3", "q4"])
    tts = _StubTTS()
    registry = ToolRegistry()
    search_tool = _SearchStubTool()
    registry.register(search_tool)
    streamer = ResponseStreamer(
        llm, tts, min_chunk_characters=1, tool_registry=registry, max_tool_rounds=10, max_searches_per_request=3
    )

    reply, _timings = streamer.stream_response([{"role": "user", "content": "search a lot"}])

    assert search_tool.received_queries == ["q1", "q2", "q3"]
    assert "Done searching." in reply


def test_stream_response_deduplicates_repeated_search_queries():
    llm = _search_calling_llm(["latest rust release", "Latest   Rust Release", "latest rust release"])
    tts = _StubTTS()
    registry = ToolRegistry()
    search_tool = _SearchStubTool()
    registry.register(search_tool)
    streamer = ResponseStreamer(
        llm, tts, min_chunk_characters=1, tool_registry=registry, max_tool_rounds=10, max_searches_per_request=5
    )

    streamer.stream_response([{"role": "user", "content": "what's new with rust"}])

    # Only the first (normalized-unique) query actually reaches the tool.
    assert search_tool.received_queries == ["latest rust release"]


def test_stream_response_tracks_search_frequency_metric():
    tts = _StubTTS()
    registry = ToolRegistry()
    registry.register(_SearchStubTool())
    streamer = ResponseStreamer(
        llm=_search_calling_llm(["q1"]), tts=tts, min_chunk_characters=1, tool_registry=registry
    )

    # Request 1: triggers a search.
    streamer.stream_response([{"role": "user", "content": "search please"}])

    # Requests 2 and 3: no search needed.
    streamer.llm = _StubLLM(["No search needed."])
    streamer.stream_response([{"role": "user", "content": "hi"}])
    streamer.llm = _StubLLM(["Still no search."])
    streamer.stream_response([{"role": "user", "content": "hi again"}])

    assert streamer.total_requests == 3
    assert streamer.search_requests == 1


class _FetchPageStubTool(Tool):
    """Fake `fetch_page` tool that records every URL it actually receives, so tests can
    verify the streamer's per-request budget and duplicate-URL dedupe."""

    name = "fetch_page"
    description = "Fake fetch_page."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def __init__(self):
        self.received_urls = []

    def run(self, arguments):
        self.received_urls.append(arguments["url"])
        return f"content for {arguments['url']}"


def _fetch_page_calling_llm(urls):
    """Builds a stub LLM that requests a fetch_page for each URL in `urls` (one per
    round), then answers in plain text once all URLs have been requested."""

    class _FetchPageLLM(LLM):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            raise NotImplementedError

        def stream_chat(self, messages, tools=None):
            self.calls += 1
            if self.calls <= len(urls):
                url = urls[self.calls - 1]
                yield StreamEvent(tool_calls=[ToolCall(name="fetch_page", arguments={"url": url})])
                return
            yield StreamEvent(content="Done fetching.")

    return _FetchPageLLM()


def test_stream_response_enforces_page_fetch_budget_per_request():
    llm = _fetch_page_calling_llm(
        ["https://example.com/1", "https://example.com/2", "https://example.com/3", "https://example.com/4"]
    )
    tts = _StubTTS()
    registry = ToolRegistry()
    fetch_tool = _FetchPageStubTool()
    registry.register(fetch_tool)
    streamer = ResponseStreamer(
        llm,
        tts,
        min_chunk_characters=1,
        tool_registry=registry,
        max_tool_rounds=10,
        max_page_fetches_per_request=3,
    )

    reply, _timings = streamer.stream_response([{"role": "user", "content": "fetch a lot"}])

    assert fetch_tool.received_urls == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert "Done fetching." in reply


def test_stream_response_deduplicates_repeated_page_fetch_urls():
    llm = _fetch_page_calling_llm(
        [
            "https://Example.com/a/?utm_source=x",
            "https://example.com/a",
            "https://example.com/a",
        ]
    )
    tts = _StubTTS()
    registry = ToolRegistry()
    fetch_tool = _FetchPageStubTool()
    registry.register(fetch_tool)
    streamer = ResponseStreamer(
        llm, tts, min_chunk_characters=1, tool_registry=registry, max_tool_rounds=10, max_page_fetches_per_request=5
    )

    streamer.stream_response([{"role": "user", "content": "fetch the same page"}])

    # Only the first (normalized-unique) URL actually reaches the tool.
    assert fetch_tool.received_urls == ["https://Example.com/a/?utm_source=x"]


def test_stream_response_tracks_page_fetch_frequency_metric():
    tts = _StubTTS()
    registry = ToolRegistry()
    registry.register(_FetchPageStubTool())
    streamer = ResponseStreamer(
        llm=_fetch_page_calling_llm(["https://example.com/1"]),
        tts=tts,
        min_chunk_characters=1,
        tool_registry=registry,
    )

    # Request 1: triggers a fetch.
    streamer.stream_response([{"role": "user", "content": "fetch please"}])

    # Requests 2 and 3: no fetch needed.
    streamer.llm = _StubLLM(["No fetch needed."])
    streamer.stream_response([{"role": "user", "content": "hi"}])
    streamer.llm = _StubLLM(["Still no fetch."])
    streamer.stream_response([{"role": "user", "content": "hi again"}])

    assert streamer.total_requests == 3
    assert streamer.page_fetch_requests == 1


class _ResearchStubTool(Tool):
    """Fake `research` tool that records every query it actually receives, so tests
    can verify the streamer's per-request budget and duplicate-query dedupe."""

    name = "research"
    description = "Fake research."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self):
        self.received_queries = []

    def run(self, arguments):
        self.received_queries.append(arguments["query"])
        return f"research context for {arguments['query']}"


def _research_calling_llm(queries):
    """Builds a stub LLM that requests `research` for each query in `queries` (one per
    round), then answers in plain text once all queries have been requested."""

    class _ResearchLLM(LLM):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            raise NotImplementedError

        def stream_chat(self, messages, tools=None):
            self.calls += 1
            if self.calls <= len(queries):
                query = queries[self.calls - 1]
                yield StreamEvent(tool_calls=[ToolCall(name="research", arguments={"query": query})])
                return
            yield StreamEvent(content="Done researching.")

    return _ResearchLLM()


def test_stream_response_enforces_research_budget_per_request():
    llm = _research_calling_llm(["go vs rust", "redis vs postgres", "one more topic"])
    tts = _StubTTS()
    registry = ToolRegistry()
    research_tool = _ResearchStubTool()
    registry.register(research_tool)
    streamer = ResponseStreamer(
        llm,
        tts,
        min_chunk_characters=1,
        tool_registry=registry,
        max_tool_rounds=10,
        max_research_per_request=2,
    )

    reply, _timings = streamer.stream_response([{"role": "user", "content": "compare a lot of things"}])

    assert research_tool.received_queries == ["go vs rust", "redis vs postgres"]
    assert "Done researching." in reply


def test_stream_response_deduplicates_repeated_research_queries():
    llm = _research_calling_llm(["  Go VS Rust  ", "go vs rust", "go vs rust"])
    tts = _StubTTS()
    registry = ToolRegistry()
    research_tool = _ResearchStubTool()
    registry.register(research_tool)
    streamer = ResponseStreamer(
        llm, tts, min_chunk_characters=1, tool_registry=registry, max_tool_rounds=10, max_research_per_request=5
    )

    streamer.stream_response([{"role": "user", "content": "compare the same thing repeatedly"}])

    # Only the first (normalized-unique) query actually reaches the tool.
    assert research_tool.received_queries == ["  Go VS Rust  "]


def test_stream_response_tracks_research_frequency_metric():
    tts = _StubTTS()
    registry = ToolRegistry()
    registry.register(_ResearchStubTool())
    streamer = ResponseStreamer(
        llm=_research_calling_llm(["go vs rust"]),
        tts=tts,
        min_chunk_characters=1,
        tool_registry=registry,
    )

    # Request 1: triggers research.
    streamer.stream_response([{"role": "user", "content": "compare please"}])

    # Requests 2 and 3: no research needed.
    streamer.llm = _StubLLM(["No research needed."])
    streamer.stream_response([{"role": "user", "content": "hi"}])
    streamer.llm = _StubLLM(["Still no research."])
    streamer.stream_response([{"role": "user", "content": "hi again"}])

    assert streamer.total_requests == 3
    assert streamer.research_requests == 1
