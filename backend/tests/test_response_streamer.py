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
