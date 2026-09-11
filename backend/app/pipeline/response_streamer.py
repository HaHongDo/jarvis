import logging
import queue
import threading
import time
import uuid
from typing import Callable, Optional

from ..config import (
    MAX_PAGE_FETCHES_PER_REQUEST,
    MAX_RESEARCH_PER_REQUEST,
    MAX_SEARCHES_PER_REQUEST,
    MAX_TOOL_ROUNDS,
    TTS_MIN_CHUNK_CHARACTERS,
)
from ..llm.base import LLM, ToolCall
from ..search.normalizer import normalize_query, normalize_url
from ..tts.base import TextToSpeech
from ..tts.preprocessing import preprocess_for_speech
from ..tools.base import ToolResult
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SENTENCE_END_CHARS = (".", "?", "!")

_QUEUE_POLL_SECONDS = 0.1

_MAX_ROUNDS_ERROR_MESSAGE = (
    "I tried using tools a few times but couldn't finish. Could you rephrase that?"
)

_SEARCH_TOOL_NAME = "search"
_SEARCH_BUDGET_EXHAUSTED_MESSAGE = (
    f"Search budget exhausted for this request (max {MAX_SEARCHES_PER_REQUEST} searches)."
)

_FETCH_PAGE_TOOL_NAME = "fetch_page"
_PAGE_FETCH_BUDGET_EXHAUSTED_MESSAGE = (
    f"Page-fetch budget exhausted for this request (max {MAX_PAGE_FETCHES_PER_REQUEST} fetches)."
)

_RESEARCH_TOOL_NAME = "research"
_RESEARCH_BUDGET_EXHAUSTED_MESSAGE = (
    f"Research budget exhausted for this request (max {MAX_RESEARCH_PER_REQUEST} calls)."
)


def should_flush(buffer: str, min_chunk_characters: int = TTS_MIN_CHUNK_CHARACTERS) -> bool:
    """Simple punctuation-based chunk boundary, held back until a minimum length."""
    stripped = buffer.strip()
    if not stripped:
        return False
    return stripped.endswith(_SENTENCE_END_CHARS) and len(stripped) >= min_chunk_characters


class ResponseStreamer:
    """Streams an LLM reply sentence-by-sentence into TTS and plays audio back in order.

    Three stages run concurrently:
      LLM producer (this thread) -> text queue -> TTS worker -> audio queue -> player worker
    so TTS/playback for earlier sentences overlaps with generation of later ones, while
    the player still consumes its queue strictly in order to guarantee sequential playback.

    When a tool registry is supplied, the producer also runs the tool-calling loop: if the
    model asks for a tool instead of answering, nothing is enqueued for TTS (raw tool output
    must never reach TTS directly), the tool is executed, and the model is asked again with
    the result added to the conversation - repeating until it answers in plain text or
    `max_tool_rounds` is exceeded.

    Each call to `stream_response` is one user turn ("request"). Within a turn, calls to
    the `search` tool are capped at `max_searches_per_request` and de-duplicated by
    normalized query + time_range, calls to the `fetch_page` tool are capped at
    `max_page_fetches_per_request` and de-duplicated by normalized URL, and calls to the
    `research` tool are capped at `max_research_per_request` and de-duplicated by
    normalized query + time_range, so a model that loops or asks near-identical
    questions can't spam the search backend, refetch the same page, or re-run an
    expensive multi-source research call; the streamer also tracks how many turns
    actually used search/fetch_page/research for the `search_requests` /
    `page_fetch_requests` / `research_requests` over `total_requests` metrics.
    """

    def __init__(
        self,
        llm: LLM,
        tts: TextToSpeech,
        min_chunk_characters: int = TTS_MIN_CHUNK_CHARACTERS,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        max_searches_per_request: int = MAX_SEARCHES_PER_REQUEST,
        max_page_fetches_per_request: int = MAX_PAGE_FETCHES_PER_REQUEST,
        max_research_per_request: int = MAX_RESEARCH_PER_REQUEST,
    ):
        self.llm = llm
        self.tts = tts
        self.min_chunk_characters = min_chunk_characters
        self.tool_registry = tool_registry
        self.max_tool_rounds = max_tool_rounds
        self.max_searches_per_request = max_searches_per_request
        self.max_page_fetches_per_request = max_page_fetches_per_request
        self.max_research_per_request = max_research_per_request
        self.total_requests = 0
        self.search_requests = 0
        self.page_fetch_requests = 0
        self.research_requests = 0

    def stream_response(
        self,
        messages: list[dict],
        cancel_event: Optional[threading.Event] = None,
        on_first_audio: Optional[Callable[[], None]] = None,
    ) -> tuple[str, dict]:
        cancel_event = cancel_event or threading.Event()
        text_queue: queue.Queue = queue.Queue()
        audio_queue: queue.Queue = queue.Queue()
        timings: dict = {}
        start = time.perf_counter()

        tts_thread = threading.Thread(
            target=self._tts_worker,
            args=(text_queue, audio_queue, cancel_event, timings, start),
            daemon=True,
        )
        player_thread = threading.Thread(
            target=self._player_worker,
            args=(audio_queue, cancel_event, timings, start, on_first_audio),
            daemon=True,
        )
        tts_thread.start()
        player_thread.start()

        request_id = uuid.uuid4().hex[:8]
        search_state = {
            "request_id": request_id,
            "count": 0,
            "seen": {},
            "page_count": 0,
            "page_seen": {},
            "research_count": 0,
            "research_seen": {},
        }

        try:
            assistant_response = self._produce_text(
                messages, text_queue, cancel_event, timings, start, search_state
            )
        finally:
            text_queue.put(None)
            tts_thread.join()
            audio_queue.put(None)
            player_thread.join()

        self.total_requests += 1
        if search_state["count"] > 0:
            self.search_requests += 1
        if search_state["page_count"] > 0:
            self.page_fetch_requests += 1
        if search_state["research_count"] > 0:
            self.research_requests += 1
        logger.info(
            "[METRICS] search_requests=%d/%d page_fetch_requests=%d/%d research_requests=%d/%d total",
            self.search_requests,
            self.total_requests,
            self.page_fetch_requests,
            self.total_requests,
            self.research_requests,
            self.total_requests,
        )
        logger.info(
            "[LATENCY] LLM first-token: %.2fs | TTS first-chunk: %.2fs | first-audio: %.2fs",
            timings.get("first_token", -1.0),
            timings.get("first_tts_chunk", -1.0),
            timings.get("first_audio", -1.0),
        )
        return assistant_response, timings

    def _produce_text(
        self,
        messages: list[dict],
        text_queue: "queue.Queue",
        cancel_event: threading.Event,
        timings: dict,
        start: float,
        search_state: dict,
    ) -> str:
        tools = self.tool_registry.schemas() if self.tool_registry else None
        buffer = ""
        assistant_response = ""

        for _round in range(self.max_tool_rounds):
            pending_tool_calls: list[ToolCall] = []

            for event in self.llm.stream_chat(messages, tools=tools):
                if cancel_event.is_set():
                    return assistant_response

                if event.has_tool_calls():
                    pending_tool_calls.extend(event.tool_calls)
                    continue

                if not event.content:
                    continue

                if "first_token" not in timings:
                    timings["first_token"] = time.perf_counter() - start

                buffer += event.content
                assistant_response += event.content

                if should_flush(buffer, self.min_chunk_characters):
                    text_queue.put(buffer)
                    buffer = ""

            if not pending_tool_calls:
                break

            if cancel_event.is_set():
                return assistant_response

            self._execute_tool_round(messages, pending_tool_calls, search_state)
        else:
            logger.error("Max tool-call rounds (%d) exceeded", self.max_tool_rounds)
            buffer += _MAX_ROUNDS_ERROR_MESSAGE
            assistant_response += _MAX_ROUNDS_ERROR_MESSAGE

        if buffer and not cancel_event.is_set():
            text_queue.put(buffer)

        return assistant_response

    def _execute_tool_round(
        self, messages: list[dict], tool_calls: list[ToolCall], search_state: dict
    ) -> None:
        """Runs every requested tool sequentially and appends the assistant's tool-call
        message plus each tool's result to `messages`, so the next LLM call sees them."""
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": call.name, "arguments": call.arguments}} for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            content = self._execute_tool_call(call, search_state)
            messages.append({"role": "tool", "name": call.name, "content": content})

    def _execute_tool_call(self, call: ToolCall, search_state: dict) -> str:
        if call.name == _SEARCH_TOOL_NAME:
            return self._execute_search_call(call, search_state)
        if call.name == _FETCH_PAGE_TOOL_NAME:
            return self._execute_fetch_page_call(call, search_state)
        if call.name == _RESEARCH_TOOL_NAME:
            return self._execute_research_call(call, search_state)
        return self.tool_registry.execute(call.name, call.arguments).to_content()

    def _execute_search_call(self, call: ToolCall, search_state: dict) -> str:
        """Applies the per-request search budget and duplicate-query detection before
        delegating to the real search tool (see Day 8: search budget + dedup)."""
        request_id = search_state["request_id"]
        query = call.arguments.get("query", "")
        dedupe_key = (normalize_query(query), call.arguments.get("time_range"))

        cached_content = search_state["seen"].get(dedupe_key)
        if cached_content is not None:
            logger.info("SEARCH request=%s query=%r decision=DUPLICATE (reusing prior result)", request_id, query)
            return cached_content

        if search_state["count"] >= self.max_searches_per_request:
            logger.warning(
                "SEARCH request=%s query=%r decision=BUDGET_EXHAUSTED (max %d)",
                request_id,
                query,
                self.max_searches_per_request,
            )
            return ToolResult(success=False, error=_SEARCH_BUDGET_EXHAUSTED_MESSAGE).to_content()

        search_state["count"] += 1
        logger.info(
            "SEARCH request=%s query=%r decision=SEARCH round=%d",
            request_id,
            query,
            search_state["count"],
        )
        content = self.tool_registry.execute(call.name, call.arguments).to_content()
        search_state["seen"][dedupe_key] = content
        return content

    def _execute_fetch_page_call(self, call: ToolCall, search_state: dict) -> str:
        """Applies the per-request page-fetch budget and duplicate-URL detection before
        delegating to the real fetch_page tool (see Day 9: fetch budget + dedup)."""
        request_id = search_state["request_id"]
        url = call.arguments.get("url", "")
        dedupe_key = normalize_url(url) if url else url

        cached_content = search_state["page_seen"].get(dedupe_key)
        if cached_content is not None:
            logger.info(
                "FETCH_PAGE request=%s url=%r decision=DUPLICATE (reusing prior result)", request_id, url
            )
            return cached_content

        if search_state["page_count"] >= self.max_page_fetches_per_request:
            logger.warning(
                "FETCH_PAGE request=%s url=%r decision=BUDGET_EXHAUSTED (max %d)",
                request_id,
                url,
                self.max_page_fetches_per_request,
            )
            return ToolResult(success=False, error=_PAGE_FETCH_BUDGET_EXHAUSTED_MESSAGE).to_content()

        search_state["page_count"] += 1
        logger.info(
            "FETCH_PAGE request=%s url=%r decision=FETCH round=%d",
            request_id,
            url,
            search_state["page_count"],
        )
        content = self.tool_registry.execute(call.name, call.arguments).to_content()
        search_state["page_seen"][dedupe_key] = content
        return content

    def _execute_research_call(self, call: ToolCall, search_state: dict) -> str:
        """Applies the per-request research budget and duplicate-query detection before
        delegating to the real research tool (see Day 10: research budget + dedup)."""
        request_id = search_state["request_id"]
        query = call.arguments.get("query", "")
        dedupe_key = (normalize_query(query), call.arguments.get("time_range"))

        cached_content = search_state["research_seen"].get(dedupe_key)
        if cached_content is not None:
            logger.info(
                "RESEARCH request=%s query=%r decision=DUPLICATE (reusing prior result)",
                request_id,
                query,
            )
            return cached_content

        if search_state["research_count"] >= self.max_research_per_request:
            logger.warning(
                "RESEARCH request=%s query=%r decision=BUDGET_EXHAUSTED (max %d)",
                request_id,
                query,
                self.max_research_per_request,
            )
            return ToolResult(success=False, error=_RESEARCH_BUDGET_EXHAUSTED_MESSAGE).to_content()

        search_state["research_count"] += 1
        logger.info(
            "RESEARCH request=%s query=%r decision=RESEARCH round=%d",
            request_id,
            query,
            search_state["research_count"],
        )
        content = self.tool_registry.execute(call.name, call.arguments).to_content()
        search_state["research_seen"][dedupe_key] = content
        return content

    def _tts_worker(
        self,
        text_queue: "queue.Queue",
        audio_queue: "queue.Queue",
        cancel_event: threading.Event,
        timings: dict,
        start: float,
    ) -> None:
        while True:
            if cancel_event.is_set():
                return
            try:
                text = text_queue.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue

            if text is None:
                return
            if cancel_event.is_set():
                return

            clean_text = preprocess_for_speech(text)
            if not clean_text:
                continue

            try:
                audio, sample_rate = self.tts.synthesize(clean_text)
            except Exception:
                logger.exception("TTS synthesis failed for chunk, skipping")
                continue

            if "first_tts_chunk" not in timings:
                timings["first_tts_chunk"] = time.perf_counter() - start

            if audio.size and not cancel_event.is_set():
                audio_queue.put((audio, sample_rate))

    def _player_worker(
        self,
        audio_queue: "queue.Queue",
        cancel_event: threading.Event,
        timings: dict,
        start: float,
        on_first_audio: Optional[Callable[[], None]],
    ) -> None:
        while True:
            if cancel_event.is_set():
                return
            try:
                item = audio_queue.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue

            if item is None:
                return
            if cancel_event.is_set():
                return

            if "first_audio" not in timings:
                timings["first_audio"] = time.perf_counter() - start
                if on_first_audio is not None:
                    on_first_audio()

            audio, sample_rate = item
            self.tts.player.play(audio, sample_rate)
