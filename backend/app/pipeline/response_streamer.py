import logging
import queue
import threading
import time
from typing import Callable, Optional

from ..config import MAX_TOOL_ROUNDS, TTS_MIN_CHUNK_CHARACTERS
from ..llm.base import LLM, ToolCall
from ..tts.base import TextToSpeech
from ..tts.preprocessing import preprocess_for_speech
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SENTENCE_END_CHARS = (".", "?", "!")

_QUEUE_POLL_SECONDS = 0.1

_MAX_ROUNDS_ERROR_MESSAGE = (
    "I tried using tools a few times but couldn't finish. Could you rephrase that?"
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
    """

    def __init__(
        self,
        llm: LLM,
        tts: TextToSpeech,
        min_chunk_characters: int = TTS_MIN_CHUNK_CHARACTERS,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ):
        self.llm = llm
        self.tts = tts
        self.min_chunk_characters = min_chunk_characters
        self.tool_registry = tool_registry
        self.max_tool_rounds = max_tool_rounds

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

        try:
            assistant_response = self._produce_text(messages, text_queue, cancel_event, timings, start)
        finally:
            text_queue.put(None)
            tts_thread.join()
            audio_queue.put(None)
            player_thread.join()

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

            self._execute_tool_round(messages, pending_tool_calls)
        else:
            logger.error("Max tool-call rounds (%d) exceeded", self.max_tool_rounds)
            buffer += _MAX_ROUNDS_ERROR_MESSAGE
            assistant_response += _MAX_ROUNDS_ERROR_MESSAGE

        if buffer and not cancel_event.is_set():
            text_queue.put(buffer)

        return assistant_response

    def _execute_tool_round(self, messages: list[dict], tool_calls: list[ToolCall]) -> None:
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
            result = self.tool_registry.execute(call.name, call.arguments)
            messages.append({"role": "tool", "name": call.name, "content": result.to_content()})

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
