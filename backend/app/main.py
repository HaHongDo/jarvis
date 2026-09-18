import logging
import threading
import time

from .audio import AudioRecorder, MicrophoneStream
from .config import (
    MAX_RECORDING_SECONDS,
    SILENCE_TIMEOUT_MS,
    SPEECH_DEBUG_LOGGING,
    SPEECH_LLM_FALLBACK_ENABLED,
    SYSTEM_PROMPT,
    TTS_MIN_CHUNK_CHARACTERS,
    WAKEWORD_ACTIVATION_DELAY_MS,
)
from .llm import OllamaConnectionError, OllamaLLM, OllamaModelNotFoundError
from .pipeline import ResponseStreamer
from .speech import LLMCorrectionFallback, SpeechNormalizer
from .state import AssistantState
from .stt import FasterWhisperSTT
from .tools import default_registry
from .tts import KokoroTTS
from .wakeword import OpenWakeWordDetector, WakeWordDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _set_state(state: AssistantState) -> None:
    logger.info("[STATE] %s", state.name)


def _listen_for_wake_word(wakeword: WakeWordDetector) -> None:
    """Block until the wake word is detected. Only runs while LISTENING."""
    with MicrophoneStream() as mic:
        for frame in mic:
            if wakeword.process(frame):
                return


def _log_latency(stt_latency: float, timings: dict) -> None:
    first_token = timings.get("first_token")
    first_tts_chunk = timings.get("first_tts_chunk")
    first_audio = timings.get("first_audio")

    logger.info("[LATENCY] STT: %.2fs", stt_latency)
    if first_token is not None:
        logger.info("[LATENCY] LLM first-token: %.2fs", first_token)
    if first_tts_chunk is not None and first_token is not None:
        logger.info("[LATENCY] TTS first-chunk: %.2fs", first_tts_chunk - first_token)
    if first_audio is not None:
        logger.info("[LATENCY] Time-to-first-audio: %.2fs", stt_latency + first_audio)


def run():
    llm = OllamaLLM()
    recorder = AudioRecorder()
    stt = FasterWhisperSTT()
    tts = KokoroTTS()
    wakeword = OpenWakeWordDetector()
    tool_registry = default_registry()
    speech_normalizer = SpeechNormalizer(
        llm_fallback=LLMCorrectionFallback(llm) if SPEECH_LLM_FALLBACK_ENABLED else None,
        debug=SPEECH_DEBUG_LOGGING,
    )
    streamer = ResponseStreamer(
        llm, tts, min_chunk_characters=TTS_MIN_CHUNK_CHARACTERS, tool_registry=tool_registry
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Jarvis is ready. Say 'Hey Jarvis' to begin (Ctrl+C to quit).")
    print()

    try:
        while True:
            _set_state(AssistantState.LISTENING)
            _listen_for_wake_word(wakeword)
            print("[Wake word detected]")

            time.sleep(WAKEWORD_ACTIVATION_DELAY_MS / 1000)

            _set_state(AssistantState.RECORDING)
            print("[Listening for a command...]")
            audio = recorder.record_command(MAX_RECORDING_SECONDS, SILENCE_TIMEOUT_MS)
            t0 = time.perf_counter()  # user stopped speaking

            _set_state(AssistantState.TRANSCRIBING)
            user_input = stt.transcribe(audio)
            stt_latency = time.perf_counter() - t0

            if not user_input:
                print("(no speech detected)")
                print()
                wakeword.reset()
                continue

            normalization = speech_normalizer.normalize(user_input)
            user_input = normalization.normalized
            if normalization.changes:
                logger.info("[SpeechNormalizer] %s -> %s", normalization.original, normalization.normalized)

            print(f"You: {user_input}")

            messages.append({"role": "user", "content": user_input})

            _set_state(AssistantState.THINKING)
            print("[Thinking...]")
            cancel_event = threading.Event()
            messages_before_turn = len(messages)
            try:
                reply, timings = streamer.stream_response(
                    messages,
                    cancel_event=cancel_event,
                    on_first_audio=lambda: _set_state(AssistantState.SPEAKING),
                )
            except OllamaConnectionError as exc:
                print(exc)
                del messages[messages_before_turn:]
                wakeword.reset()
                continue
            except OllamaModelNotFoundError as exc:
                print(exc)
                del messages[messages_before_turn:]
                wakeword.reset()
                continue

            messages.append({"role": "assistant", "content": reply})
            print(f"Jarvis: {reply}")

            _log_latency(stt_latency, timings)

            print()
            wakeword.reset()
    except KeyboardInterrupt:
        print()
        print("Goodbye.")


if __name__ == "__main__":
    run()
