import logging

from .audio import AudioRecorder
from .config import MAX_RECORDING_SECONDS, SYSTEM_PROMPT
from .llm import OllamaConnectionError, OllamaLLM, OllamaModelNotFoundError
from .state import AssistantState
from .stt import FasterWhisperSTT
from .tts import KokoroTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit"}


def _set_state(state: AssistantState) -> None:
    logger.info("[STATE] %s", state.name)


def run():
    llm = OllamaLLM()
    recorder = AudioRecorder()
    stt = FasterWhisperSTT()
    tts = KokoroTTS()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Jarvis is ready.")
    print()
    _set_state(AssistantState.IDLE)

    while True:
        command = input("Press Enter to speak (or type 'exit')... ").strip()

        if command.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            break

        _set_state(AssistantState.RECORDING)
        print("[Recording...]")
        audio = recorder.record(MAX_RECORDING_SECONDS)

        _set_state(AssistantState.TRANSCRIBING)
        user_input = stt.transcribe(audio)

        if not user_input:
            print("(no speech detected)")
            print()
            _set_state(AssistantState.IDLE)
            continue

        print(f"You: {user_input}")

        messages.append({"role": "user", "content": user_input})

        _set_state(AssistantState.THINKING)
        print("[Thinking...]")
        try:
            reply = llm.chat(messages)
        except OllamaConnectionError as exc:
            print(exc)
            messages.pop()
            _set_state(AssistantState.IDLE)
            continue
        except OllamaModelNotFoundError as exc:
            print(exc)
            messages.pop()
            _set_state(AssistantState.IDLE)
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"Jarvis: {reply}")

        _set_state(AssistantState.SPEAKING)
        tts.speak(reply)

        print()
        _set_state(AssistantState.IDLE)


if __name__ == "__main__":
    run()
