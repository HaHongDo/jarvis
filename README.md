# Jarvis

A local, voice-driven assistant:

```text
Microphone -> faster-whisper (STT) -> Ollama (LLM) -> Kokoro (TTS) -> Speaker
```

Everything runs locally — no cloud APIs.

## Quick start

Prerequisites:

- Python 3.10+
- [Ollama](https://docs.ollama.com/quickstart) installed and running
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) installed (used by Kokoro for TTS phonemes)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull gemma3

python -m app.main
```

Press Enter to speak, wait for the transcription and spoken reply, and type `exit` (or `quit`) to end the session.

See [`backend/README.md`](backend/README.md) for configuration (`config.yaml`, env vars), running tests, and trying individual pieces (mic, STT, TTS) standalone.
